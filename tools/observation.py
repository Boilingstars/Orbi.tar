"""Observation collected from one shift tick.

`collect` reads the simulator: fleet state, open jobs, legal actions, and
visibility. A user package turns that snapshot into the tensors its agents
read. The simulator only calls `encode`.
"""

from __future__ import annotations

import numpy as np

from orbi.satellite import job_points, job_visible


class Observation:
    def encode(self, env) -> dict:
        raise NotImplementedError("Provide an observation encoder")

    def known_jobs(self, env) -> list[dict]:
        k = env.k
        return [
            j
            for j in env.jobs.values()
            if j["completed_step"] is None
            and j["remaining_steps"] > 0
            and j["deadline_step"] > k
        ]


def job_view(env, job: dict, sat_ids: list[str] | None = None) -> dict:
    """How the fleet sees one job now. Ranking uses who has contact; windows live on edges."""
    ids = list(sat_ids) if sat_ids is not None else list(job.get("eligible_satellites") or env.sat_ids)
    points = job_points(job)
    addressed = job.get("addressed") is not None and not points
    period = float(np.mean(env.world.periods_s)) if env.n_sats else 1.0
    dt = max(float(getattr(env, "dt", 300.0)), 1.0)
    visible: list[int] = []
    windows: dict[int, tuple[float, float]] = {}
    remainings: list[float] = []
    for sid in ids:
        sat = env.sats.get(sid)
        if sat is None or not sat.available(env.k):
            continue

        if addressed:
            if int(sat.index) != int(job["addressed"]):
                continue
            visible.append(sat.index)
            windows[sat.index] = (dt, 0.0)
            remainings.append(dt)
            continue

        if not points or not job_visible(env.world, sat.index, job):
            continue

        visible.append(sat.index)

        # Ranking uses the pass length only for a job that is open on this tick.
        # A later slot still asks access_windows for its own edge.
        if not (job["release_step"] <= env.k < job["deadline_step"]):
            continue
        remaining, _nxt = env.world.access_windows(sat.index, points, env.world.t)
        windows[sat.index] = (float(remaining), 0.0)
        remainings.append(float(remaining))

    n_visible = len(visible)
    return {
        "n_visible": n_visible,
        "remaining_s": float(min(remainings) if remainings else 0.0),
        "next_s": 0.0 if n_visible else period,
        "visible": visible,
        "windows": windows,
    }


def collect(env, control) -> dict:
    """Snapshot of the shift at `env.k`. Also stores the action mask and job slots on `env`."""
    k = min(env.k, env.steps_total - 1)
    finished = env.k >= env.steps_total
    dt = max(env.dt, 1.0)
    period = float(np.mean(env.world.periods_s)) if env.n_sats else dt

    # Live jobs only. Misses stay in `env.jobs` for the summary, not the ranker.
    known = [] if finished else [
        j
        for j in env.jobs.values()
        if j["completed_step"] is None and j["remaining_steps"] > 0 and j["deadline_step"] > env.k
    ]
    open_jobs = [j for j in known if j["release_step"] <= env.k < j["deadline_step"]]

    # Admissibility counts live for this snapshot only. `reset` must not see them.
    env._executable_cache = {}
    env._execute_cache = {}
    env.world._visible_cache = {}
    try:
        views = {} if finished else {job["id"]: job_view(env, job, control.sat_ids) for job in known}
        env._job_views = views
        candidates = [] if finished else control.select_jobs(env, known)
        env._candidates = candidates
        packed = {cand.job_id for cand in candidates}
        return _snapshot(env, control, k, finished, dt, period, known, open_jobs, views, candidates, packed)
    finally:
        env._executable_cache = None
        env._execute_cache = None
        env.world._visible_cache = None


def _snapshot(env, control, k, finished, dt, period, known, open_jobs, views, candidates, packed) -> dict:
    """Fleet rows, legal mask, and packed edges. The admissibility cache is still open."""
    # One row per satellite: charge, contact, and time until the light changes.
    fleet = []
    soc = np.zeros(env.n_sats, dtype=np.float32)
    for i, sat in enumerate(env.fleet):
        soc[i] = sat.soc
        solar = sat.panel_w() if env.world.in_sun(i) else 0.0
        heater = sat.heater_w if sat.temp_c < env.model["heater_below_c"] else 0.0
        tts = env.world.time_to_sun_change(i) / dt
        if env.world.in_sun(i):
            to_dark, to_sun = tts, 0.0
        else:
            to_dark, to_sun = 0.0, tts
        can_calibrate = False
        if not finished:
            can_calibrate, _, _ = env.can_execute(sat.sat_id, {"action": "calibrate"})
        fleet.append(
            {
                "soc": float(soc[i]),
                "temp_c": float(sat.temp_c),
                "calibration_age_steps": int(sat.calibration_age_steps),
                "available": bool(sat.available(env.k)),
                "contact": bool(env.world.sees_any(i, env.active_points)),
                "solar_w": float(solar),
                "heater_w": float(heater),
                "to_dark": float(to_dark),
                "to_sun": float(to_sun),
                "capacity_wh": float(sat.capacity_wh),
                "base_w": float(sat.base_w),
                "can_calibrate": bool(can_calibrate),
            }
        )

    # Idle is always legal. A job slot is legal only where can_execute accepts it.
    mask = np.zeros((env.N, env.K + 2), dtype=bool)
    mask[:, env.idle_index] = True
    slots = []
    downlink_demand = 0
    n_prio3 = 0
    plan_jobs: list[dict] = []
    if not finished:
        for i, row in enumerate(fleet):
            mask[i, env.calibrate_index] = row["can_calibrate"]
        n_prio3 = sum(j["priority"] == 3 for j in open_jobs)
        ranked = sorted(known, key=lambda job: control.job_rank(env, job))
        plan_jobs = [job for job in ranked if job["id"] not in packed]
        for slot, cand in enumerate(candidates):
            job = cand.job
            view = views.get(job["id"]) or job_view(env, job, control.sat_ids)
            uniqueness = view["n_visible"] / max(env.n_sats, 1)
            points = job_points(job)
            addressed = job.get("addressed") is not None and not points
            sat_windows = view.get("windows") or {}
            edges = []
            for i in cand.sats:
                sat = env.fleet[i]
                ok, reason, _ = env.can_execute(sat.sat_id, {"action": "job", "job_id": job["id"]})
                sees = i in view["visible"] if "visible" in view else bool(job_visible(env.world, i, job))
                stored = sat_windows.get(i)

                # A satellite already in view keeps that pass. The others get time to the next rise.
                if stored is not None:
                    remaining_s, next_s = stored
                elif addressed or not points:
                    remaining_s, next_s = (dt, 0.0) if sees else (0.0, period)
                else:
                    remaining_s, next_s = env.world.access_windows(i, points, env.world.t)
                edges.append(
                    {
                        "sat": i,
                        "ok": bool(ok),
                        "reason": reason,
                        "sees": sees,
                        "last_executor": env._last_executor.get(job["id"]) == sat.sat_id,
                        "remaining_s": float(remaining_s),
                        "next_s": float(next_s),
                        "uniqueness": float(uniqueness),
                    }
                )
                mask[i, slot] = ok
            if job["kind"] == "downlink" and mask[:, slot].any():
                downlink_demand += 1
            slots.append({"job": job, "edges": edges, "view": view})

        # Queue pressure for the episode log. Counts are already on this snapshot.
        if getattr(env, "_queue", None) is not None:
            counts = env._executable_cache or {}
            exec_ids = [job_id for job_id, n in counts.items() if n > 0]
            env._queue["ticks"] += 1.0
            env._queue["feasible_edges"] += float(mask[: env.n_sats, : env.K].sum())
            env._queue["executable_jobs"] += float(len(exec_ids))
            env._queue["executors"] += float(sum(counts[job_id] for job_id in exec_ids))
            env._queue["outside_slots"] += float(sum(job_id not in packed for job_id in exec_ids))
    env._mask = mask
    return {
        "k": k,
        "finished": finished,
        "open_jobs": open_jobs,
        "plan_jobs": plan_jobs,
        "slots": slots,
        "fleet": fleet,
        "soc": soc,
        "mask": mask,
        "n_prio3": n_prio3,
        "downlink_demand": downlink_demand,
        "period_s": period,
    }
