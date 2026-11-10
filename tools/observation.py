"""Observation collected from one shift tick.

`collect` reads the simulator: fleet state, open jobs, legal actions, and
visibility. A user package turns that snapshot into the tensors its agents
read. The simulator only calls `encode`.
"""

from __future__ import annotations

import numpy as np

from orbi.satellite import job_visible


class Observation:
    def encode(self, env) -> dict:
        raise NotImplementedError("Provide an observation encoder")

    def known_jobs(self, env) -> list[dict]:
        return [j for j in env.jobs.values() if j["completed_step"] is None and j["remaining_steps"] > 0]


def collect(env, control) -> dict:
    """Snapshot of the shift at `env.k`. Also stores the action mask and job slots on `env`."""
    k = min(env.k, env.steps_total - 1)
    finished = env.k >= env.steps_total

    # Open jobs and the K slots the dispatcher packed for this tick.
    known = [] if finished else [
        j for j in env.jobs.values() if j["completed_step"] is None and j["remaining_steps"] > 0
    ]
    open_jobs = [j for j in known if j["release_step"] <= env.k < j["deadline_step"]]
    candidates = [] if finished else control.select_jobs(env, known)
    env._candidates = candidates

    # One row per satellite: charge, contact, and time until the light changes.
    step_s = max(env.dt, 1.0)
    fleet = []
    soc = np.zeros(env.n_sats, dtype=np.float32)
    for i, sat in enumerate(env.fleet):
        soc[i] = sat.soc
        solar = sat.panel_w() if env.world.in_sun(i) else 0.0
        heater = sat.heater_w if sat.temp_c < env.model["heater_below_c"] else 0.0
        tts = env.world.time_to_sun_change(i) / step_s
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
        plan_jobs = sorted(known, key=lambda job: control.job_rank(env, job))
        for slot, cand in enumerate(candidates):
            job = cand.job
            edges = []
            for i in cand.sats:
                sat = env.fleet[i]
                ok, reason, _ = env.can_execute(sat.sat_id, {"action": "job", "job_id": job["id"]})
                edges.append(
                    {
                        "sat": i,
                        "ok": bool(ok),
                        "reason": reason,
                        "sees": bool(job_visible(env.world, i, job)),
                        "last_executor": env._last_executor.get(job["id"]) == sat.sat_id,
                    }
                )
                mask[i, slot] = ok
            if job["kind"] == "downlink" and mask[:, slot].any():
                downlink_demand += 1
            slots.append({"job": job, "edges": edges})
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
    }
