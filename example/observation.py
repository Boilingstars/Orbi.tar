"""Shift observation: satellite rows, job slots, edges, and the action mask."""

from __future__ import annotations

import numpy as np

from example.control import GroundControl
from example.satellite import NeighborView
from orbi.env import (
    EDGE_FEATURES,
    JOB_FEATURES,
    NEIGHBOR_FEATURES,
    Observation,
    PLAN_MAX,
    SAT_FEATURES,
)

_HORIZON = 24.0
_WORK_SCALE = 4.0


class ShiftObservation(Observation):
    def encode(self, env) -> dict:
        return _Host(env)._observe()

    def known_jobs(self, env) -> list[dict]:
        return [j for j in env.jobs.values() if j["completed_step"] is None and j["remaining_steps"] > 0]


class _Host:
    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_env"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_env"), name, value)

    def _control(self) -> GroundControl:
        env = object.__getattribute__(self, "_env")
        if env.ground_control is None:
            env.ground_control = GroundControl(top_k=env.K, sat_ids=env.cluster)
        return env.ground_control

    def _neighbors(self) -> NeighborView:
        env = object.__getattribute__(self, "_env")
        if env.neighbor_view is None:
            env.neighbor_view = NeighborView(method=env.neighbor_method, memory=env.neighbor_memory)
        return env.neighbor_view

    def _job_rank(self, job: dict) -> tuple:
        return self._control().job_rank(object.__getattribute__(self, "_env"), job)

    def _job_features(self, job: dict, k: int) -> tuple[float, ...]:
        done = 1.0 - job["remaining_steps"] / job["work_steps"]
        slack = job["deadline_step"] - k - job["remaining_steps"]
        to_release = max(job["release_step"] - k, 0)
        priority = job["priority"]
        return (
            float(priority == 1),
            float(priority == 2),
            float(priority == 3),
            job["value_usd"] / 100.0,
            done,
            min(job["remaining_steps"], _WORK_SCALE) / _WORK_SCALE,
            float(np.clip(slack / _WORK_SCALE, -1.0, 1.0)),
            min(max(job["deadline_step"] - k, 0), _HORIZON) / _HORIZON,
            float(job["kind"] == "downlink"),
            float(job["kind"] == "relay"),
            len(job["eligible_satellites"]) / self.n_sats,
            float(self._last_executor.get(job["id"]) is not None),
            min(to_release, _HORIZON) / _HORIZON,
        )

    def _observe(self) -> dict[str, np.ndarray]:
        env = object.__getattribute__(self, "_env")
        k = min(env.k, env.steps_total - 1)
        finished = env.k >= env.steps_total
        known = [] if finished else [
            j for j in env.jobs.values() if j["completed_step"] is None and j["remaining_steps"] > 0
        ]
        open_jobs = [j for j in known if j["release_step"] <= env.k < j["deadline_step"]]
        self._candidates = [] if finished else self._control().select_jobs(env, known)

        sats = np.zeros((self.N, SAT_FEATURES), dtype=np.float32)
        jobs = np.zeros((self.K, JOB_FEATURES), dtype=np.float32)
        plan = np.zeros((PLAN_MAX, JOB_FEATURES), dtype=np.float32)
        edges = np.zeros((self.N, self.K, EDGE_FEATURES), dtype=np.float32)
        mask = np.zeros((self.N, self.K + 2), dtype=bool)
        mask[:, self.idle_index] = True

        soc = np.zeros(self.n_sats, dtype=np.float32)
        step_s = max(env.dt, 1.0)
        for i, sat in enumerate(env.fleet):
            soc[i] = sat.soc
            solar = sat.panel_w() if env.world.in_sun(i) else 0.0
            heater = sat.heater_w if sat.temp_c < env.model["heater_below_c"] else 0.0
            tts = env.world.time_to_sun_change(i) / step_s
            if env.world.in_sun(i):
                to_dark, to_sun = tts, 0.0
            else:
                to_dark, to_sun = 0.0, tts
            contact = float(env.world.sees_any(i, env.active_points))
            sats[i] = (
                soc[i],
                soc[i] - self.reserve_pct / 100.0,
                sat.temp_c / 50.0,
                sat.calibration_age_steps / self.calibration_valid,
                np.clip((self.calibration_valid - sat.calibration_age_steps) / self.calibration_valid, -1.0, 1.0),
                float(sat.available(env.k)),
                contact,
                contact,
                solar / 100.0,
                min(to_dark, _HORIZON) / _HORIZON,
                min(to_sun, _HORIZON) / _HORIZON,
                sat.capacity_wh / 200.0,
                float(heater > 0.0),
                (solar - sat.base_w - heater) / 100.0,
            )

        if not finished:
            for i, sat in enumerate(env.fleet):
                ok, _, _ = env.can_execute(sat.sat_id, {"action": "calibrate"})
                mask[i, self.calibrate_index] = ok
            n_prio3 = sum(j["priority"] == 3 for j in open_jobs)
            downlink_demand = 0
            ranked_plan = sorted(known, key=self._job_rank)
            for row, job in enumerate(ranked_plan[:PLAN_MAX]):
                plan[row] = self._job_features(job, k)
            for slot, cand in enumerate(self._candidates):
                j = cand.job
                jobs[slot] = self._job_features(j, k)
                for i in cand.sats:
                    sat = env.fleet[i]
                    ok, reason, _ = env.can_execute(sat.sat_id, {"action": "job", "job_id": j["id"]})
                    sees = float(env.world.sees_job(i, j.get("point_a"), j.get("point_b")))
                    edges[i, slot] = (
                        1.0,
                        float(ok),
                        float(reason != "energy_reserve"),
                        sees,
                        float(self._last_executor.get(j["id"]) == sat.sat_id),
                    )
                    mask[i, slot] = ok
                if j["kind"] == "downlink" and mask[:, slot].any():
                    downlink_demand += 1
        else:
            n_prio3 = 0
            downlink_demand = 0

        glob = np.array(
            [
                env.k / self.steps_total,
                min(len(open_jobs), 64) / 64.0,
                min(n_prio3, 32) / 32.0,
                min(downlink_demand / self.downlink_limit, 2.0),
                float(soc.mean()) if self.n_sats else 0.0,
                float(soc.min()) if self.n_sats else 0.0,
                float(self.goal == "priority"),
                float(self.goal == "revenue"),
            ],
            dtype=np.float32,
        )
        self._mask = mask
        view = self._neighbors()
        neighbors = (
            np.zeros((self.N, view.memory, NEIGHBOR_FEATURES), dtype=np.float32)
            if finished
            else view.encode_all(env, mask)
        )
        return {
            "sats": sats,
            "jobs": jobs,
            "plan": plan,
            "edges": edges,
            "neighbors": neighbors,
            "global": glob,
            "mask": mask.astype(np.float32),
        }
