"""Post-processing of a shift snapshot into the tensors this package's agents read."""

from __future__ import annotations

import numpy as np

from example.control import GroundControl
from example.satellite import NeighborView
from orbi.env import EDGE_FEATURES, JOB_FEATURES, NEIGHBOR_FEATURES, PLAN_MAX, SAT_FEATURES
from tools.observation import Observation, collect

_HORIZON = 24.0
_WORK_SCALE = 4.0


class ShiftObservation(Observation):
    def encode(self, env) -> dict:
        return self._present(env, collect(env, self._control(env)))

    def _control(self, env) -> GroundControl:
        if env.ground_control is None:
            env.ground_control = GroundControl(top_k=env.K, sat_ids=env.cluster)
        return env.ground_control

    def _neighbors(self, env) -> NeighborView:
        if env.neighbor_view is None:
            env.neighbor_view = NeighborView(method=env.neighbor_method, memory=env.neighbor_memory)
        return env.neighbor_view

    def _job_features(self, env, job: dict, k: int) -> tuple[float, ...]:
        done = 1.0 - job["remaining_steps"] / job["work_steps"]
        slack = job["deadline_step"] - k - job["remaining_steps"]
        to_release = max(job["release_step"] - k, 0)
        priority = job["priority"]
        return (
            float(priority == 1),
            float(priority == 2),
            float(priority == 3),
            float(job.get("value_usd", 0.0)) / 100.0,
            done,
            min(job["remaining_steps"], _WORK_SCALE) / _WORK_SCALE,
            float(np.clip(slack / _WORK_SCALE, -1.0, 1.0)),
            min(max(job["deadline_step"] - k, 0), _HORIZON) / _HORIZON,
            float(job["kind"] == "downlink"),
            float(job["kind"] == "relay"),
            len(job["eligible_satellites"]) / env.n_sats,
            float(env._last_executor.get(job["id"]) is not None),
            min(to_release, _HORIZON) / _HORIZON,
        )

    def _present(self, env, raw: dict) -> dict[str, np.ndarray]:
        k = raw["k"]

        # Fixed-size tables. Rows past the real fleet and the packed slots stay zero.
        sats = np.zeros((env.N, SAT_FEATURES), dtype=np.float32)
        jobs = np.zeros((env.K, JOB_FEATURES), dtype=np.float32)
        plan = np.zeros((PLAN_MAX, JOB_FEATURES), dtype=np.float32)
        edges = np.zeros((env.N, env.K, EDGE_FEATURES), dtype=np.float32)
        valid = env.calibration_valid
        for i, row in enumerate(raw["fleet"]):
            sats[i] = (
                row["soc"],
                row["soc"] - env.reserve_pct / 100.0,
                row["temp_c"] / 50.0,
                row["calibration_age_steps"] / valid,
                np.clip((valid - row["calibration_age_steps"]) / valid, -1.0, 1.0),
                float(row["available"]),
                float(row["contact"]),
                float(row["contact"]),
                row["solar_w"] / 100.0,
                min(row["to_dark"], _HORIZON) / _HORIZON,
                min(row["to_sun"], _HORIZON) / _HORIZON,
                row["capacity_wh"] / 200.0,
                float(row["heater_w"] > 0.0),
                (row["solar_w"] - row["base_w"] - row["heater_w"]) / 100.0,
            )

        # The plan is every unfinished job. Slots are only the ones the dispatcher packed.
        for row, job in enumerate(raw["plan_jobs"][:PLAN_MAX]):
            plan[row] = self._job_features(env, job, k)
        for slot, item in enumerate(raw["slots"]):
            jobs[slot] = self._job_features(env, item["job"], k)
            for edge in item["edges"]:
                i = edge["sat"]
                edges[i, slot] = (
                    1.0,
                    float(edge["ok"]),
                    float(edge["reason"] != "energy_reserve"),
                    float(edge["sees"]),
                    float(edge["last_executor"]),
                )

        # Fleet totals, then each satellite's neighbor memory.
        soc = raw["soc"]
        glob = np.array(
            [
                env.k / env.steps_total,
                min(len(raw["open_jobs"]), 64) / 64.0,
                min(raw["n_prio3"], 32) / 32.0,
                min(raw["downlink_demand"] / env.downlink_limit, 2.0),
                float(soc.mean()) if env.n_sats else 0.0,
                float(soc.min()) if env.n_sats else 0.0,
            ],
            dtype=np.float32,
        )
        view = self._neighbors(env)
        neighbors = (
            np.zeros((env.N, view.memory, NEIGHBOR_FEATURES), dtype=np.float32)
            if raw["finished"]
            else view.encode_all(env, raw["mask"])
        )
        return {
            "sats": sats,
            "jobs": jobs,
            "plan": plan,
            "edges": edges,
            "neighbors": neighbors,
            "global": glob,
            "mask": raw["mask"].astype(np.float32),
        }
