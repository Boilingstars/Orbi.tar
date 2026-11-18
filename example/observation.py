"""Post-processing of a shift snapshot into the tensors this package's agents read."""

from __future__ import annotations

import numpy as np

from example.control import GroundControl
from example.jobs import PRICE_DIV, work_span_ticks
from example.reward import job_worth
from example.satellite import NeighborView
from orbi.env import (
    EDGE_FEATURES,
    GLOBAL_DOWNLINK_CAP,
    GLOBAL_FEATURES,
    JOB_FEATURES,
    N_MAX,
    NEIGHBOR_FEATURES,
    PLAN_MAX,
    SAT_FEATURES,
)
from orbi.satellite import downlink_required_steps
from tools.observation import Observation, collect

# Sun and eclipse windows are a fraction of an orbit, not of the shift.
_LIGHT_HORIZON = 24.0
_WORTH_OBS = 100.0 / PRICE_DIV

# Fixed v2 scales. Not estimated from test episodes.
VALUE_NORM = 8.0
RHO_NORM = 4.0
JOB_FEATURES_V2 = 18
EDGE_FEATURES_V2 = 15


def observation_from_version(version: str = "v1"):
    if version in ("v1", "", None):
        return ShiftObservation()

    if version == "v2":
        return ShiftObservationV2()
    raise ValueError(f"unknown observation version {version}")


def _asinh_unit(value: float, scale: float) -> float:
    return float(np.arcsinh(float(value) / max(float(scale), 1e-6)) / np.arcsinh(1.0))


class ShiftObservation(Observation):
    version = "v1"
    job_features = JOB_FEATURES
    edge_features = EDGE_FEATURES

    def encode(self, env) -> dict:
        raw = collect(env, self._control(env))
        for cand in getattr(env, "_candidates", []) or []:
            cand.job["ever_packed"] = True
        return self._present(env, raw)

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
        span = max(float(env.steps_total), 1.0)
        work = max(float(job["work_steps"]), 1.0)
        slack = (job["deadline_step"] - k - job["remaining_steps"]) / span
        slack_work = (job["deadline_step"] - k - job["remaining_steps"]) / work
        to_release = max(job["release_step"] - k, 0) / span
        time_left = max(job["deadline_step"] - k, 0) / span
        priority = job["priority"]
        fleet = max(env.n_sats, 1)
        return (
            float(priority == 1),
            float(priority == 2),
            float(priority == 3),
            float(np.clip(job_worth(job) / _WORTH_OBS, -5.0, 5.0)),
            done,
            job["remaining_steps"] / work_span_ticks(env),
            float(np.clip(slack, -1.0, 1.0)),
            float(np.clip(time_left, 0.0, 1.0)),
            float(job["kind"] == "downlink"),
            float(job["kind"] == "relay"),
            len(job["eligible_satellites"]) / fleet,
            float(env._last_executor.get(job["id"]) is not None),
            float(np.clip(to_release, 0.0, 1.0)),
            float(np.clip(slack_work, -2.0, 4.0)),
        )

    def _present(self, env, raw: dict) -> dict[str, np.ndarray]:
        k = raw["k"]

        # Fixed-size tables. Rows past the real fleet and the packed slots stay zero.
        sats = np.zeros((env.N, SAT_FEATURES), dtype=np.float32)
        jobs = np.zeros((env.K, self.job_features), dtype=np.float32)
        plan = np.zeros((PLAN_MAX, self.job_features), dtype=np.float32)
        edges = np.zeros((env.N, env.K, self.edge_features), dtype=np.float32)
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
                min(row["to_dark"], _LIGHT_HORIZON) / _LIGHT_HORIZON,
                min(row["to_sun"], _LIGHT_HORIZON) / _LIGHT_HORIZON,
                row["capacity_wh"] / 200.0,
                float(row["heater_w"] > 0.0),
                (row["solar_w"] - row["base_w"] - row["heater_w"]) / 100.0,
            )

        # The plan is unfinished work that did not fit in the action slots.
        period = max(float(raw.get("period_s") or 1.0), 1.0)
        dt = max(float(getattr(env, "dt", 300.0)), 1.0)
        for row, job in enumerate(raw["plan_jobs"][:PLAN_MAX]):
            plan[row, :JOB_FEATURES] = self._job_features(env, job, k)
            if self.job_features > JOB_FEATURES:
                plan[row, JOB_FEATURES:] = self._job_v2(env, job, k)
        for slot, item in enumerate(raw["slots"]):
            jobs[slot, :JOB_FEATURES] = self._job_features(env, item["job"], k)
            if self.job_features > JOB_FEATURES:
                jobs[slot, JOB_FEATURES:] = self._job_v2(env, item["job"], k)
            for edge in item["edges"]:
                i = edge["sat"]
                edges[i, slot, :EDGE_FEATURES] = (
                    1.0,
                    float(edge["ok"]),
                    float(edge["reason"] != "energy_reserve"),
                    float(edge["sees"]),
                    float(edge["last_executor"]),
                    min(float(edge.get("remaining_s", 0.0)), period) / period,
                    min(float(edge.get("next_s", period)), period) / period,
                    float(edge.get("uniqueness", 0.0)),
                )
                if self.edge_features > EDGE_FEATURES:
                    edges[i, slot, EDGE_FEATURES:] = self._edge_v2(env, item["job"], edge, dt)

        # Fleet totals, then each satellite's neighbor memory.
        soc = raw["soc"]
        fleet = max(env.n_sats, 1)
        glob = np.zeros(GLOBAL_FEATURES, dtype=np.float32)
        glob[0] = env.k / env.steps_total
        glob[1] = min(len(raw["open_jobs"]), 4 * fleet) / (4 * fleet)
        glob[2] = min(raw["n_prio3"], 2 * fleet) / (2 * fleet)
        glob[3] = raw["downlink_demand"] / fleet
        glob[GLOBAL_DOWNLINK_CAP] = env.downlink_limit / N_MAX
        glob[5] = float(soc.mean()) if env.n_sats else 0.0
        glob[6] = float(soc.min()) if env.n_sats else 0.0
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

    def _job_v2(self, env, job: dict, k: int) -> tuple[float, ...]:
        """Appended job channels (v1 columns stay 0..13):

        value_usd (no priority bonus), work_steps, remaining_steps, priority/3.
        Time-to-deadline, kind, and current-executor flags already live in v1.
        """
        span = max(work_span_ticks(env), 1.0)
        return (
            _asinh_unit(float(job.get("value_usd", 0.0)), VALUE_NORM),
            float(job["work_steps"]) / span,
            float(job["remaining_steps"]) / span,
            float(job["priority"]) / 3.0,
        )

    def _edge_v2(self, env, job: dict, edge: dict, dt: float) -> tuple[float, ...]:
        """Appended pair channels (v1 columns stay 0..7):

        required_work_ticks, value_per_required_tick, remaining_usable_window_ticks,
        window_margin_ticks, is_current_downlink_holder, switch_would_reset_progress,
        switch_lost_work_ticks. Geometry only; no future arrivals or outages.
        """
        sat = env.fleet[int(edge["sat"])]
        holder = env._downlink_holder.get(job["id"])
        required = max(int(downlink_required_steps(job, sat.sat_id, holder)) if job.get("kind") == "downlink" else int(job["remaining_steps"]), 1)
        window = max(float(edge.get("remaining_s", 0.0)), 0.0) / dt
        is_holder = holder == sat.sat_id
        switch_reset = float(job.get("kind") == "downlink" and holder is not None and not is_holder)
        lost = max(int(job["work_steps"]) - int(job["remaining_steps"]), 0) if switch_reset else 0
        span = max(work_span_ticks(env), 1.0)
        rho = float(job.get("value_usd", 0.0)) / required
        return (
            required / span,
            _asinh_unit(rho, RHO_NORM),
            min(window, span) / span,
            float(np.clip((window - required) / span, -2.0, 4.0)),
            float(is_holder),
            switch_reset,
            lost / span,
        )


class ShiftObservationV2(ShiftObservation):
    """Same v1 columns, then USD/work/window pair features that a v1 clip would flatten."""

    version = "v2"
    job_features = JOB_FEATURES_V2
    edge_features = EDGE_FEATURES_V2

