"""Gym shift: energy, calibration, downlink cap, seed jobs."""

from __future__ import annotations

import numpy as np

from orbi.env import EventConfig, OpsEnv
from orbi.greedy import greedy_action, random_masked_action


def _env(scenario="p01", **kwargs) -> OpsEnv:
    kwargs.setdefault("events", EventConfig.off())
    kwargs.setdefault("seed", 0)
    return OpsEnv(scenario=scenario, **kwargs)


def _all_idle(env: OpsEnv) -> np.ndarray:
    return np.full(env.N, env.idle_index, dtype=np.int64)


def test_spaces_and_padding():
    env = _env("p01")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "neighbors" in obs
    assert not obs["neighbors"][env.n_sats :].any()
    mask = env.action_masks().reshape(env.N, env.K + 2)
    assert mask[env.n_sats :, env.idle_index].all()
    assert not mask[env.n_sats :, : env.idle_index].any()
    assert mask[:, env.idle_index].all()
    assert env.jobs


def test_battery_charges_only_in_sunlight():
    env = _env("p01")
    env.reset(seed=0)
    charged_in_sun = discharged_in_dark = False
    for _ in range(env.steps_total):
        start = len(env.trace)
        env.step(_all_idle(env))
        for row in env.trace[start:]:
            delta = row["energy_after_wh"] - row["energy_before_wh"]
            if row["solar_w"] == 0.0:
                assert delta <= 1e-9
                discharged_in_dark = True
            elif row["solar_w"] > row["load_w"] and delta > 0:
                charged_in_sun = True
    assert charged_in_sun and discharged_in_dark


def test_calibration_expiry_blocks_jobs_and_calibrate_clears_it():
    env = _env("p01")
    env.reset(seed=0)
    sid = env.sat_ids[0]
    job = next(j for j in env.jobs.values() if j["kind"] == "relay")
    env.sats[sid].calibration_age_steps = env.calibration_valid
    env.k = job["release_step"]
    ok, reason, _ = env.can_execute(sid, {"action": "job", "job_id": job["id"]})
    assert not ok and reason == "calibration_required"
    ok, _, _ = env.can_execute(sid, {"action": "calibrate"})
    assert ok
    env._advance({sid: {"action": "calibrate"}})
    assert env.sats[sid].calibration_age_steps == 0


def test_third_downlink_in_a_tick_becomes_idle():
    env = _env("p01", baseline_repair=True)
    env.reset(seed=0)
    k = env.k
    chosen = env.sat_ids[:3]
    jobs = []
    for i, sid in enumerate(chosen):
        env.sats[sid].calibration_age_steps = 0
        lat, lon, _ = env.world.lla(env.sats[sid].index)
        jobs.append(
            {
                "id": f"DL-{i}",
                "kind": "downlink",
                "release_step": k,
                "deadline_step": k + 3,
                "work_steps": 1,
                "eligible_satellites": [sid],
                "priority": 3,
                "value_usd": 10.0 * (i + 1),
                "point_a": (lat, lon),
                "point_b": (lat, lon),
            }
        )
    env._apply_event({"type": "add_jobs", "jobs": jobs})
    env._observe()
    slots = {c.job_id: slot for slot, c in enumerate(env._candidates)}
    action = _all_idle(env)
    for job in jobs:
        action[env.sat_ids.index(job["eligible_satellites"][0])] = slots[job["id"]]
    env.step(action)
    rows = env.trace[-env.n_sats :]
    executed = [r["requested"]["job_id"] for r in rows if r["executed"] == "job"]
    assert len(executed) == env.downlink_limit == 2
    assert set(executed) == {"DL-2", "DL-1"}


def test_greedy_and_random_finish_episode():
    env = _env("p01")
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    while True:
        action = greedy_action(obs, downlink_limit=env.downlink_limit)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            assert "summary" in info
            break
    env = _env("p01")
    obs, _ = env.reset(seed=1)
    obs, _, _, _, _ = env.step(random_masked_action(obs, rng))
    assert env.observation_space.contains(obs)


def test_p02_padding_and_job_sample():
    env = _env("p02")
    obs, _ = env.reset(seed=0)
    assert env.n_sats == 48
    assert env.jobs
    assert env.ground_region is None
    assert not obs["sats"][env.n_sats :].any()


def test_jobs_use_active_ground_points():
    env = _env("p01", ground_points=8, job_difficulty=0.0)
    env.reset(seed=0)
    assert len(env.active_points) == 8
    for job in env.jobs.values():
        if job.get("addressed") is not None and not job.get("points"):
            assert job["point_a"] is None and len(job["eligible_satellites"]) == 1
            continue
        a, b = job["point_a"], job["point_b"]
        assert any(abs(a[0] - p[0]) < 1e-9 and abs(a[1] - p[1]) < 1e-9 for p in env.active_points)
        assert any(abs(b[0] - p[0]) < 1e-9 and abs(b[1] - p[1]) < 1e-9 for p in env.active_points)


def test_no_contact_blocks_job():
    env = _env("p01")
    env.reset(seed=0)
    sid = env.sat_ids[0]
    job = next(j for j in env.jobs.values() if len(j.get("points") or []) <= 2 and j.get("point_a") is not None)
    job["point_a"] = (1.2, 0.0)
    job["point_b"] = (-1.2, 0.0)
    env.sats[sid].calibration_age_steps = 0
    env.k = job["release_step"]
    ok, reason, _ = env.can_execute(sid, {"action": "job", "job_id": job["id"]})
    assert not ok and reason == "no_contact"


def test_packed_slots_set_ever_packed():
    env = _env("p01")
    env.reset(seed=0)
    packed = [cand.job for cand in env._candidates]
    assert packed
    assert all(job.get("ever_packed") for job in packed)
