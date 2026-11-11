"""CTDE slice: optional repair, neighbors, ground-control sample."""

from __future__ import annotations

import numpy as np
import torch

from example.control import GroundControl
from example.models import CtdeNet, LocalActor
from example.satellite import NeighborView
from orbi.env import EventConfig, NEIGHBOR_FEATURES, OpsEnv

_SMALL = dict(d=32, heads=2, blocks=1, ff=64, d_pointer=16)


def _env(**kwargs) -> OpsEnv:
    kwargs.setdefault("events", EventConfig.off())
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("scenario", "p01")
    return OpsEnv(**kwargs)


def _all_idle(env: OpsEnv) -> np.ndarray:
    return np.full(env.N, env.idle_index, dtype=np.int64)


def _inject_relay(env: OpsEnv, sats: list[str], job_id: str = "RL-0") -> str:
    k = env.k
    for sid in sats:
        env.sats[sid].calibration_age_steps = 0
    lat, lon, _ = env.world.lla(env.sats[sats[0]].index)
    env.world.sees_job = lambda *args, **kwargs: True
    env._apply_event(
        {
            "type": "add_jobs",
            "jobs": [
                {
                    "id": job_id,
                    "kind": "relay",
                    "release_step": k,
                    "deadline_step": k + 3,
                    "work_steps": 2,
                    "eligible_satellites": list(sats),
                    "priority": 3,
                    "value_usd": 10.0,
                    "point_a": (lat, lon),
                    "point_b": (lat, lon),
                }
            ],
        }
    )
    env._observe()
    return job_id


def test_baseline_repair_off_leaves_duplicate_to_environment():
    env = _env(baseline_repair=False)
    env.reset(seed=0)
    sats = env.sat_ids[:2]
    job_id = _inject_relay(env, sats)
    slot = {c.job_id: s for s, c in enumerate(env._candidates)}[job_id]
    action = _all_idle(env)
    action[0] = slot
    action[1] = slot
    env.step(action)
    assert env.ops_stats["conflicts_repaired"] == 0
    rows = {r["satellite_id"]: r for r in env.trace[-env.n_sats :]}
    executed = [r for r in rows.values() if r["executed"] == "job"]
    assert len(executed) == 1
    reasons = {rows[sid]["reason"] for sid in sats}
    assert "duplicate_job_in_step" in reasons
    assert "accepted" in reasons


def test_baseline_repair_on_drops_the_duplicate():
    env = _env(baseline_repair=True)
    env.reset(seed=0)
    sats = env.sat_ids[:2]
    job_id = _inject_relay(env, sats)
    slot = {c.job_id: s for s, c in enumerate(env._candidates)}[job_id]
    action = _all_idle(env)
    action[0] = slot
    action[1] = slot
    env.step(action)
    assert env.ops_stats["conflicts_repaired"] == 1
    rows = env.trace[-env.n_sats :]
    assert all(r["reason"] in ("accepted", "idle") for r in rows)


def test_abandon_job_for_calibrate_keeps_progress():
    env = _env()
    env.reset(seed=0)
    first, second = env.sat_ids[:2]
    job_id = _inject_relay(env, [first, second], "RL-hold")
    job = env.jobs[job_id]
    remaining = job["work_steps"]
    env._advance({first: {"action": "job", "job_id": job_id}})
    assert job["remaining_steps"] == remaining - 1
    env._advance({first: {"action": "calibrate"}})
    assert job["remaining_steps"] == remaining - 1
    env._advance({second: {"action": "job", "job_id": job_id}})
    assert job["remaining_steps"] == remaining - 2


def test_neighbor_methods_and_padding():
    env = _env(neighbor_method="competitors", neighbor_memory=12)
    obs, _ = env.reset(seed=0)
    assert obs["neighbors"].shape == (env.N, 12, NEIGHBOR_FEATURES)
    assert not obs["neighbors"][env.n_sats :].any()
    env = _env(neighbor_method="nearest_index", neighbor_memory=5)
    obs, _ = env.reset(seed=0)
    assert obs["neighbors"].shape == (env.N, 5, NEIGHBOR_FEATURES)
    assert obs["neighbors"][0, :, 0].sum() == 5


def test_ground_control_visible_outranks_worth():
    env = _env()
    env.reset(seed=0)
    control = GroundControl(top_k=env.K)
    known = env.observation.known_jobs(env)
    env._job_views = {}
    visible = [job for job in known if control.has_contact(env, job)]
    hidden = [job for job in known if not control.has_contact(env, job)]
    if visible and hidden:
        cheap = min(visible, key=lambda job: job.get("value_usd", 0.0))
        rich = max(hidden, key=lambda job: job.get("value_usd", 0.0))
        assert control.job_rank(env, cheap) < control.job_rank(env, rich)


def test_executable_job_is_not_displaced():
    env = _env()
    env.reset(seed=0)
    first, second = env.sat_ids[:2]
    rich = _inject_relay(env, [first], "RL-rich")
    poor = _inject_relay(env, [second], "RL-poor")
    env.jobs[rich]["value_usd"] = 500.0
    env.jobs[poor]["value_usd"] = 1.0
    env.sats[first].calibration_age_steps = env.calibration_valid
    env._job_views = {}
    control = GroundControl(top_k=1)
    picked = control.select_jobs(env, [env.jobs[rich], env.jobs[poor]])
    assert [cand.job_id for cand in picked] == [poor]
    assert control.job_rank(env, env.jobs[poor]) < control.job_rank(env, env.jobs[rich])


def test_packed_job_keeps_every_eligible_satellite():
    env = _env()
    env.reset(seed=0)
    sats = env.sat_ids[:2]
    job_id = _inject_relay(env, sats, "RL-both")
    picked = GroundControl(top_k=4).select_jobs(env, [env.jobs[job_id]])
    assert picked[0].sats == [0, 1]


def test_ground_control_top_k_and_cluster():
    env = _env()
    env.reset(seed=0)
    known = env.observation.known_jobs(env)
    full = GroundControl(top_k=4).select_jobs(env, known)
    assert len(full) == 4
    cluster_ids = env.sat_ids[:3]
    clustered = GroundControl(top_k=8, sat_ids=cluster_ids).select_jobs(env, known)
    allowed = {env.sat_ids.index(sid) for sid in cluster_ids}
    for cand in clustered:
        assert cand.sats
        assert set(cand.sats) <= allowed


def test_local_actor_and_ctde_are_finite():
    env = _env()
    obs, _ = env.reset(seed=0)
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    actor = LocalActor(**_SMALL)
    net = CtdeNet(**_SMALL)
    logits = actor(batch)
    both, value = net(batch)
    assert logits.shape == (1, env.N * (env.K + 2))
    assert both.shape == logits.shape and value.shape == (1, 1)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_executable_count_follows_calibration_on_the_same_step():
    env = _env()
    env.reset(seed=0)
    control = env.ground_control
    job = next(item for item in env.jobs.values() if control._executable(env, item) > 0)
    assert env._executable_cache is None
    for sid in control._eligible(env, job):
        env.sats[sid].calibration_age_steps = env.calibration_valid
    assert control._executable(env, job) == 0


def test_executable_count_does_not_leak_across_reset():
    env = _env()
    env.reset(seed=0)
    env.ground_control._executable_cache = (0, {"J000": 999})
    env.reset(seed=1)
    control = env.ground_control
    job = env.jobs["J000"]
    count = control._executable(env, job)
    live = 0
    action = {"action": "job", "job_id": job["id"]}
    for sid in control._eligible(env, job):
        ok, _reason, _power = env.can_execute(sid, action)
        live += int(ok)
    assert count == live
    assert count != 999


def test_snapshot_reuses_the_executable_count():
    from tools.observation import collect

    env = _env()
    env.reset(seed=0)
    calls = {"n": 0}
    real = env.can_execute

    def wrapped(sid, action):
        if action.get("action") == "job":
            calls["n"] += 1
        return real(sid, action)

    env.can_execute = wrapped
    collect(env, env.ground_control)
    assert calls["n"] > 0
    assert env._executable_cache is None

    # The second rank of this snapshot must not ask the satellites again.
    known = env.observation.known_jobs(env)
    env._executable_cache = {}
    calls["n"] = 0
    for job in known:
        env.ground_control.job_rank(env, job)
    assert calls["n"] > 0
    calls["n"] = 0
    for job in known:
        env.ground_control.job_rank(env, job)
    assert calls["n"] == 0
    env._executable_cache = None


def test_neighbor_view_rejects_unknown_method():
    try:
        NeighborView(method="radio")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
