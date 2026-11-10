"""Job subclasses from example land in the generator with their probabilities."""

from __future__ import annotations

from example.jobs import (
    DownlinkChain,
    DownlinkPair,
    DownlinkSource,
    RelayAddressed,
    RelayPair,
    UrgentDownlink,
    UrgentRelay,
    catalog,
    work_ref_ticks,
    zone_dwell_ticks,
)
from example.plots import run_label
from orbi.env import EventConfig, OpsEnv


def test_catalog_is_every_concrete_subclass():
    names = {job_type.name for job_type in catalog().types}
    assert names == {
        RelayPair.name,
        DownlinkPair.name,
        DownlinkSource.name,
        DownlinkChain.name,
        RelayAddressed.name,
        UrgentRelay.name,
        UrgentDownlink.name,
    }
    downlink = sum(job.probability for job in catalog().types if job.family == "downlink")
    relay = sum(job.probability for job in catalog().types if job.family == "relay")
    assert abs(downlink - 0.75) < 1e-9
    assert abs(relay - 0.25) < 1e-9
    assert abs(sum(job_type.probability for job_type in catalog().types) - 1.0) < 1e-9


def test_generated_jobs_follow_their_subclass():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=1)
    env.reset(seed=1)
    kinds = {job["type"] for job in env.jobs.values()}
    assert kinds <= {
        "relay_pair",
        "downlink_pair",
        "downlink_source",
        "downlink_chain",
        "relay_addressed",
        "relay_urgent",
        "downlink_urgent",
    }
    assert "revenue_usd" in env.summary()
    for job in env.jobs.values():
        assert job["kind"] in ("downlink", "relay")
        assert "value_usd" in job["extras"]
        if job["type"] == "downlink_source":
            assert job["point_a"] == job["point_b"]
            assert len(job["points"]) == 1

        if job["type"] == "downlink_chain":
            assert len(job["points"]) == 3

        if job["type"] == "relay_addressed":
            assert job["points"] == []
            assert job["addressed"] in range(env.n_sats)
            assert job["eligible_satellites"] == [env.sat_ids[job["addressed"]]]


def test_work_mix_follows_zone_dwell():
    env = OpsEnv(scenario="p02", events=EventConfig.off(), seed=0)
    env.reset(seed=0)
    dwell = zone_dwell_ticks(env)
    ref = work_ref_ticks(env)
    assert ref == max(1, int(round(dwell / 3.0)))
    works = [job["work_steps"] for job in env.jobs.values()]
    short = sum(1 for work in works if work <= ref + 1)
    assert short / max(len(works), 1) >= 0.65
    assert max(works) <= 3 * dwell + 1


def test_short_jobs_live_until_a_window():
    env = OpsEnv(scenario="p02", events=EventConfig.off(), seed=1)
    env.reset(seed=1)
    ref = work_ref_ticks(env)
    short = [job for job in env.jobs.values() if job["work_steps"] <= ref + 1]
    assert short
    lives = [job["deadline_step"] - job["release_step"] for job in short]
    assert sum(life >= job["work_steps"] for job, life in zip(short, lives)) == len(short)
    assert max(lives) > 4


def test_p02_world_spawn_and_jammed_initial_pool():
    env = OpsEnv(scenario="p02", events=EventConfig.off(), seed=0, ground_region="world")
    env.reset(seed=0)
    assert env.ground_region is None
    open_n = env._open_job_count()
    assert 200 <= open_n <= 350
    assert abs(open_n - 5 * env.n_sats) <= 2


def test_event_refill_stays_inside_pool_band():
    from orbi.greedy import greedy_action

    env = OpsEnv(scenario="p02", events=EventConfig(), seed=0, ground_region="world")
    obs, _ = env.reset(seed=0)
    lo, hi = env.events_cfg.pool_min, env.events_cfg.pool_max
    for _ in range(8):
        obs, _, term, trunc, _ = env.step(greedy_action(obs, downlink_limit=env.downlink_limit))
        assert lo <= env._open_job_count() <= hi
        if term or trunc:
            break


def test_run_label_uses_suite_and_policy():
    assert run_label("agent/runs/07_11_26_19_30_example_dispatcher") == "07.11.26 19:30 example dispatcher"
    assert run_label("agent/runs/07_11_26_19_30_example_ctde_2") == "07.11.26 19:30 example ctde (2)"
    assert run_label("agent/runs/07_11_26_09_04_lr_1e-4") == "07.11.26 09:04 lr=1e-4"
