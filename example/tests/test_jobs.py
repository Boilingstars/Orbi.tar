"""Job subclasses from example land in the generator with their probabilities."""

from __future__ import annotations

from example.jobs import (
    DownlinkChain,
    DownlinkPair,
    DownlinkSource,
    RelayAddressed,
    RelayPair,
    catalog,
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
    }
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


def test_run_label_uses_suite_and_policy():
    assert run_label("agent/runs/07_11_26_19_30_example_dispatcher") == "07.11.26 19:30 example dispatcher"
    assert run_label("agent/runs/07_11_26_19_30_example_ctde_2") == "07.11.26 19:30 example ctde (2)"
    assert run_label("agent/runs/07_11_26_09_04_lr_1e-4") == "07.11.26 09:04 lr=1e-4"
