"""observation_v2 widens tensors without changing v1 columns or leaking future events."""

from __future__ import annotations

import numpy as np
import torch

from example.migrate import migrate_observation_v1_to_v2, overlay_policy_weights
from example.models import OpsAttentionNet
from example.observation import (
    EDGE_FEATURES_V2,
    JOB_FEATURES_V2,
    ShiftObservation,
    ShiftObservationV2,
    observation_from_version,
)
from orbi.env import EDGE_FEATURES, EventConfig, JOB_FEATURES, OpsEnv


_SMALL = dict(d=32, heads=2, blocks=1, ff=64, d_pointer=16)


def test_v1_feature_widths_unchanged():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    assert obs["jobs"].shape[-1] == JOB_FEATURES
    assert obs["edges"].shape[-1] == EDGE_FEATURES
    assert env.observation.version == "v1"


def test_v2_appends_channels():
    env = OpsEnv(
        scenario="p01",
        events=EventConfig.off(),
        seed=0,
        observation=ShiftObservationV2(),
    )
    obs, _ = env.reset(seed=0)
    assert obs["jobs"].shape[-1] == JOB_FEATURES_V2
    assert obs["edges"].shape[-1] == EDGE_FEATURES_V2
    assert env.observation_space["jobs"].shape[-1] == JOB_FEATURES_V2


def test_v1_columns_match_on_the_same_env_state():
    v1 = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0, observation=ShiftObservation())
    obs1, _ = v1.reset(seed=0)
    encoded = ShiftObservationV2().encode(v1)
    assert np.allclose(encoded["jobs"][:, :JOB_FEATURES], obs1["jobs"])
    assert np.allclose(encoded["edges"][:, :, :EDGE_FEATURES], obs1["edges"])


def test_high_value_jobs_are_not_clipped_together():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0, observation=ShiftObservationV2())
    env.reset(seed=0)
    cheap_job = {"value_usd": 18.0, "work_steps": 4, "remaining_steps": 4, "priority": 3}
    rich_job = {"value_usd": 40.0, "work_steps": 4, "remaining_steps": 4, "priority": 3}
    cheap = ShiftObservationV2()._job_v2(env, cheap_job, 0)[0]
    rich = ShiftObservationV2()._job_v2(env, rich_job, 0)[0]
    assert rich > cheap + 1e-5
    from example.observation import _WORTH_OBS
    from example.reward import job_worth

    v1_cheap = float(np.clip(job_worth(cheap_job) / _WORTH_OBS, -5.0, 5.0))
    v1_rich = float(np.clip(job_worth(rich_job) / _WORTH_OBS, -5.0, 5.0))
    assert v1_rich == v1_cheap == 5.0


def test_zero_new_channels_match_v1_network():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    net = OpsAttentionNet(**_SMALL)
    net.eval()
    with torch.no_grad():
        logits_v1, value_v1 = net(batch)
        pair_v1 = net.pointer.pair_scores(*net.encoder(batch)[:3])

    migrate_observation_v1_to_v2(net, JOB_FEATURES_V2, EDGE_FEATURES_V2)
    wide = {key: tensor.clone() for key, tensor in batch.items()}
    b, k, _ = wide["jobs"].shape
    n = wide["sats"].shape[1]
    pad_jobs = torch.zeros(b, k, JOB_FEATURES_V2 - JOB_FEATURES)
    pad_edges = torch.zeros(b, n, k, EDGE_FEATURES_V2 - EDGE_FEATURES)
    wide["jobs"] = torch.cat([wide["jobs"], pad_jobs], dim=-1)
    wide["plan"] = torch.cat([wide["plan"], torch.zeros(*wide["plan"].shape[:2], JOB_FEATURES_V2 - JOB_FEATURES)], dim=-1)
    wide["edges"] = torch.cat([wide["edges"], pad_edges], dim=-1)
    with torch.no_grad():
        logits_v2, value_v2 = net(wide)
        pair_v2 = net.pointer.pair_scores(*net.encoder(wide)[:3])
    assert torch.allclose(logits_v1, logits_v2, atol=1e-5)
    assert torch.allclose(value_v1, value_v2, atol=1e-5)
    assert torch.allclose(pair_v1, pair_v2, atol=1e-5)


def test_observation_from_version():
    assert observation_from_version("v1").version == "v1"
    assert observation_from_version("v2").version == "v2"


def test_overlay_fails_fast_on_shape_mismatch():
    v1 = OpsAttentionNet(**_SMALL)
    v2 = OpsAttentionNet(**_SMALL, job_dim=JOB_FEATURES_V2, edge_dim=EDGE_FEATURES_V2)
    try:
        overlay_policy_weights(v2, v1.state_dict(), allow_widen=False)
    except ValueError as exc:
        assert "incompatible tensor" in str(exc)
    else:
        raise AssertionError("expected fail-fast on v1->v2 without migrate")


def test_overlay_zero_new_channels_matches_v1():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    v1 = OpsAttentionNet(**_SMALL)
    v1.eval()
    v2 = OpsAttentionNet(**_SMALL, job_dim=JOB_FEATURES_V2, edge_dim=EDGE_FEATURES_V2)
    overlay_policy_weights(v2, v1.state_dict(), allow_widen=True)
    v2.eval()
    wide = {key: tensor.clone() for key, tensor in batch.items()}
    pad_jobs = torch.zeros(1, wide["jobs"].shape[1], JOB_FEATURES_V2 - JOB_FEATURES)
    pad_edges = torch.zeros(1, wide["edges"].shape[1], wide["edges"].shape[2], EDGE_FEATURES_V2 - EDGE_FEATURES)
    wide["jobs"] = torch.cat([wide["jobs"], pad_jobs], dim=-1)
    wide["plan"] = torch.cat(
        [wide["plan"], torch.zeros(*wide["plan"].shape[:2], JOB_FEATURES_V2 - JOB_FEATURES)], dim=-1
    )
    wide["edges"] = torch.cat([wide["edges"], pad_edges], dim=-1)
    with torch.no_grad():
        logits_v1, value_v1 = v1(batch)
        logits_v2, value_v2 = v2(wide)
        pair_v1 = v1.pointer.pair_scores(*v1.encoder(batch)[:3])
        pair_v2 = v2.pointer.pair_scores(*v2.encoder(wide)[:3])
    assert torch.allclose(logits_v1, logits_v2, atol=1e-5)
    assert torch.allclose(value_v1, value_v2, atol=1e-5)
    assert torch.allclose(pair_v1, pair_v2, atol=1e-5)
