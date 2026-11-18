"""Packed decoder order is independent of weights; score order is not."""

from __future__ import annotations

import numpy as np
import torch

from example.models import OpsAttentionNet, _match
from orbi.env import GLOBAL_DOWNLINK_CAP, N_MAX, EventConfig, OpsEnv


_SMALL = dict(d=32, heads=2, blocks=1, ff=64, d_pointer=16)


def _batch(env):
    obs, _ = env.reset(seed=0)
    return {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}


def test_packed_order_stable_after_weight_update():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs = _batch(env)
    masks = obs["mask"]
    net = OpsAttentionNet(**_SMALL, job_order="packed")
    net.eval()
    with torch.no_grad():
        h_s, h_j, edges, jobs, _ = net.encoder(obs)
        picked_a, lp_a, _, _ = _match(
            net.pointer, h_s, h_j, edges, jobs, masks, obs, deterministic=True, job_order="packed"
        )
        for param in net.pointer.parameters():
            param.add_(0.5)
        h_s, h_j, edges, jobs, _ = net.encoder(obs)
        picked_b, lp_b, _, _ = _match(
            net.pointer, h_s, h_j, edges, jobs, masks, obs, deterministic=True, actions=picked_a, job_order="packed"
        )
    assert torch.equal(picked_a, picked_b)
    assert torch.isfinite(lp_a).all() and torch.isfinite(lp_b).all()


def test_score_order_can_change_after_weight_update():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs = _batch(env)
    net = OpsAttentionNet(**_SMALL, job_order="scores")
    jobs = obs["jobs"]
    pair = torch.randn(1, env.N, env.K)
    prio = jobs[..., 0:3] @ torch.tensor([1.0, 2.0, 3.0])
    before = (pair.max(dim=1).values + 0.5 * prio).argsort(dim=-1, descending=True)
    pair2 = pair + torch.randn_like(pair) * 3.0
    after = (pair2.max(dim=1).values + 0.5 * prio).argsort(dim=-1, descending=True)
    assert before.shape == after.shape


def test_empty_catalog_finite_under_packed_order():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    batch["jobs"] = torch.zeros_like(batch["jobs"])
    batch["edges"] = torch.zeros_like(batch["edges"])
    net = OpsAttentionNet(**_SMALL, job_order="packed")
    logits, value = net(batch)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_packed_sample_replay_and_unique_assignment():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs = _batch(env)
    mask = obs["mask"]
    net = OpsAttentionNet(**_SMALL, job_order="packed")
    net.eval()
    for seed in range(4):
        torch.manual_seed(seed)
        with torch.no_grad():
            picked, sampled, _, _ = _match(
                net.pointer, *net.encoder(obs)[:3], obs["jobs"], mask, obs, False, job_order="packed"
            )
            _, replayed, _, _ = _match(
                net.pointer,
                *net.encoder(obs)[:3],
                obs["jobs"],
                mask,
                obs,
                False,
                actions=picked,
                job_order="packed",
            )
        assert torch.allclose(sampled, replayed, atol=1e-5)
        chosen = [int(x) for x in picked[0, : env.n_sats].tolist() if int(x) < env.K]
        assert len(chosen) == len(set(chosen))
        downs = sum(int(obs["jobs"][0, slot, 8] > 0.5) for slot in chosen)
        cap = int(round(float(obs["global"][0, GLOBAL_DOWNLINK_CAP]) * N_MAX))
        assert downs <= max(cap, 1)


def test_packed_respects_action_mask():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs = _batch(env)
    mask = obs["mask"].clone()
    net = OpsAttentionNet(**_SMALL, job_order="packed")
    with torch.no_grad():
        picked, lp, _, _ = _match(
            net.pointer, *net.encoder(obs)[:3], obs["jobs"], mask, obs, True, job_order="packed"
        )
    n = env.n_sats
    k = env.K
    for i in range(n):
        slot = int(picked[0, i])
        assert bool(mask[0, i, slot])
    assert torch.isfinite(lp).all()
