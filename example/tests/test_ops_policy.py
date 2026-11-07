"""Attention policy: shapes, masks, transfer of a trained model from 48 sats to 16."""

from __future__ import annotations

import numpy as np
import torch

from orbi.env import EventConfig, OpsEnv, P01_INTRO, P02_SHIFT, load
from example.models import CtdeNet, OpsAttentionNet, OpsAttentionPolicy, count_parameters

_SMALL = dict(d=32, heads=2, blocks=1, ff=64, d_pointer=16)


def _obs_batch(env: OpsEnv, batch: int = 3) -> dict[str, torch.Tensor]:
    obs, _ = env.reset(seed=0)
    return {
        key: torch.from_numpy(np.repeat(value[None], batch, axis=0)).float()
        for key, value in obs.items()
    }


def test_forward_shapes_and_finiteness():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    net = OpsAttentionNet(**_SMALL)
    logits, value = net(_obs_batch(env))
    assert logits.shape == (3, env.N * (env.K + 2))
    assert value.shape == (3, 1)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_empty_job_queue_does_not_produce_nan():
    """Empty job queue — softmax over an empty set of keys."""
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    blank = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    blank["jobs"] = torch.zeros_like(blank["jobs"])
    blank["edges"] = torch.zeros_like(blank["edges"])
    logits, value = OpsAttentionNet(**_SMALL)(blank)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_env_zero_fills_padding_rows():
    """Presence contract: padding rows are exactly zero, real capacity > 0."""
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    assert not obs["sats"][env.n_sats :].any()
    assert not obs["edges"][env.n_sats :].any()
    assert not obs["neighbors"][env.n_sats :].any()
    assert (obs["sats"][: env.n_sats, 11] > 0).all()


def test_padding_satellites_do_not_change_real_rows():
    """Noise in padding rows must not leak into decisions for real satellites."""
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    net = OpsAttentionNet(**_SMALL).eval()
    base = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    noisy = {key: tensor.clone() for key, tensor in base.items()}
    noise = torch.randn_like(noisy["sats"][:, env.n_sats :])
    noise[..., 11] = 0.0  # capacity stays zero: no spacecraft in this row
    noisy["sats"][:, env.n_sats :] = noise
    noisy["edges"][:, env.n_sats :] = torch.rand_like(noisy["edges"][:, env.n_sats :])
    with torch.no_grad():
        a = net(base)[0].reshape(1, env.N, env.K + 2)[:, : env.n_sats]
        b = net(noisy)[0].reshape(1, env.N, env.K + 2)[:, : env.n_sats]
    assert torch.allclose(a, b, atol=1e-5)


def test_one_model_runs_on_48_and_16_satellites():
    """Train on P02 (48 sats), play on P01 (16 sats) — same weights."""
    big = OpsEnv(scenario=load(P02_SHIFT), events=EventConfig.off(), seed=0)
    small = OpsEnv(scenario=load(P01_INTRO), events=EventConfig.off(), seed=0)
    assert big.n_sats == 48 and small.n_sats == 16

    net = OpsAttentionNet(**_SMALL).eval()
    with torch.no_grad():
        for env in (big, small):
            logits, value = net(_obs_batch(env, batch=1))
            assert logits.shape == (1, env.N * (env.K + 2))
            assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_maskable_ppo_short_learn_and_transfer(tmp_path):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor

    from example.train import mask_fn

    train_env = ActionMasker(
        Monitor(OpsEnv(scenario=load(P02_SHIFT), events=EventConfig.off(), seed=0, max_steps=8)),
        mask_fn,
    )
    model = MaskablePPO(
        OpsAttentionPolicy,
        train_env,
        policy_kwargs=dict(_SMALL),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        verbose=0,
    )
    assert count_parameters(model.policy.net) > 0
    model.learn(total_timesteps=8)
    path = tmp_path / "ops.zip"
    model.save(str(path))

    # Same model on 16 satellites without retraining.
    play = OpsEnv(scenario=load(P01_INTRO), events=EventConfig.off(), seed=1)
    loaded = MaskablePPO.load(str(path))
    obs, _ = play.reset(seed=1)
    action, _ = loaded.predict(obs, action_masks=play.action_masks(), deterministic=True)
    assert action.shape == (play.N,)
    mask = play.action_masks().reshape(play.N, play.K + 2)
    assert all(mask[i, int(action[i])] for i in range(play.N)), "only valid actions were chosen"
    play.step(action)


def test_ctde_policy_forward_and_transfer():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    net = CtdeNet(**_SMALL)
    logits, value = net(_obs_batch(env, batch=1))
    assert logits.shape == (1, env.N * (env.K + 2))
    assert value.shape == (1, 1)
    assert torch.isfinite(logits).all()

    big = OpsEnv(scenario=load(P02_SHIFT), events=EventConfig.off(), seed=0)
    small = OpsEnv(scenario=load(P01_INTRO), events=EventConfig.off(), seed=0)
    with torch.no_grad():
        for item in (big, small):
            out, val = net(_obs_batch(item, batch=1))
            assert out.shape == (1, item.N * (item.K + 2))
            assert torch.isfinite(out).all() and torch.isfinite(val).all()
