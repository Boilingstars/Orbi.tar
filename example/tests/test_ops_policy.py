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
    play = OpsEnv(scenario=load(P01_INTRO), events=EventConfig.off(), seed=1, top_k=48)
    loaded = MaskablePPO.load(str(path))
    obs, _ = play.reset(seed=1)
    action, _ = loaded.predict(obs, action_masks=play.action_masks(), deterministic=True)
    assert action.shape == (play.N,)
    mask = play.action_masks().reshape(play.N, play.K + 2)
    assert all(mask[i, int(action[i])] for i in range(play.N)), "only valid actions were chosen"
    play.step(action)


def test_empty_plan_does_not_nan_on_p02():
    env = OpsEnv(scenario=load(P02_SHIFT), events=EventConfig.off(), seed=0)
    batch = _obs_batch(env, batch=2)
    batch["plan"] = torch.zeros_like(batch["plan"])
    logits, value = OpsAttentionNet(**_SMALL)(batch)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_matching_is_conflict_free_and_masked():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    net = OpsAttentionNet(**_SMALL).eval()
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    masks = np.asarray(obs["mask"], dtype=bool).reshape(-1)
    with torch.no_grad():
        actions, _value, _lp = net.act(batch, masks, deterministic=True)
    picked = actions[0].cpu().numpy()
    mask = obs["mask"].astype(bool)
    used = {}
    downs = 0
    cap = max(int(round(float(obs["global"][4]) * 48)), 1)
    for i, a in enumerate(picked[: env.n_sats]):
        assert mask[i, int(a)]
        if a < env.K:
            assert int(a) not in used
            used[int(a)] = i
            if obs["jobs"][int(a), 8] > 0.5:
                downs += 1
    assert downs <= cap


def test_greedy_feature_indices_are_stable():
    from orbi.greedy import _EDGE_LAST_EXECUTOR, _JOB_DEADLINE, _JOB_DOWNLINK, _JOB_REMAINING, _JOB_VALUE

    assert _JOB_VALUE == 3
    assert _JOB_REMAINING == 5
    assert _JOB_DEADLINE == 7
    assert _JOB_DOWNLINK == 8
    assert _EDGE_LAST_EXECUTOR == 4


def test_greedy_ignores_window_edge_columns():
    from orbi.greedy import greedy_action

    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    base = greedy_action(obs, downlink_limit=env.downlink_limit)
    noisy = {key: np.array(value, copy=True) for key, value in obs.items()}
    noisy["edges"][..., 5:] = np.random.default_rng(0).random(noisy["edges"][..., 5:].shape)
    assert np.array_equal(base, greedy_action(noisy, downlink_limit=env.downlink_limit))


def test_pair_scores_read_window_columns():
    from example.models import _Pointer

    pointer = _Pointer(d=8, edge_dim=8, d_pointer=4)
    h_s = torch.randn(1, 4, 8)
    h_j = torch.randn(1, 3, 8)
    short = torch.zeros(1, 4, 3, 8)
    long = torch.zeros(1, 4, 3, 8)
    short[..., 5] = 0.1
    long[..., 5] = 0.9
    with torch.no_grad():
        a = pointer.pair_scores(h_s, h_j, short)
        b = pointer.pair_scores(h_s, h_j, long)
    assert not torch.allclose(a, b)


def _skip_pointer(skip: float):
    class SkipPointer:
        def pair_scores(self, h_s, h_j, edges):
            return torch.zeros(h_s.shape[0], h_s.shape[1], h_j.shape[1], device=h_s.device)

        def side_scores(self, h_s, h_j):
            b, n, _ = h_s.shape
            k = h_j.shape[1]
            z = h_s.new_zeros
            return z((b, n)) - 8, z((b, n)) - 8, z((b, k)) + skip

    return SkipPointer()


def _legal_slots(env, obs):
    mask = np.asarray(obs["mask"], dtype=bool)
    return np.flatnonzero(mask[: env.n_sats, : env.K].any(axis=0)), mask


def test_high_skip_can_leave_a_legal_slot():
    from example.models import _match

    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    legal_slots, mask = _legal_slots(env, obs)
    if legal_slots.size == 0:
        return
    h_s = torch.zeros(1, env.N, 8)
    h_j = torch.zeros(1, env.K, 8)
    torch.manual_seed(0)
    with torch.no_grad():
        picked, _, _, _ = _match(_skip_pointer(8.0), h_s, h_j, batch["edges"], batch["jobs"], mask, batch, False)
    assigned = {int(x) for x in picked[0, : env.n_sats].tolist() if int(x) < env.K}
    assert not any(int(slot) in assigned for slot in legal_slots)


def test_low_skip_still_assigns_a_legal_satellite():
    from example.models import _match

    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    legal_slots, mask = _legal_slots(env, obs)
    if legal_slots.size == 0:
        return
    h_s = torch.zeros(1, env.N, 8)
    h_j = torch.zeros(1, env.K, 8)
    torch.manual_seed(0)
    with torch.no_grad():
        picked, _, _, _ = _match(_skip_pointer(-8.0), h_s, h_j, batch["edges"], batch["jobs"], mask, batch, False)
    assigned = {int(x) for x in picked[0, : env.n_sats].tolist() if int(x) < env.K}
    assert any(int(slot) in assigned for slot in legal_slots)


def test_evaluate_can_score_a_stored_skip():
    from example.models import _match

    class SkipPointer:
        def pair_scores(self, h_s, h_j, edges):
            return torch.zeros(h_s.shape[0], h_s.shape[1], h_j.shape[1], device=h_s.device)

        def side_scores(self, h_s, h_j):
            b, n, _ = h_s.shape
            k = h_j.shape[1]
            z = h_s.new_zeros
            return z((b, n)), z((b, n)), z((b, k))

    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    mask = np.asarray(obs["mask"], dtype=bool)
    idle = np.full((1, env.N), env.idle_index, dtype=np.int64)
    h_s = torch.zeros(1, env.N, 8)
    h_j = torch.zeros(1, env.K, 8)
    with torch.no_grad():
        _, logp, _, _ = _match(
            SkipPointer(), h_s, h_j, batch["edges"], batch["jobs"], mask, batch, False, torch.from_numpy(idle)
        )
    assert torch.isfinite(logp).all()


def test_kl_budget_is_per_satellite():
    from gymnasium.spaces import MultiDiscrete

    from example.train import _action_rows

    class _Dummy:
        action_space = MultiDiscrete([50] * 48)

    assert _action_rows(_Dummy()) == 48


def test_deterministic_matching_follows_the_skip_logit():
    from example.models import _match

    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    legal_slots, mask = _legal_slots(env, obs)
    if legal_slots.size == 0:
        return
    h_s = torch.zeros(1, env.N, 8)
    h_j = torch.zeros(1, env.K, 8)
    with torch.no_grad():
        low, _, _, _ = _match(_skip_pointer(-8.0), h_s, h_j, batch["edges"], batch["jobs"], mask, batch, True)
        high, _, _, _ = _match(_skip_pointer(8.0), h_s, h_j, batch["edges"], batch["jobs"], mask, batch, True)
    low_jobs = {int(x) for x in low[0, : env.n_sats].tolist() if int(x) < env.K}
    high_jobs = {int(x) for x in high[0, : env.n_sats].tolist() if int(x) < env.K}
    assert any(int(slot) in low_jobs for slot in legal_slots)
    assert not any(int(slot) in high_jobs for slot in legal_slots)


def test_pointer_and_encoder_receive_grad():
    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    net = OpsAttentionNet(**_SMALL)
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    mask = np.asarray(obs["mask"], dtype=bool)
    torch.manual_seed(0)
    actions, value, _logp = net.act(batch, mask, deterministic=False)
    _value, logp, entropy = net.evaluate(batch, actions.detach(), mask)
    loss = -logp.mean() - 0.01 * entropy.mean() + value.mean()
    loss.backward()
    pointer = [p.grad for p in net.pointer.parameters() if p.grad is not None]
    encoder = [p.grad for p in net.encoder.parameters() if p.grad is not None]
    assert pointer and any(float(grad.abs().sum()) > 0 for grad in pointer)
    assert encoder and any(float(grad.abs().sum()) > 0 for grad in encoder)


def _replay_matches(scenario, **kwargs) -> None:
    env = OpsEnv(scenario=scenario, events=EventConfig.off(), seed=0, **kwargs)
    obs, _ = env.reset(seed=0)
    net = OpsAttentionNet(**_SMALL).eval()
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    mask = np.asarray(obs["mask"], dtype=bool)
    for seed in range(4):
        torch.manual_seed(seed)
        with torch.no_grad():
            actions, _, sampled = net.act(batch, mask, deterministic=False)
            _, replayed, _ = net.evaluate(batch, actions, mask)
        assert torch.allclose(sampled, replayed, atol=1e-5)
        ratio = torch.exp(replayed - sampled)
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4)


def test_replay_log_prob_matches_the_sample():
    _replay_matches("p01")


def test_replay_log_prob_matches_on_p03_k64():
    from orbi.env import P03_ENERGY

    _replay_matches(load(P03_ENERGY), top_k=64)


def test_matching_short_window_can_beat_worth():
    from example.models import _match

    class WindowPointer:
        def pair_scores(self, h_s, h_j, edges):
            return -10.0 * edges[..., 5]

        def side_scores(self, h_s, h_j):
            b, n, _ = h_s.shape
            k = h_j.shape[1]
            z = h_s.new_zeros
            return z((b, n)), z((b, n)) - 8, z((b, k)) - 8

    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    batch = {key: torch.from_numpy(np.array(value, copy=True)[None]).float() for key, value in obs.items()}
    batch["jobs"][:] = 0
    batch["jobs"][0, 0, 0] = 1.0
    batch["jobs"][0, 1, 0] = 1.0
    batch["jobs"][0, 0, 3] = 4.0
    batch["jobs"][0, 1, 3] = 0.2
    batch["jobs"][0, :, 5] = 0.2
    batch["edges"][:] = 0
    batch["edges"][0, 0, 0, 0] = 1.0
    batch["edges"][0, 0, 1, 0] = 1.0
    batch["edges"][0, 0, 0, 5] = 0.9
    batch["edges"][0, 0, 1, 5] = 0.1
    mask = np.zeros((env.N, env.K + 2), dtype=bool)
    mask[:, env.idle_index] = True
    mask[0, 0] = True
    mask[0, 1] = True
    h_s = torch.zeros(1, env.N, 8)
    h_j = torch.zeros(1, env.K, 8)
    with torch.no_grad():
        picked, _, _, _ = _match(WindowPointer(), h_s, h_j, batch["edges"], batch["jobs"], mask, batch, True)
    assert int(picked[0, 0]) == 1


def test_assignment_loss_falls_on_greedy_pairs():
    from example.train import _assignment_loss
    from orbi.greedy import greedy_action

    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    act = greedy_action(obs, downlink_limit=env.downlink_limit)
    net = OpsAttentionNet(**_SMALL)
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    act_t = torch.from_numpy(act[None])
    mask = np.asarray(obs["mask"], dtype=bool).reshape(-1)
    loss = _assignment_loss(net, batch, act_t, mask)
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0


def test_matching_disagrees_with_greedy_on_window_obs():
    from orbi.greedy import greedy_action

    env = OpsEnv(scenario="p01", events=EventConfig.off(), seed=0)
    obs, _ = env.reset(seed=0)
    greedy = greedy_action(obs, downlink_limit=env.downlink_limit)
    net = OpsAttentionNet(**_SMALL).eval()
    batch = {key: torch.from_numpy(value[None]).float() for key, value in obs.items()}
    masks = np.asarray(obs["mask"], dtype=bool).reshape(-1)
    with torch.no_grad():
        actions, _, _ = net.act(batch, masks, deterministic=True)
    picked = actions[0].cpu().numpy()[: env.n_sats]
    assert not np.array_equal(picked, greedy[: env.n_sats])


def test_pointer_has_no_worth_logit_bias():
    import example.models as models
    from example.models import _Pointer

    assert not hasattr(models, "_WORTH_GAIN")
    pointer = _Pointer(d=8, edge_dim=8, d_pointer=4)
    h_s = torch.zeros(2, 4, 8)
    h_j = torch.zeros(2, 3, 8)
    edges = torch.zeros(2, 4, 3, 8)
    with torch.no_grad():
        a = pointer.pair_scores(h_s, h_j, edges)
        b = pointer.pair_scores(h_s, h_j, edges)
    assert torch.allclose(a, b)


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
