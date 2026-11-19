"""Configurable reward modes. PBRS with γ<1 does not telescope to zero."""

from __future__ import annotations

import pytest

from example.reward import MyReward, REWARD_MODES
from example.tests.test_reward import _Env, _job, _row


def test_reward_modes_are_named():
    assert "current_pbrs" in REWARD_MODES
    assert "dollars" in REWARD_MODES
    assert "feasibility_pbrs" in REWARD_MODES


def test_dollars_mode_has_no_potential():
    reward = MyReward(mode="dollars", gamma=1.0, beta=1.0)
    job = _job(remaining=2, work=4, usd=8.0)
    env = _Env(job, k=0)
    reward.reset_potential(env)
    job["remaining_steps"] = 1
    env.k = 1
    got = reward.score(env, [_row(job["id"])], {}, {"conflicts": 0.0, "early_calibrations": 0.0})
    assert got == pytest.approx(0.0)
    assert reward.potential(env) == 0.0


def test_gamma_less_than_one_does_not_claim_a_zero_telescope():
    reward = MyReward(mode="pbrs", gamma=0.998, beta=1.0)
    job = _job(remaining=4, work=4, usd=8.0, deadline=40)
    env = _Env(job, k=0, steps=80)
    reward.reset_potential(env)
    job["remaining_steps"] = 3
    env.k = 1
    up = reward.score(env, [_row(job["id"])], {}, {"conflicts": 0.0, "early_calibrations": 0.0})
    job["remaining_steps"] = 4
    env.k = 2
    down = reward.score(env, [_row(job["id"], executed="idle")], {}, {"conflicts": 0.0, "early_calibrations": 0.0})
    assert up + down == pytest.approx((0.998 - 1.0) * 8.0 / 4)


def test_unknown_mode_fails_fast():
    with pytest.raises(ValueError, match="reward_mode"):
        MyReward(mode="not_a_mode")


def test_feasibility_pbrs_is_a_state_potential():
    reward = MyReward(mode="feasibility_pbrs", gamma=1.0, beta=0.5)
    job = _job(remaining=2, work=4, usd=8.0, deadline=40)
    job["kind"] = "relay"
    env = _Env(job, k=1, steps=80)
    assert reward.potential(env) == pytest.approx(0.5 * 8.0 * 2.0 / 4.0)
    env.k = 1
    again = reward.potential(env)
    assert again == pytest.approx(reward.potential(env))


def test_pbrs_and_ppo_gamma_are_the_same_field():
    reward = MyReward(mode="pbrs", gamma=0.998, beta=1.0)
    assert reward.gamma == 0.998


def test_dollars_undiscounted_sum_equals_shift_revenue():
    """Raw env return is USD. PPO still sees that value divided by 100."""
    from example.train import _REWARD_SCALE, _ScaleReward
    from orbi.env import EventConfig, OpsEnv
    from orbi.greedy import greedy_action

    assert _REWARD_SCALE == 100.0
    env = OpsEnv(
        scenario="p01",
        events=EventConfig.off(),
        seed=0,
        reward=MyReward(mode="dollars", gamma=0.998),
    )
    obs, _ = env.reset(seed=0)
    total = 0.0
    info = {}
    while True:
        action = greedy_action(obs, downlink_limit=env.downlink_limit)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    revenue = float(info["summary"]["revenue_usd"])
    assert total == pytest.approx(revenue, abs=1e-6)
    assert env.reward.potential(env) == 0.0
    assert env.reward.progress == env.reward.conflict == env.reward.early_calibrate == 0.0

    scaled = _ScaleReward(
        OpsEnv(
            scenario="p01",
            events=EventConfig.off(),
            seed=0,
            max_steps=1,
            reward=MyReward(mode="dollars", gamma=0.998),
        )
    )
    obs, _ = scaled.reset(seed=0)
    action = greedy_action(obs, downlink_limit=scaled.unwrapped.downlink_limit)
    _, scaled_r, _, _, _ = scaled.step(action)
    raw = OpsEnv(
        scenario="p01",
        events=EventConfig.off(),
        seed=0,
        max_steps=1,
        reward=MyReward(mode="dollars", gamma=0.998),
    )
    obs, _ = raw.reset(seed=0)
    _, raw_r, _, _, _ = raw.step(action)
    assert scaled_r == pytest.approx(raw_r / 100.0)


def test_dollars_progress_ticks_do_not_add_pbrs():
    reward = MyReward(mode="dollars", gamma=0.998, beta=1.0)
    job = _job(remaining=4, work=4, usd=8.0, deadline=40)
    env = _Env(job, k=0, steps=80)
    reward.reset_potential(env)
    job["remaining_steps"] = 3
    env.k = 1
    assert reward.score(env, [_row(job["id"])], {}, {"conflicts": 4.0, "early_calibrations": 3.0}) == 0.0
    job["remaining_steps"] = 0
    job["completed_step"] = 2
    env.k = 2
    assert reward.score(env, [_row(job["id"], completed=job["id"])], {}, {"conflicts": 0.0, "early_calibrations": 0.0}) == pytest.approx(8.0)
