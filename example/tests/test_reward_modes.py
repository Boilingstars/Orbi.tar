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
