"""Completions pay dollars. Default beta=0 does not credit unfinished work."""

import pytest

from example.jobs import PRICE_DIV
from example.reward import MyReward, job_worth


def _job(remaining: int, work: int = 4, usd: float = 8.0, deadline: int = 40) -> dict:
    return {
        "id": "J",
        "kind": "downlink",
        "value_usd": usd,
        "priority": 1,
        "work_steps": work,
        "remaining_steps": remaining,
        "deadline_step": deadline,
        "completed_step": None,
        "release_step": 0,
    }


class _Env:
    def __init__(self, job: dict, k: int = 0, steps: int = 40) -> None:
        self.jobs = {job["id"]: job}
        self.k = k
        self.steps_total = steps
        self.max_steps = None
        self.ops_stats = {"executor_switches": 0.0}
        self._last_executor = {}


def _row(job_id: str, completed: str | None = None, executed: str = "job") -> dict:
    return {
        "executed": executed,
        "requested": {"job_id": job_id},
        "satellite_id": "S01",
        "below_reserve": True,
        "brownout": True,
        "reason": "accepted" if executed == "job" else "idle",
        "completed_job": completed,
    }


def test_reward_worth_is_the_dollar_price():
    reward = MyReward()
    job = _job(remaining=1, work=1, usd=80.0)
    job["priority"] = 3
    assert reward.worth(job) == pytest.approx(80.0)
    assert job_worth(job) == pytest.approx(80.0 + 80.0 / PRICE_DIV)


def test_miss_pays_nothing():
    reward = MyReward()
    untouched = _job(remaining=4)
    packed = _job(remaining=4)
    packed["ever_packed"] = True
    critical = _job(remaining=4)
    critical["priority"] = 3
    started = _job(remaining=2, work=4)
    assert reward.on_miss(untouched) == 0.0
    assert reward.on_miss(packed) == 0.0
    assert reward.on_miss(critical) == 0.0
    assert reward.on_miss(started) == 0.0


def test_penalties_are_off():
    reward = MyReward()
    assert reward.progress == 0.0
    assert reward.below_reserve == 0.0
    assert reward.brownout == 0.0
    assert reward.blocked == 0.0
    assert reward.executor_switch == 0.0
    assert reward.conflict == 0.0
    assert reward.early_calibrate == 0.0
    assert not hasattr(reward, "tick_share")


def test_complete_pays_the_price_once():
    reward = MyReward()
    job = _job(remaining=0, work=4, usd=8.0)
    job["completed_step"] = 4
    env = _Env(job, k=4)
    reward.reset_potential(env)
    got = reward.score(env, [_row(job["id"], completed=job["id"])], {}, {"conflicts": 4.0, "early_calibrations": 3.0})
    assert got == pytest.approx(8.0)
    assert reward.beta == 0.0


def test_beta_zero_does_not_credit_progress():
    reward = MyReward()
    job = _job(remaining=4, work=4, usd=8.0, deadline=10)
    env = _Env(job, k=0, steps=20)
    reward.reset_potential(env)
    job["remaining_steps"] = 3
    env.k = 1
    got = reward.score(env, [_row(job["id"])], {}, {"conflicts": 0.0, "early_calibrations": 0.0})
    assert got == pytest.approx(0.0)
    assert reward._phi == pytest.approx(0.0)


def test_shaping_telescopes_on_a_finished_job():
    reward = MyReward(gamma=1.0, beta=1.0)
    job = _job(remaining=4, work=4, usd=8.0, deadline=10)
    env = _Env(job, k=0, steps=20)
    reward.reset_potential(env)
    total = 0.0
    for step in range(4):
        job["remaining_steps"] = 4 - (step + 1)
        env.k = step + 1
        completed = job["id"] if job["remaining_steps"] == 0 else None
        if completed:
            job["completed_step"] = env.k
        total += reward.score(env, [_row(job["id"], completed=completed)], {}, {"conflicts": 0.0, "early_calibrations": 0.0})
    assert total == pytest.approx(8.0)
    assert reward._phi == pytest.approx(0.0)


def test_shaping_telescopes_on_a_reset_then_miss():
    reward = MyReward(gamma=1.0, beta=1.0)
    job = _job(remaining=4, work=4, usd=8.0, deadline=5)
    env = _Env(job, k=0, steps=20)
    reward.reset_potential(env)
    total = 0.0

    job["remaining_steps"] = 3
    env.k = 1
    total += reward.score(env, [_row(job["id"])], {}, {"conflicts": 0.0, "early_calibrations": 0.0})

    job["remaining_steps"] = 2
    env.k = 2
    total += reward.score(env, [_row(job["id"])], {}, {"conflicts": 0.0, "early_calibrations": 0.0})

    # Pause drops the downlink back to zero progress.
    job["remaining_steps"] = 4
    env.k = 3
    total += reward.score(env, [_row(job["id"], executed="idle")], {}, {"conflicts": 0.0, "early_calibrations": 0.0})

    env.k = 5
    total += reward.score(env, [_row(job["id"], executed="idle")], {}, {"conflicts": 0.0, "early_calibrations": 0.0})
    assert total == pytest.approx(0.0)
    assert reward.potential(env) == pytest.approx(0.0)


def test_restart_cycles_do_not_print_money():
    reward = MyReward(gamma=0.998, beta=1.0)
    job = _job(remaining=4, work=4, usd=8.0, deadline=40)
    env = _Env(job, k=0, steps=80)
    reward.reset_potential(env)
    total = 0.0
    k = 0
    for _ in range(6):
        job["remaining_steps"] = 3
        k += 1
        env.k = k
        total += reward.score(env, [_row(job["id"])], {}, {"conflicts": 0.0, "early_calibrations": 0.0})

        job["remaining_steps"] = 4
        k += 1
        env.k = k
        total += reward.score(env, [_row(job["id"], executed="idle")], {}, {"conflicts": 0.0, "early_calibrations": 0.0})
    assert total < 0.0
    assert total == pytest.approx((0.998 - 1.0) * 6 * 8.0 / 4)


def test_terminal_potential_is_zero():
    reward = MyReward(gamma=1.0, beta=1.0)
    job = _job(remaining=1, work=4, usd=8.0, deadline=40)
    env = _Env(job, k=0, steps=1)
    reward.reset_potential(env)
    env.k = 1
    got = reward.score(env, [_row(job["id"])], {}, {"conflicts": 0.0, "early_calibrations": 0.0})
    assert reward._phi == 0.0
    assert got == pytest.approx(-6.0)
