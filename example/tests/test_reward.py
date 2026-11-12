"""The return is the dollar price of work ticks and completion."""

import pytest

from example.jobs import PRICE_DIV
from example.reward import MyReward, job_worth


def _job(remaining: int, work: int = 12, usd: float = 100.0) -> dict:
    return {
        "id": "J",
        "value_usd": usd,
        "priority": 1,
        "work_steps": work,
        "remaining_steps": remaining,
        "deadline_step": 40,
        "completed_step": None,
    }


def test_reward_worth_is_the_dollar_price():
    reward = MyReward()
    job = _job(remaining=1, work=1, usd=80.0)
    job["priority"] = 3
    assert reward.worth(job) == pytest.approx(80.0)
    assert job_worth(job) == pytest.approx(80.0 + 80.0 / PRICE_DIV)


def test_miss_pays_nothing():
    reward = MyReward()
    untouched = _job(remaining=12)
    packed = _job(remaining=12)
    packed["ever_packed"] = True
    critical = _job(remaining=12)
    critical["priority"] = 3
    started = _job(remaining=6, work=12)
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


def test_work_ticks_and_completion_sum_to_the_price():
    reward = MyReward()
    job = _job(remaining=4, work=4, usd=8.0)
    job["priority"] = 3
    tick = reward.tick_share * reward.worth(job) / job["work_steps"]
    assert 4 * tick + reward.on_complete(job) == pytest.approx(8.0)


class _Env:
    def __init__(self, job: dict) -> None:
        self.jobs = {job["id"]: job}
        self.k = job["deadline_step"]
        self.ops_stats = {"executor_switches": 0.0}
        self._last_executor = {}


def _row(job_id: str, completed: str | None = None) -> dict:
    return {
        "executed": "job",
        "requested": {"job_id": job_id},
        "satellite_id": "S01",
        "below_reserve": True,
        "brownout": True,
        "reason": "blocked",
        "completed_job": completed,
    }


def test_score_pays_a_work_tick_and_ignores_penalties():
    reward = MyReward()
    job = _job(remaining=4, work=4, usd=8.0)
    got = reward.score(_Env(job), [_row(job["id"])], {}, {"conflicts": 4.0, "early_calibrations": 3.0})
    assert got == pytest.approx(reward.tick_share * 8.0 / 4)


def test_score_pays_the_rest_when_the_job_finishes():
    reward = MyReward()
    job = _job(remaining=0, work=4, usd=8.0)
    got = reward.score(_Env(job), [_row(job["id"], completed=job["id"])], {}, {"conflicts": 0.0, "early_calibrations": 0.0})
    tick = reward.tick_share * 8.0 / 4
    assert got == pytest.approx(tick + (1.0 - reward.tick_share) * 8.0)
