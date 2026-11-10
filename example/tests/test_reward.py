"""Worth miss terms and the small calibration/reserve penalties."""

import pytest

from example.jobs import PRICE_DIV
from example.reward import MyReward, _UNTOUCHED_MISS


def _job(remaining: int, work: int = 12, usd: float = 100.0) -> dict:
    return {
        "value_usd": usd,
        "priority": 1,
        "work_steps": work,
        "remaining_steps": remaining,
    }


def test_untouched_pool_tail_is_not_penalized():
    reward = MyReward()
    job = _job(remaining=12)
    assert reward.on_miss(job) == 0.0


def test_untouched_packed_miss_is_a_fraction_of_worth():
    reward = MyReward()
    job = _job(remaining=12)
    job["ever_packed"] = True
    assert reward.on_miss(job) == pytest.approx(-_UNTOUCHED_MISS * reward.worth(job))


def test_untouched_p3_miss_is_a_fraction_of_worth():
    reward = MyReward()
    job = _job(remaining=12)
    job["priority"] = 3
    assert reward.on_miss(job) == pytest.approx(-_UNTOUCHED_MISS * reward.worth(job))


def test_started_miss_is_the_unfinished_slice():
    reward = MyReward()
    job = _job(remaining=6, work=12)
    assert reward.on_miss(job) == pytest.approx(-0.5 * reward.worth(job))


def test_aux_terms_stay_small_versus_dollars():
    reward = MyReward()
    assert reward.early_calibrate == pytest.approx(4.0 / PRICE_DIV)
    assert reward.below_reserve == pytest.approx(1.0)
    assert reward.worth(_job(remaining=1, usd=100.0, work=1)) == pytest.approx(100.0)
    p3 = reward.worth({"value_usd": 80.0, "priority": 3, "work_steps": 1, "remaining_steps": 1})
    assert 80.0 < p3 < 200.0
