"""Reward for this package.

A finished job pays its dollar price and nothing else. A share is spread
across the ticks that actually execute it, and the rest is paid when it
completes. Taking a job, missing a deadline, switching executor, calibrating,
and sitting below reserve do not move the return. Priority is not added on
top of the price: the return matches revenue.

`job_worth` still adds the priority bonus. The observation and the slot
ranker read that, so a loaded policy keeps the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from example.jobs import PRICE_DIV
from tools.reward import Reward


# Same divisor as job USD. The observation clips worth against this scale.
_PRIORITY_BONUS = (0.0, 25.0 / PRICE_DIV, 80.0 / PRICE_DIV)


def job_worth(job: dict, priority_bonus: tuple[float, float, float] = _PRIORITY_BONUS) -> float:
    """Dollars plus priority, for the observation and the ranker. Not the return."""
    rank = min(max(int(job.get("priority", 1)) - 1, 0), len(priority_bonus) - 1)
    return float(job.get("value_usd", 0.0)) + float(priority_bonus[rank])


@dataclass
class MyReward(Reward):
    """Dollar price of executed work. No penalties and no priority bonus."""

    progress: float = 0.0
    below_reserve: float = 0.0
    brownout: float = 0.0
    blocked: float = 0.0
    executor_switch: float = 0.0
    conflict: float = 0.0
    early_calibrate: float = 0.0
    tick_share: float = 0.25

    def worth(self, job: dict) -> float:
        return float(job.get("value_usd", 0.0))

    def on_complete(self, job: dict) -> float:
        return (1.0 - self.tick_share) * self.worth(job)

    def on_miss(self, job: dict) -> float:
        return 0.0

    def score(self, env, rows, prev_executor, stats):
        reward = super().score(env, rows, prev_executor, stats)

        # One slice per job that ran this tick. A second satellite on the same job does not pay twice.
        paid: set[str] = set()
        for row in rows:
            if row["executed"] != "job":
                continue

            job_id = row["requested"]["job_id"]
            if job_id in paid:
                continue

            paid.add(job_id)
            job = env.jobs[job_id]
            steps = max(float(job["work_steps"]), 1.0)
            reward += self.tick_share * self.worth(job) / steps
        return reward
