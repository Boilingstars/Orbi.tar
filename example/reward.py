"""Reward for this package.

A job pays a slice of worth on every executed tick and the rest when it
finishes. Worth is dollars plus a priority bonus, without scaling by length:
a short job is cheaper to finish, not three times more valuable. An untouched
deadline costs a fraction of worth only if the job was packed or P3; the
unscheduled pool tail is free. Started-then-dropped work pays the unfinished
slice immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

from example.jobs import PRICE_DIV, work_ref_ticks
from tools.reward import Reward


# Untouched deadline: enough that ignoring the queue is visible next to dollars.
_UNTOUCHED_MISS = 0.40

# 80% work bin in ticks. Tick credit uses the job's own length; this is the unit.
_WORK_REF = work_ref_ticks

# Same divisor as job USD, so P3 bonus and calibrate stay next to dollars.
_PRIORITY_BONUS = (0.0, 25.0 / PRICE_DIV, 80.0 / PRICE_DIV)


def job_worth(job: dict, priority_bonus: tuple[float, float, float] = _PRIORITY_BONUS) -> float:
    """Dollars plus priority. Length is not a multiplier."""
    rank = min(max(int(job.get("priority", 1)) - 1, 0), len(priority_bonus) - 1)
    return float(job.get("value_usd", 0.0)) + float(priority_bonus[rank])


@dataclass
class MyReward(Reward):
    """Tick slice of worth, remainder on finish, unfinished slice on an abandon."""

    progress: float = 0.0
    executor_switch: float = 0.25
    priority_bonus: tuple[float, float, float] = _PRIORITY_BONUS
    advance: float = 0.0
    tick_share: float = 0.25
    early_calibrate: float = 4.0 / PRICE_DIV
    below_reserve: float = 1.0

    def worth(self, job: dict) -> float:
        return job_worth(job, self.priority_bonus)

    def on_complete(self, job: dict) -> float:
        return (1.0 - self.tick_share) * self.worth(job)

    def on_miss(self, job: dict) -> float:
        worth = self.worth(job)
        started = int(job["remaining_steps"]) < int(job["work_steps"])
        if started:
            leftover = job["remaining_steps"] / max(float(job["work_steps"]), 1.0)
            return -leftover * worth
        # Jammed pool tail was never shown. Only packed, started, or P3 misses count.
        if int(job.get("priority", 1)) == 3 or job.get("ever_packed"):
            return -_UNTOUCHED_MISS * worth
        return 0.0

    def score(self, env, rows, prev_executor, stats):
        reward = super().score(env, rows, prev_executor, stats)
        mask = getattr(env, "_mask", None)
        executable = bool(mask is not None and getattr(env, "K", 0) and mask[:, : env.K].any())

        # Early calibrate in idle is free; with a legal job it must show next to dollars.
        if not executable:
            reward += self.early_calibrate * float(stats.get("early_calibrations", 0.0))

        paid: set[str] = set()
        work_ref = max(float(_WORK_REF(env)), 1.0)
        for row in rows:
            if row["executed"] != "job":
                continue

            job_id = row["requested"]["job_id"]
            if job_id in paid:
                continue

            job = env.jobs[job_id]
            paid.add(job_id)
            reward += self.tick_share * self.worth(job) / max(float(job["work_steps"]), work_ref)
        return reward
