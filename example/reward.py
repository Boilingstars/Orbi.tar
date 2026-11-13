"""Reward for this package.

A finished job pays its dollar price. Progress on unfinished work is a
potential: completing a tick raises it, pausing a relay keeps it, and
resetting a downlink drops it. The step bonus is γΦ(s') − Φ(s), with γ
the PPO discount, so the undiscounted sum of bonuses telescopes to zero
when the episode potential starts and ends at zero.

`job_worth` still adds the priority bonus. The observation and the slot
ranker read that, so a loaded policy keeps the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from example.jobs import PRICE_DIV
from tools.reward import Reward


# Same divisor as job USD. The observation clips worth against this scale.
_PRIORITY_BONUS = (0.0, 25.0 / PRICE_DIV, 80.0 / PRICE_DIV)

# Matches PPOTrainer.gamma. A restart cycle then pays (γ − 1) β V / L < 0.
_GAMMA = 0.998


def job_worth(job: dict, priority_bonus: tuple[float, float, float] = _PRIORITY_BONUS) -> float:
    """Dollars plus priority, for the observation and the ranker. Not the return."""
    rank = min(max(int(job.get("priority", 1)) - 1, 0), len(priority_bonus) - 1)
    return float(job.get("value_usd", 0.0)) + float(priority_bonus[rank])


@dataclass
class MyReward(Reward):
    """Dollar completions plus potential-based progress. No penalties."""

    progress: float = 0.0
    below_reserve: float = 0.0
    brownout: float = 0.0
    blocked: float = 0.0
    executor_switch: float = 0.0
    conflict: float = 0.0
    early_calibrate: float = 0.0
    gamma: float = _GAMMA
    beta: float = 1.0
    _phi: float = field(default=0.0, repr=False, compare=False)

    def worth(self, job: dict) -> float:
        return float(job.get("value_usd", 0.0))

    def on_complete(self, job: dict) -> float:
        return self.worth(job)

    def on_miss(self, job: dict) -> float:
        return 0.0

    def potential(self, env) -> float:
        """β Σ V_j (L_j − r_j) / L_j over unfinished jobs still inside their window."""
        total = 0.0
        k = int(env.k)
        for job in env.jobs.values():
            if job["completed_step"] is not None:
                continue

            if int(job["remaining_steps"]) <= 0:
                continue

            if k >= int(job["deadline_step"]):
                continue

            length = max(float(job["work_steps"]), 1.0)
            done = max(length - float(job["remaining_steps"]), 0.0)
            total += self.worth(job) * done / length
        return self.beta * total

    def reset_potential(self, env) -> None:
        self._phi = self.potential(env)

    def score(self, env, rows, prev_executor, stats):
        reward = super().score(env, rows, prev_executor, stats)

        # Φ(s') is zero on the terminal tick so leftover progress does not leak past T.
        terminal = env.k >= env.steps_total or (
            env.max_steps is not None and env.k >= env.max_steps
        )
        phi_next = 0.0 if terminal else self.potential(env)
        reward += self.gamma * phi_next - self._phi
        self._phi = phi_next
        return reward
