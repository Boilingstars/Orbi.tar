"""Reward functions for this package."""

from __future__ import annotations

from dataclasses import dataclass

from tools.reward import Reward


@dataclass
class MyReward(Reward):
    """Priority on a sparse mask: a legal job must beat a fleet of idle/calibrate.

    `opportunity` is weak (one tick on a slot); `on_complete` is strong (finishing).
    Fresh calibration costs `early_calibrate` until `calibrate_free_at` of the window.
    """

    opportunity: float = 1.0
    complete_priority: tuple[float, float, float] = (12.0, 24.0, 48.0)
    miss_priority: tuple[float, float, float] = (0.0, 1.0, 4.0)
    value_scale: float = 0.04
    early_calibrate: float = 1.0

    def on_complete(self, job: dict) -> float:
        return self.complete_priority[job["priority"] - 1] + self.value_scale * job["value_usd"]

    def on_miss(self, job: dict) -> float:
        return -self.miss_priority[job["priority"] - 1]

    def score(self, env, rows, prev_executor, stats):
        # Progress, conflicts, early calibration, completions, and misses.
        reward = super().score(env, rows, prev_executor, stats)

        # One taken legal job is worth more than a tick of idle.
        n_sats = env.n_sats
        legal = int(env._mask[:n_sats, : env.K].any(axis=1).sum())
        taken = int(stats.get("job", 0.0))
        if legal > 0:
            reward += self.opportunity * taken / legal
        return reward
