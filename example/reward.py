"""Две награды примера: приоритетное обслуживание и коммерческая отдача."""

from __future__ import annotations

from dataclasses import dataclass

from orbi.env import Reward


@dataclass
class PriorityReward(Reward):
    name: str = "priority"
    complete_priority: tuple[float, float, float] = (1.5, 4.0, 10.0)
    miss_priority: tuple[float, float, float] = (0.5, 2.0, 8.0)
    value_scale: float = 0.01

    def on_complete(self, job: dict) -> float:
        return self.complete_priority[job["priority"] - 1] + self.value_scale * job["value_usd"]

    def on_miss(self, job: dict) -> float:
        return -self.miss_priority[job["priority"] - 1]


@dataclass
class RevenueReward(Reward):
    name: str = "revenue"
    value_scale: float = 0.04
    miss_value: float = 0.02

    def on_complete(self, job: dict) -> float:
        return self.value_scale * job["value_usd"]

    def on_miss(self, job: dict) -> float:
        return -self.miss_value * job["value_usd"]
