"""Задачи примера: priority и revenue на одном сценарии смены."""

from __future__ import annotations

from example.observation import ShiftObservation
from example.reward import PriorityReward, RevenueReward
from orbi.env import Task

_REWARDS = {"priority": PriorityReward, "revenue": RevenueReward}


def bind(goal: str = "priority", reward=None, observation=None):
    if goal not in _REWARDS:
        raise ValueError(f"goal must be one of {tuple(_REWARDS)}")
    if reward is None:
        reward = _REWARDS[goal]()
    if observation is None:
        observation = ShiftObservation()
    return reward, observation


def make_task(scenario="p02", goal: str = "priority", **options) -> Task:
    reward, observation = bind(goal)
    return Task(scenario, reward, observation, **options)
