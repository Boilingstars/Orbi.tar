"""Package wiring. Supplies this package's reward and observation."""

from __future__ import annotations

from example.observation import ShiftObservation
from example.reward import MyReward
from orbi.env import Task


def bind(reward=None, observation=None):
    if isinstance(reward, type):
        reward = reward()
    return reward or MyReward(), observation or ShiftObservation()


def make_task(scenario="p02", reward=None, observation=None, **options) -> Task:
    reward, observation = bind(reward, observation)
    return Task(scenario, reward, observation, **options)
