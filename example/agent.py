"""Actors for this package. Each returns one action integer per satellite."""

from __future__ import annotations

import numpy as np

from orbi.greedy import greedy_action, random_masked_action
from tools.agent import Agent


class GreedyAgent(Agent):
    name = "greedy"

    def __init__(self, downlink_limit: int) -> None:
        self.downlink_limit = int(downlink_limit)

    def act(self, obs: dict, *, deterministic: bool = False) -> np.ndarray:
        return greedy_action(obs, downlink_limit=self.downlink_limit)


class RandomAgent(Agent):
    name = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, obs: dict, *, deterministic: bool = False) -> np.ndarray:
        return random_masked_action(obs, self.rng)


class PPOAgent(Agent):
    """A loaded policy. Sampling matches training; argmax is optional."""

    controller = "dispatcher"

    def __init__(self, model, name: str = "ppo", deterministic: bool = False) -> None:
        self.model = model
        self.name = name
        self.deterministic = bool(deterministic)

    def act(self, obs: dict, *, deterministic: bool = False) -> np.ndarray:
        masks = np.asarray(obs["mask"], dtype=bool).reshape(-1)
        action, _ = self.model.predict(
            obs,
            action_masks=masks,
            deterministic=deterministic or self.deterministic,
        )
        return np.asarray(action)


class Dispatcher(PPOAgent):
    """Fleet policy: one encoder, then satellites choose in index order."""

    controller = "dispatcher"

    def __init__(self, model, name: str = "dispatcher", deterministic: bool = False) -> None:
        super().__init__(model, name=name, deterministic=deterministic)


class Ctde(PPOAgent):
    """On-board policy: the same decoder on a local view, critic on the whole fleet."""

    controller = "ctde"

    def __init__(self, model, name: str = "ctde", deterministic: bool = False) -> None:
        super().__init__(model, name=name, deterministic=deterministic)
