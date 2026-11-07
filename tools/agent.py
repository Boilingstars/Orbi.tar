"""Agent contract.

One `act` call returns one integer per satellite (job slot, idle, or calibrate).
A dispatcher, a CTDE actor, and a future autoregressive decoder all use this
shape. The network that produces the integers stays in the user package.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Agent:
    name: str = "agent"

    def act(self, obs: dict, *, deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, obs: dict, *, deterministic: bool = False) -> np.ndarray:
        return self.act(obs, deterministic=deterministic)

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not save weights")

    def load(self, path: str) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not load weights")
