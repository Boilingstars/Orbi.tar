"""Custom satellite: extra attributes, neighbor memory, extra take-job rules."""

from __future__ import annotations

import numpy as np

from orbi.env import NEIGHBOR_FEATURES, NEIGHBOR_MEMORY
from orbi.satellite import Satellite as SatelliteShell

METHODS = ("competitors", "nearest_index")


class Satellite(SatelliteShell):
    """Shift satellite with grouping extras. Override `can_take` for extra rules."""

    def can_take(self, world, job: dict | None, model: dict, k: int) -> tuple[bool, str]:
        ok, reason = super().can_take(world, job, model, k)
        if not ok:
            return ok, reason
        return True, reason


class NeighborView:
    """Fixed-size memory of other satellites for one on-board actor."""

    def __init__(self, method: str = "competitors", memory: int = NEIGHBOR_MEMORY) -> None:
        if method not in METHODS:
            raise ValueError(f"neighbor_method must be one of {METHODS}")
        if memory < 1:
            raise ValueError("neighbor_memory must be >= 1")
        self.method = method
        self.memory = int(memory)

    def select(self, env, sat_i: int, mask: np.ndarray) -> list[int]:
        if sat_i >= env.n_sats:
            return []
        if self.method == "nearest_index":
            return self._nearest_index(env, sat_i)
        return self._competitors(env, sat_i, mask)

    def encode_all(self, env, mask: np.ndarray) -> np.ndarray:
        out = np.zeros((env.N, self.memory, NEIGHBOR_FEATURES), dtype=np.float32)
        for i in range(env.n_sats):
            chosen = self.select(env, i, mask)
            out[i] = self._features(env, i, chosen, mask)
        return out

    def _nearest_index(self, env, sat_i: int) -> list[int]:
        n = env.n_sats
        if n <= 1:
            return []
        chosen: list[int] = []
        offset = 1
        while len(chosen) < self.memory and offset < n:
            for other in ((sat_i - offset) % n, (sat_i + offset) % n):
                if other == sat_i or other in chosen:
                    continue
                chosen.append(other)
                if len(chosen) >= self.memory:
                    break
            offset += 1
        return chosen

    def _competitors(self, env, sat_i: int, mask: np.ndarray) -> list[int]:
        mine = np.flatnonzero(mask[sat_i, : env.K])
        if mine.size == 0:
            return []
        overlap = np.zeros(env.n_sats, dtype=np.int32)
        for slot in mine:
            overlap[np.flatnonzero(mask[: env.n_sats, slot])] += 1
        overlap[sat_i] = 0
        order = np.lexsort((np.arange(env.n_sats), -overlap))
        return [int(j) for j in order if overlap[j] > 0][: self.memory]

    def _features(self, env, sat_i: int, chosen: list[int], mask: np.ndarray) -> np.ndarray:
        rows = np.zeros((self.memory, NEIGHBOR_FEATURES), dtype=np.float32)
        own_slots = set(np.flatnonzero(mask[sat_i, : env.K]).tolist())
        pool = [sat_i] + chosen
        soc = np.array([env.fleet[j].soc for j in pool], dtype=np.float32)
        denom = max(len(pool) - 1, 1)
        ranks = np.argsort(np.argsort(-soc)).astype(np.float32) / denom
        valid = max(env.calibration_valid, 1)
        last = env._last_executor
        for row, other in enumerate(chosen):
            sat = env.fleet[other]
            shared = own_slots.intersection(np.flatnonzero(mask[other, : env.K]).tolist())
            holds = False
            for slot in shared:
                if slot >= len(env._candidates):
                    continue
                if last.get(env._candidates[slot].job_id) == sat.sat_id:
                    holds = True
                    break
            rows[row] = (
                1.0,
                soc[row + 1],
                np.clip((valid - sat.calibration_age_steps) / valid, -1.0, 1.0),
                float(sat.available(env.k)),
                min(len(shared), env.K) / env.K,
                float(holds),
                float(ranks[row + 1]),
            )
        return rows


__all__ = ["METHODS", "NeighborView", "Satellite"]
