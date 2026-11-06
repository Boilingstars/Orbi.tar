"""Greedy template: JSON initial state in, one action vector per tick."""

from __future__ import annotations

import numpy as np


_JOB_PRIO = slice(0, 3)
_JOB_VALUE = 3
_JOB_REMAINING = 5
_JOB_DEADLINE = 7
_JOB_DOWNLINK = 8
_SAT_SOC = 0
_SAT_CAL_AGE = 3
_EDGE_LAST_EXECUTOR = 4

CALIBRATE_AT = 0.85


def greedy_action(
    obs: dict[str, np.ndarray],
    downlink_limit: int = 2,
    calibrate_at: float = CALIBRATE_AT,
) -> np.ndarray:
    mask = np.asarray(obs["mask"], dtype=bool)
    jobs = np.asarray(obs["jobs"], dtype=np.float32)
    sats = np.asarray(obs["sats"], dtype=np.float32)
    edges = np.asarray(obs["edges"], dtype=np.float32)
    n, width = mask.shape
    k = width - 2
    idle, calibrate = k, k + 1
    action = np.full(n, idle, dtype=np.int64)
    priority = jobs[:, _JOB_PRIO].argmax(axis=1) + 1
    has_job = jobs[:, _JOB_PRIO].sum(axis=1) > 0
    remaining = np.maximum(jobs[:, _JOB_REMAINING], 1e-3)
    rate = jobs[:, _JOB_VALUE] / remaining
    order = sorted(
        (slot for slot in range(k) if has_job[slot] and mask[:, slot].any()),
        key=lambda s: (-priority[s], -rate[s], jobs[s, _JOB_DEADLINE]),
    )
    taken = np.zeros(n, dtype=bool)
    downlinks = 0
    for slot in order:
        if jobs[slot, _JOB_DOWNLINK] > 0.5 and downlinks >= downlink_limit:
            continue
        free = np.flatnonzero(mask[:, slot] & ~taken)
        if free.size == 0:
            continue
        pick = int(free[np.lexsort((-sats[free, _SAT_SOC], -edges[free, slot, _EDGE_LAST_EXECUTOR]))[0]])
        action[pick] = slot
        taken[pick] = True
        downlinks += int(jobs[slot, _JOB_DOWNLINK] > 0.5)
    for i in range(n):
        if taken[i] or not mask[i, calibrate]:
            continue
        if sats[i, _SAT_CAL_AGE] >= calibrate_at:
            action[i] = calibrate
    return action


def random_masked_action(obs: dict[str, np.ndarray], rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    mask = np.asarray(obs["mask"], dtype=bool)
    n, width = mask.shape
    action = np.full(n, width - 2, dtype=np.int64)
    for i in range(n):
        valid = np.flatnonzero(mask[i])
        if valid.size:
            action[i] = int(rng.choice(valid))
    return action


def run(scenario="p02", seed: int = 0, goal: str = "priority") -> dict:
    """Solve one episode from initial JSON, choosing a command each tick."""
    from orbi.env import EventConfig, OpsEnv, load, resolve_scenario

    spec = load(resolve_scenario(scenario) if not isinstance(scenario, dict) else scenario)
    env = OpsEnv(scenario=spec, goal=goal, events=EventConfig.off(), seed=seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        action = greedy_action(obs, downlink_limit=env.downlink_limit)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
    return {"return": total, "summary": info.get("summary", {}), "ops": info.get("ops", {})}


__all__ = ["CALIBRATE_AT", "greedy_action", "random_masked_action", "run"]
