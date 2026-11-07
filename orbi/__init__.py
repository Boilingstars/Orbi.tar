"""Orbi.tar: live Kepler gym for training and evaluating agents."""

from orbi.condition import Heartbeat, SatCondition, TaskCondition, sat_condition, world_heartbeat
from orbi.kepler import KeplerWorld, kepler_from_config
from orbi.satellite import Satellite

__all__ = [
    "EventConfig",
    "Heartbeat",
    "KeplerWorld",
    "Observation",
    "OpsEnv",
    "P01_INTRO",
    "P02_SHIFT",
    "P03_ENERGY",
    "P04_DEMAND",
    "Reward",
    "SCENARIOS_DIR",
    "SatCondition",
    "Satellite",
    "Task",
    "TaskCondition",
    "digest",
    "greedy_action",
    "kepler_from_config",
    "load",
    "random_masked_action",
    "resolve_scenario",
    "sat_condition",
    "validate",
    "world_heartbeat",
]

_LAZY = {
    "EventConfig",
    "Observation",
    "OpsEnv",
    "P01_INTRO",
    "P02_SHIFT",
    "P03_ENERGY",
    "P04_DEMAND",
    "Reward",
    "SCENARIOS_DIR",
    "Task",
    "digest",
    "greedy_action",
    "load",
    "random_masked_action",
    "resolve_scenario",
    "validate",
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in {"greedy_action", "random_masked_action"}:
        from orbi.greedy import greedy_action, random_masked_action

        return greedy_action if name == "greedy_action" else random_masked_action
    from orbi.env import (
        EventConfig,
        Observation,
        OpsEnv,
        P01_INTRO,
        P02_SHIFT,
        P03_ENERGY,
        P04_DEMAND,
        Reward,
        SCENARIOS_DIR,
        Task,
        digest,
        load,
        resolve_scenario,
        validate,
    )

    return locals()[name]
