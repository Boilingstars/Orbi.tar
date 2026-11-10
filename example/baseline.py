"""Compare agents on one scenario.

Edit `SCENARIO` and `AGENTS`, then run this file. Each row is an agent class
from `example.agent` and a weights path. Greedy and Random take no weights.
Episode count and the other shift settings stay at the evaluator defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from example.agent import GreedyAgent, PPOAgent, RandomAgent
from orbi.env import EventConfig, OpsEnv, load, resolve_scenario
from tools.evaluating import Eval, load_ppo, run_episode, shift_metrics


SCENARIO = "p02"

AGENTS = [
    (GreedyAgent, None),
    (RandomAgent, None),
    # (PPOAgent, "agent/models/ops.zip"),
]


def _agent(agent_cls, weights, env: OpsEnv, rng: np.random.Generator):
    if issubclass(agent_cls, RandomAgent):
        return agent_cls(rng)

    if issubclass(agent_cls, GreedyAgent):
        return agent_cls(env.downlink_limit)

    if not weights:
        raise SystemExit(f"{agent_cls.__name__} needs a weights path")
    return agent_cls(load_ppo(weights), name=Path(weights).stem)


def _summarize(name: str, rows: list[dict], settings: Eval) -> dict:
    report = {
        "agent": name,
        "scenario": settings.scenario,
        "episodes": settings.episodes,
    }
    report.update(shift_metrics(rows))
    return report


def main() -> None:
    settings = Eval(scenario=SCENARIO)
    spec = load(resolve_scenario(settings.scenario))

    # Same event switch the evaluator uses. The default is a scripted demo, which is off here.
    events = EventConfig() if settings.events == "random" else EventConfig.off()
    rng = np.random.default_rng(settings.seed)
    env = OpsEnv(
        scenario=spec,
        events=events,
        top_k=settings.top_k,
        seed=settings.seed,
        ground_points=settings.ground_points,
        job_difficulty=settings.job_difficulty,
    )
    report = []

    for agent_cls, weights in AGENTS:
        agent = _agent(agent_cls, weights, env, rng)

        # PPOAgent's default is stochastic, so seed the RNG if needed.
        if isinstance(agent, PPOAgent) and not agent.deterministic:
            import torch

            torch.manual_seed(settings.seed)
        rows = [run_episode(env, agent, seed=settings.seed + i) for i in range(settings.episodes)]
        name = Path(weights).stem if weights else agent.name
        report.append(_summarize(name, rows, settings))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
