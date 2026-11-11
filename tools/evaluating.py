"""Evaluate one actor on a shift.

The command line is parsed here. A user package such as `example` is only the
entry point: it passes its model classes and calls `launch`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orbi.env import (
    GROUND_POINTS,
    JOB_DIFFICULTY,
    NEIGHBOR_MEMORY,
    EventConfig,
    OpsEnv,
    load,
    resolve_scenario,
)
from orbi.greedy import greedy_action, random_masked_action


# Same names as `Run`. Train, the greedy anchor, and evaluation read this list.
SHIFT_FIELDS = (
    "top_k",
    "baseline_repair",
    "neighbor_method",
    "neighbor_memory",
    "ground_points",
    "ground_region",
    "job_difficulty",
)


def shift_kwargs(source) -> dict:
    """One shift's sampling and slot fields. Callers still pass the scenario and events."""
    return {name: getattr(source, name) for name in SHIFT_FIELDS}


@dataclass
class Eval:
    """One evaluation. `launch` fills this from the command line."""

    scenario: str = "p03"
    policy: str = "greedy"
    model: str | None = None
    episodes: int = 1
    events: str = "demo"
    seed: int = 0
    top_k: int | None = 64
    ground_points: int = GROUND_POINTS
    ground_region: str | None = "world"
    job_difficulty: float = JOB_DIFFICULTY
    baseline_repair: bool = False
    neighbor_method: str = "competitors"
    neighbor_memory: int = NEIGHBOR_MEMORY
    scripted_events: list | None = None
    journal: str | None = None
    deterministic: bool = False


def load_ppo(model_path: str | Path):
    """Policy weights without the optimizer: a checkpoint may have two Adam groups."""
    from sb3_contrib import MaskablePPO

    original = MaskablePPO.set_parameters

    def _set(self, params, exact_match=True, device="auto"):
        policy_only = {key: value for key, value in params.items() if "optimizer" not in key}
        return original(self, policy_only, exact_match=False, device=device)

    MaskablePPO.set_parameters = _set  # type: ignore[method-assign]
    try:
        return MaskablePPO.load(str(model_path))
    finally:
        MaskablePPO.set_parameters = original  # type: ignore[method-assign]


def parse(models: list, argv: list[str] | None = None) -> Eval:
    """Read the fixed evaluation flags. `models` are the package's neural policies."""
    names = ["greedy", "random", "ppo", *(model.__name__ for model in models)]
    parser = argparse.ArgumentParser(description="Evaluate a policy on a shift scenario")
    parser.add_argument("--scenario", default="p03")
    parser.add_argument("--policy", default="greedy", choices=names)
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--events", default="demo", choices=["demo", "random", "none"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--ground-points", type=int, default=GROUND_POINTS)
    parser.add_argument("--job-difficulty", type=float, default=JOB_DIFFICULTY)
    parser.add_argument("--journal", default=None, help="directory for journal_<episode>.json")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="argmax instead of sampling; for a neural policy this is almost always idle",
    )
    args = parser.parse_args(argv)
    return Eval(
        scenario=args.scenario,
        policy=args.policy,
        model=args.model,
        episodes=args.episodes,
        events=args.events,
        seed=args.seed,
        top_k=args.top_k,
        ground_points=args.ground_points,
        job_difficulty=args.job_difficulty,
        journal=args.journal,
        deterministic=args.deterministic,
    )


def _neural(policy: str, models: list) -> bool:
    return policy == "ppo" or any(model.__name__ == policy for model in models)


def _actor(settings: Eval, env: OpsEnv, rng: np.random.Generator, models: list):
    if settings.policy == "random":
        return lambda obs: random_masked_action(obs, rng)

    if not _neural(settings.policy, models):
        limit = env.downlink_limit
        return lambda obs: greedy_action(obs, downlink_limit=limit)

    if not settings.model:
        raise ValueError(f"{settings.policy} requires --model")
    loaded = load_ppo(settings.model)
    deterministic = settings.deterministic

    def act(obs):
        masks = np.asarray(obs["mask"], dtype=bool).reshape(-1)
        action, _ = loaded.predict(obs, action_masks=masks, deterministic=deterministic)
        return np.asarray(action)

    return act


_SUMMARY = (
    ("jobs_completed", "jobs_completed"),
    ("jobs_due_missed", "jobs_due_missed"),
    ("critical_done_on_time", "critical_jobs_completed_on_time"),
    ("critical_due", "critical_jobs_due"),
    ("revenue_usd", "revenue_usd"),
    ("below_reserve_steps", "below_reserve_satellite_steps"),
    ("brownout_steps", "brownout_satellite_steps"),
    ("min_soc_pct", "minimum_soc_pct"),
    ("blocked_commands", "blocked_command_count"),
)
_OPS = ("idle_share", "calibrate_share", "job_share", "executor_switches", "conflicts_repaired")


def shift_metrics(rows: list[dict]) -> dict:
    """Means a baseline table and an evaluation report both print."""

    def mean(section: str, key: str) -> float:
        return float(np.mean([row[section].get(key, 0.0) for row in rows]))

    report = {"mean_return": float(np.mean([row["return"] for row in rows]))}
    for out, src in _SUMMARY:
        report[out] = mean("summary", src)
    for key in _OPS:
        report[key] = mean("ops", key)
    return report


def run_episode(env: OpsEnv, actor, seed: int | None = None) -> dict:
    obs, _ = env.reset(seed=seed)
    total = 0.0
    steps = 0
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(actor(obs))
        total += float(reward)
        steps += 1
    return {
        "return": total,
        "steps": steps,
        "summary": info.get("summary", {}),
        "ops": info.get("ops", {}),
    }


def _write_journal(env: OpsEnv, out_dir: Path, episode: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"journal_{episode}.json"
    path.write_text(json.dumps(env.journal(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def evaluate(settings: Eval, models: list | None = None) -> dict:
    """Run `settings` for one actor. `models` names the package's neural policies."""
    models = list(models or [])
    path = resolve_scenario(settings.scenario)
    spec = load(path)

    # One environment. `random` is the only event mode that keeps drawing.
    events = EventConfig() if settings.events == "random" else EventConfig.off()
    rng = np.random.default_rng(settings.seed)
    env = OpsEnv(
        scenario=spec,
        events=events,
        seed=settings.seed,
        scripted_events=list(settings.scripted_events or []),
        **shift_kwargs(settings),
    )
    actor = _actor(settings, env, rng, models)
    if _neural(settings.policy, models) and not settings.deterministic:
        import torch

        torch.manual_seed(settings.seed)

    # One row per episode, optionally with the journal path beside it.
    rows = []
    for i in range(settings.episodes):
        row = run_episode(env, actor, seed=settings.seed + i)
        if settings.journal is not None:
            row["journal"] = str(_write_journal(env, Path(settings.journal), i))
        rows.append(row)

    report = {
        "scenario": spec["meta"]["id"],
        "policy": settings.policy,
        "events": settings.events,
        "episodes": settings.episodes,
        "deterministic": bool(settings.deterministic) if _neural(settings.policy, models) else True,
    }
    report.update(shift_metrics(rows))
    report["episodes_detail"] = [{k: v for k, v in row.items() if k != "ops"} for row in rows]
    return report


def launch(models: list) -> dict:
    """Parse the command line, evaluate, and print the report."""
    report = evaluate(parse(models), models)
    report.pop("episodes_detail", None)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report
