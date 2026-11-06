"""Run a policy on a shift and optionally write the command journal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from orbi.env import FAR_JOB_P, GROUND_POINTS, EventConfig, OpsEnv, load, resolve_scenario
from orbi.greedy import greedy_action, random_masked_action

Actor = Callable[[dict], np.ndarray]


def run_episode(env: OpsEnv, actor: Actor, seed: int | None = None) -> dict:
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


def _load_ppo(model_path: str | Path):
    """Веса политики без оптимизатора: чекпоинт может быть с двумя группами Adam."""
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


def make_actor(
    policy: str,
    model_path: str | Path | None,
    downlink_limit: int,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> Actor:
    if policy == "ppo":
        if model_path is None:
            raise ValueError("Для policy=ppo нужен --model")
        model = _load_ppo(model_path)

        def actor(obs: dict) -> np.ndarray:
            masks = np.asarray(obs["mask"], dtype=bool).reshape(-1)
            # По умолчанию сэмплируем, как MaskablePPO при обучении. Argmax по
            # 48 категориальным (idle — один класс, задания размазаны по K)
            # почти всегда выбирает ожидание, даже когда доля заданий ~13%.
            action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
            return np.asarray(action)

        return actor
    if policy == "random":
        return lambda obs: random_masked_action(obs, rng)
    return lambda obs: greedy_action(obs, downlink_limit=downlink_limit)


def evaluate(
    scenario: str | Path = "p02",
    goal: str = "priority",
    policy: str = "greedy",
    model_path: str | Path | None = None,
    episodes: int = 1,
    events: str = "demo",
    seed: int = 0,
    top_k: int = 32,
    journal_dir: str | Path | None = None,
    deterministic: bool = False,
    ground_points: int = GROUND_POINTS,
    far_job_p: float = FAR_JOB_P,
) -> dict:
    path = resolve_scenario(scenario)
    spec = load(path)
    event_cfg = EventConfig() if events == "random" else EventConfig.off()

    rng = np.random.default_rng(seed)
    env = OpsEnv(
        scenario=spec,
        goal=goal,
        events=event_cfg,
        top_k=top_k,
        seed=seed,
        ground_points=ground_points,
        far_job_p=far_job_p,
    )
    actor = make_actor(policy, model_path, env.downlink_limit, rng, deterministic=deterministic)
    if policy == "ppo" and not deterministic:
        import torch

        torch.manual_seed(seed)

    rows = []
    for i in range(episodes):
        row = run_episode(env, actor, seed=seed + i)
        if journal_dir is not None:
            row["journal"] = str(_write_journal(env, Path(journal_dir), i))
        rows.append(row)

    def mean(getter) -> float:
        return float(np.mean([getter(r) for r in rows]))

    return {
        "scenario": spec["meta"]["id"],
        "policy": policy,
        "goal": goal,
        "events": events,
        "episodes": episodes,
        "deterministic": bool(deterministic) if policy == "ppo" else True,
        "mean_return": mean(lambda r: r["return"]),
        "jobs_completed": mean(lambda r: r["summary"].get("jobs_completed", 0)),
        "jobs_due_missed": mean(lambda r: r["summary"].get("jobs_due_missed", 0)),
        "critical_done_on_time": mean(
            lambda r: r["summary"].get("critical_jobs_completed_on_time", 0)
        ),
        "critical_due": mean(lambda r: r["summary"].get("critical_jobs_due", 0)),
        "revenue_usd": mean(lambda r: r["summary"].get("revenue_usd", 0.0)),
        "below_reserve_steps": mean(lambda r: r["summary"].get("below_reserve_satellite_steps", 0)),
        "brownout_steps": mean(lambda r: r["summary"].get("brownout_satellite_steps", 0)),
        "min_soc_pct": mean(lambda r: r["summary"].get("minimum_soc_pct", 0.0)),
        "blocked_commands": mean(lambda r: r["summary"].get("blocked_command_count", 0)),
        "idle_share": mean(lambda r: r["ops"].get("idle_share", 0.0)),
        "calibrate_share": mean(lambda r: r["ops"].get("calibrate_share", 0.0)),
        "job_share": mean(lambda r: r["ops"].get("job_share", 0.0)),
        "executor_switches": mean(lambda r: r["ops"].get("executor_switches", 0.0)),
        "conflicts_repaired": mean(lambda r: r["ops"].get("conflicts_repaired", 0.0)),
        "episodes_detail": [{k: v for k, v in r.items() if k != "ops"} for r in rows],
    }


def _write_journal(env: OpsEnv, out_dir: Path, episode: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"journal_{episode}.json"
    path.write_text(json.dumps(env.journal(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка политики на сценарии кейса")
    parser.add_argument("--scenario", default="p02")
    parser.add_argument("--goal", default="priority", choices=["priority", "revenue"])
    parser.add_argument("--policy", default="greedy", choices=["greedy", "random", "ppo"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--events", default="demo", choices=["demo", "random", "none"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--ground-points", type=int, default=GROUND_POINTS)
    parser.add_argument("--far-job-p", type=float, default=FAR_JOB_P)
    parser.add_argument("--journal", default=None, help="каталог для journal_<episode>.json")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="argmax вместо сэмпла; для PPO почти всегда idle",
    )
    args = parser.parse_args()
    report = evaluate(
        scenario=args.scenario,
        goal=args.goal,
        policy=args.policy,
        model_path=args.model,
        episodes=args.episodes,
        events=args.events,
        seed=args.seed,
        top_k=args.top_k,
        journal_dir=args.journal,
        deterministic=args.deterministic,
        ground_points=args.ground_points,
        far_job_p=args.far_job_p,
    )
    report.pop("episodes_detail", None)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
