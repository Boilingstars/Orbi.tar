"""Сравнение greedy, random и обученной политики на одном сценарии.

Таблица в `agent/figures/comparison.json`, столбики — в `agent/figures/baseline_*.png`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from example.evaluate import evaluate  # noqa: E402
from example.telemetry import FIGURES_DIR  # noqa: E402

_REPO_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = _REPO_DIR / "agent" / "models"

_BARS = (
    ("critical_done_on_time", "prio 3 в срок"),
    ("jobs_completed", "заданий выполнено"),
    ("jobs_due_missed", "просрочено"),
    ("revenue_usd", "выручка, USD"),
)


def _bar_chart(report: dict, out_dir: Path, tag: str) -> Path:
    policies = [k for k in ("greedy", "random", "ppo") if report.get(k)]
    fig, axes = plt.subplots(1, len(_BARS), figsize=(4 * len(_BARS), 3.6))
    for ax, (key, title) in zip(axes, _BARS):
        values = [report[p][key] for p in policies]
        ax.bar(policies, values, color=["#c7c7c7", "#e6b0aa", "#58d68d"][: len(policies)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    mode = "argmax" if report.get("deterministic") else "сэмпл"
    fig.suptitle(f"Бейзлайн против RL ({mode}) — {tag}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"baseline_{tag}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def compare(
    scenario: str = "p02",
    goal: str = "priority",
    episodes: int = 1,
    events: str = "demo",
    model_path: str | Path | None = None,
    seed: int = 0,
    top_k: int = 32,
    deterministic: bool = False,
) -> dict:
    common = dict(scenario=scenario, goal=goal, episodes=episodes, events=events, seed=seed, top_k=top_k)
    report: dict = {
        "when": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "goal": goal,
        "events": events,
        "deterministic": deterministic,
        "greedy": evaluate(policy="greedy", **common),
        "random": evaluate(policy="random", **common),
    }
    model = Path(model_path) if model_path else MODELS_DIR / f"ops_{goal}.zip"
    if model.exists():
        report["ppo"] = evaluate(policy="ppo", model_path=model, deterministic=deterministic, **common)
        report["model"] = str(model)
    for key in ("greedy", "random", "ppo"):
        if key in report:
            report[key].pop("episodes_detail", None)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Бейзлайн против обученной политики")
    parser.add_argument("--scenario", default="p02")
    parser.add_argument("--goal", default="priority", choices=["priority", "revenue"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--events", default="demo", choices=["demo", "random", "none"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="argmax PPO; по умолчанию сэмпл, как при обучении",
    )
    parser.add_argument("--out", default=None, help="каталог для comparison.json и графика")
    args = parser.parse_args()

    report = compare(
        scenario=args.scenario,
        goal=args.goal,
        episodes=args.episodes,
        events=args.events,
        model_path=args.model,
        seed=args.seed,
        top_k=args.top_k,
        deterministic=args.deterministic,
    )
    out_dir = Path(args.out) if args.out else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{report['greedy']['scenario']}_{args.goal}"
    chart = _bar_chart(report, out_dir, tag)
    out = out_dir / "comparison.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nСохранено: {out}\n            {chart}")


if __name__ == "__main__":
    main()
