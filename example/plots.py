"""Training plots from `progress.csv`. PNGs go in `<run>/plots/`. Two runs can be overlaid with `--run A --run B`."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_X = "time/total_timesteps"
_DEFAULT_CRITICAL_SOC = 20.0
_RUN_NAME = re.compile(
    r"^(?P<dd>\d{2})_(?P<mm>\d{2})_(?P<yy>\d{2})_(?P<hh>\d{2})_(?P<mi>\d{2})"
    r"(?:_lr_(?P<lr>[^_]+(?:_[^_]+)*)|_(?P<suite>[A-Za-z][A-Za-z0-9]*)_(?P<model>.+?))"
    r"(?:_(?P<n>\d+))?$"
)


def run_label(run_dir: Path | str) -> str:
    name = Path(run_dir).name
    match = _RUN_NAME.fullmatch(name)
    if not match:
        return name
    label = f"{match['dd']}.{match['mm']}.{match['yy']} {match['hh']}:{match['mi']}"
    if match["lr"]:
        label += f" lr={match['lr']}"
    else:
        label += f" {match['suite']} {match['model']}"

    if match["n"]:
        label += f" ({match['n']})"
    return label


def _critical_soc_pct(run_dir: Path) -> float:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return _DEFAULT_CRITICAL_SOC
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _DEFAULT_CRITICAL_SOC
    scenario = config.get("scenario")
    if not scenario:
        return _DEFAULT_CRITICAL_SOC
    path = Path(scenario)
    if not path.exists():
        return _DEFAULT_CRITICAL_SOC
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _DEFAULT_CRITICAL_SOC
    value = spec.get("model", {}).get("critical_soc_pct")
    return float(value) if value is not None else _DEFAULT_CRITICAL_SOC


def read_progress(run_dir: Path | str) -> dict[str, list[float]]:
    path = Path(run_dir) / "progress.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: run example.train first")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    columns: dict[str, list[float]] = {}
    for key in rows[0] if rows else []:
        values = []
        for row in rows:
            raw = row.get(key, "")
            values.append(float(raw) if raw not in ("", None) else float("nan"))
        columns[key] = values
    return columns


def _series(data: dict[str, list[float]], key: str) -> tuple[list[float], list[float]] | None:
    if key not in data or _X not in data:
        return None
    xs, ys = [], []
    for x, y in zip(data[_X], data[key]):
        if y == y:  # not NaN
            xs.append(x)
            ys.append(y)
    return (xs, ys) if ys else None


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _line(ax, data, key, label, **kw) -> bool:
    series = _series(data, key)
    if series is None:
        return False
    ax.plot(series[0], series[1], label=label, **kw)
    return True


def _chart(out_dir: Path, name: str, draw) -> Path | None:
    """One figure. `draw` returns false when the series is missing, and the file is not written."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if not draw(fig, ax):
        plt.close(fig)
        return None
    return _save(fig, out_dir, name)


def plot_run(run_dir: Path | str) -> list[Path]:
    run_dir = Path(run_dir)
    data = read_progress(run_dir)
    out_dir = run_dir / "plots"
    tag = run_label(run_dir)
    leftover = out_dir / "action_shares.png"
    if leftover.exists():
        leftover.unlink()

    def learning(_fig, ax) -> bool:
        if not _line(ax, data, "rollout/ep_rew_mean", "ep_rew_mean", color="tab:blue"):
            return False
        ax.set_title(f"Learning curve — {tag}")
        ax.set_xlabel("Timesteps (5-minute ticks)")
        ax.set_ylabel("Mean episode reward")
        ax.grid(alpha=0.3)
        return True

    def priority(_fig, ax) -> bool:
        ok = _line(ax, data, "ops/critical_done_on_time", "Priority 3 on time", color="tab:green")
        ok |= _line(ax, data, "ops/jobs_due_missed", "Missed jobs", color="tab:red")
        if not ok:
            return False
        ax.set_title(f"Priority service — {tag}")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Jobs per episode")
        ax.legend()
        ax.grid(alpha=0.3)
        return True

    def revenue(_fig, ax) -> bool:
        if not _line(ax, data, "ops/revenue_usd", "Revenue", color="tab:orange"):
            return False
        ax2 = ax.twinx()
        _line(ax2, data, "ops/jobs_completed", "Jobs completed", color="tab:green")
        ax.set_title(f"Revenue — {tag}")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("USD per episode")
        ax2.set_ylabel("Jobs completed")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
        ax.grid(alpha=0.3)
        return True

    def energy(_fig, ax) -> bool:
        if not _line(ax, data, "ops/min_soc_pct", "Min SoC", color="tab:blue"):
            return False
        critical = _critical_soc_pct(run_dir)
        ax.axhspan(0, critical, color="tab:red", alpha=0.12, zorder=0, label=f"Critical SoC ({critical:g}%)")
        ax.set_title(f"Energy — {tag}")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Min SoC, %")
        ax.set_ylim(0, 100)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        return True

    made = [
        path
        for path in (
            _chart(out_dir, "learning_curve.png", learning),
            _chart(out_dir, "priority_service.png", priority),
            _chart(out_dir, "revenue.png", revenue),
            _chart(out_dir, "energy.png", energy),
        )
        if path is not None
    ]
    return made


def compare_runs(runs: list[Path | str]) -> list[Path]:
    out_dir = Path(runs[0]) / "plots"
    made: list[Path] = []
    for key, name, title, ylabel in (
        ("rollout/ep_rew_mean", "compare_reward.png", "Learning curve", "Mean episode reward"),
        ("ops/critical_done_on_time", "compare_critical.png", "Priority service", "Priority 3 on time"),
        ("ops/revenue_usd", "compare_revenue.png", "Revenue", "USD per episode"),
    ):

        def draw(_fig, ax, key=key, title=title, ylabel=ylabel) -> bool:
            drawn = False
            for run in runs:
                drawn |= _line(ax, read_progress(run), key, run_label(run))
            if not drawn:
                return False
            ax.set_title(title)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(alpha=0.3)
            return True

        path = _chart(out_dir, name, draw)
        if path is not None:
            made.append(path)
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description="Training plots from progress.csv")
    parser.add_argument("--run", action="append", required=True, help="experiments/runs/<run> directory")
    args = parser.parse_args()
    made: list[Path] = []
    for run in args.run:
        made.extend(plot_run(run))
    if len(args.run) > 1:
        made.extend(compare_runs(args.run))
    for path in made:
        print(path)


if __name__ == "__main__":
    main()
