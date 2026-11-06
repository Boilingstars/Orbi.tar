"""Графики обучения из `progress.csv`.

PNG кладутся в `<run>/plots/` и копируются в `agent/figures/`. Два запуска
сравниваются на одном полотне: `--run A --run B`.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from example.telemetry import FIGURES_DIR  # noqa: E402

_X = "time/total_timesteps"


def read_progress(run_dir: Path | str) -> dict[str, list[float]]:
    path = Path(run_dir) / "progress.csv"
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}: сначала запустите agent.train")
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
        if y == y:  # не NaN
            xs.append(x)
            ys.append(y)
    return (xs, ys) if ys else None


def _save(fig, out_dir: Path, name: str, copies: list[Path], prefix: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    for target in copies:
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / f"{prefix}{name}")
    return path


def _line(ax, data, key, label, **kw) -> bool:
    series = _series(data, key)
    if series is None:
        return False
    ax.plot(series[0], series[1], label=label, **kw)
    return True


def plot_run(run_dir: Path | str, figures_dir: Path | str | None = None) -> list[Path]:
    run_dir = Path(run_dir)
    data = read_progress(run_dir)
    out_dir = run_dir / "plots"
    copies = [Path(figures_dir) if figures_dir is not None else FIGURES_DIR]
    tag = run_dir.name
    # В agent/figures/ рядом лежат графики разных запусков, поэтому имя копии
    # префиксуется целью смены: priority_learning_curve.png, revenue_*.png.
    prefix = f"{tag.rsplit('_', 1)[-1]}_"
    made: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    if _line(ax, data, "rollout/ep_rew_mean", "ep_rew_mean", color="tab:blue"):
        ax.set_title(f"Кривая обучения — {tag}")
        ax.set_xlabel("таймстепы (тики по 5 минут)")
        ax.set_ylabel("средняя награда за эпизод")
        ax.grid(alpha=0.3)
        made.append(_save(fig, out_dir, "learning_curve.png", copies, prefix))
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ok = _line(ax, data, "ops/critical_done_on_time", "prio 3 в срок", color="tab:green")
    ok |= _line(ax, data, "ops/jobs_due_missed", "просрочено заданий", color="tab:red")
    if ok:
        ax.set_title(f"Приоритетное обслуживание — {tag}")
        ax.set_xlabel("таймстепы")
        ax.set_ylabel("заданий за эпизод")
        ax.legend()
        ax.grid(alpha=0.3)
        made.append(_save(fig, out_dir, "priority_service.png", copies, prefix))
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if _line(ax, data, "ops/revenue_usd", "revenue", color="tab:orange"):
        ax.set_title(f"Коммерческая отдача — {tag}")
        ax.set_xlabel("таймстепы")
        ax.set_ylabel("USD за эпизод")
        ax.grid(alpha=0.3)
        made.append(_save(fig, out_dir, "revenue.png", copies, prefix))
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if _line(ax, data, "ops/min_soc_pct", "минимальный SoC, %", color="tab:blue"):
        ax2 = ax.twinx()
        _line(ax2, data, "ops/below_reserve_steps", "ниже резерва, шагов", color="tab:red", ls="--")
        _line(ax2, data, "ops/brownout_steps", "brownout, шагов", color="black", ls=":")
        ax.set_title(f"Энергия — {tag}")
        ax.set_xlabel("таймстепы")
        ax.set_ylabel("SoC, %")
        ax2.set_ylabel("КА-шагов за эпизод")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.grid(alpha=0.3)
        made.append(_save(fig, out_dir, "energy.png", copies, prefix))
    else:
        plt.close(fig)

    shares = [
        _series(data, "act/idle_share"),
        _series(data, "act/calibrate_share"),
        _series(data, "act/job_share"),
    ]
    if all(s is not None for s in shares):
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = shares[0][0]  # type: ignore[index]
        ax.stackplot(
            xs,
            shares[0][1],  # type: ignore[index]
            shares[1][1],  # type: ignore[index]
            shares[2][1],  # type: ignore[index]
            labels=["ожидание", "калибровка", "задание"],
            colors=["#c7c7c7", "#7fb3d5", "#58d68d"],
        )
        ax.set_title(f"Структура команд — {tag}")
        ax.set_xlabel("таймстепы")
        ax.set_ylabel("доля команд")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")
        made.append(_save(fig, out_dir, "action_shares.png", copies, prefix))

    return made


def compare_runs(runs: list[Path | str], figures_dir: Path | str | None = None) -> list[Path]:
    out_dir = Path(runs[0]).parent / "_compare"
    copies = [Path(figures_dir) if figures_dir is not None else FIGURES_DIR]
    made: list[Path] = []
    for key, name, ylabel in (
        ("rollout/ep_rew_mean", "compare_reward.png", "средняя награда"),
        ("ops/critical_done_on_time", "compare_critical.png", "prio 3 в срок"),
        ("ops/revenue_usd", "compare_revenue.png", "USD за эпизод"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        drawn = False
        for run in runs:
            drawn |= _line(ax, read_progress(run), key, Path(run).name)
        if not drawn:
            plt.close(fig)
            continue
        ax.set_title(key)
        ax.set_xlabel("таймстепы")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        made.append(_save(fig, out_dir, name, copies))
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description="Графики обучения из progress.csv")
    parser.add_argument("--run", action="append", required=True, help="каталог agent/runs/<run>")
    parser.add_argument("--figures", default=None, help="куда копировать PNG")
    args = parser.parse_args()
    made: list[Path] = []
    for run in args.run:
        made.extend(plot_run(run, args.figures))
    if len(args.run) > 1:
        made.extend(compare_runs(args.run, args.figures))
    for path in made:
        print(path)


if __name__ == "__main__":
    main()
