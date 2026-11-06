"""Логирование обучения: каталог запуска, логгер SB3 и доменные метрики.

На каждый запуск заводится `agent/runs/<дата-время>_<сценарий>_<goal>/`:

- `train.log`   — человекочитаемый поток SB3, то же, что в stdout
- `progress.csv`— машинный ряд, из него `agent/plots.py` строит графики
- `tb/`         — tensorboard, включается флагом
- `config.json` — снимок гиперпараметров, seed, сценария, вероятностей событий
"""

from __future__ import annotations

import json
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import (
    CSVOutputFormat,
    HumanOutputFormat,
    Logger,
    TensorBoardOutputFormat,
)

REPO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_DIR / "agent" / "runs"
FIGURES_DIR = REPO_DIR / "agent" / "figures"

# summary() организаторов -> имена в progress.csv
_SUMMARY_KEYS = {
    "jobs_completed": "ops/jobs_completed",
    "jobs_due_missed": "ops/jobs_due_missed",
    "critical_jobs_completed_on_time": "ops/critical_done_on_time",
    "critical_jobs_due": "ops/critical_due",
    "revenue_usd": "ops/revenue_usd",
    "below_reserve_satellite_steps": "ops/below_reserve_steps",
    "brownout_satellite_steps": "ops/brownout_steps",
    "minimum_soc_pct": "ops/min_soc_pct",
    "blocked_command_count": "ops/blocked_commands",
    "work_steps_in_missed_jobs": "ops/work_lost_steps",
}

_OPS_KEYS = {
    "idle_share": "act/idle_share",
    "calibrate_share": "act/calibrate_share",
    "job_share": "act/job_share",
    "conflicts_repaired": "act/conflicts_repaired",
    "downlink_slots_used": "act/downlink_slots_used",
    "executor_switches": "act/executor_switches",
    "events_add_jobs": "events/add_jobs",
    "events_outage": "events/outage",
    "events_close_downlink": "events/close_downlink",
    "events_skipped": "events/skipped",
}


def make_run_dir(scenario_id: str, goal: str, root: Path | str | None = None) -> Path:
    root = Path(root) if root is not None else RUNS_DIR
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{stamp}_{scenario_id}_{goal}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class _ResumingCSVOutputFormat(CSVOutputFormat):
    """Дописывает progress.csv, не обнуляя уже записанные rollout."""

    def __init__(self, filename: str):
        self.file = open(filename, "r+", encoding="utf-8", newline="")
        header = self.file.readline()
        self.keys = [key.strip() for key in header.strip().split(",")] if header.strip() else []
        self.separator = ","
        self.quotechar = '"'
        self.file.seek(0, 2)


def last_logged_timesteps(run_dir: Path | str) -> int:
    path = Path(run_dir) / "progress.csv"
    if not path.exists() or path.stat().st_size == 0:
        return 0
    last = ""
    with path.open(encoding="utf-8", newline="") as fh:
        header = fh.readline()
        keys = [key.strip() for key in header.strip().split(",")]
        for line in fh:
            if line.strip():
                last = line
    if not last or "time/total_timesteps" not in keys:
        return 0
    cols = next(csv.reader([last]))
    idx = keys.index("time/total_timesteps")
    return int(float(cols[idx]))


def configure_run_logger(
    run_dir: Path | str, tensorboard: bool = False, resume: bool = False
) -> Logger:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "progress.csv"
    log_path = run_dir / "train.log"
    if resume and csv_path.exists() and csv_path.stat().st_size > 0:
        csv_writer = _ResumingCSVOutputFormat(str(csv_path))
        log_file = open(log_path, "a", encoding="utf-8")
        log_writer = HumanOutputFormat(log_file)
    else:
        csv_writer = CSVOutputFormat(str(csv_path))
        log_writer = HumanOutputFormat(str(log_path))
    outputs = [
        HumanOutputFormat(sys.stdout),
        log_writer,
        csv_writer,
    ]
    if tensorboard:
        tb = run_dir / "tb"
        tb.mkdir(exist_ok=True)
        outputs.append(TensorBoardOutputFormat(str(tb)))
    return Logger(folder=str(run_dir), output_formats=outputs)


def dump_config(run_dir: Path | str, config: dict[str, Any]) -> Path:
    path = Path(run_dir) / "config.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


class OpsMetricsCallback(BaseCallback):
    """Забирает `Environment.summary()` из info в конце эпизода.

    Стоковые `rollout/ep_rew_mean` и `train/*` SB3 пишет сам; здесь добавляются
    показатели, по которым оценивается смена: критичные задания в срок, выручка,
    провалы по энергии и структура команд.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._buffer: dict[str, list[float]] = {}
        self.episodes = 0

    def _add(self, key: str, value: float) -> None:
        self._buffer.setdefault(key, []).append(float(value))

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            summary = info.get("summary")
            if summary is None:
                continue
            self.episodes += 1
            for src, dst in _SUMMARY_KEYS.items():
                if src in summary:
                    self._add(dst, summary[src])
            due = max(summary.get("jobs_due", 0), 1)
            self._add("ops/completed_share", summary.get("jobs_completed", 0) / due)
            critical_due = max(summary.get("critical_jobs_due", 0), 1)
            self._add(
                "ops/critical_on_time_share",
                summary.get("critical_jobs_completed_on_time", 0) / critical_due,
            )
            for src, dst in _OPS_KEYS.items():
                value = info.get("ops", {}).get(src)
                if value is not None:
                    self._add(dst, value)
        return True

    def _on_rollout_end(self) -> None:
        for key, values in self._buffer.items():
            self.logger.record(key, float(np.mean(values)))
        self.logger.record("ops/episodes", self.episodes)
        self._buffer.clear()


__all__ = [
    "AGENT_DIR",
    "FIGURES_DIR",
    "OpsMetricsCallback",
    "RUNS_DIR",
    "configure_run_logger",
    "dump_config",
    "make_run_dir",
]
