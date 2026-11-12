"""Run directory, SB3 logger, and the core shift metrics.

Custom job fields are not listed here. A user package registers them with
`tools.jobs.register_job_field`; this writer appends whatever is registered.

A run is stored under `experiments/runs/<dd_mm_yy_HH_MM>_<suite>_<policy>/`:

- `train.log`     — human-readable SB3 stream, same as stdout
- `progress.csv`  — numeric series
- `config.json`   — hyperparameters, seed, scenario, event rates
- `checkpoints/`  — periodic weight zips
- `plots/`        — charts drawn from `progress.csv`
- `model.zip`     — weights at the end of the run
- `tb/`           — TensorBoard, enabled by flag
"""

from __future__ import annotations

import csv
import json
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

from tools.jobs import job_field_log_map

REPO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_DIR / "experiments" / "runs"

# Environment.summary() keys -> progress.csv names. Custom fields are separate.
_SUMMARY_KEYS = {
    "jobs_completed": "ops/jobs_completed",
    "jobs_due_missed": "ops/jobs_due_missed",
    "critical_jobs_completed_on_time": "ops/critical_done_on_time",
    "critical_jobs_due": "ops/critical_due",
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
    "cal_while_valid": "act/cal_while_valid",
    "cal_with_job": "act/cal_with_job",
    "cal_without_job": "act/cal_without_job",
    "cal_soon": "act/cal_soon",
    "cal_age_mean": "act/cal_age_mean",
    "revenue_per_downlink_tick": "ops/revenue_per_downlink_tick",
    "done_relay": "ops/done_relay",
    "revenue_relay": "ops/revenue_relay",
    "done_downlink": "ops/done_downlink",
    "revenue_downlink": "ops/revenue_downlink",
    "done_p1": "ops/done_p1",
    "revenue_p1": "ops/revenue_p1",
    "done_p2": "ops/done_p2",
    "revenue_p2": "ops/revenue_p2",
    "done_p3": "ops/done_p3",
    "revenue_p3": "ops/revenue_p3",
    "feasible_edges": "ops/feasible_edges",
    "executable_jobs": "ops/executable_jobs",
    "executors_per_job": "ops/executors_per_job",
    "executable_outside_slots": "ops/executable_outside_slots",
    "downlink_resets": "ops/downlink_resets",
    "downlink_lost_ticks": "ops/downlink_lost_ticks",
    "downlink_lost_usd": "ops/downlink_lost_usd",
    "downlink_work_ticks": "ops/downlink_work_ticks",
    "downlink_first_try_share": "ops/downlink_first_try_share",
    "downlink_tick_yield": "ops/downlink_tick_yield",
    "revenue_per_occupied_downlink_tick": "ops/revenue_per_occupied_downlink_tick",
    "revenue_per_productive_downlink_tick": "ops/revenue_per_productive_downlink_tick",
    "mean_value_relay": "ops/mean_value_relay",
    "mean_value_downlink": "ops/mean_value_downlink",
    "p3_rate": "ops/p3_rate",
}


def make_run_dir(suite: str, policy_name: str, root: Path | str | None = None) -> Path:
    """`<stamp>_<suite>_<policy>`, for example `07_11_26_19_30_example_dispatcher`."""

    root = Path(root) if root is not None else RUNS_DIR
    stamp = datetime.now().strftime("%d_%m_%y_%H_%M")
    suite_tag = str(suite).strip().replace("/", "-") or "suite"
    policy_tag = str(policy_name).strip().replace("/", "-") or "policy"
    name = f"{stamp}_{suite_tag}_{policy_tag}"
    run_dir = root / name
    suffix = 2
    while run_dir.exists():
        run_dir = root / f"{name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class _ResumingCSVOutputFormat(CSVOutputFormat):
    """Append to progress.csv without wiping logged rollouts."""

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


def configure_run_logger(run_dir: Path | str, tensorboard: bool = False, resume: bool = False) -> Logger:
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
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        log_writer = HumanOutputFormat(log_file)
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
    """Copy `Environment.summary()` from info at episode end.

    SB3 already logs `rollout/ep_rew_mean` and `train/*`. This adds the core
    shift metrics. Registered job fields are appended under their log keys, so
    they show up in the same cmd table.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._buffer: dict[str, list[float]] = {}
        self.episodes = 0
        self._new_episodes = False

    def _add(self, key: str, value: float) -> None:
        self._buffer.setdefault(key, []).append(float(value))

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            summary = info.get("summary")
            if summary is None:
                continue
            self.episodes += 1

            # Core summary keys, plus any job field a package registered.
            for src, dst in {**_SUMMARY_KEYS, **job_field_log_map()}.items():
                if src in summary:
                    self._add(dst, summary[src])
            due = max(summary.get("jobs_due", 0), 1)
            self._add("ops/completed_share", summary.get("jobs_completed", 0) / due)
            critical_due = max(summary.get("critical_jobs_due", 0), 1)
            self._add(
                "ops/critical_on_time_share",
                summary.get("critical_jobs_completed_on_time", 0) / critical_due,
            )

            # Action mix and event counts live on info["ops"], not on the summary.
            for src, dst in _OPS_KEYS.items():
                value = info.get("ops", {}).get(src)
                if value is not None:
                    self._add(dst, value)
        return True

    def _on_training_start(self) -> None:
        # Write one table only when at least one env finished a shift this
        # rollout. A rollout as long as the episode always has summaries; a
        # shorter one waits, so train/ and ops/ stay on the same row.
        original = self.logger.dump

        def _dump(step: int = 0, _write=original) -> None:
            if not self._new_episodes:
                return
            _write(step)
            self._new_episodes = False

        self.logger.dump = _dump

    def _on_rollout_end(self) -> None:
        self._new_episodes = bool(self._buffer)
        if not self._new_episodes:
            return
        for key, values in self._buffer.items():
            self.logger.record(key, float(np.mean(values)))
        self.logger.record("ops/episodes", self.episodes)
        self._buffer.clear()


__all__ = [
    "OpsMetricsCallback",
    "RUNS_DIR",
    "configure_run_logger",
    "dump_config",
    "last_logged_timesteps",
    "make_run_dir",
]
