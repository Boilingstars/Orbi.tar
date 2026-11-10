"""Launch settings and the run around a user model.

Edit `Run`. The command line does not override it. Checkpoint lines below can
be commented out; a checkpoint is written when any enabled rule fires.
`launch` prepares the run directory and calls the package's `build`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from tools.telemetry import (
    OpsMetricsCallback,
    configure_run_logger,
    dump_config,
    last_logged_timesteps,
    make_run_dir,
)


# Checkpoint rules. Comment out a line to turn that rule off.
# A checkpoint is written when any enabled rule fires.

# CHECKPOINT_EVERY_S = 20 * 60
# CHECKPOINT_EVERY_STEPS = 10_000

# New maximum of this logged column. EVERY_N_MAX = 1 keeps every new maximum.
CHECKPOINT_MAX_METRIC = "ops/revenue_usd"
CHECKPOINT_EVERY_N_MAX = 1


# Plot rules. Comment out a line to turn that rule off.

# PLOT_EVERY_S = 20 * 60

# Redraw charts each time progress.csv gains a row.
PLOT_ON_LOG = True


# Stop once the learning curve has flattened. Large values keep a fixed-length run.

PATIENCE = 1_000_000_000
MIN_ITERS = 1_000_000_000
MIN_DELTA = 0.02

_FRAMEWORK = {
    "suite",
    "policy_name",
    "patience",
    "min_iters",
    "min_delta",
    "checkpoint_every_s",
    "checkpoint_every_steps",
    "checkpoint_every_n_max",
    "checkpoint_max_metric",
    "plot_every_s",
    "plot_on_log",
}


@dataclass
class Run:
    """One launch. These fields are the whole command-line set."""

    scenario: str = "p03"
    timesteps: int = 120_000
    n_envs: int = 8
    top_k: int | None = 64
    seed: int = 0
    profile: str = "full"
    events: bool = True
    controller: str = "dispatcher"
    baseline_repair: bool = False
    neighbor_method: str = "competitors"
    neighbor_memory: int = 12
    ground_points: int = 64
    ground_region: str | None = "world"
    job_difficulty: float = 0.15
    max_steps: int | None = None
    tensorboard: bool = False
    subproc: bool = True
    load: str | None = None
    save: str | None = None
    run_dir: str | None = None
    resume: bool = False


def plateau_reached(
    history: list[float],
    patience: int = PATIENCE,
    min_delta: float = MIN_DELTA,
    min_iters: int = MIN_ITERS,
) -> bool:
    """True when `patience` rollouts in a row fail to beat the best so far by `min_delta`."""
    if len(history) < max(min_iters, patience + 1):
        return False
    best_before = max(history[:-patience])
    return max(history[-patience:]) < best_before * (1.0 + min_delta)


class Trainer:
    """Algorithm hyperparameters for one user model. Launch settings are `Run`."""

    suite: str = "example"
    policy_name: str = "policy"

    patience = PATIENCE
    min_iters = MIN_ITERS
    min_delta = MIN_DELTA

    checkpoint_every_s = globals().get("CHECKPOINT_EVERY_S")
    checkpoint_every_steps = globals().get("CHECKPOINT_EVERY_STEPS")
    checkpoint_every_n_max = globals().get("CHECKPOINT_EVERY_N_MAX")
    checkpoint_max_metric = globals().get("CHECKPOINT_MAX_METRIC", "rollout/ep_rew_mean")
    plot_every_s = globals().get("PLOT_EVERY_S")
    plot_on_log = globals().get("PLOT_ON_LOG")


class Session:
    """The run `build` trains inside. `notes` are written next to `Run`."""

    def __init__(self, run: Run, trainer: Trainer) -> None:
        self.run = run
        self.trainer = trainer
        self.notes: dict = {}


def launch(build, trainer: Trainer, plot=None):
    """Train `build(session) -> model` on the fixed `Run`."""
    run = Run()
    if trainer.policy_name and trainer.policy_name != "policy":
        name = trainer.policy_name
        run.controller = name[: -len("_greedy")] if name.endswith("_greedy") else name
    else:
        trainer.policy_name = run.controller

    session = Session(run, trainer)

    # Log, checkpoints, plots, and the final zip all live in this directory.
    run_dir = Path(run.run_dir) if run.run_dir else make_run_dir(trainer.suite, trainer.policy_name)
    run.run_dir = str(run_dir)
    run.save = str(run_dir / "model.zip")
    model = build(session)

    # A fresh directory, or the steps already written when this launch resumes.
    started = last_logged_timesteps(run_dir) if run.resume else 0
    if run.resume:
        model.num_timesteps = started
        remaining = max(int(run.timesteps) - started, 0)
        print(f"Resume {run_dir}: {started} steps done, {remaining} left", flush=True)
    else:
        remaining = int(run.timesteps)
    model.set_logger(configure_run_logger(run_dir, tensorboard=run.tensorboard, resume=run.resume))
    dump_config(run_dir, _config(session, started))

    # Checkpoints, plots, and the plateau stop ride along as SB3 callbacks.
    extra = list(getattr(session, "callbacks", None) or [])
    model.learn(
        total_timesteps=remaining,
        callback=[
            OpsMetricsCallback(),
            RunArtifactsCallback(run_dir, trainer, plot),
            PlateauStopCallback(trainer),
            *extra,
        ],
        progress_bar=True,
        reset_num_timesteps=not run.resume,
    )

    save_path = Path(run.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    (run_dir / "model.txt").write_text(str(save_path), encoding="utf-8")
    print(f"Model: {save_path}\nTraining log: {run_dir}")
    if plot is not None:
        try:
            for path in plot(run_dir):
                print(f"Plot: {path}")
        except Exception as exc:
            print(f"Plot failed: {exc}", flush=True)
    return model


class PlateauStopCallback(BaseCallback):
    """Stop `learn()` when rollout/ep_rew_mean has not improved for `patience` rollouts."""

    def __init__(self, trainer: Trainer) -> None:
        super().__init__()
        self.patience = trainer.patience
        self.min_delta = trainer.min_delta
        self.min_iters = trainer.min_iters
        self.history: list[float] = []
        self._stop = False

    def _on_rollout_end(self) -> None:
        value = self.logger.name_to_value.get("rollout/ep_rew_mean")
        if value is None:
            return
        self.history.append(float(value))
        if plateau_reached(self.history, self.patience, self.min_delta, self.min_iters):
            print(
                f"Learning curve plateau: ep_rew_mean={self.history[-1]:.1f} "
                f"best={max(self.history):.1f} after {len(self.history)} rollouts; stopping",
                flush=True,
            )
            self._stop = True

    def _on_step(self) -> bool:
        return not self._stop


class RunArtifactsCallback(BaseCallback):
    """Checkpoint weights by the trainer's rules and redraw plots on a timer.

    `plot` is the user package's chart function. Tools does not draw them.
    """

    def __init__(self, run_dir: Path | str, trainer: Trainer, plot=None) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.plot = plot
        self.every_s = trainer.checkpoint_every_s
        self.every_steps = trainer.checkpoint_every_steps
        self.every_n_max = trainer.checkpoint_every_n_max
        self.max_metric = trainer.checkpoint_max_metric
        self.plot_every_s = trainer.plot_every_s
        self.plot_on_log = bool(trainer.plot_on_log)
        self._next_checkpoint = 0.0
        self._next_plot = 0.0
        self._steps_mark = 0
        self._best: float | None = None
        self._max_count = 0

    def _on_training_start(self) -> None:
        now = time.monotonic()
        if self.every_s:
            self._next_checkpoint = now + self.every_s

        if self.plot_every_s:
            self._next_plot = now + self.plot_every_s

        # Charts follow a real log row, not an empty train-only dump.
        if self.plot_on_log and self.plot is not None:
            original = self.logger.dump
            progress = self.run_dir / "progress.csv"

            def _dump(step: int = 0, _write=original) -> None:
                before = progress.stat().st_size if progress.exists() else 0
                _write(step)
                after = progress.stat().st_size if progress.exists() else 0
                if after > before:
                    self._write_plots()

            self.logger.dump = _dump
        self._steps_mark = self.num_timesteps
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        now = time.monotonic()

        # Redraw the package's charts on their own timer.
        if self.plot_every_s and now >= self._next_plot:
            self._write_plots()
            while self._next_plot <= now:
                self._next_plot += self.plot_every_s

        # A checkpoint when either the clock or the step counter crosses its mark.
        save = False
        if self.every_s and now >= self._next_checkpoint:
            save = True
            while self._next_checkpoint <= now:
                self._next_checkpoint += self.every_s

        if self.every_steps:
            current = self.num_timesteps
            if self._steps_mark // self.every_steps < current // self.every_steps:
                save = True
            self._steps_mark = current

        if save:
            self._save_checkpoint()
        return True

    def _on_rollout_end(self) -> None:
        if not self.every_n_max:
            return
        raw = self.logger.name_to_value.get(self.max_metric)
        if raw is None:
            return
        value = float(raw)
        if self._best is not None and value <= self._best:
            return
        self._best = value
        self._max_count += 1
        if self._max_count % self.every_n_max == 0:
            self._save_checkpoint()

    def _write_plots(self) -> None:
        if self.plot is None:
            return
        progress = self.run_dir / "progress.csv"
        if not progress.exists() or progress.stat().st_size == 0:
            return
        try:
            made = self.plot(self.run_dir)
        except Exception as exc:
            print(f"Plots failed: {exc}", flush=True)
            return
        print(f"Plots updated: {len(made)} files in {self.run_dir / 'plots'}", flush=True)

    def _save_checkpoint(self) -> None:
        path = self.run_dir / "checkpoints" / f"ckpt_{self.num_timesteps}.zip"
        self.model.save(str(path))
        print(f"Weight checkpoint: {path}", flush=True)


def _algorithm(trainer: Trainer) -> dict:
    found = {}
    for cls in type(trainer).__mro__:
        if cls is object:
            break
        for name, value in vars(cls).items():
            if name.startswith("_") or name in _FRAMEWORK or callable(value):
                continue
            found[name] = getattr(trainer, name)
    return found


def _config(session: Session, started: int) -> dict:
    # Run fields, then whatever the package wrote into session.notes, then the algorithm.
    trainer = session.trainer
    payload = asdict(session.run)
    payload.update(session.notes)
    payload.update(
        suite=trainer.suite,
        policy_name=trainer.policy_name,
        resumed_from=started if session.run.resume else None,
        algorithm=_algorithm(trainer),
        plateau={
            "patience": trainer.patience,
            "min_iters": trainer.min_iters,
            "min_delta": trainer.min_delta,
            "metric": "rollout/ep_rew_mean",
        },
        checkpoints={
            "every_s": trainer.checkpoint_every_s,
            "every_steps": trainer.checkpoint_every_steps,
            "every_n_max": trainer.checkpoint_every_n_max,
            "max_metric": trainer.checkpoint_max_metric,
        },
        plots={
            "every_s": trainer.plot_every_s,
            "on_log": bool(trainer.plot_on_log),
        },
    )
    return payload
