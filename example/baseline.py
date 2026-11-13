"""Benchmark Greedy, Hungarian + capacity repair, and RL (MaskablePPO) on P03 fleet subsets.

Every method plays the same full shift (288 ticks of 300 s) for each fleet size
and seed. A smaller fleet is a stratified, nested subset of the 48 P03 satellites
over all six planes. Orbits, physics, prices, `K = 64` and `N_max = 48` stay as
loaded; only the satellite list, the job load, and the downlink slots follow N.

Event modes:

- `common` (primary): arrivals and outages for one (N, seed) are recorded once,
  before any policy acts, and replayed into every method's env through
  `scripted_events`. All methods get the same offered jobs and outages.
- `native` (supplementary): `EventConfig` draws inside each env. Seeds match,
  but arrivals and outages then depend on each policy's own decisions.

Methods:

- Greedy: `orbi.greedy.greedy_action` with the env's downlink limit read after `reset`.
- Hungarian + capacity repair: one-tick assignment, `w_ij = V_j / T_ij`, solved by
  `scipy.optimize.linear_sum_assignment` with idle columns. The downlink cap is
  enforced by a deterministic repair, so it is not an exact solution of the capped
  problem and it is not a whole-shift plan.
- RL: a trained MaskablePPO checkpoint, no training or fine-tuning here. On N < 48
  it is a zero-shot transfer of a policy trained on 48 satellites.

    python -m example.baseline --model PATH_TO_MODEL.zip
    python -m example.baseline --model PATH.zip --fleet-sizes 16,24,32,40,48 --seeds 0:29 \\
        --event-mode common --rl-mode stochastic --output experiments/benchmarks
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t

from example.agent import PPOAgent
from example.jobs import downlink_slots
from orbi.env import N_MAX as ENV_N_MAX
from orbi.env import EventConfig, OpsEnv, digest, load, resolve_scenario
from orbi.greedy import _SAT_CAL_AGE, CALIBRATE_AT, greedy_action
from orbi.satellite import downlink_required_steps
from tools.evaluating import Eval, load_ppo, run_episode, shift_kwargs


FLEET_SIZES = (16, 24, 32, 40, 48)
N_MAX = 48
TOP_K = 64
SCENARIO = "p03"
SHIFT_STEPS = 288
STEP_S = 300

# Set a checkpoint here to run a bare `python -m example.baseline`. `--model` overrides it.
MODEL: str | None = None

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "experiments" / "benchmarks"

# P03 load at 48 satellites. Smaller fleets scale it by n_sats / 48.
REFERENCE_SATS = 48
INITIAL_JOBS_PER_SAT = 5
BASE_POOL_MIN = 200
BASE_POOL_MAX = 350
BASE_ADDED_JOBS_MIN = 16
BASE_ADDED_JOBS_MAX = 28
BASE_OUTAGE_SATS = (1, 3)

# Hungarian weight w_ij = V_j / T_ij + tie-breaks, in USD per tick. Fixed before any run.
HUNGARIAN_TIE_SOC = 1e-4
HUNGARIAN_TIE_HOLDER = 1e-3
_FORBIDDEN = -1e9

# Entropy for the stochastic RL policy seed, mixed with (n_sats, seed).
RL_POLICY_SEED = 7919

# Fewer seeds than this mark every table and plot as preliminary.
PRELIMINARY_SEEDS = 10
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261113

GREEDY = "Greedy"
HUNGARIAN = "Hungarian"
RL_STOCHASTIC = "RL_stochastic"
RL_DETERMINISTIC = "RL_deterministic"
RL_METHODS = {"stochastic": RL_STOCHASTIC, "deterministic": RL_DETERMINISTIC}
METHOD_LABELS = {
    GREEDY: "Greedy",
    HUNGARIAN: "Hungarian + capacity repair",
    RL_STOCHASTIC: "RL (MaskablePPO, stochastic)",
    RL_DETERMINISTIC: "RL (MaskablePPO, deterministic)",
}
METHOD_STYLE = {
    GREEDY: ("#1f77b4", "o"),
    HUNGARIAN: ("#ff7f0e", "s"),
    RL_STOCHASTIC: ("#2ca02c", "^"),
    RL_DETERMINISTIC: ("#2ca02c", "v"),
}

KINDS = ("relay", "downlink")
EVENT_MODES = ("common", "native")
_INITIAL_ID = re.compile(r"^J\d+$")
_STATIC_FIELDS = (
    "id",
    "kind",
    "type",
    "release_step",
    "deadline_step",
    "work_steps",
    "priority",
    "value_usd",
    "points",
    "point_a",
    "point_b",
    "addressed",
    "eligible_satellites",
)

# The four criteria of the comparison. Everything else in a row is diagnostic.
MAIN_METRICS = ("completed_total", "revenue_usd", "revenue_per_completed_job", "missed_total")
SUMMARY_METRICS = (
    "completed_total",
    "completed_relay",
    "completed_downlink",
    "completion_rate_total",
    "completion_rate_relay",
    "completion_rate_downlink",
    "revenue_usd",
    "revenue_relay_usd",
    "revenue_downlink_usd",
    "revenue_per_completed_job",
    "revenue_per_completed_relay",
    "revenue_per_completed_downlink",
    "q_selection_total",
    "q_selection_relay",
    "q_selection_downlink",
    "missed_total",
    "missed_relay",
    "missed_downlink",
    "miss_rate_total",
    "miss_rate_relay",
    "miss_rate_downlink",
    "unfinished_at_shift_end",
    "pending_with_future_deadline",
    "jobs_generated_total",
    "jobs_due_total",
    "revenue_per_satellite",
    "completed_per_satellite",
    "missed_per_satellite",
    "completed_p3",
    "p3_completion_rate",
    "downlink_resets",
    "downlink_lost_ticks",
    "downlink_capacity_utilization",
    "active_queue_mean",
    "executable_jobs_mean",
    "executable_outside_k_mean",
    "capacity_repair_ticks",
    "capacity_repair_dropped",
    "ground_capacity_rejections",
    "blocked_commands",
    "decision_ms_mean",
    "decision_ms_p95",
    "shift_wall_s",
    "reward_return",
)
DELTA_METRICS = (
    "completed_total",
    "completed_relay",
    "completed_downlink",
    "revenue_usd",
    "revenue_per_completed_job",
    "q_selection_total",
    "missed_total",
    "missed_relay",
    "missed_downlink",
    "revenue_per_satellite",
)


@dataclass
class Options:
    """One benchmark launch. `parse` fills it from the command line."""

    scenario: str = SCENARIO
    model: str | None = MODEL
    fleet_sizes: tuple[int, ...] = FLEET_SIZES
    seeds: tuple[int, ...] = tuple(range(30))
    event_modes: tuple[str, ...] = ("common",)
    rl_mode: str = "stochastic"
    rl_secondary: bool = False
    methods: tuple[str, ...] = ("greedy", "hungarian", "rl")
    output: str = str(OUTPUT)
    into: str | None = None
    workers: int = 1
    plots: bool = True
    smoke: bool = False
    # Tests only. A benchmark run always plays the whole shift.
    max_steps: int | None = None

    @property
    def primary_rl(self) -> str:
        return RL_METHODS[self.rl_mode]

    def method_names(self) -> list[str]:
        names = []
        if "greedy" in self.methods:
            names.append(GREEDY)

        if "hungarian" in self.methods:
            names.append(HUNGARIAN)

        if "rl" in self.methods:
            names.append(self.primary_rl)
            if self.rl_secondary:
                names.extend(name for name in RL_METHODS.values() if name != self.primary_rl)
        return names


def run_benchmark(opts: Options) -> Path:
    """Play every (event mode, N, seed, method), then write tables, the report, and plots."""
    _check_options(opts)
    out_dir = _run_dir(opts)
    base = load(resolve_scenario(opts.scenario))

    # The checkpoint is checked once here. Workers only load the same file.
    model = None
    model_info = None
    if any(name in RL_METHODS.values() for name in opts.method_names()):
        model, model_info = load_policy(opts.model, base)
    if opts.into:
        previous = out_dir / "config_before_update.json"
        current = out_dir / "config.json"
        if current.is_file() and not previous.exists():
            previous.write_bytes(current.read_bytes())
    config = _config(opts, base, model_info)
    _write_json(out_dir / "config.json", config)

    # One task is one (event mode, N, seed): the tape, then each method on it.
    tasks = [(mode, n, seed) for mode in opts.event_modes for n in opts.fleet_sizes for seed in opts.seeds]
    played = set(opts.method_names())
    rows: list[dict] = _kept_rows(out_dir, played) if opts.into else []
    tapes: list[dict] = []
    started = time.perf_counter()
    print(f"benchmark: {len(tasks)} tasks x {len(opts.method_names())} methods -> {out_dir}", flush=True)
    for done, result in enumerate(_execute(opts, tasks, model), start=1):
        rows.extend(result["rows"])
        if result["tape"] is not None:
            tapes.append(result["tape"])
        _write_csv(out_dir / "episodes.csv", _sorted_rows(rows))
        mode, n, seed = result["task"]
        elapsed = time.perf_counter() - started
        print(f"  [{done}/{len(tasks)}] {mode} N={n} seed={seed} done, {elapsed / 60:.1f} min", flush=True)

    # Aggregates over seeds, the paired differences, and the files around them.
    rows = _sorted_rows(rows)
    report_opts = _report_options(opts, rows)
    config = _config(report_opts, base, model_info)
    summary = summarize(rows)
    deltas = paired_deltas(rows, report_opts)
    _write_csv(out_dir / "summary.csv", summary)
    _write_csv(out_dir / "paired_deltas.csv", deltas)
    _write_json(out_dir / "config.json", config)
    _write_json(out_dir / "workload.json", workload_report(report_opts, base, rows, tapes))
    report = build_report(report_opts, config, rows, summary, deltas, tapes)
    _write_json(out_dir / "report.json", report)
    if opts.plots:
        for mode in report_opts.event_modes:
            plot_dir = out_dir / "plots" if mode == report_opts.event_modes[0] else out_dir / "plots" / mode
            plot_all(summary, report_opts, mode, plot_dir)
    print_table(summary, report_opts)
    print(f"results: {out_dir}")
    return out_dir


def _check_options(opts: Options) -> None:
    if not opts.method_names():
        raise SystemExit("No methods selected")

    if "rl" in opts.methods and not opts.model:
        raise SystemExit(
            "RL needs a trained checkpoint: pass --model PATH_TO_MODEL.zip "
            "(or run only the classical methods with --methods greedy,hungarian)"
        )

    if any(mode not in EVENT_MODES for mode in opts.event_modes):
        raise SystemExit(f"event mode must be one of {EVENT_MODES}")

    if ENV_N_MAX != N_MAX:
        raise SystemExit(f"orbi.env.N_MAX={ENV_N_MAX}, the benchmark and the RL tensors need {N_MAX}")

    out = Path(opts.into or opts.output).resolve()
    if out == REPO:
        raise SystemExit("Write results into a subdirectory such as experiments/benchmarks, not the repository root")


def _run_dir(opts: Options) -> Path:
    if opts.into:
        path = Path(opts.into).resolve()
        if not path.is_dir():
            raise SystemExit(f"--into directory does not exist: {path}")
        return path
    root = Path(opts.output)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"benchmark_{stamp}"
    suffix = 1
    while path.exists():
        suffix += 1
        path = root / f"benchmark_{stamp}_{suffix}"
    path.mkdir(parents=True)
    return path


def _kept_rows(out_dir: Path, played: set[str]) -> list[dict]:
    """Keep already-written episodes whose method is not being replayed."""
    path = out_dir / "episodes.csv"
    if not path.is_file():
        return []
    backup = out_dir / "episodes_before_update.csv"
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    kept = [row for row in _read_episode_rows(path) if row["method"] not in played]
    print(f"updating {out_dir}: kept {len(kept)} rows, replacing {sorted(played)}", flush=True)
    return kept


def _read_episode_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append({key: _csv_value(value) for key, value in raw.items()})
    return rows


def _csv_value(value: str):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and "." not in value and "e" not in value.lower():
        return int(number)
    return number


def _report_options(opts: Options, rows: list[dict]) -> Options:
    """After `--into`, the report covers every method still in episodes.csv."""
    names = {row["method"] for row in rows}
    methods = []
    if GREEDY in names:
        methods.append("greedy")
    if HUNGARIAN in names:
        methods.append("hungarian")
    if names & set(RL_METHODS.values()):
        methods.append("rl")
    secondary = RL_STOCHASTIC in names and RL_DETERMINISTIC in names
    return replace(opts, methods=tuple(methods) or opts.methods, rl_secondary=secondary)


def _execute(opts: Options, tasks: list[tuple], model):
    """Yield task results in completion order. One worker runs in this process."""
    if opts.workers <= 1:
        _init_worker(opts, model)
        for task in tasks:
            yield _run_task(task)
        return

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(opts.workers, initializer=_init_worker, initargs=(opts, None)) as pool:
        yield from pool.imap_unordered(_run_task, tasks)


_WORKER: dict = {}


def _init_worker(opts: Options, model=None) -> None:
    # One BLAS thread per worker keeps timings comparable and seeds reproducible.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    needs_rl = any(name in RL_METHODS.values() for name in opts.method_names())
    if needs_rl:
        import torch

        torch.set_num_threads(1)
    _WORKER["opts"] = opts
    _WORKER["base"] = load(resolve_scenario(opts.scenario))
    _WORKER["model"] = model if model is not None or not needs_rl else load_ppo(opts.model)


def _run_task(task: tuple) -> dict:
    mode, n_sats, seed = task
    opts = _WORKER["opts"]
    cfg = fleet_config(_WORKER["base"], n_sats)
    work = Workload.for_fleet(n_sats)
    tape = EventTape.record(cfg, work, seed) if mode == "common" else None
    rows = [
        play(make_method(name, _WORKER["model"]), cfg, work, seed, mode, tape, opts.max_steps)
        for name in opts.method_names()
    ]
    return {"task": task, "rows": rows, "tape": tape.summary() if tape is not None else None}


@dataclass(frozen=True)
class Workload:
    """Job load and events for one fleet size, scaled from P03 at 48 satellites."""

    n_sats: int
    initial_jobs: int
    pool_min: int
    pool_max: int
    add_min: int
    add_max: int
    downlink_limit: int
    outage_sats: tuple[int, int]
    outage_len: tuple[int, int]
    p_add_jobs: float
    p_outage: float

    @classmethod
    def for_fleet(cls, n_sats: int) -> "Workload":
        base = EventConfig()
        ratio = n_sats / REFERENCE_SATS

        # Outage size keeps roughly the same share of the fleet. Halves round up.
        out_lo = max(1, _half_up(BASE_OUTAGE_SATS[0] * ratio))
        out_hi = max(out_lo, _half_up(BASE_OUTAGE_SATS[1] * ratio))
        return cls(
            n_sats=int(n_sats),
            initial_jobs=INITIAL_JOBS_PER_SAT * n_sats,
            pool_min=round(BASE_POOL_MIN * ratio),
            pool_max=round(BASE_POOL_MAX * ratio),
            add_min=round(BASE_ADDED_JOBS_MIN * ratio),
            add_max=round(BASE_ADDED_JOBS_MAX * ratio),
            downlink_limit=downlink_slots(n_sats),
            outage_sats=(out_lo, out_hi),
            outage_len=tuple(base.outage_len),
            p_add_jobs=float(base.p_add_jobs),
            p_outage=float(base.p_outage),
        )

    def events(self, live: bool) -> EventConfig:
        """`live=False` keeps the pool bounds for the initial batch and draws nothing later."""
        return EventConfig(
            p_add_jobs=self.p_add_jobs if live else 0.0,
            p_outage=self.p_outage if live else 0.0,
            add_jobs_count=(self.add_min, self.add_max),
            outage_sats=self.outage_sats,
            outage_len=self.outage_len,
            pool_min=self.pool_min,
            pool_max=self.pool_max,
        )


def _half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def fleet_config(base: dict, n_sats: int) -> dict:
    """Copy of the scenario with a stratified subset of `n_sats` satellites, in original order."""
    sats = base["satellites"]
    if not 1 <= n_sats <= len(sats):
        raise ValueError(f"fleet size {n_sats} is outside 1..{len(sats)}")

    # Orbits and every satellite field stay as loaded. Only the list and the cap change.
    keep = sorted(fleet_order(sats)[:n_sats])
    cfg = copy.deepcopy(base)
    cfg["satellites"] = [copy.deepcopy(sats[i]) for i in keep]
    cfg["model"]["downlink_parallel_limit"] = downlink_slots(n_sats)
    cfg.setdefault("meta", {})["fleet_size"] = int(n_sats)
    return cfg


def fleet_order(sats: list[dict]) -> list[int]:
    """Nested pick order. Every prefix spreads over all planes in RAAN and phase.

    Each pick goes to a plane with the fewest chosen satellites, so plane counts
    never differ by more than one. Inside those planes the pick is the satellite
    farthest from every chosen one on the (RAAN, mean anomaly) torus; ties take
    the lower index. The first pick is satellite 0.
    """
    raan = np.array([float(s.get("orbit", s)["raan_deg"]) for s in sats])
    phase = np.array([float(s.get("orbit", s).get("mean_anomaly_deg", 0.0)) for s in sats])
    planes = plane_index(sats)
    n_planes = int(planes.max()) + 1

    order = [0]
    while len(order) < len(sats):
        counts = np.bincount(planes[order], minlength=n_planes)
        low = counts.min()
        chosen = set(order)
        candidates = [i for i in range(len(sats)) if i not in chosen and counts[planes[i]] == low]
        far = [min(_torus_deg(raan[i], phase[i], raan[j], phase[j]) for j in order) for i in candidates]
        best = max(range(len(candidates)), key=lambda c: (round(far[c], 9), -candidates[c]))
        order.append(candidates[best])
    return order


def plane_index(sats: list[dict]) -> np.ndarray:
    """Orbital plane of each satellite, numbered by ascending RAAN."""
    raan = [round(float(s.get("orbit", s)["raan_deg"]) % 360.0, 6) for s in sats]
    levels = sorted(set(raan))
    return np.array([levels.index(r) for r in raan], dtype=int)


def _torus_deg(raan_a: float, phase_a: float, raan_b: float, phase_b: float) -> float:
    def wrap(d: float) -> float:
        d = abs(d) % 360.0
        return min(d, 360.0 - d)

    return math.hypot(wrap(raan_a - raan_b), wrap(phase_a - phase_b))


def make_env(cfg: dict, events: EventConfig, seed: int, scripted: list[dict] | None = None, max_steps: int | None = None) -> OpsEnv:
    """The evaluator's shift settings with K = 64 and N_max = 48."""
    settings = Eval(scenario=SCENARIO, top_k=TOP_K)
    return OpsEnv(
        scenario=cfg,
        events=events,
        seed=seed,
        scripted_events=scripted,
        max_steps=max_steps,
        n_max=N_MAX,
        **shift_kwargs(settings),
    )


@dataclass
class EventTape:
    """External events of one (fleet, seed), drawn before any policy acts.

    A private env with the same fleet and seed resets, so its initial catalog is
    the one every method gets. Then only its clock moves and its own `EventConfig`
    generator runs at each tick, exactly where `OpsEnv.step` would call it. No job
    in that env ever completes and no satellite is ever busy, so the pool cap reads
    a virtual queue in which every offered job stays open until its deadline. The
    draws therefore do not depend on any policy. Each recorded event is replayed at
    its own tick through `scripted_events`, never earlier.
    """

    n_sats: int
    seed: int
    events: list[dict]
    initial_hash: str
    tape_hash: str
    arrivals_hash: str
    stats: dict

    @classmethod
    def record(cls, cfg: dict, work: Workload, seed: int) -> "EventTape":
        started = time.perf_counter()
        env = make_env(cfg, work.events(live=True), seed)
        events: list[dict] = []
        apply = env._apply_event

        def capture(event: dict) -> None:
            events.append({"at_step": int(env.k), **copy.deepcopy(event)})
            apply(event)

        # Reset draws the initial catalog and the tick-0 events. Later ticks only move the clock.
        env._apply_event = capture
        env.reset(seed=seed)
        open_counts = [env._open_job_count()]
        for k in range(1, env.steps_total):
            env.world.step(env.dt)
            env.k = k
            env._deliver_events()
            open_counts.append(env._open_job_count())

        # Hashes of the initial catalog, the whole tape, and the offered jobs alone.
        arrivals = [job for event in events if event["type"] == "add_jobs" for job in event["jobs"]]
        initial = [job for job_id, job in env.jobs.items() if _INITIAL_ID.match(job_id)]
        outages = [event for event in events if event["type"] == "satellite_outage"]
        stats = {
            "initial_jobs": len(initial),
            "add_events": sum(event["type"] == "add_jobs" for event in events),
            "offered_jobs": len(arrivals),
            "offered_relay": sum(job["kind"] == "relay" for job in arrivals),
            "offered_downlink": sum(job["kind"] == "downlink" for job in arrivals),
            "outage_events": len(outages),
            "outage_satellites": sum(len(event["satellite_ids"]) for event in outages),
            "outage_satellite_ticks": sum(
                len(event["satellite_ids"]) * (int(event["end_step"]) - int(event["at_step"])) for event in outages
            ),
            "virtual_open_queue_mean": float(np.mean(open_counts)),
            "virtual_open_queue_max": int(np.max(open_counts)),
            "record_s": time.perf_counter() - started,
        }
        return cls(
            n_sats=work.n_sats,
            seed=int(seed),
            events=events,
            initial_hash=jobs_hash(initial),
            tape_hash=digest(_plain(events)),
            arrivals_hash=jobs_hash(arrivals),
            stats=stats,
        )

    def delivered_hash(self, steps: int) -> str:
        """Offered jobs an env has received after `steps` ticks: events at ticks 0..steps-1."""
        return jobs_hash(job for event in self.events if event["type"] == "add_jobs" and event["at_step"] < steps for job in event["jobs"])

    def arrival_steps(self) -> dict[str, int]:
        return {job["id"]: int(event["at_step"]) for event in self.events if event["type"] == "add_jobs" for job in event["jobs"]}

    def summary(self) -> dict:
        return {
            "n_sats": self.n_sats,
            "seed": self.seed,
            "tape_hash": self.tape_hash,
            "initial_hash": self.initial_hash,
            "arrivals_hash": self.arrivals_hash,
            **self.stats,
        }


def jobs_hash(jobs) -> str:
    """Hash of the fields a job is born with. Progress and completion are left out."""
    rows = sorted((_plain({key: job.get(key) for key in _STATIC_FIELDS}) for job in jobs), key=lambda row: row["id"])
    return digest(rows)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]

    if isinstance(value, np.generic):
        return value.item()
    return value


class GreedyMethod:
    """`greedy_action` as is. The downlink limit is read from the env on every tick, after `reset`."""

    name = GREEDY

    def begin(self, n_sats: int, seed: int) -> None:
        self.limits: set[int] = set()

    def act(self, obs: dict, env: OpsEnv) -> np.ndarray:
        limit = int(env.downlink_limit)
        self.limits.add(limit)
        return greedy_action(obs, downlink_limit=limit)

    def diagnostics(self) -> dict:
        return {"downlink_limit_used": max(self.limits) if len(self.limits) == 1 else None}


@dataclass
class TickView:
    """Everything Hungarian reads on one tick: packed slots, holders, SoC. No future events."""

    sat_ids: list[str]
    soc: np.ndarray
    slots: list[dict]
    holders: dict[str, str]
    last_executor: dict[str, str]
    downlink_limit: int

    @classmethod
    def from_env(cls, env: OpsEnv) -> "TickView":
        slots = [
            {
                "id": cand.job["id"],
                "kind": cand.job["kind"],
                "value_usd": float(cand.job.get("value_usd", 0.0)),
                "remaining_steps": int(cand.job["remaining_steps"]),
                "work_steps": int(cand.job["work_steps"]),
            }
            for cand in env._candidates[: env.K]
        ]
        return cls(
            sat_ids=list(env.sat_ids),
            soc=np.array([sat.soc for sat in env.fleet], dtype=float),
            slots=slots,
            holders=dict(env._downlink_holder),
            last_executor=dict(env._last_executor),
            downlink_limit=int(env.downlink_limit),
        )


class HungarianMethod:
    """Hungarian + capacity repair. One tick, no lookahead."""

    name = HUNGARIAN

    def begin(self, n_sats: int, seed: int) -> None:
        self.repair_ticks = 0
        self.dropped = 0

    def act(self, obs: dict, env: OpsEnv) -> np.ndarray:
        action, info = hungarian_action(obs, TickView.from_env(env))
        self.repair_ticks += int(info["capacity_repaired"])
        self.dropped += int(info["dropped_downlinks"])
        return action

    def diagnostics(self) -> dict:
        return {"capacity_repair_ticks": self.repair_ticks, "capacity_repair_dropped": self.dropped}


def hungarian_action(obs: dict, view: TickView) -> tuple[np.ndarray, dict]:
    """Max-weight satellite-slot matching for this tick, then the downlink cap.

    Each satellite also has its own idle column of weight 0, so a pair is taken
    only when its weight is positive. If the matching holds more downlinks than
    the cap, the heaviest `downlink_limit` downlinks stay and every other
    satellite is matched again over relay slots only.
    """
    mask = np.asarray(obs["mask"], dtype=bool)
    rows, width = mask.shape
    k = width - 2
    idle, calibrate = k, k + 1
    n = len(view.sat_ids)
    action = np.full(rows, idle, dtype=np.int64)

    # Dollars per tick of work this satellite still needs. Masked pairs stay forbidden.
    weights = pair_weights(mask, view)
    is_down = np.array([slot["kind"] == "downlink" for slot in view.slots], dtype=bool)
    match = _assign(weights, list(range(n)), list(range(len(view.slots))))

    # Past the cap, keep the heaviest downlinks and rematch everyone else on relays.
    downs = sorted(
        ((weights[i, c], c, i) for i, c in match.items() if is_down[c]),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    dropped = max(len(downs) - view.downlink_limit, 0)
    if dropped:
        kept = {i: c for _w, c, i in downs[: view.downlink_limit]}
        free = [i for i in range(n) if i not in kept]
        relays = [c for c in range(len(view.slots)) if not is_down[c]]
        match = {**kept, **_assign(weights, free, relays)}
    for i, c in match.items():
        action[i] = c

    # A satellite left idle calibrates under the greedy rule.
    sats = np.asarray(obs["sats"], dtype=np.float32)
    for i in range(n):
        if i in match or not mask[i, calibrate]:
            continue

        if sats[i, _SAT_CAL_AGE] >= CALIBRATE_AT:
            action[i] = calibrate
    info = {
        "capacity_repaired": bool(dropped),
        "dropped_downlinks": int(dropped),
        "downlinks": int(sum(bool(is_down[c]) for c in match.values())),
        "assigned": len(match),
    }
    return action, info


def pair_weights(mask: np.ndarray, view: TickView) -> np.ndarray:
    """w_ij = V_j / T_ij plus the fixed SoC and holder tie-breaks. -inf where the mask says no."""
    n = len(view.sat_ids)
    weights = np.full((n, len(view.slots)), -np.inf)
    for col, job in enumerate(view.slots):
        if job["value_usd"] <= 0.0:
            continue

        # A relay keeps progress. A downlink restarts unless its holder continues.
        holder = view.holders.get(job["id"])
        for i in np.flatnonzero(mask[:n, col]):
            sid = view.sat_ids[i]
            if job["kind"] == "downlink":
                ticks = downlink_required_steps(job, sid, holder)
            else:
                ticks = int(job["remaining_steps"])
            weights[i, col] = (
                job["value_usd"] / max(ticks, 1)
                + HUNGARIAN_TIE_SOC * float(view.soc[i])
                + HUNGARIAN_TIE_HOLDER * float(view.last_executor.get(job["id"]) == sid)
            )
    return weights


def _assign(weights: np.ndarray, rows: list[int], cols: list[int]) -> dict[int, int]:
    """Hungarian over `rows` x (`cols` + one idle column per row). Returns row -> col."""
    if not rows or not cols:
        return {}
    sub = weights[np.ix_(rows, cols)]
    allowed = np.isfinite(sub) & (sub > 0.0)
    if not allowed.any():
        return {}
    table = np.zeros((len(rows), len(cols) + len(rows)))
    table[:, : len(cols)] = np.where(allowed, sub, _FORBIDDEN)
    r_idx, c_idx = linear_sum_assignment(table, maximize=True)
    return {rows[r]: cols[c] for r, c in zip(r_idx, c_idx) if c < len(cols) and allowed[r, c]}


class RLMethod:
    """A loaded MaskablePPO policy through `PPOAgent`. Stochastic sampling gets its own seed."""

    def __init__(self, model, deterministic: bool) -> None:
        self.name = RL_DETERMINISTIC if deterministic else RL_STOCHASTIC
        self.deterministic = bool(deterministic)
        self.agent = PPOAgent(model, name=self.name, deterministic=deterministic)
        self.device = str(getattr(model, "device", "cpu"))

    def begin(self, n_sats: int, seed: int) -> None:
        import torch

        self.policy_seed = rl_policy_seed(n_sats, seed)
        torch.manual_seed(self.policy_seed)

    def act(self, obs: dict, env: OpsEnv) -> np.ndarray:
        action = self.agent.act(obs)
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
        return action

    def diagnostics(self) -> dict:
        return {"policy_seed": self.policy_seed if not self.deterministic else None}


def rl_policy_seed(n_sats: int, seed: int) -> int:
    return int(np.random.SeedSequence([RL_POLICY_SEED, int(n_sats), int(seed)]).generate_state(1)[0])


def make_method(name: str, model=None):
    if name == GREEDY:
        return GreedyMethod()

    if name == HUNGARIAN:
        return HungarianMethod()

    if name in RL_METHODS.values():
        if model is None:
            raise SystemExit(f"{name} needs --model")
        return RLMethod(model, deterministic=name == RL_DETERMINISTIC)
    raise ValueError(f"unknown method {name}")


def load_policy(path: str | None, base: dict) -> tuple[object, dict]:
    """Load the checkpoint and stop with a clear error if it does not fit N_max, K, or the decoder."""
    if not path:
        raise SystemExit("RL needs --model PATH_TO_MODEL.zip")
    model_path = Path(path)
    if not model_path.is_file():
        raise SystemExit(f"RL checkpoint not found: {model_path}")
    model = load_ppo(model_path)
    info = check_checkpoint(model, model_path, base)
    return model, info


def check_checkpoint(model, model_path: Path, base: dict) -> dict:
    """Tensor shapes, saved weights, decoder class, and the training run's scenario."""
    from stable_baselines3.common.save_util import load_from_zip_file

    from example.models import OpsAttentionPolicy

    errors: list[str] = []
    warnings: list[str] = []
    probe = make_env(fleet_config(base, len(base["satellites"])), Workload.for_fleet(N_MAX).events(live=False), 0)

    # Observation and action tensors must match the env exactly. Nothing is resized here.
    want, have = probe.observation_space, model.observation_space
    if set(want.spaces) != set(getattr(have, "spaces", {})):
        errors.append(f"observation keys {sorted(getattr(have, 'spaces', {}))} != {sorted(want.spaces)}")
    else:
        for key, space in want.spaces.items():
            got = have.spaces[key]
            if got.shape != space.shape or not np.array_equal(got.low, space.low) or not np.array_equal(got.high, space.high):
                errors.append(f"observation '{key}' {got.shape} != {space.shape}")
    nvec = np.asarray(getattr(model.action_space, "nvec", []))
    if nvec.shape != (N_MAX,) or not np.all(nvec == TOP_K + 2):
        errors.append(f"action space {nvec.tolist()} != [{TOP_K + 2}] * {N_MAX}")

    # Decoder: the package policy with its matching head. Saved weights must fit it key by key.
    policy = model.policy
    controller = getattr(policy, "_controller", None)
    if not isinstance(policy, OpsAttentionPolicy):
        errors.append(f"policy class {type(policy).__name__} is not OpsAttentionPolicy")

    if controller not in ("dispatcher", "ctde"):
        errors.append(f"unknown controller {controller!r}")
    try:
        _data, params, _vars = load_from_zip_file(str(model_path), device="cpu", load_data=False)
    except Exception as exc:
        params = None
        errors.append(f"cannot read saved weights: {exc}")

    if params is not None and hasattr(policy, "state_dict"):
        saved = params.get("policy", {})
        current = policy.state_dict()
        missing = sorted(set(current) - set(saved))
        extra = sorted(set(saved) - set(current))
        shapes = sorted(key for key in set(saved) & set(current) if tuple(saved[key].shape) != tuple(current[key].shape))
        if missing or extra or shapes:
            errors.append(f"weights do not fit the network: missing={missing[:5]} extra={extra[:5]} shape={shapes[:5]}")

    # The training run's own config, when it sits beside the checkpoint.
    run_cfg_path = next((p / "config.json" for p in (model_path.parent, model_path.parent.parent) if (p / "config.json").is_file()), None)
    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8")) if run_cfg_path else None
    if run_cfg is None:
        warnings.append("no training config.json beside the checkpoint; scenario not cross-checked")
    else:
        expected = {
            "n_max": N_MAX,
            "top_k": TOP_K,
            "steps": int(base["time"]["steps"]),
            "scenario_id": base.get("id", base.get("meta", {}).get("id")),
        }
        for key, value in expected.items():
            if key in run_cfg and run_cfg[key] != value:
                errors.append(f"training {key}={run_cfg[key]!r}, benchmark uses {value!r}")
        for key, value in shift_kwargs(Eval(scenario=SCENARIO, top_k=TOP_K)).items():
            if key in run_cfg and run_cfg[key] != value:
                warnings.append(f"training {key}={run_cfg[key]!r}, benchmark uses {value!r}")

    # One real prediction: right width and only legal actions.
    if not errors:
        obs, _ = probe.reset(seed=0)
        mask = np.asarray(obs["mask"], dtype=bool)
        action, _ = model.predict(obs, action_masks=mask.reshape(-1), deterministic=True)
        action = np.asarray(action)
        if action.shape != (N_MAX,) or not mask[np.arange(N_MAX), action].all():
            errors.append("checkpoint returned an action outside the mask")

    if errors:
        raise SystemExit("Incompatible RL checkpoint " + str(model_path) + ":\n  - " + "\n  - ".join(errors))
    return {
        "path": str(model_path.resolve()),
        "sha256": _sha256(model_path),
        "controller": controller,
        "policy_class": type(policy).__name__,
        "parameters": int(sum(p.numel() for p in policy.parameters())),
        "device": str(model.device),
        "training_config": str(run_cfg_path) if run_cfg_path else None,
        "training_scenario": (run_cfg or {}).get("scenario_id"),
        "training_n_satellites": (run_cfg or {}).get("n_satellites"),
        "training_events": (run_cfg or {}).get("events"),
        "warnings": warnings,
    }


def play(method, cfg: dict, work: Workload, seed: int, mode: str, tape: EventTape | None = None, max_steps: int | None = None) -> dict:
    """One full shift of one method. Returns one `episodes.csv` row."""
    scripted = copy.deepcopy(tape.events) if tape is not None else []
    env = make_env(cfg, work.events(live=tape is None), seed, scripted, max_steps)
    probe = _Probe(env, method, tape)
    method.begin(work.n_sats, seed)

    started = time.perf_counter()
    result = run_episode(env, probe, seed=seed)
    wall = time.perf_counter() - started

    row = {
        "event_mode": mode,
        "n_sats": work.n_sats,
        "seed": int(seed),
        "method": method.name,
        "method_label": METHOD_LABELS[method.name],
        "tape_hash": tape.tape_hash if tape is not None else None,
    }
    row.update(probe.after_reset)
    row.update(shift_outcome(env))
    row.update(probe.metrics())
    row.update(_env_diagnostics(env, result))
    row.update(method.diagnostics())
    row["tape_delivered_ok"] = tape.delivered_hash(env.k) == row["arrivals_hash"] if tape is not None else None
    row["shift_wall_s"] = wall
    row["consistency_errors"] = ";".join(check_counts(row))
    return row


class _Probe:
    """The actor `run_episode` calls. Times the decision and logs what the queue looked like."""

    def __init__(self, env: OpsEnv, method, tape: EventTape | None) -> None:
        self.env = env
        self.method = method
        self.arrival_step = tape.arrival_steps() if tape is not None else {}
        self.after_reset: dict | None = None
        self.times: list[float] = []
        self.active: list[int] = []
        self.known: list[int] = []
        self.invalid = 0
        self.leaks = 0

    def __call__(self, obs: dict) -> np.ndarray:
        env = self.env
        if self.after_reset is None:
            self.after_reset = self._after_reset()

        # Open work on this tick, and any replayed job that showed up before its own tick.
        k = env.k
        live = [j for j in env.jobs.values() if j["completed_step"] is None and j["remaining_steps"] > 0 and j["deadline_step"] > k]
        self.known.append(len(live))
        self.active.append(sum(j["release_step"] <= k for j in live))
        self.leaks += sum(self.arrival_step.get(job_id, -1) > k for job_id in env.jobs)

        # Only the method's own call is timed: not the step, not the observation.
        started = time.perf_counter()
        action = np.asarray(self.method.act(obs, env), dtype=np.int64)
        self.times.append(time.perf_counter() - started)

        mask = np.asarray(obs["mask"], dtype=bool)
        self.invalid += int((~mask[np.arange(action.shape[0]), action]).sum())
        return action

    def _after_reset(self) -> dict:
        env = self.env
        initial = [job for job_id, job in env.jobs.items() if _INITIAL_ID.match(job_id)]
        cfg = env.events_cfg
        return {
            "downlink_limit": int(env.downlink_limit),
            "initial_jobs": len(initial),
            "initial_hash": jobs_hash(initial),
            "events_p_add_jobs": float(cfg.p_add_jobs),
            "events_p_outage": float(cfg.p_outage),
            "events_add_jobs_count": f"{cfg.add_jobs_count[0]}-{cfg.add_jobs_count[1]}",
            "events_pool": f"{cfg.pool_min}-{cfg.pool_max}",
            "events_outage_sats": f"{cfg.outage_sats[0]}-{cfg.outage_sats[1]}",
        }

    def metrics(self) -> dict:
        # The first tick is warm-up for every method and is reported apart.
        times = np.asarray(self.times[1:] or self.times, dtype=float) * 1e3
        return {
            "active_queue_mean": float(np.mean(self.active)),
            "known_queue_mean": float(np.mean(self.known)),
            "decision_ms_mean": float(times.mean()),
            "decision_ms_median": float(np.median(times)),
            "decision_ms_p95": float(np.percentile(times, 95)),
            "decision_ms_max": float(times.max()),
            "decision_ms_first": float(self.times[0] * 1e3),
            "decisions": len(self.times),
            "invalid_actions": self.invalid,
            "future_event_leaks": self.leaks,
        }


def shift_outcome(env: OpsEnv) -> dict:
    """Completions, misses, and revenue at the end of the shift, from `env.jobs`.

    A job is completed when `completed_step` is set and not later than its
    deadline. It is due when `deadline_step <= env.k`. Missed means due and never
    completed. Rates divide by the due jobs of the same kind.
    """
    k = int(env.k)
    jobs = list(env.jobs.values())
    out: dict = {"steps_executed": k}
    for kind in ("total", *KINDS):
        group = jobs if kind == "total" else [job for job in jobs if job["kind"] == kind]
        done = [job for job in group if _completed(job)]
        due = [job for job in group if int(job["deadline_step"]) <= k]
        missed = [job for job in due if job["completed_step"] is None]
        revenue = float(sum(float(job.get("value_usd", 0.0)) for job in done))
        due_value = _ratio(sum(float(job.get("value_usd", 0.0)) for job in due), len(due))
        per_job = _ratio(revenue, len(done))
        suffix = "" if kind == "total" else f"_{kind}"
        out[f"jobs_generated_{kind}"] = len(group)
        out[f"jobs_due_{kind}"] = len(due)
        out[f"completed_{kind}"] = len(done)
        out[f"missed_{kind}"] = len(missed)
        out[f"completion_rate_{kind}"] = _ratio(sum(_completed(job) for job in due), len(due))
        out[f"miss_rate_{kind}"] = _ratio(len(missed), len(due))
        out[f"revenue{suffix}_usd"] = revenue
        out["revenue_per_completed_job" if kind == "total" else f"revenue_per_completed_{kind}"] = per_job
        out[f"mean_value_due_{kind}"] = due_value
        out[f"q_selection_{kind}"] = _ratio(per_job, due_value) if per_job is not None and due_value else None

    # Not done at the end: missed, or still inside a window that closes later.
    open_end = [job for job in jobs if job["completed_step"] is None]
    future = [job for job in jobs if int(job["deadline_step"]) > k]
    out["unfinished_at_shift_end"] = len(open_end)
    out["pending_with_future_deadline"] = sum(int(job["deadline_step"]) > k for job in open_end)
    out["jobs_future_deadline"] = len(future)
    out["completed_future_deadline"] = sum(_completed(job) for job in future)
    out["completed_late"] = sum(job["completed_step"] is not None and not _completed(job) for job in jobs)

    # Priority 3 within the shift, then the per-satellite scale.
    p3_due = [job for job in jobs if int(job.get("priority", 1)) == 3 and int(job["deadline_step"]) <= k]
    out["due_p3"] = len(p3_due)
    out["completed_p3"] = sum(_completed(job) for job in p3_due)
    out["p3_completion_rate"] = _ratio(out["completed_p3"], len(p3_due))
    n = max(int(env.n_sats), 1)
    out["revenue_per_satellite"] = out["revenue_usd"] / n
    out["completed_per_satellite"] = out["completed_total"] / n
    out["missed_per_satellite"] = out["missed_total"] / n
    out["arrivals_jobs"] = sum(not _INITIAL_ID.match(job["id"]) for job in jobs)
    out["arrivals_hash"] = jobs_hash(job for job in jobs if not _INITIAL_ID.match(job["id"]))
    return out


def _completed(job: dict) -> bool:
    return job["completed_step"] is not None and int(job["completed_step"]) <= int(job["deadline_step"])


def _ratio(num: float, den: float) -> float | None:
    return float(num) / float(den) if den else None


def _env_diagnostics(env: OpsEnv, result: dict) -> dict:
    ops, summary = result["ops"], result["summary"]
    reasons = defaultdict(int)
    for row in env.trace:
        reasons[row["reason"]] += 1
    steps = max(int(env.k), 1)
    return {
        "reward_return": float(result["return"]),
        "downlink_resets": float(ops.get("downlink_resets", 0.0)),
        "downlink_lost_ticks": float(ops.get("downlink_lost_ticks", 0.0)),
        "downlink_work_ticks": float(ops.get("downlink_work_ticks", 0.0)),
        "downlink_capacity_utilization": float(ops.get("downlink_work_ticks", 0.0)) / (int(env.downlink_limit) * steps),
        "executable_jobs_mean": float(ops.get("executable_jobs", 0.0)),
        "executable_outside_k_mean": float(ops.get("executable_outside_slots", 0.0)),
        "feasible_edges_mean": float(ops.get("feasible_edges", 0.0)),
        "executor_switches": float(ops.get("executor_switches", 0.0)),
        "idle_share": float(ops.get("idle_share", 0.0)),
        "calibrate_share": float(ops.get("calibrate_share", 0.0)),
        "job_share": float(ops.get("job_share", 0.0)),
        "events_add_jobs": float(ops.get("events_add_jobs", 0.0)),
        "events_outage": float(ops.get("events_outage", 0.0)),
        "blocked_commands": int(summary.get("blocked_command_count", 0)),
        "ground_capacity_rejections": int(reasons["ground_capacity"]),
        "duplicate_job_rejections": int(reasons["duplicate_job_in_step"]),
        "below_reserve_satellite_steps": int(summary.get("below_reserve_satellite_steps", 0)),
        "brownout_satellite_steps": int(summary.get("brownout_satellite_steps", 0)),
        "minimum_soc_pct": float(summary.get("minimum_soc_pct", 0.0)),
    }


def check_counts(row: dict) -> list[str]:
    """Identities every episode row must satisfy. Returns the names of the broken ones."""
    errors = []

    def same(a, b, name: str) -> None:
        if a is None or b is None:
            if a is not b:
                errors.append(name)
            return

        if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-6):
            errors.append(name)

    for stem in ("jobs_generated", "jobs_due", "completed", "missed"):
        same(row[f"{stem}_total"], row[f"{stem}_relay"] + row[f"{stem}_downlink"], f"{stem}_kinds")
    same(row["revenue_usd"], row["revenue_relay_usd"] + row["revenue_downlink_usd"], "revenue_kinds")

    # Every job is due or has a later deadline; a due job is completed or missed, never both.
    same(row["jobs_due_total"] + row["jobs_future_deadline"], row["jobs_generated_total"], "generated")
    same(row["completed_total"] - row["completed_future_deadline"] + row["missed_total"], row["jobs_due_total"], "due_total")
    same(row["completed_future_deadline"] + row["pending_with_future_deadline"], row["jobs_future_deadline"], "future")
    same(row["missed_total"] + row["pending_with_future_deadline"], row["unfinished_at_shift_end"], "unfinished")
    if not row["completed_future_deadline"]:
        for kind in KINDS:
            same(row[f"completed_{kind}"] + row[f"missed_{kind}"], row[f"jobs_due_{kind}"], f"due_{kind}")
    same(row["completed_late"], 0, "completed_late")
    if row["completed_total"]:
        same(row["revenue_per_completed_job"] * row["completed_total"], row["revenue_usd"], "revenue_per_job")
    else:
        same(row["revenue_per_completed_job"], None, "revenue_per_job_empty")
    return errors


def describe(values: list[float]) -> dict:
    """Mean, sample SD, median, and the t-based 95% CI of the mean over seeds."""
    x = np.asarray(values, dtype=float)
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "ci95_low": None, "ci95_high": None}
    mean = float(x.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "std": None, "median": mean, "ci95_low": None, "ci95_high": None}
    std = float(x.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1)) * std / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "median": float(np.median(x)), "ci95_low": mean - half, "ci95_high": mean + half}


def summarize(rows: list[dict]) -> list[dict]:
    """One row per (event mode, N, method, metric)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["event_mode"], row["n_sats"], row["method"])].append(row)

    out = []
    for (mode, n, method), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], _method_rank(item[0][2]))):
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in group if _finite(row.get(metric))]
            if not values:
                continue
            out.append({"event_mode": mode, "n_sats": n, "method": method, "metric": metric, **describe(values)})
    return out


def paired_deltas(rows: list[dict], opts: Options) -> list[dict]:
    """Per-seed differences A - B inside one (event mode, N), with a paired bootstrap CI."""
    names = opts.method_names()
    rl_names = [name for name in names if name in RL_METHODS.values()]
    pairs = [(rl, other) for rl in rl_names for other in (GREEDY, HUNGARIAN) if other in names]
    if HUNGARIAN in names and GREEDY in names:
        pairs.append((HUNGARIAN, GREEDY))
    index = {(row["event_mode"], row["n_sats"], row["method"], row["seed"]): row for row in rows}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    out = []
    for mode in opts.event_modes:
        for n in opts.fleet_sizes:
            for a, b in pairs:
                for metric in DELTA_METRICS:
                    deltas = []
                    for seed in opts.seeds:
                        ra, rb = index.get((mode, n, a, seed)), index.get((mode, n, b, seed))
                        if ra is None or rb is None or not (_finite(ra.get(metric)) and _finite(rb.get(metric))):
                            continue
                        deltas.append(float(ra[metric]) - float(rb[metric]))
                    if not deltas:
                        continue
                    out.append(
                        {
                            "event_mode": mode,
                            "n_sats": n,
                            "comparison": f"{a} - {b}",
                            "role": "primary" if a in (opts.primary_rl, HUNGARIAN) else "supplementary",
                            "metric": metric,
                            **_paired(np.asarray(deltas), rng),
                            "preliminary": len(deltas) < PRELIMINARY_SEEDS,
                        }
                    )
    return out


def _paired(deltas: np.ndarray, rng: np.random.Generator) -> dict:
    n = int(deltas.size)
    out = {
        "n_pairs": n,
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std(ddof=1)) if n > 1 else None,
        "median_delta": float(np.median(deltas)),
        "wins": int((deltas > 0).sum()),
        "losses": int((deltas < 0).sum()),
        "ties": int((deltas == 0).sum()),
        "bootstrap_ci95_low": None,
        "bootstrap_ci95_high": None,
    }
    if n > 1:
        means = deltas[rng.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))].mean(axis=1)
        out["bootstrap_ci95_low"] = float(np.percentile(means, 2.5))
        out["bootstrap_ci95_high"] = float(np.percentile(means, 97.5))
    return out


def _finite(value) -> bool:
    return value is not None and value != "" and math.isfinite(float(value))


def _method_rank(name: str) -> int:
    order = [GREEDY, HUNGARIAN, RL_STOCHASTIC, RL_DETERMINISTIC]
    return order.index(name) if name in order else len(order)


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (EVENT_MODES.index(r["event_mode"]), r["n_sats"], r["seed"], _method_rank(r["method"])))


def build_report(opts: Options, config: dict, rows: list[dict], summary: list[dict], deltas: list[dict], tapes: list[dict]) -> dict:
    """Run-level facts for the paper: settings, integrity checks, main tables, caveats."""
    preliminary = len(opts.seeds) < PRELIMINARY_SEEDS
    lookup = {(s["event_mode"], s["n_sats"], s["method"], s["metric"]): s for s in summary}
    main_table = [
        {
            "event_mode": mode,
            "n_sats": n,
            "method": method,
            "label": METHOD_LABELS[method],
            **{metric: _stats_only(lookup.get((mode, n, method, metric))) for metric in MAIN_METRICS},
        }
        for mode in opts.event_modes
        for n in opts.fleet_sizes
        for method in opts.method_names()
    ]
    return {
        "title": "Greedy vs Hungarian + capacity repair vs RL (MaskablePPO) on P03 fleet subsets",
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": "preliminary" if preliminary else "final",
        "n_seeds": len(opts.seeds),
        "seeds": list(opts.seeds),
        "fleet_sizes": list(opts.fleet_sizes),
        "primary_event_mode": opts.event_modes[0],
        "event_modes": list(opts.event_modes),
        "primary_rl_mode": opts.primary_rl if "rl" in opts.methods else None,
        "methods": {name: METHOD_LABELS[name] for name in opts.method_names()},
        "notes": _notes(opts),
        "metric_definitions": {
            "completed": "completed_step is set and completed_step <= deadline_step",
            "due": "deadline_step <= env.k at shift end (the denominator of every completion and miss rate, per kind)",
            "missed": "due and completed_step is None",
            "pending_with_future_deadline": "not completed and deadline_step > env.k (never counted as missed)",
            "revenue_usd": "sum of simulator value_usd over completed jobs; no PRICE_DIV rescaling",
            "revenue_per_completed_job": "revenue_usd / completed_total; null when nothing completed",
            "q_selection": "mean value of completed jobs / mean value of due jobs, per kind",
            "per_satellite": "absolute metric / n_sats (revenue, completed, missed)",
            "downlink_capacity_utilization": "executed downlink ticks / (downlink_limit * ticks)",
            "decision_ms": "wall time of the method call only (no env.step, no observation); first tick excluded as warm-up",
            "ci95": "t-interval of the mean over seeds; paired deltas use a percentile bootstrap over seeds",
        },
        "hungarian": {
            "label": METHOD_LABELS[HUNGARIAN],
            "weight": "w_ij = value_usd_j / T_ij + 1e-4 * soc_i + 1e-3 * [i executed j last tick]",
            "T_ij": "relay: remaining_steps; downlink: remaining_steps for the current holder, else work_steps",
            "tie_soc": HUNGARIAN_TIE_SOC,
            "tie_holder": HUNGARIAN_TIE_HOLDER,
            "idle_columns": "one zero-weight idle column per satellite; only pairs with w > 0 and mask = 1 can be chosen",
            "capacity_repair": "keep the downlink_limit heaviest downlinks, then rerun the assignment for every other satellite over relay slots",
            "calibration": f"idle satellites calibrate when allowed and calibration age >= {CALIBRATE_AT} of validity (greedy rule)",
            "repair_ticks_mean": _per_n(summary, HUNGARIAN, "capacity_repair_ticks"),
            "repair_dropped_mean": _per_n(summary, HUNGARIAN, "capacity_repair_dropped"),
        },
        "checks": integrity_checks(opts, rows, tapes),
        "timing": {
            "device": (config.get("model") or {}).get("device", "cpu"),
            "workers": opts.workers,
            "torch_threads_per_worker": 1,
            "method": "time.perf_counter around the method call; RL includes tensor conversion and decoding, Hungarian includes building its tick view",
            "warmup": "the first decision of each shift is excluded and stored as decision_ms_first",
            "load_average_at_start": config["host"]["load_average"],
            "decision_ms_mean": {name: _per_n(summary, name, "decision_ms_mean") for name in opts.method_names()},
        },
        "main_table": main_table,
        "paired_deltas_main": [d for d in deltas if d["metric"] in MAIN_METRICS],
    }


def _stats_only(stats: dict | None) -> dict | None:
    if stats is None:
        return None
    return {key: stats[key] for key in ("n", "mean", "std", "median", "ci95_low", "ci95_high")}


def _notes(opts: Options) -> list[str]:
    notes = [
        "Common event tape is the primary mode: for each (N, seed) the initial catalog, arrivals, and outages are identical for every method; the tape hash is stored per episode.",
        "The tape's pool cap uses a virtual queue in which offered jobs never complete, and outages may hit any satellite; native mode instead caps on the live queue and spares busy satellites.",
        "Native mode shares seeds and parameters only: later arrivals and outages depend on each policy's decisions, so it is not a comparison on an identical job set.",
        "Hungarian + capacity repair solves a one-tick assignment. It is not a whole-shift planner, and with the repair it is not an exact solution of the capped one-tick problem.",
        "Greedy reads env.downlink_limit on every tick after reset (n_sats // 8), not the scenario's initial value.",
    ]
    if "rl" in opts.methods:
        notes.append(
            f"Primary RL mode is {opts.primary_rl}, fixed before the run."
            + (" The other mode is supplementary." if opts.rl_secondary else "")
        )
        notes.append("RL was trained on 48 satellites; N < 48 is zero-shot transfer, so a drop there may come from the shifted observation distribution.")
    if len(opts.seeds) < PRELIMINARY_SEEDS:
        notes.append(f"Only {len(opts.seeds)} seeds: all tables and plots are preliminary.")
    return notes


def _per_n(summary: list[dict], method: str, metric: str) -> dict:
    return {f"{s['event_mode']}/{s['n_sats']}": s["mean"] for s in summary if s["method"] == method and s["metric"] == metric}


def integrity_checks(opts: Options, rows: list[dict], tapes: list[dict]) -> dict:
    """Same tape and catalog per (N, seed), consistent counters, no leaks, the right caps."""
    by_task: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[(row["event_mode"], row["n_sats"], row["seed"])].append(row)
    tape_index = {(t["n_sats"], t["seed"]): t for t in tapes}

    common_same = True
    initial_same = True
    native_differs = 0
    for (mode, n, seed), group in by_task.items():
        initial_same &= len({row["initial_hash"] for row in group}) == 1
        if mode == "common":
            tape = tape_index.get((n, seed), {})
            common_same &= {row["tape_hash"] for row in group} == {tape.get("tape_hash")}
            common_same &= len({row["arrivals_hash"] for row in group}) == 1
            common_same &= all(row["tape_delivered_ok"] for row in group)
            initial_same &= {row["initial_hash"] for row in group} == {tape.get("initial_hash")}
        else:
            native_differs += int(len({row["arrivals_hash"] for row in group}) > 1)

    work_ok = all(
        row["downlink_limit"] == downlink_slots(row["n_sats"])
        and row["initial_jobs"] == min(Workload.for_fleet(row["n_sats"]).initial_jobs, Workload.for_fleet(row["n_sats"]).pool_max)
        and row["events_pool"] == f"{Workload.for_fleet(row['n_sats']).pool_min}-{Workload.for_fleet(row['n_sats']).pool_max}"
        for row in rows
    )
    greedy_ok = all(row.get("downlink_limit_used") == downlink_slots(row["n_sats"]) for row in rows if row["method"] == GREEDY)
    return {
        "common_tape_identical_across_methods": bool(common_same),
        "initial_catalog_identical_across_methods": bool(initial_same),
        "native_tasks_with_diverging_arrivals": native_differs,
        "counter_consistency_errors": sum(bool(row["consistency_errors"]) for row in rows),
        "future_event_leaks": sum(int(row["future_event_leaks"]) for row in rows),
        "invalid_actions": sum(int(row["invalid_actions"]) for row in rows),
        "env_ground_capacity_rejections": _by_method(rows, "ground_capacity_rejections"),
        "env_duplicate_job_rejections": _by_method(rows, "duplicate_job_rejections"),
        "workload_after_reset_ok": bool(work_ok),
        "greedy_downlink_limit_ok": bool(greedy_ok),
        "full_shifts": all(row["steps_executed"] == SHIFT_STEPS for row in rows) if opts.max_steps is None else None,
    }


def _by_method(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = defaultdict(int)
    for row in rows:
        out[row["method"]] += int(row[key])
    return dict(out)


def workload_report(opts: Options, base: dict, rows: list[dict], tapes: list[dict]) -> dict:
    """Target load per N, what the env had after reset, the tape, and the realised queue."""
    out = {}
    planes = plane_index(base["satellites"])
    for n in opts.fleet_sizes:
        work = Workload.for_fleet(n)
        cfg = fleet_config(base, n)
        keep = [i for i, s in enumerate(base["satellites"]) if s["id"] in {c["id"] for c in cfg["satellites"]}]
        group = [row for row in rows if row["n_sats"] == n]
        realised = {}
        for (mode, method), items in _group(group, ("event_mode", "method")).items():
            realised[f"{mode}/{method}"] = {
                key: float(np.mean([row[key] for row in items]))
                for key in ("jobs_generated_total", "arrivals_jobs", "active_queue_mean", "known_queue_mean", "executable_jobs_mean", "executable_outside_k_mean", "events_outage")
            }
        out[str(n)] = {
            "target": asdict(work),
            "satellites": [base["satellites"][i]["id"] for i in keep],
            "plane_counts": np.bincount(planes[keep], minlength=int(planes.max()) + 1).tolist(),
            "after_reset": {
                mode: {
                    key: sorted({row[key] for row in items})
                    for key in ("downlink_limit", "initial_jobs", "events_pool", "events_add_jobs_count", "events_outage_sats", "events_p_add_jobs", "events_p_outage")
                }
                for (mode,), items in _group(group, ("event_mode",)).items()
            },
            "tapes": [t for t in tapes if t["n_sats"] == n],
            "realised_mean": realised,
        }
    return out


def _group(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        out[tuple(row[key] for key in keys)].append(row)
    return out


def plot_all(summary: list[dict], opts: Options, mode: str, out_dir: Path) -> list[Path]:
    """The four required figures plus revenue per satellite, means with 95% CIs over seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = {(s["n_sats"], s["method"], s["metric"]): s for s in summary if s["event_mode"] == mode}
    methods = [name for name in opts.method_names() if name in (GREEDY, HUNGARIAN, opts.primary_rl)]
    tag = f"event mode: {mode}, {len(opts.seeds)} seeds" + (", PRELIMINARY" if len(opts.seeds) < PRELIMINARY_SEEDS else "")
    figures = [
        ("completed_tasks.png", "Completed jobs per shift", "Completed jobs",
         [("completed_total", "All jobs"), ("completed_relay", "Relay"), ("completed_downlink", "Downlink")]),
        ("revenue.png", "Revenue per shift", "Revenue, USD", [("revenue_usd", "All jobs")]),
        ("revenue_per_task.png", "Mean value of a completed job", "USD per completed job",
         [("revenue_per_completed_job", "All jobs"), ("revenue_per_completed_relay", "Relay"), ("revenue_per_completed_downlink", "Downlink")]),
        ("missed_tasks.png", "Missed jobs per shift (deadline passed, not completed)", "Missed jobs",
         [("missed_total", "All jobs"), ("missed_relay", "Relay"), ("missed_downlink", "Downlink")]),
        ("revenue_per_satellite.png", "Revenue per satellite per shift", "USD per satellite", [("revenue_per_satellite", "All jobs")]),
    ]
    paths = []
    sizes = list(opts.fleet_sizes)
    spread = 0.35 * (min(np.diff(sizes)) if len(sizes) > 1 else 4)
    for filename, title, ylabel, panels in figures:
        fig, axes = plt.subplots(1, len(panels), figsize=(max(4.6 * len(panels) + 0.6, 7.5), 4.6), squeeze=False)
        for ax, (metric, panel_title) in zip(axes[0], panels):
            for j, method in enumerate(methods):
                color, marker = METHOD_STYLE[method]
                offset = (j - (len(methods) - 1) / 2) * spread / max(len(methods), 1)
                xs, ys, lo, hi = [], [], [], []
                for n in sizes:
                    s = lookup.get((n, method, metric))
                    if s is None or s["mean"] is None:
                        continue
                    xs.append(n + offset)
                    ys.append(s["mean"])
                    lo.append(s["mean"] - s["ci95_low"] if s["ci95_low"] is not None else 0.0)
                    hi.append(s["ci95_high"] - s["mean"] if s["ci95_high"] is not None else 0.0)
                if xs:
                    ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, capsize=3, linewidth=1.6, markersize=6, label=METHOD_LABELS[method])
            ax.set_xticks(sizes)
            ax.set_xlabel("Satellites in the fleet, N")
            ax.set_ylabel(ylabel)
            if len(panels) > 1:
                ax.set_title(panel_title)
            ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=8, loc="best")
        fig.suptitle(f"{title}\n{tag}\nmean and 95% CI over seeds; RL is zero-shot for N < 48", fontsize=10)
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def print_table(summary: list[dict], opts: Options) -> None:
    """Four main criteria per N and method, mean +- CI half-width, for each event mode."""
    lookup = {(s["event_mode"], s["n_sats"], s["method"], s["metric"]): s for s in summary}
    for mode in opts.event_modes:
        status = " (PRELIMINARY)" if len(opts.seeds) < PRELIMINARY_SEEDS else ""
        print(f"\n== event mode: {mode}, {len(opts.seeds)} seeds{status} ==")
        header = f"{'N':>3}  {'method':<30} {'completed':>16} {'revenue, USD':>20} {'USD / completed':>18} {'missed':>16}"
        print(header)
        print("-" * len(header))
        for n in opts.fleet_sizes:
            for method in opts.method_names():
                cells = [_cell(lookup.get((mode, n, method, metric))) for metric in MAIN_METRICS]
                print(f"{n:>3}  {METHOD_LABELS[method]:<30} {cells[0]:>16} {cells[1]:>20} {cells[2]:>18} {cells[3]:>16}")


def _cell(stats: dict | None) -> str:
    if not stats or stats["mean"] is None:
        return "-"

    if stats["ci95_high"] is None:
        return f"{stats['mean']:.1f}"
    return f"{stats['mean']:.1f} +- {stats['ci95_high'] - stats['mean']:.1f}"


def _config(opts: Options, base: dict, model_info: dict | None) -> dict:
    planes = plane_index(base["satellites"])
    fleets = {}
    for n in opts.fleet_sizes:
        ids = {s["id"] for s in fleet_config(base, n)["satellites"]}
        keep = [i for i, s in enumerate(base["satellites"]) if s["id"] in ids]
        fleets[str(n)] = {
            "satellites": [base["satellites"][i]["id"] for i in keep],
            "plane_counts": np.bincount(planes[keep], minlength=int(planes.max()) + 1).tolist(),
        }
    return {
        "options": {**asdict(opts), "method_names": opts.method_names()},
        "scenario_id": base.get("id", base.get("meta", {}).get("id")),
        "scenario_sha256": digest(base),
        "constants": {
            "N_MAX": N_MAX,
            "TOP_K": TOP_K,
            "SHIFT_STEPS": SHIFT_STEPS,
            "STEP_S": STEP_S,
            "REFERENCE_SATS": REFERENCE_SATS,
            "INITIAL_JOBS_PER_SAT": INITIAL_JOBS_PER_SAT,
            "BASE_POOL_MIN": BASE_POOL_MIN,
            "BASE_POOL_MAX": BASE_POOL_MAX,
            "BASE_ADDED_JOBS_MIN": BASE_ADDED_JOBS_MIN,
            "BASE_ADDED_JOBS_MAX": BASE_ADDED_JOBS_MAX,
            "BASE_OUTAGE_SATS": BASE_OUTAGE_SATS,
            "HUNGARIAN_TIE_SOC": HUNGARIAN_TIE_SOC,
            "HUNGARIAN_TIE_HOLDER": HUNGARIAN_TIE_HOLDER,
            "RL_POLICY_SEED": RL_POLICY_SEED,
            "PRELIMINARY_SEEDS": PRELIMINARY_SEEDS,
            "BOOTSTRAP_SAMPLES": BOOTSTRAP_SAMPLES,
            "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        },
        "shift_settings": shift_kwargs(Eval(scenario=SCENARIO, top_k=TOP_K)),
        "workload": {str(n): asdict(Workload.for_fleet(n)) for n in opts.fleet_sizes},
        "fleets": fleets,
        "model": model_info,
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        },
        "versions": _versions(),
        "git": _git(),
    }


def _versions() -> dict:
    out = {}
    for name in ("numpy", "scipy", "torch", "stable_baselines3", "sb3_contrib", "gymnasium", "matplotlib"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def _git() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    status = run("status", "--porcelain", "--", "example", "orbi", "tools")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(status)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def parse(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description="Greedy vs Hungarian + capacity repair vs RL on P03 fleet subsets")
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--model", default=MODEL, help="trained MaskablePPO checkpoint (.zip); required for RL")
    parser.add_argument("--fleet-sizes", default=",".join(map(str, FLEET_SIZES)))
    parser.add_argument("--seeds", default="0:29", help="inclusive range a:b, or a comma list")
    parser.add_argument("--event-mode", default="common", choices=["common", "native", "both"])
    parser.add_argument("--rl-mode", default="stochastic", choices=list(RL_METHODS), help="primary RL mode, fixed before the run")
    parser.add_argument("--rl-secondary", action="store_true", help="also play the other RL mode as a supplementary result")
    parser.add_argument("--methods", default="greedy,hungarian,rl", help="subset of greedy,hungarian,rl")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--into", default=None, help="update this existing benchmark_* directory instead of creating a new one")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes, one (event mode, N, seed) each")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="seeds 0,1,2; results are marked preliminary")
    args = parser.parse_args(argv)

    methods = tuple(m.strip().lower() for m in args.methods.split(",") if m.strip())
    unknown = set(methods) - {"greedy", "hungarian", "rl"}
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}")
    sizes = tuple(int(x) for x in args.fleet_sizes.split(",") if x.strip())
    opts = Options(
        scenario=args.scenario,
        model=args.model,
        fleet_sizes=sizes,
        seeds=(0, 1, 2) if args.smoke else _seeds(args.seeds),
        event_modes=EVENT_MODES if args.event_mode == "both" else (args.event_mode,),
        rl_mode=args.rl_mode,
        rl_secondary=args.rl_secondary,
        methods=methods,
        output=args.output,
        into=args.into,
        workers=max(int(args.workers), 1),
        plots=not args.no_plots,
        smoke=args.smoke,
    )
    return opts


def _seeds(text: str) -> tuple[int, ...]:
    text = text.strip()
    if ":" in text:
        lo, hi = (int(x) for x in text.split(":", 1))
        return tuple(range(lo, hi + 1))
    return tuple(int(x) for x in text.split(",") if x.strip())


def main(argv: list[str] | None = None) -> Path:
    return run_benchmark(parse(argv))


if __name__ == "__main__":
    main()
