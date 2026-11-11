"""Gymnasium environment. One step is one tick for the whole fleet.

Geometry comes from KeplerWorld. Jobs and events are sampled from seed.
`OpsEnv` is the single model entry for train, greedy, and console.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from orbi.coverage import load_ground_points, load_ground_region, pair_ground_points
from orbi.kepler import KeplerWorld
from orbi.satellite import Satellite as SatelliteTemplate
from tools.observation import Observation
from tools.reward import Reward

N_MAX = 48


def job_slots(n_sats: int) -> int:
    """Job rows in the network input, one per satellite."""
    return max(int(n_sats), 1)
PLAN_MAX = 192
SAT_FEATURES = 14
JOB_FEATURES = 14
EDGE_FEATURES = 8
GLOBAL_FEATURES = 7
# global[GLOBAL_DOWNLINK_CAP] is downlink_limit / N_MAX. The decoder reads that cap back.
GLOBAL_DOWNLINK_CAP = 4
NEIGHBOR_MEMORY = 12
NEIGHBOR_FEATURES = 7
GROUND_POINTS = 64
JOB_DIFFICULTY = 0.15

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "example" / "scenarios"
_ALIASES = {
    "p01": "P01_intro.json",
    "p01_intro": "P01_intro.json",
    "intro": "P01_intro.json",
    "p02": "P02_shift.json",
    "p02_shift": "P02_shift.json",
    "shift": "P02_shift.json",
    "p03": "P03_energy.json",
    "p03_energy": "P03_energy.json",
    "energy": "P03_energy.json",
    "p04": "P04_demand.json",
    "p04_demand": "P04_demand.json",
    "demand": "P04_demand.json",
}
P01_INTRO = SCENARIOS_DIR / "P01_intro.json"
P02_SHIFT = SCENARIOS_DIR / "P02_shift.json"
P03_ENERGY = SCENARIOS_DIR / "P03_energy.json"
P04_DEMAND = SCENARIOS_DIR / "P04_demand.json"


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_scenario(name: str | Path) -> Path:
    key = str(name).strip().lower()
    if key in _ALIASES:
        return SCENARIOS_DIR / _ALIASES[key]
    path = Path(name)
    if not path.exists():
        raise FileNotFoundError(f"Scenario not found: {name}")
    return path


def validate(cfg: dict) -> None:
    if not isinstance(cfg, dict) or "satellites" not in cfg or "time" not in cfg or "model" not in cfg:
        raise ValueError("Initial JSON needs satellites, time, and model")

    # Each satellite needs a unique id, an orbit, and a battery.
    sats = cfg["satellites"]
    if not isinstance(sats, list) or not 1 <= len(sats) <= N_MAX:
        raise ValueError("satellites must contain 1..48 objects")
    ids = []
    for spec in sats:
        sid = spec.get("id")
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError("satellite.id must be a nonempty string")
        ids.append(sid)
        orbit = spec.get("orbit", spec)
        for key in ("raan_deg", "period_min", "inclination_deg", "altitude_km"):
            if not isinstance(orbit.get(key), (int, float)) or isinstance(orbit.get(key), bool):
                raise ValueError(f"Missing orbit field {key}")
        if float(spec.get("capacity_wh", 0)) <= 0:
            raise ValueError("Invalid battery capacity")
    if len(ids) != len(set(ids)):
        raise ValueError("Satellite IDs must be unique")

    steps, dt = cfg["time"].get("steps"), cfg["time"].get("step_s")
    if type(steps) is not int or steps < 1:
        raise ValueError("time.steps must be a positive integer")

    if not isinstance(dt, (int, float)) or dt <= 0:
        raise ValueError("time.step_s must be positive")


def load(path: str | Path | dict) -> dict:
    if isinstance(path, dict):
        validate(path)
        return path
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate(cfg)
    return cfg


class Task:
    def __init__(self, scenario, reward, observation, **options):
        self.scenario = scenario
        self.reward = reward
        self.observation = observation
        self.options = options

    def make_env(self, **kwargs) -> "OpsEnv":
        opts = {**self.options, **kwargs}
        return OpsEnv(scenario=self.scenario, reward=self.reward, observation=self.observation, **opts)


@dataclass
class EventConfig:
    p_add_jobs: float = 1.0
    p_outage: float = 0.05
    add_jobs_count: tuple[int, int] = (16, 28)
    outage_sats: tuple[int, int] = (1, 3)
    outage_len: tuple[int, int] = (2, 8)
    pool_min: int = 200
    pool_max: int = 350

    @classmethod
    def off(cls) -> "EventConfig":
        return cls(p_add_jobs=0.0, p_outage=0.0)


@dataclass
class _Candidate:
    job_id: str
    job: dict
    sats: list[int] = field(default_factory=list)


class OpsEnv(gym.Env):
    """One shift tick. Reward and observation come from the task."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str | Path | dict = "p02",
        reward=None,
        observation=None,
        top_k: int | None = None,
        n_max: int = N_MAX,
        events: EventConfig | None = None,
        scripted_events: list[dict] | None = None,
        max_steps: int | None = None,
        seed: int | None = None,
        baseline_repair: bool = False,
        neighbor_method: str = "competitors",
        neighbor_memory: int = NEIGHBOR_MEMORY,
        ground_points: int = GROUND_POINTS,
        ground_region: str | None = None,
        job_difficulty: float = JOB_DIFFICULTY,
        cluster: list[str] | None = None,
        sat_cls=None,
        job_catalog=None,
    ) -> None:
        super().__init__()

        # Reward and observation come from the task package when the caller omits them.
        if reward is None or observation is None:
            from example.task import bind

            reward, observation = bind(reward, observation)
        self.reward = reward
        self.observation = observation
        self.cfg = scenario if isinstance(scenario, dict) else load(resolve_scenario(scenario))
        self.scenario = self.cfg

        # Fleet width and the sampling knobs for this launch. Slot count follows the fleet.
        self.N = int(n_max)
        self.events_cfg = events if events is not None else EventConfig()
        self.scripted_events = list(scripted_events or [])
        self.max_steps = max_steps
        self.baseline_repair = bool(baseline_repair)
        self.neighbor_method = str(neighbor_method)
        self.neighbor_memory = int(neighbor_memory)
        self.n_ground = max(2, int(ground_points))
        self.job_difficulty = float(np.clip(job_difficulty, 0.0, 1.0))

        # `world` keeps the random catalog draw. A region name takes the centre and its neighbours.
        named = ground_region if ground_region is not None else (self.cfg.get("ground") or {}).get("region")
        self.ground_region = None if named in (None, "", "world", "random") else str(named)
        self.ground_catalog = load_ground_points()

        self.sat_cls = sat_cls
        self.job_catalog = job_catalog
        self._seed = seed

        specs = list(self.cfg["satellites"])
        self.n_sats = len(specs)
        self.K = int(top_k) if top_k is not None else job_slots(self.n_sats)
        if self.n_sats > self.N:
            raise ValueError(f"Scenario has {self.n_sats} satellites, n_max={self.N}")
        self.sat_ids = [str(s["id"]) for s in specs]
        self.steps_total = int(self.cfg["time"]["steps"])
        self.dt = float(self.cfg["time"]["step_s"])
        self.model = dict(self.cfg["model"])
        self.model.setdefault("sun_target_c", 20.0)
        self.model.setdefault("eclipse_target_c", 5.0)
        self.downlink_limit = int(self.model["downlink_parallel_limit"])
        self.calibration_valid = int(self.model["calibration_valid_steps"])
        self.reserve_pct = float(self.model["reserve_soc_pct"])
        self.idle_index = self.K
        self.calibrate_index = self.K + 1

        # One categorical per satellite: K job slots, idle, calibrate.
        self.action_space = gym.spaces.MultiDiscrete([self.K + 2] * self.N)
        self.observation_space = gym.spaces.Dict(
            {
                "sats": gym.spaces.Box(-5.0, 5.0, (self.N, SAT_FEATURES), np.float32),
                "jobs": gym.spaces.Box(-5.0, 5.0, (self.K, JOB_FEATURES), np.float32),
                "plan": gym.spaces.Box(-5.0, 5.0, (PLAN_MAX, JOB_FEATURES), np.float32),
                "edges": gym.spaces.Box(0.0, 1.0, (self.N, self.K, EDGE_FEATURES), np.float32),
                "neighbors": gym.spaces.Box(
                    -5.0, 5.0, (self.N, self.neighbor_memory, NEIGHBOR_FEATURES), np.float32
                ),
                "global": gym.spaces.Box(-5.0, 5.0, (GLOBAL_FEATURES,), np.float32),
                "mask": gym.spaces.Box(0.0, 1.0, (self.N, self.K + 2), np.float32),
            }
        )
        # Episode state. reset fills the fleet, the jobs, and the mask.
        self.cluster = list(cluster) if cluster is not None else None
        self.ground_control = None
        self.neighbor_view = None
        self.rng = np.random.default_rng(seed)
        self.world = KeplerWorld.from_config(self.cfg)
        self.fleet: list[SatelliteTemplate] = []
        self.sats: dict[str, SatelliteTemplate] = {}
        self.jobs: dict[str, dict] = {}
        self.active_points = np.zeros((0, 2), dtype=float)
        self.k = 0
        self.trace: list[dict] = []
        self._mask = np.zeros((self.N, self.K + 2), dtype=bool)
        self._candidates: list[_Candidate] = []
        self._last_executor: dict[str, str] = {}
        self._event_counter = 0
        self._commands: list[dict] = []
        self.ops_stats: dict[str, float] = {}
        self.capacity = np.array([float(s["capacity_wh"]) for s in specs], dtype=np.float32)
        self.base_w = np.array([float(s.get("base_w", 18.0)) for s in specs], dtype=np.float32)
        self.heater_w = np.array([float(s.get("heater_w", 30.0)) for s in specs], dtype=np.float32)

    @property
    def state(self) -> dict[str, dict]:
        return {
            sat.sat_id: {
                "energy_wh": sat.energy_wh,
                "temp_c": sat.temp_c,
                "calibration_age_steps": sat.calibration_age_steps,
            }
            for sat in self.fleet
        }

    def _make_sats(self) -> None:
        cls = self.sat_cls
        if cls is None:
            from example.satellite import Satellite as cls
        self.fleet = [cls.from_spec(spec, i) for i, spec in enumerate(self.cfg["satellites"])]
        self.sats = {sat.sat_id: sat for sat in self.fleet}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)

        # New world and fleet. Orbits stay as loaded; sun and jobs are redrawn.
        self.world = KeplerWorld.from_config(self.cfg)
        self.world.reset(seed=self._seed)
        self._make_sats()
        self.active_points = self._pick_active_points()
        self.k = 0
        self.trace = []
        self._commands = []
        self._last_executor = {}
        self._event_counter = 0
        self.jobs = self._sample_jobs(int(self.cfg.get("jobs", {}).get("initial", max(8, self.n_sats))))
        self.ops_stats = {
            "act_idle": 0.0,
            "act_calibrate": 0.0,
            "act_job": 0.0,
            "conflicts_repaired": 0.0,
            "downlink_slots_used": 0.0,
            "executor_switches": 0.0,
            "events_add_jobs": 0.0,
            "events_outage": 0.0,
            "events_skipped": 0.0,
            "reward_total": 0.0,
            "cal_while_valid": 0.0,
            "cal_with_job": 0.0,
            "cal_without_job": 0.0,
            "cal_soon": 0.0,
            "cal_age_sum": 0.0,
            "cal_count": 0.0,
        }
        self._queue = {
            "ticks": 0.0,
            "feasible_edges": 0.0,
            "executable_jobs": 0.0,
            "executors": 0.0,
            "outside_slots": 0.0,
        }
        self._deliver_events()
        return self._observe(), {"step": 0}

    def action_masks(self) -> np.ndarray:
        return self._mask.reshape(-1).copy()

    def available(self, sid: str) -> bool:
        return self.sats[sid].available(self.k)

    def can_execute(self, sid: str, action: dict) -> tuple[bool, str, float]:
        cache = getattr(self, "_execute_cache", None)
        key = (sid, action.get("action"), action.get("job_id"))

        if isinstance(cache, dict) and key in cache:
            return cache[key]

        result = self._can_execute(sid, action)

        if isinstance(cache, dict):
            cache[key] = result
        return result

    def _can_execute(self, sid: str, action: dict) -> tuple[bool, str, float]:
        sat = self.sats.get(sid)
        if sat is None:
            return False, "unknown_satellite", 0.0
        kind = action.get("action", "idle")
        if kind == "idle":
            return True, "idle", 0.0

        if not sat.available(self.k):
            return False, "satellite_unavailable", 0.0

        # Calibrate has a fixed load. A job must pass the satellite's own check.
        job = self.jobs.get(action.get("job_id")) if kind == "job" else None
        if kind == "calibrate":
            power = sat.calibration_w
        elif kind == "job":
            ok, reason = sat.can_take(self.world, job, self.model, self.k)
            if not ok:
                return False, reason, 0.0
            power = sat.payload_w(kind, job)
        else:
            return False, "unknown_action", 0.0

        # The tick must leave the battery above reserve and the temperature in range.
        solar = sat.panel_w() if self.world.in_sun(sat.index) else 0.0
        energy, temp, _, _ = sat.transition(solar, power, self.model, self.dt)
        reserve = sat.capacity_wh * self.model["reserve_soc_pct"] / 100.0
        if sat.energy_wh < reserve - 1e-9 or energy < reserve - 1e-9:
            return False, "energy_reserve", 0.0
        lo, hi = self.model["payload_min_c"], self.model["payload_max_c"]
        if not (lo <= sat.temp_c <= hi and lo <= temp <= hi):
            return False, "thermal_limit", 0.0
        return True, "accepted", power

    def step(self, action):
        # Decode the integers, advance physics, then score the tick.
        commands, stats = self._apply_actions(np.asarray(action, dtype=np.int64))
        prev = dict(self._last_executor)
        rows = self._advance(commands)
        self._commands.append(commands)
        reward = self.reward.score(self, rows, prev, stats)

        # Running totals for the episode log.
        self.ops_stats["act_idle"] += stats["idle"]
        self.ops_stats["act_calibrate"] += stats["calibrate"]
        self.ops_stats["act_job"] += stats["job"]
        self.ops_stats["conflicts_repaired"] += stats["conflicts"]
        self.ops_stats["downlink_slots_used"] += stats["downlinks"]
        self.ops_stats["cal_while_valid"] += stats["cal_while_valid"]
        self.ops_stats["cal_with_job"] += stats["cal_with_job"]
        self.ops_stats["cal_without_job"] += stats["cal_without_job"]
        self.ops_stats["cal_soon"] += stats["cal_soon"]
        self.ops_stats["cal_age_sum"] += stats["cal_age_sum"]
        self.ops_stats["cal_count"] += stats["cal_count"]
        self.ops_stats["reward_total"] += reward
        # Events keep arriving until the shift or the step cap ends.
        terminated = self.k >= self.steps_total
        truncated = bool(self.max_steps is not None and self.k >= self.max_steps)
        info: dict[str, Any] = {"step": self.k}
        if not (terminated or truncated):
            self._deliver_events()
        else:
            info["summary"] = self.summary()
            info["ops"] = self._episode_metrics()
        return self._observe(), float(reward), bool(terminated), bool(truncated), info

    def _advance(self, commands: dict[str, dict]) -> list[dict]:
        used: set[str] = set()
        downlinks = 0
        rows = []
        for sat in self.fleet:
            requested = commands.get(sat.sat_id, {"action": "idle"})
            ok, reason, power = self.can_execute(sat.sat_id, requested)
            job = self.jobs.get(requested.get("job_id")) if requested.get("action") == "job" else None

            # One holder per job, and only `downlink_limit` downlinks in the tick.
            if ok and job:
                if job["id"] in used:
                    ok, reason, power = False, "duplicate_job_in_step", 0.0
                elif job["kind"] == "downlink" and downlinks >= self.downlink_limit:
                    ok, reason, power = False, "ground_capacity", 0.0
            actual = requested.get("action", "idle") if ok else "idle"
            if not ok:
                power = 0.0

            # Battery, temperature, and calibration age for this satellite.
            solar = sat.panel_w() if self.world.in_sun(sat.index) else 0.0
            before_e, before_t = sat.energy_wh, sat.temp_c
            energy, temp, heater, load_w = sat.transition(solar, power, self.model, self.dt)
            sat.energy_wh = min(sat.capacity_wh, max(0.0, energy))
            sat.temp_c = temp
            sat.calibration_age_steps = 0 if actual == "calibrate" else sat.calibration_age_steps + 1

            # A finished job leaves the queue on the next tick.
            completed = None
            if ok and actual == "job" and job is not None:
                used.add(job["id"])
                downlinks += int(job["kind"] == "downlink")
                job["remaining_steps"] -= 1
                if job["remaining_steps"] == 0:
                    job["completed_step"] = self.k + 1
                    completed = job["id"]
            rows.append(
                {
                    "step": self.k,
                    "satellite_id": sat.sat_id,
                    "requested": requested,
                    "executed": actual,
                    "reason": reason,
                    "energy_before_wh": round(before_e, 6),
                    "energy_after_wh": round(sat.energy_wh, 6),
                    "temp_before_c": round(before_t, 6),
                    "temp_after_c": round(temp, 6),
                    "solar_w": solar,
                    "heater_w": heater,
                    "load_w": load_w,
                    "calibration_age_steps": sat.calibration_age_steps,
                    "completed_job": completed,
                    "brownout": energy < 0,
                    "below_reserve": sat.energy_wh < sat.capacity_wh * self.reserve_pct / 100.0 - 1e-9,
                }
            )
        # The constellation moves after every satellite has been updated.
        self.world.step(self.dt)
        self.trace.extend(rows)
        self.k += 1
        return rows

    def _pick_active_points(self) -> np.ndarray:
        # A named region is already ordered from its centre. The first m rows are the local group.
        if self.ground_region:
            catalog = load_ground_region(self.ground_region)
            m = min(self.n_ground, len(catalog))
            return np.array(catalog[:m], dtype=float, copy=True)

        catalog = load_ground_points()
        m = min(self.n_ground, len(catalog))
        idx = np.sort(self.rng.choice(len(catalog), size=m, replace=False))
        return catalog[idx]

    def _job_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return pair_ground_points(
            self.rng,
            self.active_points,
            self.world.coverage_radius_m(),
            self.job_difficulty,
            self.world.coverage_radius_std_m(),
            self.ground_catalog,
        )

    def _jobs(self):
        if self.job_catalog is None:
            from example.jobs import catalog

            self.job_catalog = catalog()
        return self.job_catalog

    def _sample_jobs(self, n: int) -> dict[str, dict]:
        return self._jobs().sample_initial(self.rng, self, n)

    def _deliver_events(self) -> None:
        k = self.k
        for event in self.scripted_events:
            if event.get("at_step") == k:
                self._apply_event(event)
        cfg = self.events_cfg
        if cfg.p_add_jobs and self.rng.random() < cfg.p_add_jobs:
            self._emit_add_jobs(k)

        if cfg.p_outage and self.rng.random() < cfg.p_outage:
            self._emit_outage(k)

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"RND-{self._event_counter:04d}"

    def _apply_event(self, event: dict) -> None:
        typ = event.get("type")
        if typ == "add_jobs":
            for job in event.get("jobs", []):
                live = dict(job, remaining_steps=job["work_steps"], completed_step=None)
                if live.get("addressed") is None and (live.get("point_a") is None or live.get("point_b") is None):
                    point_a, point_b = self._job_points()
                    live.setdefault("point_a", point_a)
                    live.setdefault("point_b", point_b)
                    live.setdefault("points", [live["point_a"], live["point_b"]])
                live.setdefault("eligible_satellites", list(self.sat_ids))
                self.jobs[job["id"]] = live
            self.ops_stats["events_add_jobs"] += 1.0
        elif typ == "satellite_outage":
            end = int(event["end_step"])
            for sid in event.get("satellite_ids", []):
                if sid in self.sats:
                    self.sats[sid].outage_until = max(self.sats[sid].outage_until, end)
            self.ops_stats["events_outage"] += 1.0

    def _open_job_count(self) -> int:
        k = self.k
        return sum(
            1
            for job in self.jobs.values()
            if job["completed_step"] is None
            and job["remaining_steps"] > 0
            and job["release_step"] <= k < job["deadline_step"]
        )

    def _emit_add_jobs(self, k: int) -> None:
        open_n = self._open_job_count()
        lo_pool = int(self.events_cfg.pool_min)
        hi_pool = int(self.events_cfg.pool_max)
        if hi_pool > 0 and open_n >= hi_pool:
            return
        lo, hi = self.events_cfg.add_jobs_count
        hi = max(int(hi), int(lo), 1)
        count = int(self.rng.integers(int(lo), hi + 1))
        if lo_pool > 0 and open_n < lo_pool:
            count = max(count, min(hi, lo_pool - open_n))
        if hi_pool > 0:
            count = min(count, hi_pool - open_n)
        if count <= 0:
            return
        jobs = self._jobs().sample_added(self.rng, self, k, count, self._next_event_id())
        if jobs:
            self._apply_event({"type": "add_jobs", "jobs": jobs})

    def _emit_outage(self, k: int) -> None:
        busy = set(self._last_executor.values())
        pool = [sid for sid in self.sat_ids if sid not in busy]
        if not pool:
            return
        lo, hi = self.events_cfg.outage_sats
        size = int(self.rng.integers(lo, min(hi, len(pool)) + 1))
        picks = [str(x) for x in self.rng.choice(pool, size=size, replace=False)]
        end = k + int(self.rng.integers(self.events_cfg.outage_len[0], self.events_cfg.outage_len[1] + 1))
        if end > self.steps_total or end <= k:
            return
        self._apply_event({"type": "satellite_outage", "satellite_ids": picks, "end_step": end})

    def _observe(self):
        return self.observation.encode(self)

    def _apply_actions(self, action: np.ndarray) -> tuple[dict[str, dict], dict[str, float]]:
        decoded = self._decode_actions(action)
        conflicts = 0
        if self.baseline_repair:
            decoded, conflicts = self._baseline_repair(decoded)
        return self._commands_and_stats(decoded, conflicts)

    def _decode_actions(self, action: np.ndarray) -> list[tuple[str, str | None]]:
        decoded: list[tuple[str, str | None]] = []
        for i, _sid in enumerate(self.sat_ids):
            a = int(action[i]) if i < action.shape[0] else self.idle_index
            if a == self.calibrate_index and self._mask[i, self.calibrate_index]:
                decoded.append(("calibrate", None))
            elif a < self.K and a < len(self._candidates) and self._mask[i, a]:
                decoded.append(("job", self._candidates[a].job_id))
            else:
                decoded.append(("idle", None))
        return decoded

    def _baseline_repair(self, decoded: list[tuple[str, str | None]]) -> tuple[list[tuple[str, str | None]], int]:
        decoded = list(decoded)
        conflicts = 0
        by_job: dict[str, list[int]] = {}
        for i, (kind, job_id) in enumerate(decoded):
            if kind == "job" and job_id is not None:
                by_job.setdefault(job_id, []).append(i)

        # Keep the previous executor, else the fullest battery. The rest go idle.
        forced_idle: set[int] = set()
        for job_id, holders in by_job.items():
            if len(holders) == 1:
                continue
            last = self._last_executor.get(job_id)
            holders.sort(
                key=lambda i: (
                    0 if self.sat_ids[i] == last else 1,
                    -(self.sats[self.sat_ids[i]].energy_wh / self.capacity[i]),
                    i,
                )
            )
            for i in holders[1:]:
                decoded[i] = ("idle", None)
                forced_idle.add(i)
                conflicts += 1
            by_job[job_id] = holders[:1]

        # Past the downlink cap, drop the lowest-priority passes.
        downlinks = [
            (job_id, holders[0])
            for job_id, holders in by_job.items()
            if holders and self.jobs[job_id]["kind"] == "downlink"
        ]
        if len(downlinks) > self.downlink_limit:
            downlinks.sort(
                key=lambda p: (
                    -self.jobs[p[0]]["priority"],
                    self.jobs[p[0]]["deadline_step"],
                    -float(self.jobs[p[0]].get("value_usd", 0.0)),
                )
            )
            for _, i in downlinks[self.downlink_limit :]:
                decoded[i] = ("idle", None)
                forced_idle.add(i)
                conflicts += 1

        # Hand a unique downlink to a satellite that was on a relay, then refill idle rows.
        self._yield_unique_downlinks(decoded, forced_idle)
        self._fill_forced_idle_with_relays(decoded, forced_idle)
        return decoded, conflicts

    def _commands_and_stats(self, decoded, conflicts: int):
        downlinks = [
            job_id
            for kind, job_id in decoded
            if kind == "job" and job_id is not None and self.jobs[job_id]["kind"] == "downlink"
        ]
        # Idle is the default, so only calibrate and job become commands.
        commands: dict[str, dict] = {}
        counts = {"idle": 0, "calibrate": 0, "job": 0}
        for i, (kind, job_id) in enumerate(decoded):
            counts[kind] += 1
            if kind == "calibrate":
                commands[self.sat_ids[i]] = {"action": "calibrate"}
            elif kind == "job" and job_id is not None:
                commands[self.sat_ids[i]] = {"action": "job", "job_id": job_id}

        # Fresh calibration is the part of the window before calibrate_free_at.
        early = 0
        while_valid = with_job = without_job = soon = n_cal = 0
        age_sum = 0
        valid = max(self.calibration_valid, 1)
        quarter = max(valid // 4, 1)
        free_at = self.reward.calibrate_free_at
        for i, (kind, _) in enumerate(decoded):
            if kind != "calibrate":
                continue

            n_cal += 1
            age = int(self.sats[self.sat_ids[i]].calibration_age_steps)
            age_sum += age

            if age / valid < free_at:
                early += 1

            if age < self.calibration_valid:
                while_valid += 1

            if self._mask[i, : self.K].any():
                with_job += 1
            else:
                without_job += 1

            if age < quarter:
                soon += 1
        return commands, {
            "idle": float(counts["idle"]),
            "calibrate": float(counts["calibrate"]),
            "job": float(counts["job"]),
            "conflicts": float(conflicts),
            "early_calibrations": float(early),
            "downlinks": float(len(downlinks)),
            "cal_while_valid": float(while_valid),
            "cal_with_job": float(with_job),
            "cal_without_job": float(without_job),
            "cal_soon": float(soon),
            "cal_age_sum": float(age_sum),
            "cal_count": float(n_cal),
        }

    def _slot_by_id(self) -> dict[str, int]:
        return {c.job_id: slot for slot, c in enumerate(self._candidates)}

    @staticmethod
    def _job_pref(job: dict) -> tuple:
        return (-job["priority"], job["deadline_step"], -float(job.get("value_usd", 0.0)), job["id"])

    def _yield_unique_downlinks(self, decoded, forced_idle: set[int]) -> None:
        slots = self._slot_by_id()
        taken = {job_id for kind, job_id in decoded if kind == "job" and job_id is not None}
        n_downlinks = sum(1 for job_id in taken if self.jobs[job_id]["kind"] == "downlink")

        # A downlink only one satellite can see, and that satellite is on a relay.
        unique = []
        for slot, cand in enumerate(self._candidates):
            job = cand.job
            if job["kind"] != "downlink" or job["id"] in taken:
                continue
            holders = [i for i in range(self.n_sats) if self._mask[i, slot]]
            if len(holders) == 1:
                unique.append((job, holders[0], slot))
        unique.sort(key=lambda row: self._job_pref(row[0]))
        for job, i, _slot in unique:
            if n_downlinks >= self.downlink_limit:
                break
            kind, current = decoded[i]
            if kind != "job" or current is None or self.jobs[current]["kind"] != "relay":
                continue
            decoded[i] = ("job", job["id"])
            taken.add(job["id"])
            taken.discard(current)
            n_downlinks += 1
            relay_slot = slots.get(current)
            if relay_slot is None:
                continue
            last = self._last_executor.get(current)
            recipients = [
                j
                for j, (other_kind, _) in enumerate(decoded)
                if j != i and other_kind == "idle" and self._mask[j, relay_slot]
            ]
            recipients.sort(key=lambda j: (0 if j in forced_idle else 1, 0 if self.sat_ids[j] == last else 1, j))
            if recipients:
                decoded[recipients[0]] = ("job", current)
                taken.add(current)
                forced_idle.discard(recipients[0])

    def _fill_forced_idle_with_relays(self, decoded, forced_idle: set[int]) -> None:
        taken = {job_id for kind, job_id in decoded if kind == "job" and job_id is not None}
        relays = [(cand.job, slot) for slot, cand in enumerate(self._candidates) if cand.job["kind"] == "relay" and cand.job_id not in taken]
        relays.sort(key=lambda row: self._job_pref(row[0]))
        for i in sorted(forced_idle):
            if decoded[i][0] != "idle":
                continue
            for job, slot in relays:
                if job["id"] in taken or not self._mask[i, slot]:
                    continue
                decoded[i] = ("job", job["id"])
                taken.add(job["id"])
                break

    def _episode_metrics(self) -> dict[str, float]:
        total = max(self.ops_stats["act_idle"] + self.ops_stats["act_calibrate"] + self.ops_stats["act_job"], 1.0)
        out = dict(self.ops_stats)
        out["idle_share"] = self.ops_stats["act_idle"] / total
        out["calibrate_share"] = self.ops_stats["act_calibrate"] / total
        out["job_share"] = self.ops_stats["act_job"] / total
        n_cal = self.ops_stats["cal_count"]
        out["cal_age_mean"] = self.ops_stats["cal_age_sum"] / n_cal if n_cal else 0.0

        # Means over the ticks that built an observation. Executors are per executable job.
        ticks = self._queue["ticks"]
        out["feasible_edges"] = self._queue["feasible_edges"] / ticks if ticks else 0.0
        out["executable_jobs"] = self._queue["executable_jobs"] / ticks if ticks else 0.0
        out["executable_outside_slots"] = self._queue["outside_slots"] / ticks if ticks else 0.0
        job_ticks = self._queue["executable_jobs"]
        out["executors_per_job"] = self._queue["executors"] / job_ticks if job_ticks else 0.0

        finished = [job for job in self.jobs.values() if job["completed_step"] is not None]
        revenue = sum(float(job.get("value_usd", 0.0)) for job in finished)
        used = self.ops_stats["downlink_slots_used"]
        out["revenue_per_downlink_tick"] = revenue / used if used else 0.0
        for kind in ("relay", "downlink"):
            group = [job for job in finished if job.get("kind") == kind]
            out[f"done_{kind}"] = float(len(group))
            out[f"revenue_{kind}"] = float(sum(float(job.get("value_usd", 0.0)) for job in group))
        for priority in (1, 2, 3):
            group = [job for job in finished if int(job.get("priority", 1)) == priority]
            out[f"done_p{priority}"] = float(len(group))
            out[f"revenue_p{priority}"] = float(sum(float(job.get("value_usd", 0.0)) for job in group))
        return out

    def summary(self) -> dict:
        # Completions, misses, and the worst battery. Custom job fields append themselves.
        finished = [j for j in self.jobs.values() if j["completed_step"] is not None]
        due = [j for j in self.jobs.values() if j["deadline_step"] <= self.k]
        critical = [j for j in due if j["priority"] == 3]
        socs = [100.0 * sat.energy_wh / sat.capacity_wh for sat in self.fleet]
        from tools.jobs import job_field_summary

        report = {
            "steps_executed": self.k,
            "jobs_total": len(self.jobs),
            "jobs_completed": len(finished),
            "jobs_due": len(due),
            "jobs_due_missed": sum(j["completed_step"] is None for j in due),
            "critical_jobs_due": len(critical),
            "critical_jobs_completed_on_time": sum(j["completed_step"] is not None for j in critical),
            "blocked_command_count": sum(r["reason"] not in ("accepted", "idle") for r in self.trace),
            "below_reserve_satellite_steps": sum(r["below_reserve"] for r in self.trace),
            "brownout_satellite_steps": sum(r["brownout"] for r in self.trace),
            "minimum_soc_pct": round(min(socs, default=100.0), 6),
        }
        report.update(job_field_summary(self))
        return report

    def journal(self) -> dict:
        return {
            "schema_version": "orbi-ops-result-1.0",
            "scenario_id": self.cfg.get("id", self.cfg.get("meta", {}).get("id", "")),
            "seed": self._seed,
            "commands": self._commands,
            "steps_executed": self.k,
            "summary": self.summary(),
        }


__all__ = [
    "EDGE_FEATURES",
    "EventConfig",
    "GLOBAL_DOWNLINK_CAP",
    "GLOBAL_FEATURES",
    "JOB_FEATURES",
    "N_MAX",
    "NEIGHBOR_FEATURES",
    "NEIGHBOR_MEMORY",
    "PLAN_MAX",
    "Observation",
    "OpsEnv",
    "P01_INTRO",
    "P02_SHIFT",
    "P03_ENERGY",
    "P04_DEMAND",
    "Reward",
    "SAT_FEATURES",
    "SCENARIOS_DIR",
    "job_slots",
    "Task",
    "digest",
    "load",
    "resolve_scenario",
    "validate",
    "GROUND_POINTS",
    "JOB_DIFFICULTY",
]
