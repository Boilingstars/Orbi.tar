"""Job families the simulator knows: downlink and relay.

A user package subclasses `Downlink` or `Relay`. When that package is selected
for training, every concrete subclass is drawn into the episode with its
`probability`. Subclasses differ by visibility, how far the ground points sit,
how long the job runs, and extra fields. Extra fields are not part of the core
log; register them with `register_job_field` so training prints them.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from orbi.coverage import pair_ground_points

_FIELDS: dict[str, tuple[str, Callable]] = {}


def register_job_field(name: str, log_key: str, reduce: Callable) -> None:
    """Publish one custom job field in the training log.

    `reduce(env, finished_jobs) -> float` runs at episode end. `name` is the
    key on `summary`. `log_key` is the column in `progress.csv` and the cmd table.
    """

    _FIELDS[name] = (log_key, reduce)


def job_field_summary(env) -> dict[str, float]:
    finished = [job for job in env.jobs.values() if job["completed_step"] is not None]
    return {name: float(reduce(env, finished)) for name, (_log_key, reduce) in _FIELDS.items()}


def job_field_log_map() -> dict[str, str]:
    return {name: log_key for name, (log_key, _reduce) in _FIELDS.items()}


class UniformRange:
    """Inclusive integer range."""

    def __init__(self, lo: int, hi: int) -> None:
        if hi < lo:
            raise ValueError(f"duration range {lo}..{hi} is empty")
        self.lo = int(lo)
        self.hi = int(hi)

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.lo, self.hi + 1))


class WeightedLaw:
    """Discrete law: values drawn with the given weights."""

    def __init__(self, values: tuple[int, ...] | list[int], weights: tuple[float, ...] | list[float]) -> None:
        if len(values) == 0 or len(values) != len(weights):
            raise ValueError("duration law needs one weight per value")
        self.values = [int(v) for v in values]
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("duration weights must sum to a positive number")
        self.weights = [float(w) / total for w in weights]

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.values, p=self.weights))


class SeePoints:
    """The satellite must see every ground point: A, A+B, or A+B+C, ...

    `far_p` is the chance the next point lies outside one satellite's footprint.
    `None` uses the environment's `far_job_p`.
    """

    def __init__(self, n: int, far_p: float | None = None) -> None:
        if n < 1:
            raise ValueError("see-points needs at least one ground point")
        self.n = int(n)
        self.far_p = far_p

    @property
    def name(self) -> str:
        return f"see_{self.n}"

    def place(self, rng: np.random.Generator, env) -> dict:
        far_p = env.far_job_p if self.far_p is None else float(self.far_p)
        pts = env.active_points
        if self.n == 1:
            i = int(rng.integers(len(pts)))
            point = (float(pts[i, 0]), float(pts[i, 1]))
            return {
                "points": [point],
                "point_a": point,
                "point_b": point,
                "addressed": None,
            }
        coverage = env.world.coverage_radius_m()
        first, second = pair_ground_points(rng, pts, coverage, far_p)
        points = [first, second]
        while len(points) < self.n:
            _anchor, extra = pair_ground_points(rng, pts, coverage, far_p)
            points.append(extra)
        return {
            "points": points,
            "point_a": points[0],
            "point_b": points[1],
            "addressed": None,
        }


class Addressed:
    """No visibility condition. The job names one satellite by fleet index."""

    name = "addressed"

    def place(self, rng: np.random.Generator, env) -> dict:
        index = int(rng.integers(0, env.n_sats))
        return {
            "points": [],
            "point_a": None,
            "point_b": None,
            "addressed": index,
            "eligible_satellites": [env.sat_ids[index]],
        }


class JobType:
    """One concrete task the generator can draw. Subclass `Downlink` or `Relay`."""

    family = ""
    name = ""
    probability = 0.0
    condition = SeePoints(2)
    work: UniformRange | WeightedLaw = UniformRange(1, 4)
    slack: UniformRange | WeightedLaw = UniformRange(2, 7)
    priority_p = (0.3, 0.4, 0.3)

    def sample_extras(self, rng: np.random.Generator, draft: dict) -> dict:
        return {}

    def sample(
        self,
        rng: np.random.Generator,
        env,
        job_id: str,
        release: int | None = None,
    ) -> dict | None:
        # Duration and slack. A mid-episode job that would end past the shift is dropped.
        work = int(self.work.sample(rng))
        slack = int(self.slack.sample(rng))
        if release is None:
            horizon = max(1, env.steps_total - work - slack)
            release = int(rng.integers(0, horizon))
            deadline = min(env.steps_total, release + work + slack)
        else:
            deadline = release + work + slack
            if deadline > env.steps_total:
                return None
        # Priority, where it sits, and whatever extra fields the subclass adds.
        priority = int(rng.choice([1, 2, 3], p=self.priority_p))
        placed = self.condition.place(rng, env)
        draft = {"priority": priority, "work_steps": work, "kind": self.family}
        extras = dict(self.sample_extras(rng, draft))
        job = {
            "id": job_id,
            "kind": self.family,
            "type": self.name,
            "release_step": int(release),
            "deadline_step": int(deadline),
            "work_steps": work,
            "remaining_steps": work,
            "completed_step": None,
            "priority": priority,
            "eligible_satellites": list(placed.get("eligible_satellites", env.sat_ids)),
            "points": list(placed.get("points", [])),
            "point_a": placed.get("point_a"),
            "point_b": placed.get("point_b"),
            "addressed": placed.get("addressed"),
            "condition": self.condition.name,
            "extras": extras,
        }
        job.update(extras)
        return job


class Downlink(JobType):
    """Base family. Concrete subclasses set `name` and `probability`."""

    family = "downlink"


class Relay(JobType):
    """Base family. Concrete subclasses set `name` and `probability`."""

    family = "relay"


class JobCatalog:
    """Draws concrete downlink and relay subclasses by their probabilities."""

    def __init__(self, types: list[JobType]) -> None:
        usable = [
            job_type
            for job_type in types
            if type(job_type) not in (JobType, Downlink, Relay) and job_type.probability > 0
        ]
        if not usable:
            raise ValueError("job catalog is empty")
        for job_type in usable:
            if job_type.family not in ("downlink", "relay"):
                raise ValueError(f"{job_type.name} family must be downlink or relay")
        weights = np.asarray([job_type.probability for job_type in usable], dtype=float)
        self.types = usable
        self.probs = weights / weights.sum()

    @classmethod
    def collect(cls, module) -> "JobCatalog":
        found: list[JobType] = []
        for obj in vars(module).values():
            if not isinstance(obj, type) or not issubclass(obj, JobType):
                continue

            if obj in (JobType, Downlink, Relay):
                continue

            if obj.probability <= 0 or not obj.name:
                continue
            found.append(obj())
        return cls(found)

    def pick(self, rng: np.random.Generator) -> JobType:
        return self.types[int(rng.choice(len(self.types), p=self.probs))]

    def sample_initial(self, rng: np.random.Generator, env, n: int) -> dict[str, dict]:
        jobs: dict[str, dict] = {}
        for i in range(max(0, n)):
            job_id = f"J{i:03d}"
            job = self.pick(rng).sample(rng, env, job_id, release=None)
            if job is not None:
                jobs[job_id] = job
        return jobs

    def sample_added(
        self,
        rng: np.random.Generator,
        env,
        k: int,
        count: int,
        ident,
    ) -> list[dict]:
        jobs = []
        for i in range(count):
            job = self.pick(rng).sample(rng, env, f"{ident}-J{i}", release=k)
            if job is not None:
                jobs.append(job)
        return jobs
