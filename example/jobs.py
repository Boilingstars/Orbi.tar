"""Job types for this app. Each concrete downlink and relay is drawn into an episode with their probability.

Work follows the zone pass, not the Kepler period: 80% of jobs fit in one third of a
Europe dual-access so a satellite can close 1–3 of them before it flies out.
"""

from __future__ import annotations

import sys

import numpy as np

from orbi.coverage import haversine_m, pair_ground_points
from orbi.satellite import job_points
from tools.jobs import (
    Addressed,
    Downlink,
    JobCatalog,
    Relay,
    SeePoints,
    UniformRange,
    register_job_field,
)


# Dual-access median over Europe at 1500 km / 5 min, as a fraction of the orbit.
_ZONE_ORBIT_FRAC = 0.13


def downlink_slots(n_sats: int) -> int:
    """Parallel downlink slots. One eighth of the fleet, and at least one."""
    return max(int(n_sats) // 8, 1)


def zone_dwell_ticks(env) -> int:
    """Median dual-access length in ticks for a Europe-sized job pair."""
    dt = max(float(getattr(env, "dt", 300.0)), 1.0)
    periods = getattr(getattr(env, "world", None), "periods_s", None)
    if periods is None or len(periods) == 0:
        period_s = 23.0 * dt
    else:
        period_s = float(np.mean(periods))
    return max(3, int(round(_ZONE_ORBIT_FRAC * period_s / dt)))


def work_ref_ticks(env) -> int:
    """80% bin: one third of a zone pass, so three of these fill one dwell."""
    return max(1, int(round(zone_dwell_ticks(env) / 3.0)))


def work_span_ticks(env) -> float:
    """Longest 5% job, used to scale remaining-work features."""
    return float(max(3 * zone_dwell_ticks(env), 1))


def _first_downlink_window(env, rng, points, work: int, from_step: int) -> tuple[int, int] | None:
    """One contiguous pass long enough for `work` ticks, or the longest shorter pass."""
    if not points or work < 1:
        return None

    world = env.world
    dt = max(float(env.dt), 1.0)
    t_now = float(world.t)
    t_from = t_now + (int(from_step) - int(env.k)) * dt
    t_limit = t_now + (int(env.steps_total) - int(env.k)) * dt
    sats = np.arange(env.n_sats)
    rng.shuffle(sats)

    best: tuple[int, int] | None = None
    best_ticks = 0
    for sat_i in sats:
        t = t_from
        for _ in range(8):
            if t > t_limit + 1e-9:
                break
            ticks, nxt = world.access_run_ticks(int(sat_i), points, t, step_s=dt)
            if ticks > 0:
                start = int(from_step) + int(round((t - t_from) / dt))
                end = min(int(env.steps_total), start + ticks)
                run = end - start
                if run >= work:
                    return start, end
                if run > best_ticks:
                    best_ticks = run
                    best = (start, end)

                # The next orbit is a different window. Do not stitch this run to it.
                remaining, _nxt = world.access_windows(int(sat_i), points, t, step_s=dt)
                t = t + max(float(remaining), dt) + dt
                continue

            if nxt <= 0:
                break
            t = t + float(nxt)
    return best


def _bind_downlink_window(env, rng, job: dict, from_step: int) -> dict | None:
    """Pin a downlink to one pass where some satellite can finish it without a gap."""
    points = job_points(job)
    work = max(int(job["work_steps"]), 1)
    window = _first_downlink_window(env, rng, points, work, from_step)
    if window is None:
        return None

    start, end = window
    ticks = end - start
    if ticks < 1:
        return None

    # A 2–3 pass draw cannot fit in one access. Shrink the work to this pass.
    if ticks < work:
        job["work_steps"] = ticks
        job["remaining_steps"] = ticks
    job["release_step"] = int(start)
    job["deadline_step"] = int(end)
    return job


def _deadline_on_window(env, job: dict, extra_passes: int) -> int:
    """Deadline after N zone-passes worth of wait. Avoids a Kepler scan of the whole fleet."""
    dwell = zone_dwell_ticks(env)
    need = int(np.ceil(int(job["work_steps"]) / max(dwell, 1)))
    skip = max(int(extra_passes), max(need - 1, 0))
    release = int(job["release_step"])
    dt = max(float(getattr(env, "dt", 300.0)), 1.0)
    periods = getattr(getattr(env, "world", None), "periods_s", None)
    orbit = int(round(float(np.mean(periods)) / dt)) if periods is not None and len(periods) else 23
    wait = (skip + 1) * max(orbit, dwell)
    return release + int(job["work_steps"]) + wait


class ZoneWork:
    """80% a third of the zone pass, 15% the whole pass, 5% two or three passes."""

    def sample(self, rng: np.random.Generator, env=None) -> int:
        dwell = zone_dwell_ticks(env)
        draw = float(rng.random())
        if draw < 0.80:
            centre = max(1, int(round(dwell / 3.0)))
            lo, hi = max(1, centre - 1), centre + 1
            return int(rng.integers(lo, hi + 1))

        if draw < 0.95:
            return max(1, int(round(dwell)))
        return max(1, int(round(dwell * float(rng.choice([2, 3])))))


class ZoneSlack:
    """Spare ticks after work. `passes` is extra zone-passes the job may wait."""

    def __init__(self, passes: float = 1.0) -> None:
        self.passes = float(passes)

    def sample(self, rng: np.random.Generator, env=None) -> int:
        dwell = zone_dwell_ticks(env)
        extra = max(0, int(round(self.passes * dwell)))
        if extra <= 0:
            return int(rng.integers(0, 2))
        lo = max(0, extra - 1)
        hi = extra + 1
        return int(rng.integers(lo, hi + 1))


_WORK = ZoneWork()
_SPARE_NORMAL = ZoneSlack(1.0)
_SPARE_SOURCE = ZoneSlack(1.0)
_SPARE_SHORT = ZoneSlack(0.0)

# Jobs/tick rose ~25x versus the Europe trickle; keep episode USD in the same band.
PRICE_DIV = 25.0


def _band(floor: float, per_priority: float) -> tuple[float, float]:
    return (round(floor / PRICE_DIV, 4), round(per_priority / PRICE_DIV, 4))


class _Usd:
    """Puts `value_usd` on every example job. Not itself a task.

    `price` is `(floor, per_priority)`. A downlink band sits above a relay of
    the same priority. An urgent job lifts that band.
    """

    price = _band(15.0, 40.0)
    urgent = False
    slack = _SPARE_NORMAL
    work = _WORK

    def sample(self, rng: np.random.Generator, env, job_id: str, release: int | None = None):
        job = super().sample(rng, env, job_id, release)
        if job is None:
            return job

        # Initial batch is known from tick 0. Deadline is the end of the N-th zone window.
        if release is None:
            job["release_step"] = 0

        if job["kind"] == "downlink":
            return _bind_downlink_window(env, rng, job, from_step=int(job["release_step"]))

        extra = int(round(getattr(self.slack, "passes", 0.0)))
        end = _deadline_on_window(env, job, extra)
        job["deadline_step"] = min(int(env.steps_total), max(int(job["release_step"]) + int(job["work_steps"]), end))
        return job

    def sample_extras(self, rng: np.random.Generator, draft: dict) -> dict:
        # Same-priority downlink is worth more than relay: a relay can be skipped.
        priority = int(draft["priority"])
        floor, per_priority = self.price
        high = per_priority * priority

        # Urgent starts where a normal job of this kind stops.
        if self.urgent:
            floor = high
            high *= 1.5
        return {"value_usd": round(float(rng.uniform(floor, high)), 2)}


class _Near(SeePoints):
    """A and B from the fleet difficulty. Further points stay inside the first footprint."""

    def place(self, rng: np.random.Generator, env) -> dict:
        if self.n <= 2:
            return super().place(rng, env)

        # The pair uses the run difficulty. Later points have to stay near A.
        placed = super().place(rng, env)
        mean_r = env.world.coverage_radius_m()
        std_r = env.world.coverage_radius_std_m()
        pts = env.active_points
        points = list(placed["points"])
        first = points[0]
        guard = 0
        while len(points) < self.n and guard < 40:
            guard += 1
            _, extra = pair_ground_points(rng, pts, mean_r, env.job_difficulty, std_r, env.ground_catalog)
            if haversine_m(first[0], first[1], extra[0], extra[1]) <= mean_r:
                points.append(extra)

        if len(points) < self.n:
            points.append(points[-1])
        return {
            "points": points[: self.n],
            "point_a": points[0],
            "point_b": points[1],
            "addressed": None,
        }


class RelayPair(_Usd, Relay):
    """Relay that needs both A and B in view."""

    name = "relay_pair"
    probability = 0.16
    condition = _Near(2)
    work = _WORK
    slack = _SPARE_NORMAL


class RelayAddressed(_Usd, Relay):
    """Relay with no ground points, assigned to one satellite by fleet index."""

    name = "relay_addressed"
    probability = 0.05
    slack = _SPARE_SHORT
    condition = Addressed()
    work = _WORK


class UrgentRelay(_Usd, Relay):
    """Priority-3 relay. Shorter spare time and a higher price."""

    name = "relay_urgent"
    probability = 0.04
    urgent = True
    slack = _SPARE_SHORT
    priority_p = (0.0, 0.0, 1.0)
    condition = _Near(2)
    work = _WORK


class DownlinkPair(_Usd, Downlink):
    """Downlink that needs both A and B in view."""

    name = "downlink_pair"
    probability = 0.42
    price = _band(50.0, 100.0)
    condition = _Near(2)
    work = _WORK
    slack = _SPARE_NORMAL


class DownlinkSource(_Usd, Downlink):
    """Downlink that only needs the source point A."""

    name = "downlink_source"
    probability = 0.14
    price = _band(50.0, 100.0)
    slack = _SPARE_SOURCE
    condition = SeePoints(1)
    work = _WORK


class DownlinkChain(_Usd, Downlink):
    """Downlink that needs A, B, and C, all inside one footprint."""

    name = "downlink_chain"
    probability = 0.07
    price = _band(50.0, 100.0)
    condition = _Near(3)
    work = _WORK
    slack = _SPARE_NORMAL


class UrgentDownlink(_Usd, Downlink):
    """Priority-3 downlink. Shorter spare time and a higher price than a relay."""

    name = "downlink_urgent"
    probability = 0.12
    urgent = True
    price = _band(50.0, 100.0)
    slack = _SPARE_SHORT
    priority_p = (0.0, 0.0, 1.0)
    condition = _Near(2)
    work = _WORK


class _ShiftJobs(JobCatalog):
    """One batch of about 5 satellites of open work. Events refill the jammed pool."""

    def _apply_downlink_slots(self, env) -> None:
        slots = downlink_slots(env.n_sats)
        env.downlink_limit = slots
        env.model["downlink_parallel_limit"] = slots

    def sample_initial(self, rng, env, n: int) -> dict[str, dict]:
        self._apply_downlink_slots(env)
        core = 5.0 * env.n_sats
        target = max(int(n), int(round(core)))
        cap = int(getattr(getattr(env, "events_cfg", None), "pool_max", 0) or 0)
        if cap > 0:
            target = min(target, cap)

        # A downlink with no one-pass window is dropped. Retry so the pool stays full.
        jobs: dict[str, dict] = {}
        i = 0
        attempts = 0
        limit = max(target * 12, target)
        while len(jobs) < target and attempts < limit:
            attempts += 1
            job_id = f"J{i:03d}"
            i += 1
            job = self.pick(rng).sample(rng, env, job_id, release=None)
            if job is not None:
                jobs[job_id] = job
        return jobs

    def sample_added(self, rng, env, k: int, count: int, ident) -> list[dict]:
        self._apply_downlink_slots(env)
        jobs = []
        i = 0
        attempts = 0
        limit = max(int(count) * 12, int(count))
        while len(jobs) < count and attempts < limit:
            attempts += 1
            job = self.pick(rng).sample(rng, env, f"{ident}-J{i}", release=k)
            i += 1
            if job is not None:
                jobs.append(job)
        return jobs


def catalog() -> JobCatalog:
    found = JobCatalog.collect(sys.modules[__name__])
    return _ShiftJobs(found.types)


def _revenue_usd(_env, finished: list[dict]) -> float:
    return round(sum(float(job.get("value_usd", 0.0)) for job in finished), 6)


register_job_field("revenue_usd", "ops/revenue_usd", _revenue_usd)
