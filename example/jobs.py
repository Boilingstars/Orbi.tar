"""Job types for this app. Each concrete downlink and relay is drawn into an episode with its probability."""

from __future__ import annotations

import sys

import numpy as np

from tools.jobs import (
    Addressed,
    Downlink,
    JobCatalog,
    Relay,
    SeePoints,
    UniformRange,
    WeightedLaw,
    register_job_field,
)


class _Usd:
    """Puts `value_usd` on every example job. Not itself a task."""

    def sample_extras(self, rng: np.random.Generator, draft: dict) -> dict:
        priority = int(draft["priority"])
        return {"value_usd": round(float(rng.uniform(15.0, 40.0 * priority)), 2)}


class RelayPair(_Usd, Relay):
    """Relay that needs both A and B in view."""

    name = "relay_pair"
    probability = 0.55
    condition = SeePoints(2)
    work = UniformRange(1, 4)
    slack = UniformRange(2, 7)


class DownlinkPair(_Usd, Downlink):
    """Downlink that needs both A and B in view."""

    name = "downlink_pair"
    probability = 0.20
    condition = SeePoints(2)
    work = UniformRange(1, 4)
    slack = UniformRange(2, 7)


class DownlinkSource(_Usd, Downlink):
    """Downlink that only needs the source point A."""

    name = "downlink_source"
    probability = 0.10
    condition = SeePoints(1, far_p=0.0)
    work = UniformRange(1, 3)
    slack = UniformRange(1, 6)


class DownlinkChain(_Usd, Downlink):
    """Downlink that needs A, B, and C. Far points, weighted duration."""

    name = "downlink_chain"
    probability = 0.05
    condition = SeePoints(3, far_p=1.0)
    work = WeightedLaw((2, 3, 4, 5), (0.1, 0.4, 0.4, 0.1))
    slack = UniformRange(2, 8)


class RelayAddressed(_Usd, Relay):
    """Relay with no ground points, assigned to one satellite by fleet index."""

    name = "relay_addressed"
    probability = 0.10
    condition = Addressed()
    work = UniformRange(1, 3)
    slack = UniformRange(2, 5)


def catalog() -> JobCatalog:
    return JobCatalog.collect(sys.modules[__name__])


def _revenue_usd(_env, finished: list[dict]) -> float:
    return round(sum(float(job.get("value_usd", 0.0)) for job in finished), 6)


register_job_field("revenue_usd", "ops/revenue_usd", _revenue_usd)
