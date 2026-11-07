"""Framework contracts for a shift simulator.

A user package such as `example` supplies the pieces these types describe:
an agent, a reward, a trainer, job subclasses, and any extra log fields.
Plots and baseline charts stay in that package.
"""

from tools.agent import Agent
from tools.jobs import (
    Addressed,
    Downlink,
    JobCatalog,
    JobType,
    Relay,
    SeePoints,
    UniformRange,
    WeightedLaw,
    register_job_field,
)
from tools.observation import Observation
from tools.reward import Reward
from tools.training import Trainer

__all__ = [
    "Addressed",
    "Agent",
    "Downlink",
    "JobCatalog",
    "JobType",
    "Observation",
    "Relay",
    "Reward",
    "SeePoints",
    "Trainer",
    "UniformRange",
    "WeightedLaw",
    "register_job_field",
]
