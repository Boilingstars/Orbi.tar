"""Reward contract. Every trainer reads one float per fleet tick from `score`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Reward:
    """Shared step terms plus `on_complete` / `on_miss` for the user's objective.

    Subclasses keep the same methods. Extra bonuses (priority, a custom job
    field, an opportunity term) belong on the subclass, not here.
    """

    name: str = "reward"
    progress: float = 0.15
    below_reserve: float = 0.02
    brownout: float = 0.5
    blocked: float = 0.02
    executor_switch: float = 0.05
    conflict: float = 0.15
    early_calibrate: float = 0.15
    calibrate_free_at: float = 0.85

    def on_complete(self, job: dict) -> float:
        return 0.0

    def on_miss(self, job: dict) -> float:
        return 0.0

    def score(self, env, rows: list[dict], prev_executor: dict[str, str], stats: dict[str, float]) -> float:
        # Progress for a job that actually ran, and the shared step penalties.
        reward = 0.0
        switches = 0
        executed: dict[str, str] = {}
        for row in rows:
            if row["executed"] == "job":
                job_id = row["requested"]["job_id"]
                executed[job_id] = row["satellite_id"]
                reward += self.progress
                if prev_executor.get(job_id) not in (None, row["satellite_id"]):
                    switches += 1

            if row["below_reserve"]:
                reward -= self.below_reserve

            if row["brownout"]:
                reward -= self.brownout

            if row["reason"] not in ("accepted", "idle"):
                reward -= self.blocked
        reward -= self.executor_switch * switches
        reward -= self.conflict * stats.get("conflicts", 0.0)
        reward -= self.early_calibrate * stats.get("early_calibrations", 0.0)
        env.ops_stats["executor_switches"] += switches
        env._last_executor = executed

        # Completions this tick, then jobs whose deadline is this tick and that never finished.
        for row in rows:
            job_id = row["completed_job"]
            if job_id is None:
                continue
            reward += self.on_complete(env.jobs[job_id])
        for job in env.jobs.values():
            if job["deadline_step"] == env.k and job["completed_step"] is None:
                reward += self.on_miss(job)
        return reward
