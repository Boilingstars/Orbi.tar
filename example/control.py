"""Ground control for this package. Ranks open jobs into the fixed action slots."""

from __future__ import annotations

from orbi.env import _Candidate
from orbi.satellite import job_visible


class GroundControl:
    """
    Ranks unfinished jobs into the K action slots the policies share.

    `sat_ids=None` addresses the whole fleet. A subset is a cluster: only those
    spacecraft appear as eligible executors on the packed slots.
    """

    def __init__(self, top_k: int = 32, sat_ids: list[str] | None = None) -> None:
        self.top_k = int(top_k)
        self.sat_ids = list(sat_ids) if sat_ids is not None else None
        self._cluster = set(self.sat_ids) if self.sat_ids is not None else None

    def _eligible(self, env, job: dict) -> list[str]:
        ids = job.get("eligible_satellites") or env.sat_ids
        if self._cluster is not None:
            ids = [sid for sid in ids if sid in self._cluster]
        return ids

    def has_contact(self, env, job: dict) -> bool:
        for sid in self._eligible(env, job):
            sat = env.sats[sid]
            if sat.available(env.k) and job_visible(env.world, sat.index, job):
                return True
        return False

    def job_rank(self, env, job: dict) -> tuple:
        k = env.k
        open_now = job["release_step"] <= k < job["deadline_step"]

        # In contact now, then not yet visible, then already missed.
        if open_now and self.has_contact(env, job):
            return (0, -job["priority"], job["deadline_step"], job["remaining_steps"], job["id"])

        if job["release_step"] > k or open_now:
            kind_penalty = 0 if job["kind"] == "relay" else 1
            future = 0 if job["release_step"] > k else 1
            return (1, kind_penalty, future, job["release_step"], -job["priority"], job["deadline_step"], job["id"])
        return (2, job["deadline_step"], -job["priority"], job["id"])

    def select_jobs(self, env, known: list[dict]) -> list[_Candidate]:
        ranked = sorted(known, key=lambda job: self.job_rank(env, job))
        index = {sid: i for i, sid in enumerate(env.sat_ids)}
        out: list[_Candidate] = []
        for job in ranked:
            sats = [index[sid] for sid in self._eligible(env, job) if sid in index]
            if not sats:
                continue
            out.append(_Candidate(job_id=job["id"], job=job, sats=sats))
            if len(out) >= self.top_k:
                break
        return out


__all__ = ["GroundControl"]
