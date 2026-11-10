"""Ground control for this package. Ranks open jobs into the fixed action slots."""

from __future__ import annotations

from example.reward import job_worth
from orbi.env import _Candidate
from tools.observation import job_view


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

    def _view(self, env, job: dict) -> dict:
        cached = getattr(env, "_job_views", None) or {}
        view = cached.get(job["id"])
        if view is not None:
            return view
        return job_view(env, job, self.sat_ids)

    def has_contact(self, env, job: dict) -> bool:
        return self._view(env, job)["n_visible"] > 0

    def job_rank(self, env, job: dict) -> tuple:
        k = env.k
        open_now = job["release_step"] <= k < job["deadline_step"]
        score = job_worth(job)
        view = self._view(env, job)
        n_visible = int(view["n_visible"])
        remaining_s = float(view["remaining_s"])

        # Visible work first. Closing windows, then uniqueness; dollars are a tie-break.
        if open_now and n_visible > 0:
            return (0, remaining_s, n_visible, -score, job["deadline_step"], job["id"])

        if job["release_step"] > k or open_now:
            return (1, -score, job["deadline_step"], job["id"])

        return (2, job["deadline_step"], -score, job["id"])

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
