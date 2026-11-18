"""Optional matching diagnostics for a shift. Off on the default benchmark path.

Per-shift totals describe how revenue was earned or missed. Missed value is not
claimed as avoidable: many due jobs cannot run at the same time.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from orbi.satellite import downlink_required_steps


N_QUANTILES = 4
# Observation sat table: 5 = available this tick, 11 = capacity (presence of a spacecraft).
_SAT_AVAILABLE = 5


def required_ticks(job: dict, sat_id: str, holder: str | None) -> int:
    """Ticks this satellite still needs. Relay keeps progress; a new downlink holder restarts."""
    if job.get("kind") == "downlink":
        return max(int(downlink_required_steps(job, sat_id, holder)), 1)
    return max(int(job["remaining_steps"]), 1)


def pair_rho(job: dict, sat_id: str, holder: str | None) -> float:
    """V_j / T_ij in USD per work tick."""
    return float(job.get("value_usd", 0.0)) / required_ticks(job, sat_id, holder)


def window_ticks(remaining_s: float, dt: float) -> float:
    return max(float(remaining_s), 0.0) / max(float(dt), 1.0)


def quantile_index(value: float, edges: np.ndarray) -> int:
    if edges.size == 0:
        return 0
    return int(np.searchsorted(edges, value, side="right") - 1)


def _edges(values: list[float]) -> np.ndarray:
    if len(values) < 2:
        return np.array([0.0, 1.0])
    qs = np.linspace(0.0, 1.0, N_QUANTILES + 1)
    return np.unique(np.quantile(np.asarray(values, dtype=float), qs))


class ShiftDiagnose:
    """One shift of pair-level and assignment counters. Cheap when the caller skips it."""

    def __init__(self) -> None:
        self.ticks = 0
        self.selected_rho: list[float] = []
        self.best_feasible_rho: list[float] = []
        self.worse_job = 0
        self.worse_sat = 0
        self.downlink_switches = 0
        self.downlink_lost_ticks = 0.0
        self.feasible_not_selected = 0
        self.feasible_outside_k = 0
        self.sats_unused = 0
        self.sats_available = 0
        self.downlink_cap_unused = 0
        self.downlink_cap = 0
        self.skip_on_feasible = 0
        self.feasible_slots = 0
        self.hungarian_pair_change = 0
        self.hungarian_compared = 0
        self._completed: list[dict] = []
        self._missed: list[dict] = []

    def on_tick(self, env, obs: dict, action: np.ndarray, hungarian_action=None) -> None:
        """Record this decision against the packed slots. Does not step the env."""
        mask = np.asarray(obs["mask"], dtype=bool)
        n = int(env.n_sats)
        k = int(env.K)
        dt = max(float(env.dt), 1.0)
        holders = dict(getattr(env, "_downlink_holder", {}) or {})
        candidates = list(getattr(env, "_candidates", []) or [])[:k]
        action = np.asarray(action, dtype=np.int64)
        self.ticks += 1

        selected = {}
        for i in range(n):
            slot = int(action[i]) if i < action.shape[0] else k
            if 0 <= slot < len(candidates):
                selected[i] = slot

        available = [i for i in range(n) if float(obs["sats"][i, _SAT_AVAILABLE]) > 0.5]
        self.sats_available += len(available)
        self.sats_unused += sum(i not in selected for i in available)

        downs_used = 0
        packed_ids = {cand.job["id"] for cand in candidates}
        for i, slot in selected.items():
            job = candidates[slot].job
            sid = env.sat_ids[i]
            rho = pair_rho(job, sid, holders.get(job["id"]))
            self.selected_rho.append(rho)
            if job.get("kind") == "downlink":
                downs_used += 1
                holder = holders.get(job["id"])
                if holder is not None and holder != sid:
                    self.downlink_switches += 1
                    self.downlink_lost_ticks += max(int(job["work_steps"]) - int(job["remaining_steps"]), 0)

        cap = int(env.downlink_limit)
        self.downlink_cap += cap
        skipped_down = any(
            cand.job.get("kind") == "downlink"
            and mask[:, slot].any()
            and slot not in selected.values()
            for slot, cand in enumerate(candidates)
        )
        if downs_used < cap and skipped_down:
            self.downlink_cap_unused += cap - downs_used

        for slot, cand in enumerate(candidates):
            job = cand.job
            legal = [i for i in range(n) if mask[i, slot]]
            if not legal:
                continue
            self.feasible_slots += 1
            best_i = max(legal, key=lambda i: pair_rho(job, env.sat_ids[i], holders.get(job["id"])))
            best_rho = pair_rho(job, env.sat_ids[best_i], holders.get(job["id"]))
            self.best_feasible_rho.append(best_rho)
            taken = [i for i, s in selected.items() if s == slot]
            if not taken:
                self.skip_on_feasible += 1
                self.feasible_not_selected += 1
                continue
            i = taken[0]
            sid = env.sat_ids[i]
            got = pair_rho(job, sid, holders.get(job["id"]))
            if got + 1e-9 < best_rho:
                self.worse_sat += 1
            richer = [
                other
                for other, other_cand in enumerate(candidates)
                if other != slot
                and mask[:, other].any()
                and float(other_cand.job.get("value_usd", 0.0)) > float(job.get("value_usd", 0.0)) + 1e-9
            ]
            if richer and got + 1e-9 < max(
                pair_rho(candidates[o].job, env.sat_ids[j], holders.get(candidates[o].job["id"]))
                for o in richer
                for j in range(n)
                if mask[j, o]
            ):
                self.worse_job += 1

        exec_cache = getattr(env, "_executable_cache", None) or {}
        self.feasible_outside_k += sum(
            1 for job_id, count in exec_cache.items() if count > 0 and job_id not in packed_ids
        )

        if hungarian_action is not None:
            hung = np.asarray(hungarian_action, dtype=np.int64)
            self.hungarian_compared += 1
            rl_pairs = {(i, int(action[i])) for i in range(min(n, action.shape[0])) if 0 <= int(action[i]) < k}
            hu_pairs = {(i, int(hung[i])) for i in range(min(n, hung.shape[0])) if 0 <= int(hung[i]) < k}
            if rl_pairs != hu_pairs:
                self.hungarian_pair_change += 1

    def close(self, env) -> dict:
        """Quantiles of completed vs missed jobs. Call once after the shift."""
        k = int(env.k)
        completed, missed = [], []
        for job in env.jobs.values():
            done = job["completed_step"] is not None and int(job["completed_step"]) <= int(job["deadline_step"])
            due = int(job["deadline_step"]) <= k
            row = {
                "value": float(job.get("value_usd", 0.0)),
                "work": float(job.get("work_steps", 1.0)),
                "kind": job.get("kind"),
            }
            row["vpt"] = row["value"] / max(row["work"], 1.0)
            if done:
                completed.append(row)
            elif due:
                missed.append(row)

        values = [row["value"] for row in completed + missed]
        works = [row["work"] for row in completed + missed]
        vpts = [row["vpt"] for row in completed + missed]
        value_edges, work_edges, vpt_edges = _edges(values), _edges(works), _edges(vpts)

        def _hist(rows: list[dict], key: str, edges: np.ndarray) -> dict[str, int]:
            counts = defaultdict(int)
            for row in rows:
                counts[quantile_index(row[key], edges)] += 1
            return {f"q{i}": int(counts[i]) for i in range(N_QUANTILES)}

        n_sel = max(len(self.selected_rho), 1)
        downs = [row for row in completed if row["kind"] == "downlink"]
        productive = float(sum(row["work"] for row in downs))
        dl_rev = float(sum(row["value"] for row in downs))
        return {
            "revenue_per_productive_downlink_tick": (dl_rev / productive) if productive else 0.0,
            "completed_by_value_quantile": _hist(completed, "value", value_edges),
            "missed_by_value_quantile": _hist(missed, "value", value_edges),
            "completed_by_work_quantile": _hist(completed, "work", work_edges),
            "missed_by_work_quantile": _hist(missed, "work", work_edges),
            "completed_by_value_per_tick_quantile": _hist(completed, "vpt", vpt_edges),
            "missed_by_value_per_tick_quantile": _hist(missed, "vpt", vpt_edges),
            "missed_value_usd": float(sum(row["value"] for row in missed)),
            "missed_work_steps": float(sum(row["work"] for row in missed)),
            "downlink_switches": int(self.downlink_switches),
            "downlink_lost_ticks": float(self.downlink_lost_ticks),
            "feasible_jobs_not_selected": int(self.feasible_not_selected),
            "feasible_jobs_outside_K": int(self.feasible_outside_k),
            "skip_rate_on_feasible_jobs": float(self.skip_on_feasible) / max(self.feasible_slots, 1),
            "available_satellites_unused": float(self.sats_unused) / max(self.sats_available, 1),
            "downlink_capacity_unused_with_feasible_jobs": float(self.downlink_cap_unused) / max(self.downlink_cap, 1),
            "mean_selected_rho": float(np.mean(self.selected_rho)) if self.selected_rho else None,
            "mean_best_feasible_rho": float(np.mean(self.best_feasible_rho)) if self.best_feasible_rho else None,
            "rho_gap": (float(np.mean(self.best_feasible_rho)) - float(np.mean(self.selected_rho))) if self.selected_rho and self.best_feasible_rho else None,
            "worse_satellite_rate": float(self.worse_sat) / n_sel,
            "worse_job_rate": float(self.worse_job) / n_sel,
            "hungarian_assignment_change_rate": float(self.hungarian_pair_change) / max(self.hungarian_compared, 1),
            "diagnose_ticks": int(self.ticks),
        }


def assignment_delta(order_a, order_b, picked_a, picked_b) -> dict:
    """How a decoder changed on a frozen observation. A change is not itself an error."""
    order_a = np.asarray(order_a)
    order_b = np.asarray(order_b)
    picked_a = np.asarray(picked_a)
    picked_b = np.asarray(picked_b)
    n = min(picked_a.shape[-1], picked_b.shape[-1])
    pairs_a = {(i, int(picked_a[..., i])) for i in range(n)}
    pairs_b = {(i, int(picked_b[..., i])) for i in range(n)}
    return {
        "job_order_change_rate": float(np.mean(order_a != order_b)) if order_a.size else 0.0,
        "selected_pair_change_rate": 1.0 - (len(pairs_a & pairs_b) / max(len(pairs_a | pairs_b), 1)),
    }


def flatten_diagnose(report: dict) -> dict:
    """One-level keys for episodes.csv."""
    out = {}
    for key, value in report.items():
        if isinstance(value, dict):
            for inner, count in value.items():
                out[f"{key}_{inner}"] = count
        else:
            out[key] = value
    return out
