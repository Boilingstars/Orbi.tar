"""Один прогон смены в компактный NDJSON. Повтор того же JSON берётся из кэша.

Модель грузится при первом кадре после шага 0 и только в этом процессе.
Обучение и каталог прогона не затрагиваются.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections import OrderedDict, defaultdict
from typing import Callable, Iterator

import numpy as np

from orbi import digest

RESULT_LIMIT = 120 * 1024 * 1024

Actor = Callable[[dict], object]

_CACHE_MAX = 4
_cache: OrderedDict[str, "Run"] = OrderedDict()
_cache_lock = threading.Lock()
_infer_lock = threading.Lock()
_models: dict[str, object] = {}
_model_lock = threading.Lock()
_ALT_KEY = "|alt"
_PRIMARY_MODEL = "/models/ckpt_68120.zip"
_ALT_MODEL = "/models/ckpt_35714.zip"
WHATIF_HORIZON = 10
_WHATIF_MAX = 16
_whatif_cache: OrderedDict[str, "Run"] = OrderedDict()
_whatif_lock = threading.Lock()

_ACTION = {"idle": 0, "job": 1, "calibrate": 2}


class Run:
    def __init__(self, key: str) -> None:
        self.key = key
        self.events: list[dict] = []
        self.actions: list[np.ndarray] = []
        self.done = False
        self.error: str | None = None
        self.result_bytes: bytes | None = None
        self.result_size = 0
        self.stop_after: int | None = None
        self.cond = threading.Condition()


def clear() -> None:
    with _cache_lock:
        _cache.clear()
    clear_whatif()


def clear_whatif() -> None:
    with _whatif_lock:
        _whatif_cache.clear()


def actor_for(scenario: dict) -> Actor:
    """Сэмпл политики приоритетов. Argmax почти всегда idle."""
    return _actor(_primary_path())


def actor_for_alt(scenario: dict) -> Actor:
    """Сэмпл второй политики (выручка, ORBI_ALT_MODEL)."""
    return _actor(_alt_path())


def actor_for_goal(scenario: dict, goal: str = "priority") -> Actor:
    """priority → ckpt_68120, revenue → ckpt_35714."""
    if _goal(goal) == "revenue":
        return actor_for_alt(scenario)
    return actor_for(scenario)


def _actor(path: str) -> Actor:
    model = _load_model(path)

    def actor(obs: dict) -> tuple[np.ndarray, float]:
        import torch

        masks = np.asarray(obs["mask"], dtype=bool).reshape(-1)
        with _infer_lock:
            with torch.inference_mode():
                obs_t, _ = model.policy.obs_to_tensor(obs)
                mask_t = torch.as_tensor(masks, device=model.device)
                if mask_t.ndim == 1:
                    mask_t = mask_t.unsqueeze(0)
                actions, values, _ = model.policy.forward(
                    obs_t, deterministic=False, action_masks=mask_t
                )
                vec = actions.detach().cpu().numpy().reshape(-1)
                value = float(np.asarray(values.detach().cpu()).reshape(-1)[0])
        return vec, value

    return actor


def begin(scenario: dict, goal: str = "priority", *, alt: bool = False) -> Run:
    key = digest(scenario) + (_ALT_KEY if alt else "")
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached
        run = Run(key)
        _cache[key] = run
        _evict()
        threading.Thread(
            target=_worker,
            args=(run, scenario, _goal(goal), [], None, 0, alt),
            daemon=True,
        ).start()
        return run


def resume(
    scenario: dict,
    step: int,
    events: list[dict],
    goal: str = "priority",
    *,
    alt: bool = False,
) -> tuple[Run, int]:
    """Хвост с k: прошлое — сохранённые команды, сеть только с текущего шага."""
    clear_whatif()
    origin = begin(scenario, goal, alt=alt)
    with origin.cond:
        while not origin.done and len(origin.actions) < step:
            origin.cond.wait(timeout=1.0)
        if len(origin.actions) < step:
            raise RuntimeError("Прогон ещё не дошёл до этого шага")
        origin.stop_after = step
        origin.cond.notify_all()
        while not origin.done:
            origin.cond.wait(timeout=1.0)
        kinds = ("alt",) if alt else ("frame",)
        kept = [
            item
            for item in origin.events
            if item.get("type") in kinds and int(item.get("step", -1)) < step
        ]
        replay = [np.asarray(a, dtype=np.int64) for a in origin.actions[:step]]
        origin.events = kept
        origin.actions = replay
        origin.done = False
        origin.error = None
        origin.result_bytes = None
        origin.result_size = 0
        origin.stop_after = None
        start = len(origin.events)
        origin.cond.notify_all()
    threading.Thread(
        target=_worker,
        args=(origin, scenario, _goal(goal), list(events or []), replay, step, alt),
        daemon=True,
    ).start()
    return origin, start


def begin_whatif(
    scenario: dict,
    step: int,
    events: list[dict] | None,
    goal: str = "revenue",
    horizon: int = WHATIF_HORIZON,
) -> Run:
    """Развилка: префикс живого прогона, с k — целевая сеть на horizon шагов.

    Живой Run не останавливается и не обрезается.
    """
    target = _goal(goal)
    hz = int(horizon)
    if hz < 0:
        raise ValueError("horizon")
    key = (
        f"{digest(scenario)}|{int(step)}|{hz}|{target}|"
        f"{digest(list(events or []))}"
    )
    with _whatif_lock:
        cached = _whatif_cache.get(key)
        if cached is not None:
            _whatif_cache.move_to_end(key)
            return cached
    replay = _wait_live_actions(scenario, int(step))
    with _whatif_lock:
        cached = _whatif_cache.get(key)
        if cached is not None:
            _whatif_cache.move_to_end(key)
            return cached
        run = Run(key)
        _whatif_cache[key] = run
        _evict_whatif()
        threading.Thread(
            target=_whatif_worker,
            args=(run, scenario, target, list(events or []), replay, int(step), hz),
            daemon=True,
        ).start()
        return run


def _wait_live_actions(scenario: dict, step: int) -> list[np.ndarray]:
    origin = begin(scenario)
    with origin.cond:
        while not origin.done and len(origin.actions) < step:
            origin.cond.wait(timeout=1.0)
        if origin.error:
            raise RuntimeError(origin.error)

        if len(origin.actions) < step:
            raise RuntimeError("Прогон ещё не дошёл до этого шага")
        return [np.asarray(a, dtype=np.int64) for a in origin.actions[:step]]


def _evict_whatif() -> None:
    while len(_whatif_cache) > _WHATIF_MAX:
        key, run = next(iter(_whatif_cache.items()))
        if not run.done:
            return
        _whatif_cache.pop(key)


def iter_ndjson(run: Run, from_index: int = 0) -> Iterator[bytes]:
    pos = max(0, int(from_index))
    while True:
        with run.cond:
            while pos >= len(run.events) and not run.done:
                run.cond.wait()
            batch = run.events[pos:]
            pos = len(run.events)
            finished = run.done and pos >= len(run.events)
            error = run.error
        for event in batch:
            yield _line(event)
        if finished:
            if error:
                yield _line({"type": "error", "detail": error})
            return


def iter_events(
    scenario: dict,
    actor: Actor,
    *,
    goal: str = "priority",
    scripted_events: list[dict] | None = None,
    replay_actions: list[np.ndarray] | None = None,
    from_step: int = 0,
    run: Run | None = None,
    compact: bool = False,
    compact_type: str = "alt",
    until_step: int | None = None,
    horizon: int | None = None,
    stay_step: int | None = None,
) -> Iterator[dict]:
    """Кадр k — состояние при env.k == k. Действие на k > 0 — команда тика k - 1."""
    from orbi.env import EventConfig, OpsEnv

    steps = int(scenario["time"]["steps"])
    replay = [np.asarray(a, dtype=np.int64) for a in (replay_actions or [])]
    start_k = max(0, int(from_step))
    cat = _Catalog(scenario)
    env = OpsEnv(
        scenario=scenario,
        events=EventConfig.off(),
        scripted_events=list(scripted_events or []),
        top_k=32,
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    phys = env
    cat.sync(phys)
    tally = _Tally()
    reserve = float(scenario["model"]["reserve_soc_pct"])
    kind = compact_type if compact_type in ("alt", "whatif") else "alt"
    stay = int(stay_step) if stay_step is not None else start_k
    hz = int(horizon) if horizon is not None else 0

    if not compact:
        yield _meta(scenario, cat)
    elif kind == "whatif":
        yield {
            "type": "whatif_meta",
            "steps": steps,
            "horizon": hz,
            "from_step": start_k,
            "stay_step": stay,
        }
    else:
        yield {"type": "alt_meta", "steps": steps}

    def take(obs_now: dict) -> tuple[np.ndarray, float]:
        out = actor(obs_now)
        if isinstance(out, tuple):
            return np.asarray(out[0], dtype=np.int64), float(out[1])
        return np.asarray(out, dtype=np.int64), 0.0

    def record(action: np.ndarray) -> None:
        if run is None:
            return
        with run.cond:
            run.actions.append(np.asarray(action, dtype=np.int64).copy())
            run.cond.notify_all()

    def stopped() -> bool:
        if run is None:
            return False
        with run.cond:
            return run.stop_after is not None and len(run.actions) >= run.stop_after

    action: np.ndarray | None = None
    value = 0.0

    def push_meta() -> Iterator[dict]:
        if compact:
            cat.added = False
            return
            yield

        if cat.added:
            cat.added = False
            yield _meta(scenario, cat)

    def emit(step: int, rows: list[dict] | None, value_now: float) -> dict:
        frame = _frame(step, phys, cat, rows, tally, value_now, reserve)
        if compact:
            payload = {
                "type": kind,
                "step": int(frame["step"]),
                "metrics": frame["metrics"],
            }
            if kind == "whatif":
                payload["horizon"] = hz
                payload["stay_step"] = stay
            return payload
        return frame

    def past_horizon() -> bool:
        return until_step is not None and int(phys.k) >= int(until_step)

    if start_k == 0:
        action, value = take(obs)
        record(action)
        yield emit(0, None, value)
    else:
        emit(0, None, 0.0)
        last_rows: list[dict] | None = None
        terminated = truncated = False
        for tick in range(start_k):
            start = len(phys.trace)
            obs, _, terminated, truncated, _ = env.step(replay[tick])
            last_rows = phys.trace[start:]
            cat.sync(phys)
            if tick < start_k - 1:
                emit(int(phys.k), last_rows, 0.0)
                if terminated or truncated:
                    yield {"type": "_journal", "record": env.journal()}
                    return
        yield from push_meta()
        action, value = take(obs)
        record(action)
        yield emit(int(phys.k), last_rows, value)
        if terminated or truncated:
            yield {"type": "_journal", "record": env.journal()}
            return

    while phys.k < steps:
        if stopped() or past_horizon():
            break
        assert action is not None
        start = len(phys.trace)
        obs, _, terminated, truncated, _ = env.step(action)
        rows = phys.trace[start:]
        cat.sync(phys)
        yield from push_meta()
        if terminated or truncated:
            yield emit(int(phys.k), rows, 0.0)
            break
        action, value = take(obs)
        record(action)
        yield emit(int(phys.k), rows, value)

    yield {"type": "_journal", "record": env.journal()}


def packed_result(run: Run) -> tuple[bytes | None, str]:
    with run.cond:
        if not run.done:
            return None, "not_ready"

        if run.error:
            return None, "error"

        if run.result_bytes is None:
            return None, "too_large" if run.result_size > RESULT_LIMIT else "missing"
        return run.result_bytes, "ok"


def _worker(
    run: Run,
    scenario: dict,
    goal: str,
    scripted: list[dict],
    replay: list[np.ndarray] | None,
    from_step: int,
    alt: bool = False,
) -> None:
    _quiet_torch()
    holder: dict[str, Actor] = {}

    def actor(obs: dict):
        if "fn" not in holder:
            import torch

            torch.manual_seed(0)
            holder["fn"] = actor_for_goal(scenario, goal)
        return holder["fn"](obs)

    try:
        for event in iter_events(
            scenario,
            actor,
            goal=goal,
            scripted_events=scripted,
            replay_actions=replay,
            from_step=from_step,
            run=run,
            compact=alt,
        ):
            if event.get("type") == "_journal":
                raw = json.dumps(
                    event["record"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
                with run.cond:
                    run.result_size = len(raw)
                    run.result_bytes = raw if len(raw) <= RESULT_LIMIT else None
                continue
            with run.cond:
                run.events.append(event)
                run.cond.notify_all()
    except Exception as exc:
        with run.cond:
            run.error = str(exc)
    finally:
        with run.cond:
            run.done = True
            run.cond.notify_all()


def _whatif_worker(
    run: Run,
    scenario: dict,
    goal: str,
    scripted: list[dict],
    replay: list[np.ndarray],
    from_step: int,
    horizon: int,
) -> None:
    _quiet_torch()
    holder: dict[str, Actor] = {}
    steps = int(scenario["time"]["steps"])
    stay = min(int(from_step) + int(horizon), steps)

    def actor(obs: dict):
        if "fn" not in holder:
            import torch

            torch.manual_seed(0)
            holder["fn"] = actor_for_goal(scenario, goal)
        return holder["fn"](obs)

    try:
        for event in iter_events(
            scenario,
            actor,
            goal=goal,
            scripted_events=scripted,
            replay_actions=replay,
            from_step=from_step,
            run=run,
            compact=True,
            compact_type="whatif",
            until_step=stay,
            horizon=horizon,
            stay_step=stay,
        ):
            if event.get("type") == "_journal":
                continue
            with run.cond:
                run.events.append(event)
                run.cond.notify_all()
    except Exception as exc:
        with run.cond:
            run.error = str(exc)
    finally:
        with run.cond:
            run.done = True
            run.cond.notify_all()


def _goal(raw: str | None) -> str:
    if raw in ("revenue", "money"):
        return "revenue"
    return "priority"


def _limit_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _quiet_torch() -> None:
    """Один поток BLAS в процессе API, до первого импорта torch."""
    _limit_threads()
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _primary_path() -> str:
    return os.environ.get("ORBI_MODEL", _PRIMARY_MODEL)


def _alt_path() -> str:
    return os.environ.get("ORBI_ALT_MODEL", _ALT_MODEL)


def _load_model(path: str | None = None):
    target = path or _primary_path()
    with _model_lock:
        cached = _models.get(target)
        if cached is None:
            import torch

            _limit_threads()
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            from example.models import OpsAttentionPolicy  # noqa: F401
            from tools.evaluating import load_ppo

            cached = load_ppo(target)
            cached.policy.set_training_mode(False)
            _models[target] = cached
        return cached


def _evict() -> None:
    while len(_cache) > _CACHE_MAX:
        key, run = next(iter(_cache.items()))
        if not run.done:
            return
        _cache.pop(key)


class _Catalog:
    def __init__(self, scenario: dict) -> None:
        self.sat_ids = [sat["id"] for sat in scenario["satellites"]]
        self.sat_index = {sid: i for i, sid in enumerate(self.sat_ids)}
        self.capacity = {sat["id"]: float(sat["capacity_wh"]) for sat in scenario["satellites"]}
        self.jobs_seed = int(scenario.get("jobs", {}).get("initial", 0))
        self.potential_seed = 0.0
        self.jobs: list[dict] = []
        self.job_index: dict[str, int] = {}
        self.by_deadline: dict[int, list[int]] = defaultdict(list)
        self.priority: list[int] = []
        self.value: list[float] = []
        self.prev_rem: list[int] = []
        self.prev_exec: list[int] = []
        self.prev_done: list[int] = []
        self.added = False

    def sync(self, phys) -> bool:
        grew = False
        for job_id, live in phys.jobs.items():
            if job_id in self.job_index:
                continue
            spec = {
                "id": job_id,
                "kind": live["kind"],
                "release_step": int(live["release_step"]),
                "deadline_step": int(live["deadline_step"]),
                "work_steps": int(live["work_steps"]),
                "eligible_satellites": list(live.get("eligible_satellites") or self.sat_ids),
                "priority": int(live["priority"]),
                "value_usd": float(live["value_usd"]),
            }
            index = len(self.jobs)
            self.jobs.append(spec)
            self.job_index[job_id] = index
            self.by_deadline[spec["deadline_step"]].append(index)
            self.priority.append(spec["priority"])
            self.value.append(spec["value_usd"])
            self.prev_rem.append(int(live["remaining_steps"]))
            self.prev_exec.append(-1)
            self.prev_done.append(0 if live["completed_step"] is None else 1)
            grew = True
        self.added = grew
        return grew


def _meta(scenario: dict, cat: _Catalog) -> dict:
    potential = round(sum(float(job["value_usd"]) for job in cat.jobs), 6)
    return {
        "type": "meta",
        "steps": int(scenario["time"]["steps"]),
        "ids": cat.sat_ids,
        "jobs_total": len(cat.jobs),
        "jobs_seed": cat.jobs_seed,
        "potential_revenue_usd": potential,
        "potential_seed_usd": cat.potential_seed,
        "reserve_soc_pct": float(scenario["model"]["reserve_soc_pct"]),
        "jobs": [
            {
                "id": job["id"],
                "kind": job["kind"],
                "release_step": int(job["release_step"]),
                "deadline_step": int(job["deadline_step"]),
                "work_steps": int(job["work_steps"]),
                "eligible": list(job["eligible_satellites"]),
                "priority": int(job["priority"]),
                "value_usd": float(job["value_usd"]),
            }
            for job in cat.jobs
        ],
    }


class _Tally:
    __slots__ = ("revenue", "completed", "missed", "critical_done", "critical_due", "below", "brownout", "blocked")

    def __init__(self) -> None:
        self.revenue = 0.0
        self.completed = 0
        self.missed = 0
        self.critical_done = 0
        self.critical_due = 0
        self.below = 0
        self.brownout = 0
        self.blocked = 0

    def as_dict(self) -> dict:
        return {
            "revenue_usd": round(self.revenue, 6),
            "jobs_completed": self.completed,
            "jobs_due_missed": self.missed,
            "critical_jobs_completed_on_time": self.critical_done,
            "critical_jobs_due": self.critical_due,
            "below_reserve_satellite_steps": self.below,
            "brownout_satellite_steps": self.brownout,
            "blocked_command_count": self.blocked,
        }


def _quality(
    value: float,
    phys,
    cat: _Catalog,
    executed: dict[str, tuple[str, str | None]],
    missed: list[int],
    reserve: float,
) -> float:
    remaining = 0.0
    p3 = 0
    p3_ok = 0
    step = int(phys.k)
    for spec in cat.jobs:
        live = phys.jobs[spec["id"]]
        if live["completed_step"] is None:
            remaining += float(spec["value_usd"])

        if int(spec["priority"]) != 3:
            continue
        p3 += 1
        if live["completed_step"] is not None or int(spec["deadline_step"]) > step:
            p3_ok += 1
    v_norm = 1.0 / (1.0 + math.exp(-float(value) / max(1.0, remaining)))
    n = max(1, len(cat.sat_ids))
    above = 0
    busy = 0
    for sid in cat.sat_ids:
        cap = max(cat.capacity[sid], 1e-6)
        soc = 100.0 * float(phys.state[sid]["energy_wh"]) / cap
        if soc >= reserve:
            above += 1
        kind, _ = executed.get(sid, ("idle", None))
        if kind != "idle":
            busy += 1
    heuristic = (
        0.35 * (above / n)
        + 0.35 * (p3_ok / p3 if p3 else 1.0)
        + 0.2 * (busy / n)
        + 0.1 * (0.0 if missed else 1.0)
    )
    return round(0.5 * v_norm + 0.5 * heuristic, 4)


def _frame(
    step: int,
    phys,
    cat: _Catalog,
    rows: list[dict] | None,
    tally: _Tally,
    value: float,
    reserve: float,
) -> dict:
    executed: dict[str, tuple[str, str | None]] = {}
    exec_now = [-1] * len(cat.jobs)
    if rows:
        for row in rows:
            sid = row["satellite_id"]
            kind = row["executed"]
            job_id = row["requested"].get("job_id") if kind == "job" else None
            executed[sid] = (kind, job_id)
            if kind == "job" and job_id in cat.job_index:
                exec_now[cat.job_index[job_id]] = cat.sat_index[sid]
            if row["below_reserve"]:
                tally.below += 1
            if row["brownout"]:
                tally.brownout += 1
            if row["reason"] not in ("accepted", "idle"):
                tally.blocked += 1
            done_id = row["completed_job"]
            if done_id:
                tally.completed += 1
                tally.revenue += cat.value[cat.job_index[done_id]]

    missed: list[int] = []
    for index in cat.by_deadline.get(step, ()):
        job = phys.jobs[cat.jobs[index]["id"]]
        if cat.priority[index] == 3:
            tally.critical_due += 1
            if job["completed_step"] is not None:
                tally.critical_done += 1

        if job["completed_step"] is None:
            missed.append(index)
            tally.missed += 1

    sat_rows = []
    for sid in cat.sat_ids:
        state = phys.state[sid]
        kind, job_id = executed.get(sid, ("idle", None))
        sat_rows.append([
            round(float(state["energy_wh"]), 4),
            round(float(state["temp_c"]), 3),
            int(state["calibration_age_steps"]),
            _ACTION.get(kind, 0),
            cat.job_index[job_id] if job_id in cat.job_index else -1,
        ])

    deltas = []
    for i, spec in enumerate(cat.jobs):
        live = phys.jobs[spec["id"]]
        rem = int(live["remaining_steps"])
        done = 0 if live["completed_step"] is None else 1
        ex = exec_now[i] if i < len(exec_now) else -1
        if rem != cat.prev_rem[i] or done != cat.prev_done[i] or ex != cat.prev_exec[i]:
            deltas.append([i, rem, ex, done])
            cat.prev_rem[i] = rem
            cat.prev_done[i] = done
            cat.prev_exec[i] = ex

    open_jobs, take = _access(phys, cat)
    return {
        "type": "frame",
        "step": step,
        "sat": sat_rows,
        "job": deltas,
        "missed": missed,
        "open": open_jobs,
        "take": take,
        "metrics": tally.as_dict(),
        "q": _quality(value, phys, cat, executed, missed, reserve),
    }


def _access(phys, cat: _Catalog) -> tuple[list[int], list[int]]:
    """Кто может взять задание на текущем k: `Environment.can_execute`, не маска top-K."""
    k = int(phys.k)
    take = [0] * len(cat.sat_ids)
    open_jobs: list[int] = []
    for i, spec in enumerate(cat.jobs):
        live = phys.jobs.get(spec["id"])
        if live is None or live["completed_step"] is not None:
            continue

        if not (int(spec["release_step"]) <= k < int(spec["deadline_step"])):
            continue
        found = False
        for sid in spec["eligible_satellites"]:
            if cat.sat_index.get(sid) is None:
                continue
            ok, _, _ = phys.can_execute(sid, {"action": "job", "job_id": spec["id"]})
            if ok:
                take[cat.sat_index[sid]] = 1
                found = True
        if found:
            open_jobs.append(i)
    return open_jobs, take


def _line(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode()
