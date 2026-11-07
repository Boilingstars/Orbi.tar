"""One loaded shift in memory. Globe and dispatch share KeplerWorld from OpsEnv."""

from __future__ import annotations

import json
import math

import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from orbi.constants import OMEGA_EARTH, R_EARTH
from orbi.coverage import eci_to_ecef
from orbi.env import load
from orbi.kepler import KeplerWorld

from console.api.dispatch import (
    WHATIF_HORIZON,
    begin,
    begin_whatif,
    iter_ndjson,
    packed_result,
    resume as resume_run,
)

ORBIT_SAMPLES = 64
STEP_S = 300

app = FastAPI(
    title="Orbi.tar operator API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
_shift: dict | None = None
_OPENAPI = Path(__file__).resolve().parents[1] / "openapi.yaml"


def _world_for(scenario: dict) -> KeplerWorld:
    world = KeplerWorld.from_config(scenario)
    world.reset(seed=0)
    return world


def _lla(world: KeplerWorld, sat_i: int, t: float) -> dict:
    lat, lon, alt = world.lla(sat_i, t)
    return {
        "lat": round(math.degrees(lat), 5),
        "lon": round(math.degrees(lon), 5),
        "alt_m": round(float(alt), 1),
    }


def _batch_lla(world: KeplerWorld, times: np.ndarray) -> np.ndarray:
    """Lat/lon degrees and altitude for every satellite at every time. Shape (T, N, 3)."""
    el = np.asarray(world.elements, dtype=float)
    a, e, inc, raan, argp, m0 = (el[:, i] for i in range(6))
    mean_n = np.asarray(world.mean_motions, dtype=float)
    nu = np.mod(m0 + mean_n * times[:, None], 2.0 * np.pi)
    c_o, s_o = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    rot = np.stack(
        [
            np.stack([c_o * cw - s_o * sw * ci, -c_o * sw - s_o * cw * ci, s_o * si], axis=-1),
            np.stack([s_o * cw + c_o * sw * ci, -s_o * sw + c_o * cw * ci, -c_o * si], axis=-1),
            np.stack([sw * si, cw * si, ci], axis=-1),
        ],
        axis=-2,
    )
    p = np.where(e < 1e-12, a, a * (1.0 - e**2))
    radius = p / (1.0 + e * np.cos(nu))
    pos_pqw = np.stack([radius * np.cos(nu), radius * np.sin(nu), np.zeros_like(nu)], axis=-1)
    eci = np.einsum("nij,tnj->tni", rot, pos_pqw)
    theta = world.gmst0 + OMEGA_EARTH * times
    c_t, s_t = np.cos(theta), np.sin(theta)
    x, y, z = eci[..., 0], eci[..., 1], eci[..., 2]
    ex = c_t[:, None] * x + s_t[:, None] * y
    ey = -s_t[:, None] * x + c_t[:, None] * y
    lat = np.degrees(np.arctan2(z, np.hypot(ex, ey)))
    lon = np.degrees(np.arctan2(ey, ex))
    alt = np.sqrt(ex * ex + ey * ey + z * z) - R_EARTH
    return np.stack([lat, lon, alt], axis=-1)


def _sun_rows(world: KeplerWorld, times: np.ndarray) -> np.ndarray:
    sun = np.asarray(world.sun_eci, dtype=float)
    theta = world.gmst0 + OMEGA_EARTH * times
    c_t, s_t = np.cos(theta), np.sin(theta)
    ex = c_t * sun[0] + s_t * sun[1]
    ey = -s_t * sun[0] + c_t * sun[1]
    ez = np.full(times.shape, sun[2])
    vec = np.stack([ex, ey, ez], axis=-1)
    return vec / np.linalg.norm(vec, axis=-1)[:, None]


def _ndjson(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode()


@app.post("/api/scenario")
def load_scenario(scenario: dict) -> dict:
    global _shift
    try:
        scenario = load(scenario)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    satellites = [sat["id"] for sat in scenario["satellites"]]
    _shift = {
        "scenario": scenario,
        "ids": satellites,
        "world": _world_for(scenario),
    }
    sid = scenario.get("id") or scenario.get("meta", {}).get("id", "")
    return {
        "scenario_id": sid,
        "satellites": satellites,
        "jobs": [],
    }


@app.get("/api/orbits")
def orbits(step: int = 0) -> dict:
    if _shift is None:
        raise HTTPException(status_code=409, detail="Сценарий ещё не загружен")
    steps = int(_shift["scenario"]["time"]["steps"])
    if step < 0 or step > steps:
        raise HTTPException(status_code=400, detail="Шаг вне смены")
    world: KeplerWorld = _shift["world"]
    t = float(step * STEP_S)
    sun = eci_to_ecef(world.sun_eci, t, world.gmst0)
    sun = sun / np.linalg.norm(sun)
    period = float(world.period)
    satellites = []
    for i, sat_id in enumerate(_shift["ids"]):
        polyline = [
            _lla(world, i, t + period * k / ORBIT_SAMPLES)
            for k in range(ORBIT_SAMPLES)
        ]
        satellites.append({"id": sat_id, **_lla(world, i, t), "orbit": polyline})
    return {
        "step": step,
        "t_s": step * STEP_S,
        "sun_ecef": [round(float(v), 5) for v in sun],
        "satellites": satellites,
    }


@app.get("/api/tracks")
def tracks():
    """Stream the whole shift: positions first, orbit lines once."""
    if _shift is None:
        raise HTTPException(status_code=409, detail="Сценарий ещё не загружен")
    world: KeplerWorld = _shift["world"]
    ids = list(_shift["ids"])
    steps = int(_shift["scenario"]["time"]["steps"])
    times = np.arange(steps + 1, dtype=float) * STEP_S
    table = _batch_lla(world, times)
    suns = _sun_rows(world, times)
    period = float(world.period)
    line_times = period * np.arange(ORBIT_SAMPLES, dtype=float) / ORBIT_SAMPLES
    lines = _batch_lla(world, line_times)

    def gen():
        yield _ndjson({"type": "meta", "ids": ids, "steps": steps})
        yield _ndjson(_frame(0, table[0], suns[0]))
        yield _ndjson({
            "type": "orbits",
            "path": [
                [[round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 1)] for p in lines[:, i, :]]
                for i in range(len(ids))
            ],
        })
        for step in range(1, steps + 1):
            yield _ndjson(_frame(step, table[step], suns[step]))

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _goal(raw: str | None) -> str:
    if raw in ("revenue", "money"):
        return "revenue"
    return "priority"


@app.get("/api/dispatch")
def dispatch(goal: str = "priority"):
    """Stream the shift the loaded policy just flew. Same JSON reuses the cache."""
    if _shift is None:
        raise HTTPException(status_code=409, detail="Сценарий ещё не загружен")
    run = begin(_shift["scenario"], _goal(goal))
    return StreamingResponse(
        iter_ndjson(run),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/dispatch/resume")
def dispatch_resume(body: dict):
    """Переиграть хвост с шага k: прошлые команды как были, сеть только с k."""
    if _shift is None:
        raise HTTPException(status_code=409, detail="Сценарий ещё не загружен")
    steps = int(_shift["scenario"]["time"]["steps"])
    try:
        step = int(body.get("step", -1))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Шаг вне смены") from exc
    if step < 0 or step > steps:
        raise HTTPException(status_code=400, detail="Шаг вне смены")
    events = body.get("events") or []
    if not isinstance(events, list):
        raise HTTPException(status_code=400, detail="events должен быть списком")
    live_goal = _goal(body.get("goal"))
    try:
        run, start = resume_run(
            _shift["scenario"], step, events, live_goal
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    alt_goal = "priority" if live_goal == "revenue" else "revenue"
    try:
        resume_run(_shift["scenario"], step, events, alt_goal, alt=True)
    except RuntimeError:
        pass
    return StreamingResponse(
        iter_ndjson(run, from_index=start),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/dispatch/whatif")
def dispatch_whatif(body: dict):
    """Развилка с шага k: префикс живого прогона, целевая сеть на horizon шагов."""
    if _shift is None:
        raise HTTPException(status_code=409, detail="Сценарий ещё не загружен")
    steps = int(_shift["scenario"]["time"]["steps"])
    try:
        step = int(body.get("step", -1))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Шаг вне смены") from exc
    if step < 0 or step > steps:
        raise HTTPException(status_code=400, detail="Шаг вне смены")
    events = body.get("events") or []
    if not isinstance(events, list):
        raise HTTPException(status_code=400, detail="events должен быть списком")
    try:
        horizon = int(body.get("horizon", WHATIF_HORIZON))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Неверный horizon") from exc
    if horizon < 0:
        raise HTTPException(status_code=400, detail="Неверный horizon")
    try:
        run = begin_whatif(
            _shift["scenario"],
            step,
            events,
            _goal(body.get("goal")),
            horizon,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StreamingResponse(
        iter_ndjson(run),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/dispatch/alt")
def dispatch_alt():
    """Поток второй политики (ORBI_ALT_MODEL). Только метрики кадра."""
    if _shift is None:
        raise HTTPException(status_code=409, detail="Сценарий ещё не загружен")
    run = begin(_shift["scenario"], "revenue", alt=True)
    return StreamingResponse(
        iter_ndjson(run),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/result")
def result():
    if _shift is None:
        raise HTTPException(status_code=409, detail="Сценарий ещё не загружен")
    run = begin(_shift["scenario"])
    payload, status = packed_result(run)
    if status == "not_ready":
        raise HTTPException(status_code=409, detail="Прогон ещё не готов")

    if status == "too_large":
        raise HTTPException(status_code=413, detail="Результат больше 120 Мб")

    if status != "ok" or payload is None:
        raise HTTPException(status_code=500, detail=run.error or "Нет результата прогона")
    name = f"{_shift['scenario']['meta']['id']}.result.json"
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


def _frame(step: int, row: np.ndarray, sun: np.ndarray) -> dict:
    return {
        "type": "frame",
        "step": step,
        "sun": [round(float(v), 5) for v in sun],
        "pos": [[round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 1)] for p in row],
    }


_WEB = Path(__file__).resolve().parents[1]


@app.get("/api/docs", include_in_schema=False)
def swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title="Orbi.tar operator API",
    )


@app.get("/api/redoc", include_in_schema=False)
def redoc_ui():
    return get_redoc_html(
        openapi_url="/api/openapi.json",
        title="Orbi.tar operator API",
    )


@app.get("/api/openapi.json", include_in_schema=False)
def openapi_json():
    import yaml

    return JSONResponse(yaml.safe_load(_OPENAPI.read_text(encoding="utf-8")))


@app.get("/api/openapi.yaml", include_in_schema=False)
def openapi_yaml():
    return FileResponse(_OPENAPI, media_type="application/yaml")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB / "index.html")


app.mount("/static", StaticFiles(directory=_WEB / "static"), name="static")
app.mount("/src", StaticFiles(directory=_WEB / "src"), name="src")
