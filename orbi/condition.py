"""Tick snapshots of satellites and jobs. Geometry comes from KeplerWorld."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbi.coverage import footprint_radius
from orbi.kepler import KeplerWorld, kepler_from_config
from orbi.satellite import Satellite

STATUS_IDLE = 0
STATUS_SLEEP = 1
STATUS_BUSY = 2


@dataclass
class TaskCondition:
    kind: str
    point_a: tuple[float, float]
    point_b: tuple[float, float]
    priority: int = 1
    value_usd: float = 0.0
    remaining_steps: int = 1
    deadline_step: int = 1
    work_steps: int = 1
    power_need: float = 0.0
    name: str = ""


@dataclass
class SatCondition:
    sat_i: int
    lat: float
    lon: float
    alt: float
    battery: float
    battery_wh: float
    temp_c: float
    calibration_age_steps: int
    in_sun: bool
    time_to_sun_change: float
    can_work: bool
    in_view_now: bool = False
    access_remaining: float = 0.0
    footprint_radius: float = 0.0
    status: int = STATUS_IDLE


@dataclass
class Heartbeat:
    sat_i: int
    battery_wh: float
    lat_deg: float
    lon_deg: float
    alt_km: float
    in_sun: bool
    shadow_factor: float


def as_task(task) -> TaskCondition | None:
    if task is None:
        return None

    if isinstance(task, TaskCondition):
        return task

    if isinstance(task, dict):
        a = task.get("point_a") or (0.0, 0.0)
        b = task.get("point_b") or (0.0, 0.0)
        return TaskCondition(
            kind=str(task.get("kind", "relay")),
            point_a=(float(a[0]), float(a[1])),
            point_b=(float(b[0]), float(b[1])),
            priority=int(task.get("priority", 1)),
            value_usd=float(task.get("value_usd", 0.0)),
            remaining_steps=int(task.get("remaining_steps", task.get("work_steps", 1))),
            deadline_step=int(task.get("deadline_step", 1)),
            work_steps=int(task.get("work_steps", 1)),
            power_need=float(task.get("power_need", 0.0)),
            name=str(task.get("id", "")),
        )
    return None


def sat_condition(world: KeplerWorld, sat: Satellite | int, task=None) -> SatCondition:
    """Snapshot of one satellite relative to an optional job."""
    sat_i = sat.index if isinstance(sat, Satellite) else int(sat)
    task = as_task(task)
    lat, lon, alt = world.lla(sat_i)
    in_sun = bool(world.in_sun(sat_i))

    # Visibility is only defined when a job was passed in.
    in_view = False
    access = 0.0
    can = False
    if task is not None:
        in_view = bool(world.sees_job(sat_i, task.point_a, task.point_b))
        access = min(
            float(world.access_remaining(sat_i, task.point_a[0], task.point_a[1], world.t)),
            float(world.access_remaining(sat_i, task.point_b[0], task.point_b[1], world.t)),
        )
        can = in_view

    # An index alone has geometry, not a battery.
    energy = sat.energy_wh if isinstance(sat, Satellite) else 0.0
    cap = sat.capacity_wh if isinstance(sat, Satellite) else 1.0
    temp = sat.temp_c if isinstance(sat, Satellite) else 0.0
    age = sat.calibration_age_steps if isinstance(sat, Satellite) else 0
    status = STATUS_IDLE if in_sun else STATUS_SLEEP
    return SatCondition(
        sat_i=sat_i,
        lat=lat,
        lon=lon,
        alt=alt,
        battery=energy / max(cap, 1e-9),
        battery_wh=energy,
        temp_c=temp,
        calibration_age_steps=age,
        in_sun=in_sun,
        time_to_sun_change=float(world.time_to_sun_change(sat_i)),
        can_work=can,
        in_view_now=in_view,
        access_remaining=access,
        footprint_radius=footprint_radius(alt, world.min_elev),
        status=status,
    )


def world_heartbeat(world: KeplerWorld, fleet: list[Satellite] | None = None) -> list[Heartbeat]:
    rows = []
    for i in range(world.n_sats):
        lat, lon, alt = world.lla(i)
        in_sun = bool(world.in_sun(i))
        energy = fleet[i].energy_wh if fleet is not None else 0.0
        rows.append(
            Heartbeat(
                sat_i=i,
                battery_wh=energy,
                lat_deg=float(np.degrees(lat)),
                lon_deg=float(np.degrees(lon)),
                alt_km=float(alt / 1000.0),
                in_sun=in_sun,
                shadow_factor=1.0 if in_sun else 0.0,
            )
        )
    return rows


def open_world(cfg: dict):
    world = kepler_from_config(cfg)
    world.reset(seed=0)
    return world, "kepler"
