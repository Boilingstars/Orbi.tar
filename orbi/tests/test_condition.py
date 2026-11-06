"""Shared tick snapshots used by the gym and the console."""

from __future__ import annotations

from orbi.condition import TaskCondition, sat_condition, world_heartbeat
from orbi.kepler import KeplerWorld
from orbi.satellite import Satellite


def _sat(world: KeplerWorld, i: int = 0) -> Satellite:
    row = world.orbits[i]
    return Satellite(
        sat_id=f"S{i}",
        index=i,
        raan_deg=row["raan_deg"],
        period_min=row["period_min"],
        inclination_deg=row["inclination_deg"],
        altitude_km=row["altitude_km"],
        mean_anomaly_deg=row.get("mean_anomaly_deg", 0.0),
    )


def test_nadir_snapshot_has_geometry():
    world = KeplerWorld.walker(n_sats=4, n_planes=2)
    world.reset(seed=0)
    lat, lon, _ = world.lla(0)
    task = TaskCondition(kind="relay", point_a=(lat, lon), point_b=(lat, lon))
    cond = sat_condition(world, _sat(world), task)
    assert cond.footprint_radius > 0.0
    assert cond.can_work
    assert cond.in_view_now


def test_heartbeat_length_matches_fleet():
    world = KeplerWorld.walker(n_sats=4, n_planes=2)
    world.reset(seed=1)
    rows = world_heartbeat(world)
    assert len(rows) == 4
