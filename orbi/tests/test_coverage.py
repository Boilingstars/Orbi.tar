"""Tests for coverage geometry and KeplerWorld."""

from __future__ import annotations

import numpy as np
import pytest

from orbi.constants import R_EARTH
from orbi.coverage import (
    access_remaining,
    ecef_to_lla,
    elevation,
    footprint_radius,
    haversine_m,
    in_sunlight,
    lla_to_ecef,
    load_ground_points,
    pair_ground_points,
    rv_at_time,
    walker_delta_elements,
)
from orbi.kepler import KeplerWorld


def test_nadir_elevation_is_90_deg():
    lat, lon = np.deg2rad(10.0), np.deg2rad(20.0)
    tgt = lla_to_ecef(lat, lon, 0.0)
    sat = lla_to_ecef(lat, lon, 500e3)
    assert elevation(sat, tgt) > np.deg2rad(89.0)


def test_horizon_target_has_low_elevation():
    sat = lla_to_ecef(0.0, 0.0, 500e3)
    far = lla_to_ecef(0.0, np.deg2rad(80.0), 0.0)
    assert elevation(sat, far) < 0.0


def test_footprint_grows_with_altitude():
    r_low = footprint_radius(400e3, np.deg2rad(10.0))
    r_high = footprint_radius(800e3, np.deg2rad(10.0))
    assert r_high > r_low
    assert r_low > 5e5


def test_lla_roundtrip():
    r = lla_to_ecef(np.deg2rad(45.0), np.deg2rad(-30.0), 500e3)
    lat, lon, alt = ecef_to_lla(r)
    assert abs(lat - np.deg2rad(45.0)) < 1e-6
    assert abs(lon - np.deg2rad(-30.0)) < 1e-6
    assert abs(alt - 500e3) < 1.0


def test_cylindrical_eclipse():
    sun = np.array([1.0, 0.0, 0.0])
    lit = np.array([R_EARTH + 500e3, 0.0, 0.0])
    dark = np.array([-(R_EARTH + 500e3), 0.0, 0.0])
    assert in_sunlight(lit, sun)
    assert not in_sunlight(dark, sun)


def test_walker_has_n_satellites():
    els = walker_delta_elements(4, 2, 500e3, np.deg2rad(51.6))
    assert len(els) == 4
    rs = [np.linalg.norm(rv_at_time(*el, 0.0)[0]) for el in els]
    assert all(abs(r - (R_EARTH + 500e3)) < 1.0 for r in rs)


def test_overhead_access_remaining_positive():
    world = KeplerWorld.walker(n_sats=4, n_planes=2)
    world.reset(seed=0)
    lat, lon, _ = world.lla(0)
    acc = world.access_remaining(0, lat, lon, world.t)
    assert acc > 60.0
    far_lat, far_lon = lat + np.deg2rad(80.0), lon
    assert world.access_remaining(0, far_lat, far_lon, world.t) == 0.0


def test_access_end_matches_remaining_when_in_view():
    world = KeplerWorld.walker(n_sats=4, n_planes=2, altitude_m=1500e3)
    world.reset(seed=0)
    lat, lon, _ = world.lla(0)
    remaining, nxt = world.access_windows(0, ((lat, lon),), world.t)
    end = world.access_end(0, ((lat, lon),), world.t, skip_passes=0)
    assert nxt == 0.0
    assert end == pytest.approx(world.t + remaining)
    far = (lat + np.deg2rad(80.0), lon)
    assert world.access_end(0, (far,), world.t, skip_passes=0) >= 0.0


def test_access_next_zero_when_in_view():
    world = KeplerWorld.walker(n_sats=4, n_planes=2, altitude_m=1500e3)
    world.reset(seed=0)
    lat, lon, _ = world.lla(0)
    remaining, nxt = world.access_windows(0, ((lat, lon),), world.t)
    assert remaining > 60.0
    assert nxt == 0.0
    far = (lat + np.deg2rad(80.0), lon)
    remaining, nxt = world.access_windows(0, (far,), world.t)
    assert remaining == 0.0
    assert nxt >= 0.0


def test_kepler_time_advances_on_step():
    world = KeplerWorld.walker(n_sats=4, n_planes=2)
    world.reset(seed=2)
    world.step(600.0)
    assert world.t == 600.0


def test_ground_catalog_is_dense():
    pts = load_ground_points()
    assert len(pts) > 2000


def test_europe_points_grow_from_the_center():
    from orbi.coverage import load_ground_region

    pts = load_ground_region("europe")
    assert len(pts) > 64
    dist = haversine_m(pts[0, 0], pts[0, 1], pts[:, 0], pts[:, 1])
    assert dist[0] == 0.0
    assert np.all(np.diff(dist) >= -1.0)
    assert dist[63] < dist[-1]


def test_haversine_equator_quarter():
    dist = haversine_m(0.0, 0.0, 0.0, np.pi / 2)
    assert abs(dist - 0.5 * np.pi * R_EARTH) < 1e3


def test_sees_job_requires_both_points():
    world = KeplerWorld.walker(n_sats=4, n_planes=2)
    world.reset(seed=0)
    lat, lon, _ = world.lla(0)
    assert world.sees_job(0, (lat, lon), (lat, lon))
    far = (lat + np.deg2rad(80.0), lon)
    assert not world.sees_job(0, (lat, lon), far)
    assert not world.sees_job(0, None, (lat, lon))
    radius = world.coverage_radius_m()
    assert radius > 5e5
    assert abs(world.typical_altitude_m() - 500e3) < 1.0


class _Pick:
    def __init__(self, i: int, roll: float = 0.0, draw: float | None = None) -> None:
        self.i = i
        self.roll = roll
        self.draw = draw

    def integers(self, n):
        return self.i

    def random(self):
        return self.roll

    def normal(self, loc, scale):
        return loc if self.draw is None else self.draw

    def choice(self, pool):
        return pool[0]


def test_pair_ground_points_easy_and_hard():
    pts = np.array(
        [
            [0.0, 0.0],
            [0.0, np.deg2rad(5.0)],
            [0.0, np.deg2rad(80.0)],
        ]
    )
    cover = haversine_m(0.0, 0.0, 0.0, np.deg2rad(20.0))
    ring = haversine_m(0.0, 0.0, 0.0, np.deg2rad(80.0))
    easy = pair_ground_points(_Pick(0, roll=1.0), pts, cover, difficulty=0.0)
    hard = pair_ground_points(_Pick(0, roll=0.0, draw=ring), pts, cover, difficulty=1.0, radius_std_m=1.0)
    d_easy = haversine_m(easy[0][0], easy[0][1], easy[1][0], easy[1][1])
    d_hard = haversine_m(hard[0][0], hard[0][1], hard[1][0], hard[1][1])
    assert d_easy <= cover + 1.0
    assert abs(d_hard - ring) < 1.0


def test_hard_pair_can_leave_the_local_group():
    local = np.array([[0.0, 0.0], [0.0, np.deg2rad(2.0)]])
    far_lon = np.deg2rad(40.0)
    wide = np.vstack([local, [[0.0, far_lon]]])
    ring = haversine_m(0.0, 0.0, 0.0, far_lon)
    _origin, other = pair_ground_points(
        _Pick(0, roll=0.0, draw=ring),
        local,
        ring,
        1.0,
        1.0,
        wide=wide,
    )
    assert abs(other[1] - far_lon) < 1e-9


def test_access_remaining_helper_zero_if_not_in_view():
    els = walker_delta_elements(4, 2, 500e3, np.deg2rad(51.6))

    def r_fn(t, el=els[0]):
        return rv_at_time(*el, t)[0]

    acc = access_remaining(r_fn, np.deg2rad(80.0), 0.0, 0.0, np.deg2rad(10.0))
    assert acc == 0.0
