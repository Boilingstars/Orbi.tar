"""Tests for coverage geometry and KeplerWorld."""

from __future__ import annotations

import numpy as np

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


def test_kepler_time_advances_on_step():
    world = KeplerWorld.walker(n_sats=4, n_planes=2)
    world.reset(seed=2)
    world.step(600.0)
    assert world.t == 600.0


def test_ground_catalog_is_dense():
    pts = load_ground_points()
    assert len(pts) > 2000


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
    def __init__(self, i: int) -> None:
        self.i = i

    def integers(self, n):
        return self.i

    def random(self):
        return 0.0

    def choice(self, pool):
        return pool[0]


def test_pair_ground_points_near_and_far():
    pts = np.array(
        [
            [0.0, 0.0],
            [0.0, np.deg2rad(5.0)],
            [0.0, np.deg2rad(80.0)],
        ]
    )
    cover = haversine_m(0.0, 0.0, 0.0, np.deg2rad(20.0))
    near = pair_ground_points(_Pick(0), pts, cover, far_p=0.0)
    far = pair_ground_points(_Pick(0), pts, cover, far_p=1.0)
    d_near = haversine_m(near[0][0], near[0][1], near[1][0], near[1][1])
    d_far = haversine_m(far[0][0], far[0][1], far[1][0], far[1][1])
    assert d_near <= cover + 1.0
    assert d_far > cover


def test_access_remaining_helper_zero_if_not_in_view():
    els = walker_delta_elements(4, 2, 500e3, np.deg2rad(51.6))

    def r_fn(t, el=els[0]):
        return rv_at_time(*el, t)[0]

    acc = access_remaining(r_fn, np.deg2rad(80.0), 0.0, 0.0, np.deg2rad(10.0))
    assert acc == 0.0
