"""Analytic constellation: two-body Kepler, cylindrical eclipse, live visibility.

Orbital elements come from the initial JSON and stay constant. `step` advances
time only; energy and temperature live on the satellite template.
"""

from __future__ import annotations

import numpy as np

import orbi.coverage as geo
from orbi.constants import R_EARTH, orbital_period


class KeplerWorld:
    def __init__(self, orbits: list[dict], min_elev: float = np.deg2rad(10.0)) -> None:
        if not orbits:
            raise ValueError("KeplerWorld needs at least one satellite orbit")
        self.orbits = [dict(row) for row in orbits]
        self.n_sats = len(self.orbits)
        self.min_elev = float(min_elev)
        self.t = 0.0
        self.gmst0 = 0.0
        self.sun_eci = np.array([1.0, 0.0, 0.0])
        self._rng = np.random.default_rng()
        self._access_cache: dict = {}
        self._sun_cache: dict = {}
        self._rebuild_elements()

    def _rebuild_elements(self) -> None:
        self.elements = []
        self.mean_motions = np.zeros(self.n_sats, dtype=float)
        self.periods_s = np.zeros(self.n_sats, dtype=float)
        for i, row in enumerate(self.orbits):
            a = R_EARTH + float(row["altitude_km"]) * 1e3
            period_s = float(row["period_min"]) * 60.0
            self.periods_s[i] = period_s
            self.mean_motions[i] = 2.0 * np.pi / max(period_s, 1e-6)
            self.elements.append(
                (
                    a,
                    0.0,
                    np.deg2rad(float(row["inclination_deg"])),
                    np.deg2rad(float(row["raan_deg"])),
                    0.0,
                    np.deg2rad(float(row.get("mean_anomaly_deg", 0.0))),
                )
            )

    @property
    def period(self) -> float:
        return float(self.periods_s[0])

    @classmethod
    def walker(
        cls,
        n_sats: int = 4,
        n_planes: int = 2,
        altitude_m: float = 500e3,
        inclination: float = np.deg2rad(51.6),
        min_elev: float = np.deg2rad(10.0),
        raan_offset: float = 0.0,
    ) -> KeplerWorld:
        els = geo.walker_delta_elements(n_sats, n_planes, altitude_m, inclination, raan_offset=raan_offset)
        period_min = orbital_period(R_EARTH + altitude_m) / 60.0
        orbits = [
            {
                "raan_deg": float(np.rad2deg(raan)),
                "period_min": period_min,
                "inclination_deg": float(np.rad2deg(inc)),
                "altitude_km": altitude_m / 1e3,
                "mean_anomaly_deg": float(np.rad2deg(m0)),
            }
            for (_a, _e, inc, raan, _argp, m0) in els
        ]
        return cls(orbits, min_elev=min_elev)

    @classmethod
    def from_config(cls, cfg: dict) -> KeplerWorld:
        min_elev = np.deg2rad(float(cfg.get("model", cfg).get("min_elevation_deg", 10.0)))
        orbits = []
        for spec in cfg["satellites"]:
            orbit = spec.get("orbit", spec)
            orbits.append(
                {
                    "raan_deg": float(orbit["raan_deg"]),
                    "period_min": float(orbit["period_min"]),
                    "inclination_deg": float(orbit["inclination_deg"]),
                    "altitude_km": float(orbit["altitude_km"]),
                    "mean_anomaly_deg": float(orbit.get("mean_anomaly_deg", 0.0)),
                }
            )
        return cls(orbits, min_elev=min_elev)

    def reset(self, seed: int | None = None) -> None:
        """Redraw sun and GMST. Orbital elements stay as loaded."""
        self._rng = np.random.default_rng(seed)
        self.t = 0.0
        self.gmst0 = float(self._rng.uniform(0.0, 2.0 * np.pi))
        lon = float(self._rng.uniform(0.0, 2.0 * np.pi))
        lat = float(np.arcsin(self._rng.uniform(-0.4, 0.4)))
        self.sun_eci = np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        self._access_cache = {}
        self._sun_cache = {}

    def r_eci(self, sat_i: int, t: float | None = None) -> np.ndarray:
        t_use = self.t if t is None else t
        a, e, inc, raan, argp, m0 = self.elements[sat_i]
        r, _ = geo.rv_at_time(
            a, e, inc, raan, argp, m0, t_use, 0.0, mean_motion=float(self.mean_motions[sat_i])
        )
        return r

    def r_ecef(self, sat_i: int, t: float | None = None) -> np.ndarray:
        t_use = self.t if t is None else t
        return geo.eci_to_ecef(self.r_eci(sat_i, t_use), t_use, self.gmst0)

    def lla(self, sat_i: int, t: float | None = None) -> tuple[float, float, float]:
        return geo.ecef_to_lla(self.r_ecef(sat_i, t))

    def in_sun(self, sat_i: int, t: float | None = None) -> bool:
        t_use = self.t if t is None else t
        return geo.in_sunlight(self.r_eci(sat_i, t_use), self.sun_eci)

    def typical_altitude_m(self) -> float:
        return float(np.mean([float(row["altitude_km"]) * 1e3 for row in self.orbits]))

    def coverage_radius_m(self) -> float:
        """Typical one-satellite footprint radius on the surface, metres."""
        return geo.footprint_radius(self.typical_altitude_m(), self.min_elev)

    def elevation(self, sat_i: int, lat: float, lon: float, t: float | None = None) -> float:
        t_use = self.t if t is None else t
        return geo.elevation(self.r_ecef(sat_i, t_use), geo.lla_to_ecef(lat, lon))

    def sees_point(self, sat_i: int, lat: float, lon: float, t: float | None = None) -> bool:
        return self.elevation(sat_i, lat, lon, t) >= self.min_elev

    def sees_any(self, sat_i: int, points, t: float | None = None) -> bool:
        r_sat = self.r_ecef(sat_i, t)
        for row in points:
            if geo.elevation(r_sat, geo.lla_to_ecef(float(row[0]), float(row[1]))) >= self.min_elev:
                return True
        return False

    def sees_job(self, sat_i: int, point_a, point_b, t: float | None = None) -> bool:
        """True only if this satellite sees both A and B now. No cross-sat relay."""
        a, b = _ll(point_a), _ll(point_b)
        if a is None or b is None:
            return False
        r_sat = self.r_ecef(sat_i, t)
        return (
            geo.elevation(r_sat, geo.lla_to_ecef(a[0], a[1])) >= self.min_elev
            and geo.elevation(r_sat, geo.lla_to_ecef(b[0], b[1])) >= self.min_elev
        )

    def access_remaining(self, sat_i: int, lat: float, lon: float, t_query: float) -> float:
        key = (sat_i, round(lat, 5), round(lon, 5), round(float(t_query), 1))
        cached = self._access_cache.get(key)
        if cached is not None:
            return cached
        value = geo.access_remaining(
            lambda tau, i=sat_i: self.r_eci(i, tau),
            lat,
            lon,
            t_query,
            self.min_elev,
            gmst0=self.gmst0,
            max_horizon=0.25 * self.periods_s[sat_i],
            dt=8.0,
        )
        self._access_cache[key] = value
        if len(self._access_cache) > 4096:
            self._access_cache.clear()
        return value

    def time_to_sun_change(self, sat_i: int) -> float:
        key = (sat_i, round(self.t, 1))
        cached = self._sun_cache.get(key)
        if cached is not None:
            return cached
        lit = self.in_sun(sat_i)
        value = geo.time_to_sun_change(
            lambda tau, i=sat_i: self.r_eci(i, tau),
            self.sun_eci,
            self.t,
            lit,
            max_horizon=self.periods_s[sat_i],
            dt=20.0,
        )
        self._sun_cache[key] = value
        if len(self._sun_cache) > 512:
            self._sun_cache.clear()
        return value

    def step(self, dt: float, payload_w: np.ndarray | None = None) -> None:
        if len(self.elements) != self.n_sats:
            raise ValueError("The constellation is fixed; orbits are not recomputed")
        if dt <= 0.0:
            return
        self.t += float(dt)
        self._access_cache.clear()
        self._sun_cache.clear()


def _ll(point) -> tuple[float, float] | None:
    if point is None:
        return None
    try:
        return float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return None


def kepler_from_config(cfg: dict) -> KeplerWorld:
    return KeplerWorld.from_config(cfg)
