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
        self._eci_cache: dict = {}
        self._ecef_cache: dict = {}
        self._ground_ecef: dict = {}
        self._sin_min_sq = float(np.sin(self.min_elev) ** 2)
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
    def walker(cls, n_sats: int = 4, n_planes: int = 2, altitude_m: float = 500e3, inclination: float = np.deg2rad(51.6), min_elev: float = np.deg2rad(10.0), raan_offset: float = 0.0) -> KeplerWorld:
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

        # One Greenwich angle and one sun direction for the whole episode.
        self.gmst0 = float(self._rng.uniform(0.0, 2.0 * np.pi))
        lon = float(self._rng.uniform(0.0, 2.0 * np.pi))
        lat = float(np.arcsin(self._rng.uniform(-0.4, 0.4)))
        self.sun_eci = np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        self._access_cache = {}
        self._sun_cache = {}
        self._eci_cache = {}
        self._ecef_cache = {}

    def _stored(self, cache: dict, key, limit: int, compute):
        cached = cache.get(key)
        if cached is not None:
            return cached
        value = compute()
        cache[key] = value
        if len(cache) > limit:
            cache.clear()
        return value

    def _above(self, r_sat: np.ndarray, lat: float, lon: float) -> bool:
        """Same boolean as `elevation >= min_elev`, without a trig call on the hot path."""
        if self.min_elev <= 0.0:
            return geo.elevation(r_sat, geo.lla_to_ecef(lat, lon)) >= self.min_elev

        # One ECEF point per ground site. The fleet asks the same sites many times a tick.
        key = (round(float(lat), 5), round(float(lon), 5))
        tgt = self._ground_ecef.get(key)
        if tgt is None:
            tgt = geo.lla_to_ecef(lat, lon)
            self._ground_ecef[key] = tgt
        dx = float(r_sat[0]) - float(tgt[0])
        dy = float(r_sat[1]) - float(tgt[1])
        dz = float(r_sat[2]) - float(tgt[2])
        rho2 = dx * dx + dy * dy + dz * dz
        if rho2 < 1.0:
            return False
        up2 = float(tgt[0]) ** 2 + float(tgt[1]) ** 2 + float(tgt[2]) ** 2
        if up2 < 1.0:
            return False
        dot = dx * float(tgt[0]) + dy * float(tgt[1]) + dz * float(tgt[2])

        # min elevation is above the horizon, so a downward line of sight fails.
        if dot <= 0.0:
            return False
        return dot * dot >= self._sin_min_sq * rho2 * up2

    def _time_key(self, t: float | None) -> float:
        t_use = self.t if t is None else float(t)
        return round(t_use, 1)

    def r_eci(self, sat_i: int, t: float | None = None) -> np.ndarray:
        t_use = self.t if t is None else float(t)
        key = (int(sat_i), self._time_key(t))
        hit = self._eci_cache.get(key)
        if hit is not None:
            return hit
        a, e, inc, raan, argp, m0 = self.elements[sat_i]
        r, _ = geo.rv_at_time(
            a, e, inc, raan, argp, m0, t_use, 0.0, mean_motion=float(self.mean_motions[sat_i])
        )
        self._eci_cache[key] = r
        return r

    def r_ecef(self, sat_i: int, t: float | None = None) -> np.ndarray:
        t_use = self.t if t is None else float(t)
        key = (int(sat_i), self._time_key(t))
        hit = self._ecef_cache.get(key)
        if hit is not None:
            return hit
        r = geo.eci_to_ecef(self.r_eci(sat_i, t_use), t_use, self.gmst0)
        self._ecef_cache[key] = r
        return r

    def lla(self, sat_i: int, t: float | None = None) -> tuple[float, float, float]:
        return geo.ecef_to_lla(self.r_ecef(sat_i, t))

    def in_sun(self, sat_i: int, t: float | None = None) -> bool:
        t_use = self.t if t is None else t
        return geo.in_sunlight(self.r_eci(sat_i, t_use), self.sun_eci)

    def typical_altitude_m(self) -> float:
        return float(np.mean([float(row["altitude_km"]) * 1e3 for row in self.orbits]))

    def coverage_radii_m(self) -> np.ndarray:
        """One footprint radius per satellite, from that satellite's altitude."""
        return np.array(
            [geo.footprint_radius(float(row["altitude_km"]) * 1e3, self.min_elev) for row in self.orbits],
            dtype=float,
        )

    def coverage_radius_m(self) -> float:
        """Mean footprint radius over the fleet, metres."""
        radii = self.coverage_radii_m()
        if len(radii) == 0:
            return geo.footprint_radius(self.typical_altitude_m(), self.min_elev)
        return float(np.mean(radii))

    def coverage_radius_std_m(self) -> float:
        """Spread of the per-satellite footprint radii, metres."""
        radii = self.coverage_radii_m()
        if len(radii) < 2:
            return 0.0

        # Identical altitudes leave a few ulps of noise. That is not a real spread.
        spread = float(np.std(radii))
        return 0.0 if spread < 1.0 else spread

    def elevation(self, sat_i: int, lat: float, lon: float, t: float | None = None) -> float:
        t_use = self.t if t is None else t
        return geo.elevation(self.r_ecef(sat_i, t_use), geo.lla_to_ecef(lat, lon))

    def sees_point(self, sat_i: int, lat: float, lon: float, t: float | None = None) -> bool:
        return self.elevation(sat_i, lat, lon, t) >= self.min_elev

    def sees_any(self, sat_i: int, points, t: float | None = None) -> bool:
        r_sat = self.r_ecef(sat_i, t)
        return any(self._above(r_sat, float(row[0]), float(row[1])) for row in points)

    def sees_job(self, sat_i: int, point_a, point_b, t: float | None = None) -> bool:
        """True only if this satellite sees both A and B now. No cross-sat relay."""
        a, b = _ll(point_a), _ll(point_b)
        if a is None or b is None:
            return False
        r_sat = self.r_ecef(sat_i, t)
        return self._above(r_sat, a[0], a[1]) and self._above(r_sat, b[0], b[1])

    def access_remaining(self, sat_i: int, lat: float, lon: float, t_query: float) -> float:
        remaining, _nxt = self.access_windows(sat_i, ((lat, lon),), t_query)
        return remaining

    def access_next(self, sat_i: int, lat: float, lon: float, t_query: float) -> float:
        _remaining, nxt = self.access_windows(sat_i, ((lat, lon),), t_query)
        return nxt

    def access_windows(self, sat_i: int, points, t_query: float | None = None, step_s: float = 300.0) -> tuple[float, float]:
        """Seconds this satellite keeps / waits for joint view of `points`.

        Sampled at the env tick so a shift step does not walk the orbit at 20 s.
        """
        t_use = self.t if t_query is None else float(t_query)
        pts = tuple((round(float(lat), 5), round(float(lon), 5)) for lat, lon in points)
        step = max(float(step_s), 1.0)
        key = (sat_i, pts, round(t_use, 1), round(step, 1))

        def compute() -> tuple[float, float]:
            period = float(self.periods_s[sat_i])
            horizon = max(int(round(period / step)), 1)

            def visible(t: float) -> bool:
                if not pts:
                    return False
                r_sat = self.r_ecef(sat_i, t)
                return all(self._above(r_sat, lat, lon) for lat, lon in pts)

            if visible(t_use):
                remaining = 0.5 * step
                for k in range(1, horizon + 1):
                    if not visible(t_use + k * step):
                        break
                    remaining = k * step
                return remaining, 0.0

            for k in range(1, horizon + 1):
                if visible(t_use + k * step):
                    return 0.0, k * step
            return 0.0, period

        return self._stored(self._access_cache, key, 4096, compute)

    def access_end(self, sat_i: int, points, t_query: float | None = None, skip_passes: int = 0, step_s: float = 300.0) -> float:
        """Absolute time when the (skip_passes+1)-th joint access ends. 0 if none."""
        t_use = self.t if t_query is None else float(t_query)
        skip = max(int(skip_passes), 0)
        step = max(float(step_s), 1.0)
        period = float(self.periods_s[sat_i])
        t = t_use
        seen = 0
        t_limit = t_use + (skip + 2) * period
        while t <= t_limit + 1e-9:
            remaining, nxt = self.access_windows(sat_i, points, t, step_s=step)
            if remaining > 0:
                end = t + remaining
                if seen >= skip:
                    return end
                seen += 1
                t = end + step
                continue

            if nxt <= 0 or nxt >= period - 1e-9:
                return 0.0
            t = t + nxt
        return 0.0

    def time_to_sun_change(self, sat_i: int) -> float:
        key = (sat_i, round(self.t, 1))

        def compute() -> float:
            return geo.time_to_sun_change(
                lambda tau, i=sat_i: self.r_eci(i, tau),
                self.sun_eci,
                self.t,
                self.in_sun(sat_i),
                max_horizon=self.periods_s[sat_i],
                dt=20.0,
            )

        return self._stored(self._sun_cache, key, 512, compute)

    def step(self, dt: float, payload_w: np.ndarray | None = None) -> None:
        if len(self.elements) != self.n_sats:
            raise ValueError("The constellation is fixed; orbits are not recomputed")

        if dt <= 0.0:
            return
            
        self.t += float(dt)
        self._access_cache.clear()
        self._sun_cache.clear()
        self._eci_cache.clear()
        self._ecef_cache.clear()


def _ll(point) -> tuple[float, float] | None:
    if point is None:
        return None
    try:
        return float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return None


def kepler_from_config(cfg: dict) -> KeplerWorld:
    return KeplerWorld.from_config(cfg)
