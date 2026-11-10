"""Ground-track geometry: LLA, elevation, footprint, access remaining, ground points."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from orbi.constants import MU_EARTH, OMEGA_EARTH, R_EARTH

GROUND_COVER_KM = 100.0
GROUND_POINTS_PATH = Path(__file__).with_name("ground_points.json")
REGIONS_DIR = Path(__file__).with_name("regions")

# Continental Europe, Britain through western Russia. Degrees.
EUROPE_BOX = {"lat": (36.0, 71.0), "lon": (-10.0, 40.0)}
REGION_BOXES = {"europe": EUROPE_BOX}


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def eci_to_ecef(r_eci: np.ndarray, t: float, gmst0: float = 0.0) -> np.ndarray:
    return rot_z(gmst0 + OMEGA_EARTH * t) @ np.asarray(r_eci, dtype=float)


def ecef_to_eci(r_ecef: np.ndarray, t: float, gmst0: float = 0.0) -> np.ndarray:
    return rot_z(gmst0 + OMEGA_EARTH * t).T @ np.asarray(r_ecef, dtype=float)


def ecef_to_lla(r_ecef: np.ndarray) -> tuple[float, float, float]:
    x, y, z = np.asarray(r_ecef, dtype=float)
    lon = np.arctan2(y, x)
    hyp = np.hypot(x, y)
    lat = np.arctan2(z, hyp)
    alt = np.linalg.norm(r_ecef) - R_EARTH
    return float(lat), float(lon), float(alt)


def lla_to_ecef(lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
    r = R_EARTH + alt
    clat, slat = np.cos(lat), np.sin(lat)
    clon, slon = np.cos(lon), np.sin(lon)
    return np.array([r * clat * clon, r * clat * slon, r * slat], dtype=float)


def elevation(r_sat_ecef: np.ndarray, r_tgt_ecef: np.ndarray) -> float:
    """Elevation of the satellite above the local horizon at the target, radians."""
    rho = np.asarray(r_sat_ecef, dtype=float) - np.asarray(r_tgt_ecef, dtype=float)
    up = np.asarray(r_tgt_ecef, dtype=float)
    up_norm = np.linalg.norm(up)
    rho_norm = np.linalg.norm(rho)
    if rho_norm < 1.0 or up_norm < 1.0:
        return -np.pi / 2.0
    sine = np.clip(np.dot(rho, up) / (rho_norm * up_norm), -1.0, 1.0)
    return float(np.arcsin(sine))


def earth_central_angle(alt_m: float, min_elev: float) -> float:
    r = R_EARTH + max(alt_m, 1.0)
    arg = np.clip((R_EARTH / r) * np.cos(min_elev), -1.0, 1.0)
    return float(np.arccos(arg) - min_elev)


def footprint_radius(alt_m: float, min_elev: float) -> float:
    """Approximate geodetic coverage radius along the surface, metres."""
    return R_EARTH * earth_central_angle(alt_m, min_elev)


def koe_to_rv(a: float, e: float, inc: float, raan: float, argp: float, nu: float, mu: float = MU_EARTH) -> tuple[np.ndarray, np.ndarray]:  
    """Classical orbital elements to ECI position/velocity."""
    p = a * (1.0 - e**2) if e < 0.999 else a
    if e < 1e-12:
        p = a
    r_pqw = p / (1.0 + e * np.cos(nu))
    pos_pqw = np.array([r_pqw * np.cos(nu), r_pqw * np.sin(nu), 0.0])
    vel_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    rot = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ]
    )
    return rot @ pos_pqw, rot @ vel_pqw


def true_anomaly_from_mean(M: float, e: float, iters: int = 12) -> float:
    if e < 1e-12:
        return float(np.mod(M, 2.0 * np.pi))
    E = M if e < 0.8 else np.pi
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    sin_nu = (np.sqrt(1.0 - e**2) * np.sin(E)) / (1.0 - e * np.cos(E))
    cos_nu = (np.cos(E) - e) / (1.0 - e * np.cos(E))
    return float(np.arctan2(sin_nu, cos_nu))


def rv_at_time(a: float, e: float, inc: float, raan: float, argp: float, m0: float, t: float, t0: float = 0.0, mean_motion: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = np.sqrt(MU_EARTH / a**3) if mean_motion is None else float(mean_motion)
    M = m0 + n * (t - t0)
    nu = true_anomaly_from_mean(M, e)
    return koe_to_rv(a, e, inc, raan, argp, nu)


def walker_delta_elements(n_sats: int, n_planes: int, altitude_m: float, inclination: float, phasing: int = 1, raan_offset: float = 0.0) -> list[tuple[float, float, float, float, float, float]]:
    """Return list of (a, e, i, RAAN, argp, M0) for a Walker-Delta constellation."""
    if n_sats % n_planes != 0:
        raise ValueError("n_sats must be divisible by n_planes")
    per_plane = n_sats // n_planes
    a = R_EARTH + altitude_m
    elements = []
    for p in range(n_planes):
        raan = raan_offset + 2.0 * np.pi * p / n_planes
        for k in range(per_plane):
            m0 = 2.0 * np.pi * k / per_plane + 2.0 * np.pi * phasing * p / n_sats
            elements.append((a, 0.0, inclination, raan, 0.0, float(np.mod(m0, 2.0 * np.pi))))
    return elements


def in_sunlight(r_eci: np.ndarray, sun_eci: np.ndarray, radius: float = R_EARTH) -> bool:
    """Cylindrical eclipse: False when the spacecraft is in Earth's shadow."""
    s = np.asarray(sun_eci, dtype=float)
    s_norm = np.linalg.norm(s)
    if s_norm < 1.0:
        return True
    s_hat = s / s_norm
    r = np.asarray(r_eci, dtype=float)
    if np.dot(r, s_hat) >= 0.0:
        return True
    perp = r - np.dot(r, s_hat) * s_hat
    return float(np.linalg.norm(perp)) > radius


def _seconds_until(t0: float, horizon: float, dt: float, still_true, refine: int, last_true: bool) -> float:
    """Seconds after t0 until `still_true` fails. `last_true` keeps the last good instant."""
    t = t0
    limit = t0 + horizon
    while t < limit:
        nxt = min(t + dt, limit)
        if still_true(nxt):
            t = nxt
            continue

        # Bisect the step that crossed.
        lo, hi = t, nxt
        for _ in range(refine):
            mid = 0.5 * (lo + hi)
            if still_true(mid):
                lo = mid
            else:
                hi = mid
        end = lo if last_true else hi
        return max(0.0, end - t0)
    return horizon


def _points_in_view(r_sat_fn, points, t: float, min_elev: float, gmst0: float) -> bool:
    r_ecef = eci_to_ecef(r_sat_fn(t), t, gmst0)
    return all(elevation(r_ecef, lla_to_ecef(lat, lon, 0.0)) >= min_elev for lat, lon in points)


def access_remaining(r_sat_fn, lat: float, lon: float, t0: float, min_elev: float, gmst0: float = 0.0, max_horizon: float = 900.0, dt: float = 10.0) -> float:
    """Seconds the target stays in view after t0.

    r_sat_fn(t) -> ECI position. Returns 0 if the target is not in view at t0.
    """
    remaining, _next = access_windows(r_sat_fn, ((lat, lon),), t0, min_elev, gmst0, max_horizon, dt)
    return remaining


def access_next(r_sat_fn, lat: float, lon: float, t0: float, min_elev: float, gmst0: float = 0.0, max_horizon: float = 900.0, dt: float = 10.0) -> float:
    """Seconds until the target rises. 0 if it is already in view."""
    _remaining, nxt = access_windows(r_sat_fn, ((lat, lon),), t0, min_elev, gmst0, max_horizon, dt)
    return nxt


def access_windows(r_sat_fn, points, t0: float, min_elev: float, gmst0: float = 0.0, max_horizon: float = 900.0, dt: float = 10.0) -> tuple[float, float]:
    """Joint access of every point: seconds remaining in this pass, seconds to the next rise.

    `next` is 0 while the points are in view. Both are 0 / `max_horizon` when no pass starts inside the horizon.
    """
    pts = [(float(lat), float(lon)) for lat, lon in points]
    if not pts:
        return 0.0, float(max_horizon)

    def in_view(t: float) -> bool:
        return _points_in_view(r_sat_fn, pts, t, min_elev, gmst0)

    if in_view(t0):
        remaining = _seconds_until(t0, max_horizon, dt, in_view, refine=18, last_true=True)
        return remaining, 0.0

    nxt = _seconds_until(t0, max_horizon, dt, lambda t: not in_view(t), refine=18, last_true=False)
    if nxt >= max_horizon - 1e-9 and not in_view(t0 + max_horizon):
        return 0.0, float(max_horizon)
    return 0.0, nxt


def time_to_sun_change(r_sat_fn, sun_eci: np.ndarray, t0: float, currently_in_sun: bool, max_horizon: float = 6000.0, dt: float = 20.0) -> float:
    def same_light(t: float) -> bool:
        return in_sunlight(r_sat_fn(t), sun_eci) == currently_in_sun

    return _seconds_until(t0, max_horizon, dt, same_light, refine=16, last_true=False)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Arguments are geodetic radians; lat2/lon2 may be arrays."""
    lat1, lon1, lat2, lon2 = (np.asarray(v, dtype=float) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    chord = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5) ** 2
    return (2.0 * R_EARTH) * np.arcsin(np.clip(np.sqrt(chord), 0.0, 1.0))


def load_ground_points(path: Path | None = None) -> np.ndarray:
    """Catalog rows: lat_rad, lon_rad. Built so 100 km circles cover most continental land."""
    raw = json.loads((path or GROUND_POINTS_PATH).read_text(encoding="utf-8"))
    pts = np.asarray(raw["points"], dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
        raise ValueError("ground_points.json needs at least two lat/lon rows")
    return np.column_stack((np.deg2rad(pts[:, 0]), np.deg2rad(pts[:, 1])))


def load_ground_region(name: str) -> np.ndarray:
    """Region rows in radians. Row 0 is the centre; each next row is the next-nearest neighbour."""
    path = REGIONS_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown ground region {name!r}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    pts = np.asarray(raw["points"], dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
        raise ValueError(f"{path.name} needs at least two lat/lon rows")
    return np.column_stack((np.deg2rad(pts[:, 0]), np.deg2rad(pts[:, 1])))


def write_ground_region(name: str = "europe", box: dict | None = None) -> Path:
    """Keep catalog points inside `box`, then order them from the centre outward.

    The centre is the catalog point nearest the mean of the box. Every later
    row is the next nearest neighbour of that centre, so the first `m` rows
    are the local group.
    """
    box = box or REGION_BOXES[name]
    raw = json.loads(GROUND_POINTS_PATH.read_text(encoding="utf-8"))
    pts = np.asarray(raw["points"], dtype=float)
    lat0, lat1 = box["lat"]
    lon0, lon1 = box["lon"]
    inside = (
        (pts[:, 0] >= lat0)
        & (pts[:, 0] <= lat1)
        & (pts[:, 1] >= lon0)
        & (pts[:, 1] <= lon1)
    )
    chosen = pts[inside]
    if len(chosen) < 2:
        raise ValueError(f"{name} box holds {len(chosen)} catalog points")

    # The forming point is the catalog site closest to the middle of the land box.
    mean_lat = float(chosen[:, 0].mean())
    mean_lon = float(chosen[:, 1].mean())
    to_mean = haversine_m(
        math.radians(mean_lat),
        math.radians(mean_lon),
        np.deg2rad(chosen[:, 0]),
        np.deg2rad(chosen[:, 1]),
    )
    center = chosen[int(np.argmin(to_mean))]
    to_center = haversine_m(
        math.radians(float(center[0])),
        math.radians(float(center[1])),
        np.deg2rad(chosen[:, 0]),
        np.deg2rad(chosen[:, 1]),
    )
    ordered = chosen[np.argsort(to_center, kind="mergesort")]
    payload = {
        "name": name,
        "order": "nearest_to_center",
        "box": {"lat": [lat0, lat1], "lon": [lon0, lon1]},
        "center": [round(float(center[0]), 4), round(float(center[1]), 4)],
        "points": [[round(float(lat), 4), round(float(lon), 4)] for lat, lon in ordered],
    }
    REGIONS_DIR.mkdir(parents=True, exist_ok=True)
    dest = REGIONS_DIR / f"{name}.json"
    dest.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return dest


def pair_ground_points(
    rng,
    points,
    mean_radius_m: float,
    difficulty: float,
    radius_std_m: float = 0.0,
    wide=None,
):
    """Pick A from the local group, then B.

    With probability `difficulty`, the target distance is a draw from
    Normal(mean footprint radius, spread of those radii). B is the closest
    site in `wide` (the world catalog when the caller passes it), so a long
    draw can leave the local group. Otherwise B stays in `points`, uniform
    inside the mean radius.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        raise ValueError("Need at least one active ground point")
    i = int(rng.integers(n))
    origin = (float(pts[i, 0]), float(pts[i, 1]))

    # A long draw searches the whole catalog. A short draw stays in the local group.
    if float(rng.random()) < float(difficulty):
        pool = pts if wide is None else np.asarray(wide, dtype=float)
        if len(pool) == 0:
            return origin, origin
        dist = np.array(haversine_m(origin[0], origin[1], pool[:, 0], pool[:, 1]), dtype=float, copy=True)
        dist[dist < 1.0] = np.inf
        if not np.isfinite(dist).any():
            return origin, origin
        target = max(float(rng.normal(mean_radius_m, max(float(radius_std_m), 0.0))), 0.0)
        j = int(np.argmin(np.abs(dist - target)))
        return origin, (float(pool[j, 0]), float(pool[j, 1]))

    dist = np.array(haversine_m(origin[0], origin[1], pts[:, 0], pts[:, 1]), dtype=float, copy=True)
    dist[i] = np.inf
    if not np.isfinite(dist).any():
        return origin, origin
    inside = np.flatnonzero(np.isfinite(dist) & (dist <= mean_radius_m))
    j = int(rng.choice(inside)) if inside.size else int(np.argmin(dist))
    return origin, (float(pts[j, 0]), float(pts[j, 1]))


def _point_in_ring(lon: float, lat: float, ring) -> bool:
    x, y = lon, lat
    inside = False
    j = len(ring) - 1
    for i, pt in enumerate(ring):
        xi, yi = pt[0], pt[1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-16) + xi:
            inside = not inside
        j = i
    return inside


def _in_land(lon: float, lat: float, features) -> bool:
    for feat in features:
        box = feat.get("_bbox")
        if box is None:
            geom = feat["geometry"]
            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            xs = [p[0] for poly in polys for p in poly[0]]
            ys = [p[1] for poly in polys for p in poly[0]]
            box = (min(xs), min(ys), max(xs), max(ys))
            feat["_bbox"] = box
            feat["_polys"] = polys
        else:
            polys = feat["_polys"]

        if lon < box[0] or lon > box[2] or lat < box[1] or lat > box[3]:
            continue
        for poly in polys:
            if _point_in_ring(lon, lat, poly[0]) and not any(_point_in_ring(lon, lat, hole) for hole in poly[1:]):
                return True
    return False


def _hex_land_cells(features, radius_km: float) -> np.ndarray:
    spacing = radius_km * math.sqrt(3.0)
    dlat = spacing / 111.32
    rows = []
    lat = -56.0
    odd = False
    while lat <= 75.0:
        dlon = spacing / max(111.32 * math.cos(math.radians(lat)), 20.0)
        lon = -180.0 + (0.5 * dlon if odd else 0.0)
        while lon < 180.0:
            if _in_land(lon, lat, features):
                rows.append((lat, lon))
            lon += dlon
        lat += dlat
        odd = not odd
    return np.asarray(rows, dtype=float)


def _parse_cities(raw: bytes) -> list[tuple[float, float, int, str]]:
    cities = []
    if raw[:2] == b"PK":
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".txt"))
            text = zf.read(name).decode("utf-8", "replace")
        for line in text.splitlines():
            col = line.split("\t")
            if len(col) < 15:
                continue
            cities.append((float(col[4]), float(col[5]), int(float(col[14] or 0)), col[2] or col[1]))
        return cities
    data = json.loads(raw.decode("utf-8"))
    for feat in data.get("features", []):
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"][:2]
        pop = int(props.get("POP_MAX") or props.get("pop_max") or 0)
        name = str(props.get("NAMEASCII") or props.get("NAME") or props.get("name") or "city")
        cities.append((float(lat), float(lon), pop, name))
    return cities


def write_ground_points(path: Path | None = None, radius_km: float = GROUND_COVER_KM) -> Path:
    """Download land and inhabited places, then write a 100 km covering catalog.

    Cities from GeoNames (population >= 15k) sit first; leftover continental cells
    keep their hex centre so 100 km disks cover most land outside Antarctica.
    """
    import urllib.request

    dest = path or GROUND_POINTS_PATH
    land_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson"
    city_urls = (
        "https://download.geonames.org/export/dump/cities15000.zip",
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_populated_places_simple.geojson",
    )

    # Land outline, then the first city catalog that answers.
    with urllib.request.urlopen(land_url, timeout=60) as resp:
        land = json.loads(resp.read().decode("utf-8"))["features"]
    cities: list[tuple[float, float, int, str]] = []
    for url in city_urls:
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                cities = _parse_cities(resp.read())
            if cities:
                break
        except OSError:
            continue
    if not cities:
        raise RuntimeError("Could not download inhabited places")

    # Cities cover nearby hex cells. Leftover cells keep their own centre.
    cells = _hex_land_cells(land, radius_km)
    cover_m = radius_km * 1e3
    cell_lat = np.deg2rad(cells[:, 0])
    cell_lon = np.deg2rad(cells[:, 1])
    on_land = [
        (lat, lon, pop, name)
        for lat, lon, pop, name in cities
        if -56.0 <= lat <= 75.0 and _in_land(lon, lat, land)
    ]
    on_land.sort(key=lambda row: -row[2])
    uncovered = np.ones(len(cells), dtype=bool)
    picked: list[tuple[float, float, str]] = []
    for lat, lon, _pop, name in on_land:
        dist = haversine_m(math.radians(lat), math.radians(lon), cell_lat, cell_lon)
        hit = uncovered & (dist <= cover_m)
        if not hit.any():
            continue
        uncovered[hit] = False
        picked.append((round(lat, 4), round(lon, 4), name))
        if not uncovered.any():
            break
    for i in np.flatnonzero(uncovered):
        picked.append((round(float(cells[i, 0]), 4), round(float(cells[i, 1]), 4), "land"))

    dest.write_text(
        json.dumps({"radius_km": radius_km, "points": [[p, q] for p, q, _n in picked]}, separators=(",", ":")),
        encoding="utf-8",
    )
    return dest


if __name__ == "__main__":
    out = write_ground_points()
    n = len(json.loads(out.read_text(encoding="utf-8"))["points"])
    print(f"{out} ({n} points)")
