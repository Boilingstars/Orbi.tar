"""Physical satellite template: orbit from JSON, tick energy/thermal/calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from orbi.constants import R_EARTH, SOLAR_CONSTANT


@dataclass
class Satellite:
    sat_id: str
    index: int
    raan_deg: float
    period_min: float
    inclination_deg: float
    altitude_km: float
    mean_anomaly_deg: float = 0.0
    capacity_wh: float = 140.0
    energy_wh: float = 100.0
    temp_c: float = 20.0
    calibration_age_steps: int = 0
    base_w: float = 18.0
    heater_w: float = 30.0
    calibration_w: float = 20.0
    downlink_w: float = 90.0
    relay_w: float = 65.0
    panel_area_m2: float = 0.25
    panel_efficiency: float = 0.2
    outage_until: int = 0
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: dict, index: int) -> Satellite:
        orbit = spec.get("orbit", spec)
        cap = float(spec["capacity_wh"])
        soc = float(spec.get("initial_soc_pct", 70.0))
        return cls(
            sat_id=str(spec["id"]),
            index=int(index),
            raan_deg=float(orbit["raan_deg"]),
            period_min=float(orbit["period_min"]),
            inclination_deg=float(orbit["inclination_deg"]),
            altitude_km=float(orbit["altitude_km"]),
            mean_anomaly_deg=float(orbit.get("mean_anomaly_deg", 0.0)),
            capacity_wh=cap,
            energy_wh=cap * soc / 100.0,
            temp_c=float(spec.get("initial_temp_c", 20.0)),
            calibration_age_steps=int(spec.get("initial_calibration_age_steps", 0)),
            base_w=float(spec.get("base_w", 18.0)),
            heater_w=float(spec.get("heater_w", 30.0)),
            calibration_w=float(spec.get("calibration_w", 20.0)),
            downlink_w=float(spec.get("downlink_w", 90.0)),
            relay_w=float(spec.get("relay_w", 65.0)),
            panel_area_m2=float(spec.get("panel_area_m2", 0.25)),
            panel_efficiency=float(spec.get("panel_efficiency", 0.2)),
        )

    @property
    def a_m(self) -> float:
        return R_EARTH + self.altitude_km * 1e3

    @property
    def mean_motion(self) -> float:
        return 2.0 * math.pi / max(self.period_min * 60.0, 1e-6)

    @property
    def soc(self) -> float:
        return float(self.energy_wh / max(self.capacity_wh, 1e-9))

    def available(self, k: int) -> bool:
        return k >= self.outage_until

    def payload_w(self, action: str, job: dict | None = None) -> float:
        if action == "calibrate":
            return self.calibration_w
        if action == "job" and job is not None:
            return self.downlink_w if job["kind"] == "downlink" else self.relay_w
        return 0.0

    def panel_w(self) -> float:
        return self.panel_area_m2 * self.panel_efficiency * SOLAR_CONSTANT

    def transition(self, solar_w: float, payload_w: float, model: dict, dt: float) -> tuple[float, float, float, float]:
        heater = self.heater_w if self.temp_c < model["heater_below_c"] else 0.0
        load_w = self.base_w + heater + payload_w
        delta = (solar_w - load_w) * dt / 3600.0
        if delta >= 0:
            if model["charge_min_c"] <= self.temp_c <= model["charge_max_c"]:
                delta *= model["charge_efficiency"]
            else:
                delta = 0.0
        else:
            delta /= model["discharge_efficiency"]
        energy = self.energy_wh + delta
        target = model["sun_target_c"] if solar_w > 0 else model["eclipse_target_c"]
        equilibrium = target + model["thermal_gain_c_per_w"] * load_w
        temp = equilibrium + (self.temp_c - equilibrium) * math.exp(-dt / model["thermal_tau_s"])
        return energy, temp, heater, load_w

    def can_take(self, world, job: dict | None, model: dict, k: int) -> tuple[bool, str]:
        """Physical take-job check. Subclasses add extra conditions."""
        if job is None:
            return False, "unknown_job"
        if not self.available(k):
            return False, "satellite_unavailable"
        if job["completed_step"] is not None or job["remaining_steps"] <= 0:
            return False, "already_completed"
        if not job["release_step"] <= k < job["deadline_step"]:
            return False, "outside_job_window"
        if self.calibration_age_steps >= model["calibration_valid_steps"]:
            return False, "calibration_required"
        point_a = job.get("point_a")
        point_b = job.get("point_b")
        if point_a is None or point_b is None or not world.sees_job(self.index, point_a, point_b):
            return False, "no_contact"
        return True, "accepted"
