"""Earth and orbit constants for the Kepler picture."""

from __future__ import annotations

import numpy as np

R_EARTH = 6371.0e3
MU_EARTH = 3.986004418e14
OMEGA_EARTH = 7.2921159e-5
SOLAR_CONSTANT = 1362.0
AU = 1.495978707e11


def orbital_period(semi_major_axis_m: float) -> float:
    return 2.0 * np.pi * np.sqrt(semi_major_axis_m**3 / MU_EARTH)
