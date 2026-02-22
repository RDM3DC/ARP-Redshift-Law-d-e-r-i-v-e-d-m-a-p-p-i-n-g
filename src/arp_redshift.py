"""Core ARP redshift-law utilities.

Model:
    z(t) = z0 * (1 - exp(-gamma * t))
which satisfies:
    dz/dt = gamma * (z0 - z)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RedshiftParams:
    z0: float
    gamma: float


def z_of_t(t: np.ndarray, z0: float, gamma: float) -> np.ndarray:
    """Analytic solution z(t) = z0 (1 - e^{-gamma t})."""
    t = np.asarray(t, dtype=float)
    return z0 * (1.0 - np.exp(-gamma * t))


def dz_dt(z: np.ndarray, z0: float, gamma: float) -> np.ndarray:
    """Right-hand side of ODE: dz/dt = gamma * (z0 - z)."""
    z = np.asarray(z, dtype=float)
    return gamma * (z0 - z)


def solve_euler(t: np.ndarray, z_init: float, z0: float, gamma: float) -> np.ndarray:
    """Numerical ODE solve by forward Euler over user-provided grid t."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be a 1D array with at least 2 points")

    z = np.empty_like(t)
    z[0] = float(z_init)
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            raise ValueError("t must be strictly increasing")
        z[i] = z[i - 1] + dt * gamma * (z0 - z[i - 1])
    return z


def estimate_params_from_timeseries(
    t: np.ndarray,
    z: np.ndarray,
    z0_grid_size: int = 2000,
    z0_margin: float = 0.2,
) -> Tuple[RedshiftParams, float]:
    """Fit z0,gamma without scipy via z0 grid + linearized slope fit.

    For each z0 candidate > max(z):
        ln(1 - z/z0) = -gamma * t
    and gamma is estimated by least-squares slope through origin.
    Returns (params, mse).
    """
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    if t.shape != z.shape:
        raise ValueError("t and z must have same shape")
    if t.ndim != 1 or t.size < 3:
        raise ValueError("need at least 3 samples")

    z_max = float(np.max(z))
    z0_min = z_max * (1.0 + 1e-6)
    z0_max = max(z0_min * (1.0 + z0_margin), z0_min + 1.0)

    z0_candidates = np.linspace(z0_min, z0_max, z0_grid_size)
    best = None

    for z0 in z0_candidates:
        frac = 1.0 - (z / z0)
        if np.any(frac <= 0):
            continue
        y = np.log(frac)
        denom = np.dot(t, t)
        if denom <= 0:
            continue
        slope = np.dot(t, y) / denom
        gamma = -slope
        if gamma <= 0:
            continue

        z_hat = z_of_t(t, z0=z0, gamma=gamma)
        mse = float(np.mean((z_hat - z) ** 2))

        if best is None or mse < best[2]:
            best = (z0, gamma, mse)

    if best is None:
        raise ValueError("fit failed: no valid z0/gamma pair found")

    params = RedshiftParams(z0=best[0], gamma=best[1])
    return params, best[2]


def half_time(gamma: float) -> float:
    """Return t_1/2 such that z(t_1/2) = 0.5 z0."""
    return np.log(2.0) / gamma
