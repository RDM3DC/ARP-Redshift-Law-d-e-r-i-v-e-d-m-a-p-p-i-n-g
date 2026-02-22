"""Core ARP redshift-law utilities.

Model:
    z(t) = z0 * (1 - exp(-gamma * t))
which satisfies:
    dz/dt = gamma * (z0 - z)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class RedshiftParams:
    z0: float
    gamma: float


@dataclass(frozen=True)
class TwoTimescaleParams:
    z0: float
    gamma1: float
    gamma2: float
    a: float
    w1: float
    w2: float


def z_of_t(t: np.ndarray, z0: float, gamma: float) -> np.ndarray:
    """Analytic solution z(t) = z0 (1 - e^{-gamma t})."""
    t = np.asarray(t, dtype=float)
    return z0 * (1.0 - np.exp(-gamma * t))


def z_of_t_two_timescale(
    t: np.ndarray,
    z0: float,
    a: float,
    gamma1: float,
    gamma2: float,
) -> np.ndarray:
    """Two-timescale relaxation model.

    z(t) = z0 * [1 - a*exp(-gamma1*t) - (1-a)*exp(-gamma2*t)]
    with 0 <= a <= 1 and gamma1,gamma2 > 0.
    """
    t = np.asarray(t, dtype=float)
    return z0 * (
        1.0 - a * np.exp(-gamma1 * t) - (1.0 - a) * np.exp(-gamma2 * t)
    )


def z_of_t_oscillatory(
    t: np.ndarray,
    z_horizon: float,
    gamma: float,
    omega: float,
    phi: float = 0.0,
) -> np.ndarray:
    """Oscillatory toy extension.

    z(t) = z_horizon * (1 - exp(-gamma * t)) * sin(omega * t + phi)

    This is useful as a trajectory-shaping toy model. For positive-only trajectories
    over a finite interval, tune omega and phi so the sine factor does not cross zero.
    """
    t = np.asarray(t, dtype=float)
    envelope = z_horizon * (1.0 - np.exp(-gamma * t))
    return envelope * np.sin(omega * t + phi)


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


def _fit_nonnegative_two_component(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Small nonnegative least-squares helper for 2 features.

    Attempts unconstrained least squares first; if negative coefficients appear,
    falls back to boundary solutions where one coefficient is zero.
    """
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    if np.all(w >= 0):
        return w

    x1 = X[:, 0]
    x2 = X[:, 1]

    d1 = float(np.dot(x1, x1))
    d2 = float(np.dot(x2, x2))
    w1 = max(0.0, float(np.dot(x1, y) / d1)) if d1 > 0 else 0.0
    w2 = max(0.0, float(np.dot(x2, y) / d2)) if d2 > 0 else 0.0

    cand = [
        np.array([w1, 0.0], dtype=float),
        np.array([0.0, w2], dtype=float),
        np.array([0.0, 0.0], dtype=float),
    ]

    best = min(cand, key=lambda c: float(np.mean((X @ c - y) ** 2)))
    return best


def estimate_two_timescale_params_from_timeseries(
    t: np.ndarray,
    z: np.ndarray,
    gamma_min: float = 1e-3,
    gamma_max: float = 5.0,
    gamma_grid_size: int = 120,
) -> Tuple[TwoTimescaleParams, float]:
    """Fit two-timescale model from z(t) using gamma grid + nonnegative LS.

    Model is linear in nonnegative (w1,w2) for fixed gammas:
        z(t) = w1*(1-exp(-g1*t)) + w2*(1-exp(-g2*t))
    where z0 = w1+w2 and a = w1/z0.
    """
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    if t.shape != z.shape:
        raise ValueError("t and z must have same shape")
    if t.ndim != 1 or t.size < 5:
        raise ValueError("need at least 5 samples")

    gammas = np.linspace(gamma_min, gamma_max, gamma_grid_size)
    best = None

    for i, g1 in enumerate(gammas):
        for g2 in gammas[i + 1 :]:
            x1 = 1.0 - np.exp(-g1 * t)
            x2 = 1.0 - np.exp(-g2 * t)
            X = np.column_stack([x1, x2])

            w = _fit_nonnegative_two_component(X, z)
            z_hat = X @ w
            mse = float(np.mean((z_hat - z) ** 2))

            if best is None or mse < best[0]:
                best = (mse, g1, g2, float(w[0]), float(w[1]))

    if best is None:
        raise ValueError("fit failed: no valid two-timescale solution found")

    _, g1, g2, w1, w2 = best
    z0 = w1 + w2
    a = (w1 / z0) if z0 > 0 else 0.5
    params = TwoTimescaleParams(z0=z0, gamma1=g1, gamma2=g2, a=a, w1=w1, w2=w2)
    return params, float(best[0])


def information_criteria(rss: float, n: int, k: int) -> Dict[str, float]:
    """Compute AIC and BIC from residual sum of squares."""
    if n <= 0:
        raise ValueError("n must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    rss_eff = max(float(rss), 1e-15)
    aic = n * np.log(rss_eff / n) + 2 * k
    bic = n * np.log(rss_eff / n) + k * np.log(n)
    return {"aic": float(aic), "bic": float(bic)}
