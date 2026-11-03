"""Known reference systems with established properties.

Per the complexity brief's prime directive: every measure must recover ground
truth on these before it is allowed near price data. White noise and AR(1) are
the stochastic baselines; Lorenz and Hénon are deterministic-chaos references;
``coupled_linear`` is a unidirectional-coupling reference for transfer entropy.

All generators are seedable/deterministic and return 1-D float arrays.
"""

from __future__ import annotations

import numpy as np


def white_noise(n: int, *, seed: int = 0, sigma: float = 1.0) -> np.ndarray:
    """IID Gaussian noise. Maximal entropy; Hurst ~ 0.5; no memory."""
    return np.random.default_rng(seed).normal(0.0, sigma, n)


def ar1(n: int, phi: float = 0.9, *, seed: int = 0, sigma: float = 1.0) -> np.ndarray:
    """AR(1): x_t = phi*x_{t-1} + eps. phi>0 persistent, phi<0 anti-persistent."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def random_walk(n: int, *, seed: int = 0, sigma: float = 1.0) -> np.ndarray:
    """Integrated white noise. DFA exponent ~ 1.5 (a classic ground truth)."""
    return np.cumsum(white_noise(n, seed=seed, sigma=sigma))


def lorenz(
    n: int,
    *,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: tuple[float, float, float] = (1.0, 1.0, 1.0),
    discard: int = 1000,
    component: int = 0,
) -> np.ndarray:
    """Lorenz attractor via RK4. Deterministic chaos: low ordinal complexity,
    positive largest Lyapunov exponent, strong recurrence structure.

    Returns one coordinate (default x). Use :func:`lorenz_xyz` for all three.
    """
    return lorenz_xyz(n, dt=dt, sigma=sigma, rho=rho, beta=beta, x0=x0, discard=discard)[
        :, component
    ]


def lorenz_xyz(
    n: int,
    *,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: tuple[float, float, float] = (1.0, 1.0, 1.0),
    discard: int = 1000,
) -> np.ndarray:
    """Full Lorenz trajectory, shape (n, 3)."""

    def deriv(s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    total = n + discard
    out = np.empty((total, 3))
    s = np.array(x0, dtype=float)
    for i in range(total):
        out[i] = s
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out[discard:]


def henon(
    n: int,
    *,
    a: float = 1.4,
    b: float = 0.3,
    x0: float = 0.1,
    y0: float = 0.3,
    discard: int = 100,
) -> np.ndarray:
    """Hénon map (deterministic chaos), returns the x series."""
    total = n + discard
    x, y = x0, y0
    out = np.empty(total)
    for i in range(total):
        out[i] = x
        x, y = 1.0 - a * x * x + y, b * x
    return out[discard:]


def coupled_linear(
    n: int, c: float = 0.6, *, seed: int = 0, sigma: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Unidirectional coupling X -> Y: y_t = c*x_{t-1} + noise; x is independent.

    Ground truth for transfer entropy: TE(X->Y) >> TE(Y->X) ~ 0.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, sigma, n)
    ny = rng.normal(0.0, sigma, n)
    y = np.empty(n)
    y[0] = ny[0]
    y[1:] = c * x[:-1] + np.sqrt(max(1.0 - c * c, 0.0)) * ny[1:]
    return x, y
