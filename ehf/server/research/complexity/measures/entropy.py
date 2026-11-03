"""Entropy measures. Pure, vectorized scalar estimators over a 1-D series.

Permutation entropy (Bandt-Pompe 2002) first — a cheap, robust measure of the
ordinal-pattern complexity of a series, used as a regime-character signal.
"""

from __future__ import annotations

from math import factorial, log2

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series | list[float]


def _as_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D series")
    return arr


def permutation_entropy(
    x: ArrayLike, order: int = 3, delay: int = 1, *, normalize: bool = True
) -> float:
    """Bandt-Pompe permutation entropy of a series.

    Embeds the series into ordinal patterns of length ``order`` (spacing
    ``delay``) and returns the Shannon entropy of the pattern distribution. With
    ``normalize`` the result is in [0, 1]: ~1 for white noise (all orderings
    equally likely), low for smooth/deterministic signals (Lorenz, Hénon).

    Returns NaN if the series is too short for even one embedding vector.
    """
    arr = _as_array(x)
    m = len(arr) - (order - 1) * delay
    if order < 2 or m < 1:
        return float("nan")

    # Build the (m, order) embedding and take the ordinal pattern of each row.
    idx = np.arange(0, order * delay, delay)
    embedded = arr[np.arange(m)[:, None] + idx[None, :]]
    patterns = np.argsort(embedded, axis=1, kind="stable")

    # Count distinct ordinal patterns.
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / counts.sum()
    pe = -float(np.sum(p * np.log2(p)))
    if normalize:
        pe /= log2(factorial(order))
    return pe


def shannon_entropy(x: ArrayLike, bins: int = 16, *, normalize: bool = True) -> float:
    """Histogram Shannon entropy of a series' value distribution (in bits)."""
    arr = _as_array(x)
    counts, _ = np.histogram(arr, bins=bins)
    counts = counts[counts > 0]
    if counts.size == 0:
        return float("nan")
    p = counts / counts.sum()
    h = -float(np.sum(p * np.log2(p)))
    if normalize and bins > 1:
        h /= log2(bins)
    return h
