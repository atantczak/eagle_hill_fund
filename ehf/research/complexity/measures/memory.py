"""Long-memory / persistence measures: Hurst (R/S) and DFA.

These are *scalar* estimators over a whole series — the ground-truth-validation
form. (The rolling, signal-shaped Hurst used by strategies lives in
``ehf.research.signals.hurst``; this module is the canonical math for validation and for
the complexity applications.)

  * Hurst exponent via rescaled-range (R/S): H>0.5 persistent/trending,
    H~0.5 random, H<0.5 anti-persistent/mean-reverting.
  * DFA (detrended fluctuation analysis) exponent alpha: ~0.5 white noise,
    ~1.0 1/f noise, ~1.5 a random walk (integrated noise).
"""

from __future__ import annotations

import numpy as np

from ehf.research.complexity.measures.entropy import ArrayLike, _as_array


def _log_spaced_scales(n: int, min_scale: int, max_scale: int | None, count: int) -> np.ndarray:
    hi = max_scale if max_scale is not None else n // 4
    hi = max(hi, min_scale + 1)
    scales = np.unique(np.floor(np.logspace(np.log10(min_scale), np.log10(hi), count)).astype(int))
    return scales[scales >= min_scale]


def hurst(
    x: ArrayLike, *, min_scale: int = 8, max_scale: int | None = None, count: int = 20
) -> float:
    """Hurst exponent via rescaled-range (R/S) analysis.

    For each window size ``w`` the series is split into non-overlapping chunks;
    each chunk's rescaled range R/S is averaged, then ``log(R/S)`` is regressed
    on ``log(w)`` — the slope is H.
    """
    arr = _as_array(x)
    n = len(arr)
    scales = _log_spaced_scales(n, min_scale, max_scale, count)
    if len(scales) < 2:
        return float("nan")

    rs_means: list[float] = []
    valid_scales: list[int] = []
    for w in scales:
        n_chunks = n // w
        if n_chunks < 1:
            continue
        chunks = arr[: n_chunks * w].reshape(n_chunks, w)
        rs_vals = []
        for chunk in chunks:
            mean = chunk.mean()
            dev = np.cumsum(chunk - mean)
            r = dev.max() - dev.min()
            s = chunk.std()
            if s > 0 and r > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_means.append(float(np.mean(rs_vals)))
            valid_scales.append(int(w))

    if len(valid_scales) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(valid_scales), np.log(rs_means), 1)
    return float(slope)


def dfa(
    x: ArrayLike,
    *,
    order: int = 1,
    min_scale: int = 8,
    max_scale: int | None = None,
    count: int = 20,
) -> float:
    """Detrended fluctuation analysis exponent.

    Integrates the mean-removed series into a profile, then for each scale fits
    and removes a polynomial trend of degree ``order`` per segment and measures
    the RMS fluctuation. The slope of ``log F(scale)`` vs ``log scale`` is alpha.
    """
    arr = _as_array(x)
    n = len(arr)
    profile = np.cumsum(arr - arr.mean())
    scales = _log_spaced_scales(n, min_scale, max_scale, count)
    if len(scales) < 2:
        return float("nan")

    fluct: list[float] = []
    valid_scales: list[int] = []
    for s in scales:
        n_seg = n // s
        if n_seg < 1:
            continue
        segments = profile[: n_seg * s].reshape(n_seg, s)
        t = np.arange(s)
        rms_vals = []
        for seg in segments:
            coef = np.polyfit(t, seg, order)
            trend = np.polyval(coef, t)
            rms_vals.append(np.mean((seg - trend) ** 2))
        f = float(np.sqrt(np.mean(rms_vals)))
        if f > 0:
            fluct.append(f)
            valid_scales.append(int(s))

    if len(valid_scales) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(valid_scales), np.log(fluct), 1)
    return float(slope)
