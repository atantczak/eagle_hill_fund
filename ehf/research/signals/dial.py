"""Dial state machines — pure, causal transforms from a reading to an exposure in [0, 1].

These sit in ``signals`` (not ``strategies``) because they are signal-shaped:
Series in, Series out, bar *t* using only data through *t*. That placement also
lets the daily report (which ships only ``signals`` + ``complexity`` to Lambda)
compute the live market dial with the *same* code the backtests ran.

  hysteresis_gate   out when x > out_above, back in when x < in_below, else hold
  three_state       1 below lo, 0.5 between, 0 above hi (a graded dial)
  cooldown_gate     0 for ``bars`` bars after any trigger, else 1
  min_dwell         keep any new state at least ``bars`` bars (whipsaw brake)
  band_rebalance    only move a continuous dial when it drifts > ``band``
  confirm           a boolean must hold ``n`` consecutive bars to count

NaN inputs (warm-up) resolve to exposure 0 — the conservative default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hysteresis_gate(x: pd.Series, *, out_above: float, in_below: float) -> pd.Series:
    """1/0 gate with a dead band: leave when ``x > out_above``, return when ``x < in_below``.

    Starts *in* (1). ``in_below`` is clamped to ``out_above`` so a misordered
    sweep cell degrades to a plain threshold instead of a gate that never re-enters.
    NaN bars read 0 (unknown = flat) without disturbing the remembered state.
    """
    in_below = min(in_below, out_above)
    vals = x.to_numpy(dtype=float)
    out = np.zeros(len(vals))
    state = 1.0
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        if v > out_above:
            state = 0.0
        elif v < in_below:
            state = 1.0
        out[i] = state
    return pd.Series(out, index=x.index)


def three_state(x: pd.Series, *, lo: float, hi: float, high_is_risk: bool = True) -> pd.Series:
    """Graded dial: 1 when ``x < lo``, 0.5 between, 0 when ``x > hi`` (flipped for low-is-risk)."""
    lo, hi = (lo, hi) if lo <= hi else (hi, lo)
    e = pd.Series(0.5, index=x.index)
    if high_is_risk:
        e[x < lo] = 1.0
        e[x > hi] = 0.0
    else:
        e[x < lo] = 0.0
        e[x > hi] = 1.0
    return e.where(x.notna(), 0.0)


def cooldown_gate(trigger: pd.Series, bars: int) -> pd.Series:
    """Exposure 0 on a trigger bar and for ``bars - 1`` bars after, else 1 (NaN -> 0)."""
    t = trigger.astype(float).where(trigger.notna())
    recent = t.rolling(max(bars, 1), min_periods=1).max()
    return (1.0 - recent).where(t.notna(), 0.0)


def min_dwell(exposure: pd.Series, bars: int) -> pd.Series:
    """Hold each new dial value for at least ``bars`` bars before accepting a change."""
    if bars <= 1:
        return exposure
    vals = exposure.to_numpy(dtype=float)
    out = vals.copy()
    held = 0
    for i in range(1, len(vals)):
        if vals[i] != out[i - 1] and held < bars - 1:
            out[i] = out[i - 1]
            held += 1
        elif vals[i] != out[i - 1]:
            out[i] = vals[i]
            held = 0
        else:
            held += 1
    return pd.Series(out, index=exposure.index)


def band_rebalance(target: pd.Series, band: float) -> pd.Series:
    """Track a continuous dial lazily: move only when |target - held| > ``band``."""
    if band <= 0:
        return target
    vals = target.to_numpy(dtype=float)
    out = np.zeros(len(vals))
    held = 0.0
    for i, v in enumerate(vals):
        if np.isnan(v):
            held = 0.0
        elif abs(v - held) > band:
            held = v
        out[i] = held
    return pd.Series(out, index=target.index)


def confirm(cond: pd.Series, n: int) -> pd.Series:
    """True only once ``cond`` has been True for ``n`` consecutive bars."""
    if n <= 1:
        return cond.fillna(False).astype(bool)
    c = cond.fillna(False).astype(float)
    return c.rolling(n).sum() >= n
