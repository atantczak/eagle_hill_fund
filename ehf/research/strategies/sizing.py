"""Pluggable position-sizing components. Shared across strategies, never copied.

Each takes a signal frame (and optionally the panel) and returns a target-weight
frame (timestamp index, symbol columns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ehf.research.backtest.panel import Panel
from ehf.research.signals import realized_vol


def normalize_weights(signals: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Scale each row so total absolute exposure equals ``gross`` (cash if flat).

    Fixed-fractional / equal-weight: every active name gets the same magnitude.
    """
    abs_sum = signals.abs().sum(axis=1)
    scaled = signals.div(abs_sum.where(abs_sum > 0, 1.0), axis=0) * gross
    return scaled.fillna(0.0)


def vol_target(
    signals: pd.DataFrame,
    panel: Panel,
    target_vol: float = 0.10,
    window: int = 63,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """Scale each leg toward a per-asset annualized ``target_vol``, then cap gross.

    The classic vol-targeting sleeve used by TSMOM and as an overlay.
    """
    asset_vol = realized_vol(panel.close, window, panel.periods_per_year)
    scale = (target_vol / asset_vol).replace([np.inf, -np.inf], np.nan)
    weights = (signals * scale).fillna(0.0)
    gross = weights.abs().sum(axis=1)
    over = gross > max_leverage
    weights.loc[over] = weights.loc[over].div(gross[over], axis=0) * max_leverage
    return weights.fillna(0.0)


def cross_sectional_rank(
    scores: pd.DataFrame,
    *,
    quantile: float = 0.2,
    gross: float = 1.0,
    long_short: bool = True,
) -> pd.DataFrame:
    """Rank the cross-section each bar; go long the top, short the bottom.

    The relative-value construction behind cross-sectional momentum/value: per
    bar, rank symbols by ``scores`` and equal-weight the top ``quantile`` long
    against the bottom ``quantile`` short. With ``long_short`` the two legs are
    balanced so the book nets ~zero (dollar-neutral) at gross exposure ``gross``;
    set it False for a long-only top-quantile tilt.

    Rows without a full pair of legs (e.g. a bar with too few ranked names, or
    long-only with no names) stay flat — we never run a half-hedged book. NaN
    scores are excluded from the ranking automatically.
    """
    # Percentile rank within each row; ``method="first"`` breaks ties by column
    # order so the leg sizes stay symmetric instead of bunching at a tie.
    ranks = scores.rank(axis=1, pct=True, method="first")
    longs = (ranks > 1.0 - quantile).astype(float)
    shorts = (ranks <= quantile).astype(float) if long_short else longs * 0.0

    n_long = longs.sum(axis=1)
    n_short = shorts.sum(axis=1)
    side = gross / 2.0 if long_short else gross  # split gross across the two legs

    long_w = longs.div(n_long.where(n_long > 0, 1.0), axis=0) * side
    short_w = shorts.div(n_short.where(n_short > 0, 1.0), axis=0) * side
    weights = (long_w - short_w).fillna(0.0)

    # Flat any bar that can't form the required leg(s): both for long/short,
    # just the long leg for long-only.
    incomplete = (n_long == 0) | (n_short == 0) if long_short else (n_long == 0)
    weights.loc[incomplete] = 0.0
    return weights
