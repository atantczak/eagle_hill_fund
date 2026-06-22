"""Pluggable position-sizing components. Shared across strategies, never copied.

Each takes a signal frame (and optionally the panel) and returns a target-weight
frame (timestamp index, symbol columns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ehf.server.research.backtest.panel import Panel
from ehf.server.research.signals import realized_vol


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
