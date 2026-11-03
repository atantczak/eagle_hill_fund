"""Regime breakdown — performance conditioned on market state.

An edge that only exists in one regime (e.g. only in calm bull markets) is a
different animal from one that's broad-based. This conditions a strategy's
returns on a market proxy's trend and volatility state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_regime(
    market_returns: pd.Series, trend_window: int = 63, vol_window: int = 21
) -> pd.DataFrame:
    """Label each period bull/bear/chop and calm/normal/turbulent from a proxy.

    Trend: sign of the trailing ``trend_window`` cumulative return (with a
    dead-band near zero = chop). Vol: terciles of trailing realized vol.
    """
    cum = (1.0 + market_returns).rolling(trend_window).apply(np.prod, raw=True) - 1.0
    trend = pd.Series("chop", index=market_returns.index, dtype=object)
    trend[cum > 0.02] = "bull"
    trend[cum < -0.02] = "bear"

    vol = market_returns.rolling(vol_window).std()
    q1, q2 = vol.quantile([1 / 3, 2 / 3])
    volstate = pd.Series("normal", index=market_returns.index, dtype=object)
    volstate[vol <= q1] = "calm"
    volstate[vol >= q2] = "turbulent"
    return pd.DataFrame({"trend": trend, "vol": volstate})


def regime_breakdown(
    returns: pd.Series,
    market_returns: pd.Series | None = None,
    *,
    periods_per_year: float = 252.0,
) -> pd.DataFrame:
    """Per-regime annualized return, vol, Sharpe, and time share.

    ``market_returns`` is the conditioning proxy (e.g. SPY); if omitted, the
    strategy conditions on its own returns (less independent, but still shows
    where its P&L lives).
    """
    proxy = market_returns if market_returns is not None else returns
    regimes = classify_regime(proxy).reindex(returns.index)
    out = {}
    for dim in ("trend", "vol"):
        for state, grp in returns.groupby(regimes[dim]):
            grp = grp.dropna()
            if grp.empty:
                continue
            sd = grp.std()
            out[f"{dim}:{state}"] = {
                "ann_return": float(grp.mean() * periods_per_year),
                "ann_vol": float(sd * np.sqrt(periods_per_year)),
                "sharpe": float(grp.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
                "time_share": float(len(grp) / len(returns.dropna())),
            }
    return pd.DataFrame(out).T
