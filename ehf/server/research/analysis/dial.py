"""Metrics that matter for an in/out dial, beyond Sharpe.

An overlay on one asset is judged on *what it saved* and *what it cost*: how
much of the time it held the market, how often it flipped, how deep it fell in
the episodes that actually hurt, and how it compares with simply holding. These
read stored runs (equity + trade ledger), so they work on the stitched
walk-forward curve and on a full-sample run alike.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

#: The stretches an investor remembers. Start a little before each peak so the
#: drawdown is measured from the top, not from an arbitrary January 1st.
CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "2008": ("2007-10-01", "2009-06-30"),
    "2011": ("2011-04-01", "2011-12-31"),
    "2018": ("2018-09-01", "2019-03-31"),
    "2020": ("2020-01-01", "2020-09-30"),
    "2022": ("2022-01-01", "2023-03-31"),
    "2025": ("2025-01-01", "2025-09-30"),
}


def drawdown_in(equity: pd.Series, start: str, end: str) -> float:
    """Max drawdown measured inside ``[start, end]`` from the peak inside that window."""
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    sub = equity.loc[(equity.index >= lo) & (equity.index <= hi)].dropna()
    if len(sub) < 2:
        return float("nan")
    return float((sub / sub.cummax() - 1.0).min())


def positions_from_trades(
    trades: pd.DataFrame,
    index: pd.DatetimeIndex,
    asset: str,
    resets: Iterable[pd.Timestamp] = (),
) -> pd.Series:
    """Rebuild the held weight in ``asset`` on every bar from the trade ledger.

    ``resets`` are bars where the book is known to start flat (walk-forward test
    windows open flat, and their first trade is the full opening position), so the
    cumulative sum restarts there instead of double counting.
    """
    pos = pd.Series(0.0, index=index)
    if trades.empty:
        return pos
    own = trades.loc[trades["symbol"] == asset]
    deltas = own.groupby("timestamp")["delta_weight"].sum().reindex(index).fillna(0.0)
    reset_set = {pd.Timestamp(r) for r in resets}
    held = 0.0
    out = np.zeros(len(index))
    for i, ts in enumerate(index):
        if ts in reset_set:
            held = 0.0
        held += float(deltas.iloc[i])
        out[i] = held
    return pd.Series(out, index=index)


def calendar_year_returns(net: pd.Series) -> pd.Series:
    """Compounded return per calendar year of a per-bar return series."""
    years = pd.DatetimeIndex(net.index).year
    return (1.0 + net).groupby(years).prod() - 1.0


def dial_metrics(
    net: pd.Series,
    equity: pd.Series,
    trades: pd.DataFrame,
    *,
    asset: str,
    market_net: pd.Series | None = None,
    resets: Iterable[pd.Timestamp] = (),
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """The dial scorecard: exposure, switches, crisis drawdowns, and the hold comparison."""
    net = net.dropna()
    idx = pd.DatetimeIndex(net.index)
    equity = equity.reindex(idx)
    pos = positions_from_trades(trades, idx, asset, resets)
    changes = pos.diff().abs() > 1e-9  # first bar is NaN -> the opening trade is not a switch
    reset_set = {pd.Timestamp(r) for r in resets}
    switches = int(sum(1 for ts, c in changes.items() if c and ts not in reset_set))

    out: dict[str, Any] = {
        "time_in_market": float(pos.mean()),
        "n_switches": switches,
        "switches_per_year": switches / (len(net) / periods_per_year) if len(net) else 0.0,
    }
    for name, (lo, hi) in CRISIS_WINDOWS.items():
        out[f"dd_{name}"] = drawdown_in(equity, lo, hi)

    years = calendar_year_returns(net)
    out["worst_year"] = float(years.min()) if not years.empty else float("nan")
    out["worst_year_label"] = str(years.idxmin()) if not years.empty else ""

    if market_net is not None:
        m = market_net.reindex(idx).fillna(0.0)
        m_eq = (1.0 + m).cumprod() * float(equity.iloc[0]) if len(equity) else m
        for name, (lo, hi) in CRISIS_WINDOWS.items():
            out[f"mkt_dd_{name}"] = drawdown_in(m_eq, lo, hi)
        m_years = calendar_year_returns(m)
        out["mkt_worst_year"] = float(m_years.min()) if not m_years.empty else float("nan")
        yrs = len(net) / periods_per_year
        g_s = float(equity.iloc[-1] / equity.iloc[0]) if len(equity) > 1 else 1.0
        g_m = float(m_eq.iloc[-1] / m_eq.iloc[0]) if len(m_eq) > 1 else 1.0
        out["cagr"] = g_s ** (1.0 / yrs) - 1.0 if yrs > 0 and g_s > 0 else float("nan")
        out["mkt_cagr"] = g_m ** (1.0 / yrs) - 1.0 if yrs > 0 and g_m > 0 else float("nan")
        out["mkt_max_drawdown"] = float((m_eq / m_eq.cummax() - 1.0).min())
        sd = float(m.std())
        out["mkt_sharpe"] = float(m.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
        # share of calendar years the dial beat holding
        both = pd.concat({"s": years, "m": m_years}, axis=1).dropna()
        out["years_beating_market"] = (
            float((both["s"] > both["m"]).mean()) if len(both) else float("nan")
        )
    return out
