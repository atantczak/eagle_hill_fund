"""Shared fixtures: synthetic FMP-shaped data and canonical bars (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ehf.server.research.data.fmp_provider import normalize_fmp_records
from ehf.server.research.data.schema import Timeframe


@pytest.fixture
def fmp_daily_records() -> pd.DataFrame:
    """Two symbols of raw FMP-shaped daily rows (US/Eastern naive dates)."""
    dates = pd.bdate_range("2024-01-02", periods=10).strftime("%Y-%m-%d")
    rng = np.random.default_rng(0)
    rows = []
    for sym in ("AAPL", "MSFT"):
        base = 100.0 if sym == "AAPL" else 300.0
        for d in dates:
            o = base + rng.normal(0, 1)
            c = o + rng.normal(0, 1)
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "open": o,
                    "high": max(o, c) + 0.5,
                    "low": min(o, c) - 0.5,
                    "close": c,
                    "volume": float(rng.integers(1_000_000, 5_000_000)),
                    "vwap": (o + c) / 2,  # extra vendor column that must be dropped
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def canonical_bars(fmp_daily_records: pd.DataFrame) -> pd.DataFrame:
    return normalize_fmp_records(fmp_daily_records, timeframe=Timeframe.DAY, adjusted=True)


def make_daily_bars(
    close_paths: dict[str, list[float]],
    start: str = "2020-01-02",
    open_from_prev_close: bool = True,
) -> pd.DataFrame:
    """Build canonical daily bars from per-symbol close paths (test helper).

    ``open[t] = close[t-1]`` (a typical overnight gap) when
    ``open_from_prev_close``, else ``open[t] = close[t]``.
    """
    n = len(next(iter(close_paths.values())))
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    rows = []
    for sym, closes in close_paths.items():
        prev = closes[0]
        for d, c in zip(dates, closes, strict=True):
            o = prev if open_from_prev_close else c
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "open": o,
                    "high": max(o, c) + 0.01,
                    "low": min(o, c) - 0.01,
                    "close": c,
                    "volume": 1_000_000.0,
                }
            )
            prev = c
    raw = pd.DataFrame(rows)
    return normalize_fmp_records(raw, timeframe=Timeframe.DAY, adjusted=True)
