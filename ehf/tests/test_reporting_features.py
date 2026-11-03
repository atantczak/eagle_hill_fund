"""Pure feature / market-context tests for the daily report (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ehf.research.reporting import market as mk
from ehf.research.reporting.features import FEATURE_COLUMNS, price_features, reaction_since


def make_bars(symbols: dict[str, np.ndarray], start: str = "2025-01-02") -> pd.DataFrame:
    """Long daily bars from per-symbol close paths (open=prev close, hi/lo around close)."""
    rows = []
    for sym, closes in symbols.items():
        dates = pd.bdate_range(start, periods=len(closes))
        prev = np.concatenate([[closes[0]], closes[:-1]])
        for d, o, c in zip(dates, prev, closes, strict=True):
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "open": o,
                    "high": max(o, c) * 1.01,
                    "low": min(o, c) * 0.99,
                    "close": c,
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def bars() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 300
    flat = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    trend = 100.0 * np.exp(np.linspace(0, 0.8, n))  # monotonic uptrend -> new highs
    crash = flat.copy()
    crash[-1] = crash[-2] * 0.85  # -15% on the last day
    return make_bars({"FLAT": flat, "TREND": trend, "CRASH": crash})


def test_feature_table_shape_and_columns(bars: pd.DataFrame) -> None:
    f = price_features(bars)
    assert list(f.columns) == FEATURE_COLUMNS
    assert set(f.index) == {"FLAT", "TREND", "CRASH"}
    assert (f["n_bars"] == 300).all()
    assert f["last_date"].eq(f["last_date"].iloc[0]).all()


def test_trend_symbol_is_a_new_high_and_above_averages(bars: pd.DataFrame) -> None:
    f = price_features(bars)
    t = f.loc["TREND"]
    assert bool(t["new_high_252"]) is True
    assert bool(t["above_200"]) and bool(t["above_50"])
    assert t["mom_12_1"] > 0.3
    assert abs(t["pct_from_high_252"]) < 1e-9


def test_crash_symbol_is_an_outlier(bars: pd.DataFrame) -> None:
    f = price_features(bars)
    c = f.loc["CRASH"]
    assert c["ret_1d"] == pytest.approx(-0.15, abs=1e-9)
    assert c["z_1d"] < -5  # 15% drop vs ~1% daily vol
    assert c["dd_from_high_63"] < -0.1
    assert c["vol_ratio_20"] == pytest.approx(1.0)


def test_empty_bars_give_empty_table() -> None:
    f = price_features(
        pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    )
    assert f.empty and list(f.columns) == FEATURE_COLUMNS


def test_reaction_since_anchors_on_close_before_event(bars: pd.DataFrame) -> None:
    last = bars["date"].max()
    r = reaction_since(bars, "CRASH", last)  # event on the crash day -> base is prior close
    assert r == pytest.approx(-0.15, abs=1e-9)
    assert reaction_since(bars, "CRASH", pd.Timestamp("2020-01-01")) is None  # no close before


def test_index_context_and_regime(bars: pd.DataFrame) -> None:
    spy = bars[bars["symbol"] == "TREND"]
    ctx = mk.index_context(spy)
    assert ctx["above_sma200"] is True and ctx["regime"].startswith("risk-on")
    assert mk.classify_regime(
        {"above_sma200": False, "realized_vol_20d_pctile_1y": 0.95}
    ).startswith("risk-off")
    assert mk.classify_regime({}) == "unknown"
    assert mk.index_context(spy.iloc[0:0]) == {}


def test_breadth_counts(bars: pd.DataFrame) -> None:
    b = mk.breadth(price_features(bars))
    assert b["n_symbols"] == 3
    assert b["new_52w_highs"] >= 1
    assert 0.0 <= b["pct_above_sma200"] <= 1.0


def test_treasury_sector_econ_helpers() -> None:
    tr = mk.treasury_context([{"date": "2026-10-30", "month3": 3.9, "year2": 3.5, "year10": 4.2}])
    assert tr["slope_10y_2y"] == pytest.approx(0.7) and tr["slope_10y_3m"] == pytest.approx(0.3)
    assert mk.treasury_context([]) == {}
    secs = mk.sector_context(
        [
            {"sector": "Energy", "exchange": "NASDAQ", "averageChange": -1.0},
            {"sector": "Tech", "exchange": "NASDAQ", "averageChange": 2.0},
            {"sector": "Tech", "exchange": "NYSE", "averageChange": 9.0},
        ]
    )
    assert [s["sector"] for s in secs] == ["Tech", "Energy"] and secs[0]["avg_change_pct"] == 2.0
    econ = mk.econ_context(
        [
            {"date": "2026-11-05", "country": "US", "impact": "High", "event": "CPI"},
            {"date": "2026-11-03", "country": "US", "impact": "Low", "event": "noise"},
            {"date": "2026-11-04", "country": "DE", "impact": "High", "event": "ZEW"},
        ]
    )
    assert [e["event"] for e in econ] == ["CPI"]


def test_earnings_ahead_filters_small_caps_and_past_dates() -> None:
    uni = pd.DataFrame(
        {"symbol": ["BIG", "SMALL"], "name": ["Big Co", "Small Co"], "market_cap": [50e9, 1e9]}
    )
    rows = [
        {"symbol": "BIG", "date": "2026-11-05", "epsEstimated": 1.0},
        {"symbol": "SMALL", "date": "2026-11-05", "epsEstimated": 1.0},
        {"symbol": "BIG", "date": "2026-10-20", "epsEstimated": 1.0},  # already happened
    ]
    out = mk.earnings_ahead(rows, uni, after=pd.Timestamp("2026-10-31").date())
    assert [e["symbol"] for e in out] == ["BIG"] and out[0]["name"] == "Big Co"
