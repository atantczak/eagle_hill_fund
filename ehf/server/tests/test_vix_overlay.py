"""The VIX overlay: dataset assembly, signal primitives, dial helpers, the twelve strategies.

No network: SKEW is parsed from an inline CSV, and the panel is synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ehf.server.research import signals
from ehf.server.research.analysis.dial import dial_metrics, positions_from_trades
from ehf.server.research.backtest.engine import run_backtest
from ehf.server.research.backtest.panel import Panel
from ehf.server.research.config.spec import RunSpec, UniverseSpec
from ehf.server.research.data.cboe import parse_cboe_csv
from ehf.server.research.data.vix import CASH, assemble_vix_bars, cash_bars_from_yield
from ehf.server.research.strategies.base import build_strategy
from ehf.server.research.strategies.dial import (
    band_rebalance,
    confirm,
    cooldown_gate,
    hysteresis_gate,
    min_dwell,
    three_state,
)
from ehf.server.research.strategies.vix_overlay import GRIDS, Vote
from ehf.server.tests.conftest import make_daily_bars

SYMS = ("SPY", "CASH", "^VIX", "^VIX3M", "^VVIX", "^SKEW")


def _paths(n: int = 700, seed: int = 7) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.012, n)
    spy = 100 * np.exp(np.cumsum(r))
    # VIX reacts to SPY the way it does: up on down days, mean-reverting toward 20
    v = np.empty(n)
    v[0] = 20.0
    for i in range(1, n):
        v[i] = max(
            9.0, v[i - 1] + 0.1 * (20 - v[i - 1]) - 4.5 * v[i - 1] * r[i] + rng.normal(0, 0.6)
        )
    vix3m = v * 1.05 + rng.normal(0, 0.3, n)
    vvix = 90 + 0.8 * (v - 20) + rng.normal(0, 3, n)
    skew = 130 + rng.normal(0, 4, n)
    cash = 100 * (1 + 0.04 / 252) ** np.arange(n)
    return {
        "SPY": list(spy),
        "CASH": list(cash),
        "^VIX": list(v),
        "^VIX3M": list(vix3m),
        "^VVIX": list(vvix),
        "^SKEW": list(skew),
    }


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return make_daily_bars(_paths())


@pytest.fixture(scope="module")
def panel(bars: pd.DataFrame) -> Panel:
    return Panel.from_bars(bars)


# --- dial helpers -------------------------------------------------------------------


def test_hysteresis_gate_dead_band_and_nan() -> None:
    x = pd.Series([0.9, 1.02, 0.98, 0.96, 0.94, np.nan, 0.94, 1.5])
    g = hysteresis_gate(x, out_above=1.0, in_below=0.95)
    # in, out on breach, stays out inside the band, back in below 0.95, NaN=flat w/o forgetting
    assert g.tolist() == [1, 0, 0, 0, 1, 0, 1, 0]


def test_hysteresis_gate_clamps_misordered_thresholds() -> None:
    x = pd.Series([0.5, 1.5, 0.5])
    g = hysteresis_gate(x, out_above=1.0, in_below=2.0)  # in_below > out_above -> clamp
    assert g.tolist() == [1, 0, 1]


def test_three_state_both_polarities() -> None:
    x = pd.Series([0.0, 1.0, 2.0, np.nan])
    assert three_state(x, lo=0.5, hi=1.5).tolist() == [1.0, 0.5, 0.0, 0.0]
    assert three_state(x, lo=0.5, hi=1.5, high_is_risk=False).tolist() == [0.0, 0.5, 1.0, 0.0]


def test_cooldown_gate_counts_bars() -> None:
    trig = pd.Series([False, True, False, False, False, True, False])
    assert cooldown_gate(trig, 3).tolist() == [1, 0, 0, 0, 1, 0, 0]
    assert cooldown_gate(trig, 1).tolist() == [1, 0, 1, 1, 1, 0, 1]


def test_min_dwell_holds_new_state() -> None:
    e = pd.Series([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert min_dwell(e, 1).tolist() == e.tolist()
    assert min_dwell(e, 3).tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1]


def test_band_rebalance_is_lazy() -> None:
    t = pd.Series([0.5, 0.53, 0.58, 0.7, np.nan, 0.2])
    assert band_rebalance(t, 0.1).tolist() == [0.5, 0.5, 0.5, 0.7, 0.0, 0.2]


def test_confirm_requires_consecutive_bars() -> None:
    c = pd.Series([True, True, False, True, True, True])
    assert confirm(c, 2).tolist() == [False, True, False, False, True, True]


# --- signals -------------------------------------------------------------------------


def test_vix_signals_shapes_and_ranges(panel: Panel) -> None:
    spy, vix, vix3m = panel.close["SPY"], panel.close["^VIX"], panel.close["^VIX3M"]
    ratio = signals.term_structure_ratio(vix, vix3m)
    assert ratio.index.equals(spy.index) and 0.5 < ratio.dropna().median() < 1.2
    sr = signals.surprise_rate(spy, vix, 21).dropna()
    assert ((sr >= 0) & (sr <= 1)).all()
    corr = signals.vix_spy_correlation(spy, vix, 21).dropna()
    assert corr.median() < -0.3  # the synthetic VIX is built to move against SPY
    z = signals.residual_zscore(spy, vix, 63).dropna()
    assert abs(z.mean()) < 0.5 and 0.5 < z.std() < 2.0
    assert signals.implied_daily_move(pd.Series([16.0])).iloc[0] == pytest.approx(0.01008, rel=1e-3)


def test_vix_signals_are_causal(panel: Panel) -> None:
    spy, vix = panel.close["SPY"].copy(), panel.close["^VIX"].copy()
    before = {
        "resid": signals.residual_zscore(spy, vix, 63),
        "vrp": signals.variance_risk_premium(vix, spy, 21),
        "surprise": signals.surprise_rate(spy, vix, 21),
        "corr": signals.vix_spy_correlation(spy, vix, 21),
    }
    cut = 500
    spy2, vix2 = spy.copy(), vix.copy()
    spy2.iloc[cut:] *= 0.5
    vix2.iloc[cut:] *= 3.0
    after = {
        "resid": signals.residual_zscore(spy2, vix2, 63),
        "vrp": signals.variance_risk_premium(vix2, spy2, 21),
        "surprise": signals.surprise_rate(spy2, vix2, 21),
        "corr": signals.vix_spy_correlation(spy2, vix2, 21),
    }
    for k in before:
        pd.testing.assert_series_equal(before[k].iloc[:cut], after[k].iloc[:cut])


# --- dataset assembly ------------------------------------------------------------------


def test_assemble_vix_bars_aligns_and_accrues() -> None:
    n = 30
    spy = make_daily_bars({"SPY": list(100 + np.arange(n, dtype=float))})
    sessions = spy["timestamp"]
    # VIX with a gap (missing the 5th session) and ^IRX at a constant 5%
    idx_rows = make_daily_bars({"^VIX": [20.0] * n, "^IRX": [5.0] * n})
    fifth = sessions.iloc[4]
    idx_rows = idx_rows[~((idx_rows["symbol"] == "^VIX") & (idx_rows["timestamp"] == fifth))]
    idx_rows = idx_rows.reset_index(drop=True)
    idx_rows.attrs.update(spy.attrs)
    # SKEW keyed by naive session date, with one date missing
    dates = pd.DatetimeIndex(
        sessions.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    )
    skew = pd.Series(130.0, index=dates).drop(dates[10])

    out = assemble_vix_bars(spy, idx_rows, skew)
    assert set(out["symbol"]) == {"SPY", "^VIX", "^IRX", "^SKEW", CASH}
    per = out.groupby("symbol")["timestamp"].nunique()
    assert (per == n).all()  # everyone on SPY's sessions
    vix = out[out["symbol"] == "^VIX"].set_index("timestamp")["close"]
    assert vix.loc[fifth] == 20.0  # forward-filled over the gap
    cash = out[out["symbol"] == CASH].set_index("timestamp")["close"]
    assert cash.iloc[-1] / cash.iloc[0] == pytest.approx((1 + 0.05 / 252) ** (n - 1), rel=1e-9)
    assert (out[out["symbol"] == "^SKEW"]["close"] == 130.0).all()


def test_cash_bars_clip_bad_yield_prints() -> None:
    idx = pd.date_range("2020-01-01", periods=5, tz="UTC")
    cash = cash_bars_from_yield(pd.Series([2.0, -1.0, 90.0, np.nan, 2.0], index=idx))
    assert (cash["close"].diff().dropna() >= 0).all()
    assert cash["close"].iloc[-1] < 100 * (1 + 25 / 100 / 252) ** 4 + 1e-9


def test_parse_cboe_csv() -> None:
    s = parse_cboe_csv("DATE,SKEW\n01/02/1990,126.09\n01/03/1990,123.34\nbad,1\n", "SKEW")
    assert list(s.index.strftime("%Y-%m-%d")) == ["1990-01-02", "1990-01-03"]
    assert s.iloc[0] == 126.09


# --- strategies through the engine ---------------------------------------------------


@pytest.mark.parametrize("key", sorted(GRIDS))
def test_every_dial_strategy_runs_and_weights_sum_to_one(
    key: str, bars: pd.DataFrame, panel: Panel
) -> None:
    strat = build_strategy(key, {})
    w = strat.target_weights(panel)
    assert list(w.columns) == panel.symbols
    assert ((w["SPY"] >= -1e-12) & (w["SPY"] <= 1 + 1e-12)).all()
    assert np.allclose(w["SPY"] + w["CASH"], 1.0)
    assert (w.drop(columns=["SPY", "CASH"]).abs().sum().sum()) == 0.0
    spec = RunSpec(
        strategy=key, universe=UniverseSpec(symbols=SYMS), start="2021-01-01", end="2022-09-01"
    )
    res = run_backtest(spec, bars)
    assert len(res.equity) > 200 and np.isfinite(res.metrics["sharpe"])


def test_hold_is_fully_invested_and_sma_switches(panel: Panel) -> None:
    hold = build_strategy("dial_hold", {}).target_weights(panel)
    assert (hold["SPY"] == 1.0).all()
    sma = build_strategy("dial_sma", {"window": 50}).target_weights(panel)
    assert (
        set(sma["SPY"].unique()) <= {0.0, 1.0} and sma["SPY"].iloc[:49].sum() == 0
    )  # warm-up flat


def test_vote_is_mean_of_three_rules(panel: Panel) -> None:
    v = Vote(term_out_above=1.02, resid_warn_z=1.8, vrp_out_below=1.0)
    e = v.exposure(panel)
    parts = [
        build_strategy("vix_term", {"out_above": 1.02}).exposure(panel),
        build_strategy("vix_resid", {"warn_z": 1.8}).exposure(panel),
        build_strategy("vix_vrp", {"out_below": 1.0}).exposure(panel),
    ]
    pd.testing.assert_series_equal(e, pd.concat(parts, axis=1).mean(axis=1))


def test_min_dwell_param_reduces_switches(panel: Panel) -> None:
    fast = build_strategy("vix_vrp", {"min_dwell": 1}).target_weights(panel)["SPY"]
    slow = build_strategy("vix_vrp", {"min_dwell": 10}).target_weights(panel)["SPY"]
    assert (slow.diff().abs() > 0).sum() < (fast.diff().abs() > 0).sum()


def test_missing_signal_column_is_a_clear_error(bars: pd.DataFrame) -> None:
    sub = bars[bars["symbol"].isin(["SPY", "CASH", "^VIX"])].reset_index(drop=True)
    sub.attrs.update(bars.attrs)
    with pytest.raises(KeyError, match="VIX3M"):
        build_strategy("vix_term", {}).target_weights(Panel.from_bars(sub))


# --- dial metrics ----------------------------------------------------------------------


def test_positions_from_trades_respects_resets() -> None:
    idx = pd.date_range("2024-01-01", periods=6, tz="UTC")
    trades = pd.DataFrame(
        {
            "timestamp": [idx[0], idx[2], idx[3], idx[5]],
            "symbol": ["SPY"] * 4,
            "delta_weight": [1.0, -1.0, 1.0, -0.5],  # window 2 opens flat at idx[3]
            "side": ["buy", "sell", "buy", "sell"],
            "price": [1.0] * 4,
        }
    )
    pos = positions_from_trades(trades, idx, "SPY", resets=[idx[3]])
    assert pos.tolist() == [1.0, 1.0, 0.0, 1.0, 1.0, 0.5]


def test_dial_metrics_time_in_market_and_crisis_dd(bars: pd.DataFrame) -> None:
    spec = RunSpec(
        strategy="dial_hold",
        universe=UniverseSpec(symbols=SYMS),
        start="2020-03-01",
        end="2022-09-01",
    )
    res = run_backtest(spec, bars)
    eq = res.equity.set_index("timestamp")
    mkt = bars[bars["symbol"] == "SPY"].set_index("timestamp")["close"].pct_change()
    d = dial_metrics(eq["net_return"], eq["equity"], res.trades, asset="SPY", market_net=mkt)
    assert d["time_in_market"] == pytest.approx(1.0)
    assert d["n_switches"] == 0  # the opening buy is not a decision change
    assert d["dd_2020"] <= 0 and np.isfinite(d["mkt_dd_2020"])
    assert d["dd_2020"] == pytest.approx(d["mkt_dd_2020"], abs=0.02)  # hold ~ market
