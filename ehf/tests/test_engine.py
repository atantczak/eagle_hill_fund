"""Engine: determinism, costs, look-ahead safety, and the reference strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ehf.research.backtest.engine import run_backtest
from ehf.research.backtest.panel import Panel
from ehf.research.config.spec import CostConfig, FillModel, RunSpec, UniverseSpec
from ehf.research.strategies.base import Strategy
from ehf.tests.conftest import make_daily_bars


class AlwaysLong(Strategy):
    """Test-only strategy: hold a full long in every symbol every bar."""

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        return pd.DataFrame(1.0, index=panel.index, columns=panel.symbols)


class AlwaysShort(Strategy):
    """Test-only strategy: hold a full short every bar (can be ruined by a spike)."""

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        return pd.DataFrame(-1.0, index=panel.index, columns=panel.symbols)


def _spec(fill: FillModel = FillModel.NEXT_OPEN, **kw) -> RunSpec:
    base = dict(
        strategy="always_long",
        universe=UniverseSpec(symbols=("A",)),
        start="2020-01-01",
        end="2025-01-01",
        fill_model=fill,
    )
    base.update(kw)
    return RunSpec(**base)


def _trend_bars(n: int = 60) -> pd.DataFrame:
    path = list(100 * np.cumprod(1 + np.full(n, 0.01)))  # steady uptrend
    return make_daily_bars({"A": path})


def test_engine_is_deterministic() -> None:
    bars = _trend_bars()
    e1 = run_backtest(_spec(), bars, AlwaysLong())
    e2 = run_backtest(_spec(), bars, AlwaysLong())
    pd.testing.assert_frame_equal(e1.equity, e2.equity)


def test_metrics_present() -> None:
    res = run_backtest(_spec(), _trend_bars(), AlwaysLong())
    for key in ("sharpe", "max_drawdown", "cagr", "ann_vol", "total_return", "n_trades"):
        assert key in res.metrics


def test_costs_reduce_returns() -> None:
    bars = _trend_bars()
    frictionless = run_backtest(_spec(costs=CostConfig(frictionless=True)), bars, AlwaysLong())
    with_costs = run_backtest(_spec(costs=CostConfig(slippage_bps=10.0)), bars, AlwaysLong())
    assert with_costs.metrics["total_return"] <= frictionless.metrics["total_return"]


def test_no_lookahead_next_open() -> None:
    """Under NEXT_OPEN, altering the FINAL bar's close cannot change equity."""
    bars = _trend_bars()
    base = run_backtest(_spec(FillModel.NEXT_OPEN), bars, AlwaysLong())

    bumped = bars.copy()
    last_mask = bumped["timestamp"] == bumped["timestamp"].max()
    bumped.loc[last_mask, "close"] *= 2.0  # a fat future move
    bumped.attrs.update(bars.attrs)
    after = run_backtest(_spec(FillModel.NEXT_OPEN), bumped, AlwaysLong())

    pd.testing.assert_series_equal(base.equity["equity"], after.equity["equity"])


def test_same_close_leaks_by_construction() -> None:
    """The DEBUG SAME_CLOSE fill DOES leak — proving the test above is meaningful."""
    bars = _trend_bars()
    base = run_backtest(_spec(FillModel.SAME_CLOSE), bars, AlwaysLong())

    bumped = bars.copy()
    last_mask = bumped["timestamp"] == bumped["timestamp"].max()
    bumped.loc[last_mask, "close"] *= 2.0
    bumped.attrs.update(bars.attrs)
    after = run_backtest(_spec(FillModel.SAME_CLOSE), bumped, AlwaysLong())

    assert base.equity["equity"].iloc[-1] != after.equity["equity"].iloc[-1]


def test_ruin_does_not_crash_metrics() -> None:
    """A short into a >100% up-move must not raise (was: complex-number CAGR)."""
    # Price spikes 100 -> 2000 in one bar (a +1900% squeeze / bad print).
    path = [100.0, 100.0, 100.0, 2000.0, 2000.0, 2000.0, 2000.0]
    bars = make_daily_bars({"A": path})
    res = run_backtest(_spec(), bars, AlwaysShort())

    m = res.metrics
    assert all(isinstance(v, float) and not isinstance(v, complex) for v in m.values())
    assert m["cagr"] == -1.0  # wiped out
    assert (res.equity["equity"] >= 0).all()  # no negative equity
    assert m["max_drawdown"] >= -1.0


def test_reference_strategies_run_and_register() -> None:
    """Both Rise/Fall and TSMOM build from the registry and produce equity."""
    rng = np.random.default_rng(7)
    path_a = list(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 300))))
    path_b = list(80 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, 300))))
    bars = make_daily_bars({"A": path_a, "B": path_b})

    rf = run_backtest(
        RunSpec(
            strategy="rise_fall",
            params={"trend_window": 20, "atr_window": 10, "max_hold": 10},
            universe=UniverseSpec(symbols=("A", "B")),
            start="2020-01-01",
            end="2025-01-01",
        ),
        bars,
    )
    assert len(rf.equity) == 300
    assert rf.metrics["n_bars"] > 0

    ts = run_backtest(
        RunSpec(
            strategy="tsmom",
            params={"lookback": 60, "vol_window": 20},
            universe=UniverseSpec(symbols=("A", "B")),
            start="2020-01-01",
            end="2025-01-01",
        ),
        bars,
    )
    assert len(ts.equity) == 300
    assert "sharpe" in ts.metrics
