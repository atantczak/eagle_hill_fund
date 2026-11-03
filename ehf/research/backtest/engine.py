"""Cost-aware, look-ahead-safe vectorized backtest engine (the custom approach).

Contract: a ``Strategy`` produces a **target-weight panel** ``w`` where ``w[t,
sym]`` is the fraction of equity to hold in ``sym``, decided using data through
the close of bar ``t``. The engine enforces the fill model, charges costs on
turnover, and produces equity curve + trade ledger + core metrics.

Look-ahead safety is structural: the position held over the interval that begins
at bar ``t`` is ``w.shift(1)`` — i.e. the most recent weight decided *before*
that interval. A test (`tests/test_engine.py::test_no_lookahead`) proves a future
price move cannot change a prior bar's P&L.

Evaluation window: weights are computed on *every* bar passed in, but P&L,
costs, trades and metrics are measured only inside ``[spec.start, spec.end]``.
Pass extra history before ``spec.start`` as signal warm-up (the walk-forward
harness does) — it can never leak, because the first in-window position is still
the weight decided before that bar, and the window starts flat: establishing the
opening position is charged like any other trade.

Fill models:
  * ``NEXT_OPEN``  (default, safe): trade at next bar's open; P&L measured
    open-to-open over the holding interval.
  * ``NEXT_CLOSE``: position from the prior close decision earns close-to-close.
  * ``SAME_CLOSE`` (DEBUG ONLY): weights earn the *same* bar's return — a
    deliberate look-ahead leak, kept only to demonstrate/measure the bias.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from ehf.research.backtest.metrics import core_metrics, equity_from_returns
from ehf.research.backtest.panel import Panel
from ehf.research.config.spec import CostConfig, FillModel, RunSpec
from ehf.research.results.registry import RunResult
from ehf.research.strategies.base import Strategy, build_strategy


def run_backtest(spec: RunSpec, bars: pd.DataFrame, strategy: Strategy | None = None) -> RunResult:
    """Run a single backtest from a ``RunSpec`` + canonical bars.

    ``strategy`` may be passed directly; otherwise it's built from
    ``spec.strategy`` + ``spec.params`` via the strategy registry.
    """
    panel = Panel.from_bars(bars)
    strat = strategy if strategy is not None else build_strategy(spec.strategy, spec.params)

    weights = strat.target_weights(panel)
    weights = weights.reindex(index=panel.index, columns=panel.symbols).fillna(0.0)

    gross, position = _holding_returns(weights, panel, spec.fill_model)

    # Restrict accounting to the spec's evaluation window (bars before it are
    # warm-up only). Inside the window the book starts flat, so the opening
    # position is a real, cost-bearing trade.
    lo, hi = evaluation_window(spec)
    in_window = (panel.index >= lo) & (panel.index < hi)
    if not in_window.any():
        raise ValueError(
            f"no bars inside the evaluation window {spec.start}..{spec.end} "
            f"(bars span {panel.index[0].date()}..{panel.index[-1].date()})"
        )
    panel = panel.slice(lo, hi)
    position = position.loc[in_window]
    gross = gross.loc[in_window]

    costs = _costs(position, panel, spec.costs)
    net = (gross - costs).fillna(0.0)
    # A single period can't lose more than 100% of equity (no negative equity;
    # ruin is absorbing). A short position into a violent up-move — or a bad data
    # print — can otherwise drive net < -1 and flip the cumprod negative.
    net = net.clip(lower=-1.0)

    equity = equity_from_returns(net, spec.initial_capital)
    trades = _trade_ledger(position, panel, spec.fill_model)
    metrics = core_metrics(net, equity["equity"], trades, panel.periods_per_year)
    return RunResult(metrics=metrics, equity=equity, trades=trades)


def evaluation_window(spec: RunSpec) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``[lo, hi)`` UTC bounds of the spec's accounting window (``end`` inclusive)."""
    lo = pd.Timestamp(spec.start, tz="UTC")
    hi = pd.Timestamp(spec.end, tz="UTC") + pd.Timedelta(days=1)
    return lo, hi


def _position_deltas(position: pd.DataFrame) -> pd.DataFrame:
    """Per-bar change in held weights; the first bar is a move from flat."""
    delta = position.diff()
    delta.iloc[0] = position.iloc[0]
    return delta


def _holding_returns(
    weights: pd.DataFrame, panel: Panel, fill: FillModel
) -> tuple[pd.Series, pd.DataFrame]:
    """Return (portfolio gross return per bar, position-held matrix per bar)."""
    if fill is FillModel.NEXT_OPEN:
        # Position over interval starting at open[t] is the weight decided at the
        # prior close. Asset return over that interval is open-to-open.
        position = weights.shift(1)
        asset_ret = panel.open.shift(-1) / panel.open - 1.0
    elif fill is FillModel.NEXT_CLOSE:
        position = weights.shift(1)
        asset_ret = panel.close.pct_change()
    elif fill is FillModel.SAME_CLOSE:  # DEBUG: look-ahead leak by construction
        position = weights
        asset_ret = panel.close.pct_change()
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"unknown fill model {fill!r}")

    gross = (position * asset_ret).sum(axis=1, min_count=1).fillna(0.0)
    return gross, position.fillna(0.0)


def _costs(position: pd.DataFrame, panel: Panel, cfg: CostConfig) -> pd.Series:
    """Per-bar cost as a fraction of equity (slippage on turnover + borrow)."""
    if cfg.frictionless:
        return pd.Series(0.0, index=position.index)

    deltas = _position_deltas(position).abs()
    turnover = deltas.sum(axis=1).fillna(0.0)
    slippage = turnover * (cfg.slippage_bps / 1e4)

    # Commission converted to a return drag: shares ≈ turnover_notional / price.
    if cfg.commission_per_share > 0:
        notional_turnover = deltas * panel.close
        shares = (notional_turnover / panel.close).sum(axis=1).fillna(0.0)
        commission = shares * cfg.commission_per_share / panel.close.mean(axis=1)
    else:
        commission = pd.Series(0.0, index=position.index)

    short_exposure = (-position.clip(upper=0.0)).sum(axis=1).fillna(0.0)
    borrow = short_exposure * (cfg.borrow_bps_annual / 1e4) / panel.periods_per_year

    return (slippage + commission + borrow).fillna(0.0)


def _trade_ledger(position: pd.DataFrame, panel: Panel, fill: FillModel) -> pd.DataFrame:
    """One row per (bar, symbol) where the held position changes."""
    delta = _position_deltas(position)
    fill_price = panel.open if fill is FillModel.NEXT_OPEN else panel.close
    rows: list[dict[str, object]] = []
    for ts, row in delta.iterrows():
        changed = row[row.abs() > 1e-12]
        for sym, dw in changed.items():
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "delta_weight": float(dw),
                    "side": "buy" if dw > 0 else "sell",
                    "price": float(cast(float, fill_price.at[ts, sym])),
                }
            )
    cols = ["timestamp", "symbol", "delta_weight", "side", "price"]
    return pd.DataFrame(rows, columns=cols)
