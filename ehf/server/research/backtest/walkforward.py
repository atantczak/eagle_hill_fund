"""Walk-forward harness — the out-of-sample truth serum.

A parameter sweep tells you the best params *in hindsight*. Walk-forward asks
the only question that matters: had you re-optimized on the past as time went
by and traded the winner forward, what would you have earned? Every window is a
real experiment logged to the registry, so the whole study is reproducible and
queryable.

    window i:  [ warm-up | train  (optimize: sweep) ][ test  (validate: best cell) ]
    window i+1:          [ warm-up | train ...                ][ test ... ]
    ...
    stitched OOS curve = test_0 + test_1 + ... + test_n  (non-overlapping)

Design (DRY on purpose):
  * optimize  = :func:`ehf.server.research.backtest.sweep.run_sweep` on the train slice
  * validate  = :func:`ehf.server.research.backtest.engine.run_backtest` on the test slice
  * the engine's evaluation window does the warm-up/no-leak work; this module
    only decides *which bars* and *which params* go into each call.
  * the stitched out-of-sample curve is logged as its own run (``status =
    "walk_forward"``) whose spec carries the grid + walk-forward settings, so
    ``registry.get_run(hash)`` regenerates it like any other run.

Identifiers (all deterministic from the inputs):
  ``wf_id``                 ``wf_<strategy>_<8-hex>``        the study
  ``<wf_id>:w03:train``     sweep_id of window 3's optimize sweep
  ``<wf_id>:oos``           sweep_id shared by every test-window run
  ``<wf_id>:oos:x2``        same, at 2x costs (stress)
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from ehf.server.research.backtest.engine import run_backtest
from ehf.server.research.backtest.metrics import core_metrics, equity_from_returns
from ehf.server.research.backtest.sweep import run_sweep
from ehf.server.research.config.spec import RunSpec
from ehf.server.research.data.schema import Timeframe
from ehf.server.research.results.registry import ExperimentRegistry, RunResult

Grid = dict[str, Sequence[Any]]


class WalkForwardSpec(BaseModel):
    """How to cut time into optimize -> validate windows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_months: int = 36
    test_months: int = 12
    anchored: bool = False  # True: train always starts at the first bar (expanding)
    warmup_bars: int = 252  # bars prepended to every slice for signal warm-up (no P&L)
    objective: str = "sharpe"  # metric maximized in-sample (a ``m_<name>`` column)
    min_trades: int = 10  # cells below this are ineligible for selection (if any pass)
    stress_cost_multipliers: tuple[float, ...] = (2.0,)  # OOS reruns at scaled slippage


@dataclass(frozen=True)
class Window:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "window": self.index,
            "train_start": self.train_start.date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "test_start": self.test_start.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
        }


@dataclass
class WalkForwardResult:
    """Everything a scorecard or the gauntlet needs, plus the registry handles."""

    wf_id: str
    base_spec: RunSpec
    grid: Grid
    wf_spec: WalkForwardSpec
    windows: pd.DataFrame  # one row per window: dates, chosen params, IS/OOS metrics, hashes
    oos_returns: pd.Series  # stitched per-bar net returns (UTC index)
    oos_metrics: dict[str, float]
    stitched_hash: str  # registry hash of the stitched OOS run
    trial_sharpes: np.ndarray  # every train cell's Sharpe across all windows (for DSR)
    train_sweep_ids: list[str]
    stress: dict[float, dict[str, float]] = field(default_factory=dict)  # multiplier -> metrics

    @property
    def n_windows(self) -> int:
        return int(len(self.windows))

    @property
    def oos_sharpe(self) -> float:
        return float(self.oos_metrics.get("sharpe", float("nan")))

    @property
    def mean_is_sharpe(self) -> float:
        return float(self.windows["is_sharpe"].mean()) if self.n_windows else float("nan")

    @property
    def efficiency(self) -> float:
        """Walk-forward efficiency: OOS Sharpe / mean in-sample Sharpe of the chosen cells."""
        is_sr = self.mean_is_sharpe
        if not math.isfinite(is_sr) or is_sr <= 0:
            return float("nan")
        return self.oos_sharpe / is_sr


# --- window construction (pure) ---------------------------------------------------


def make_windows(
    index: pd.DatetimeIndex,
    wf: WalkForwardSpec,
    *,
    origin: pd.Timestamp | None = None,
    horizon: pd.Timestamp | None = None,
) -> list[Window]:
    """Tile non-overlapping test windows across ``index`` (pure; no I/O).

    Training starts at ``origin`` (the base spec's ``start``) provided
    ``warmup_bars`` sessions precede it in ``index``; otherwise at the first bar
    that has that much warm-up behind it. Anchoring to the spec, not to the first
    cached bar, keeps a study's windows stable as the cache grows. Windows are
    calendar-month aligned on that start; the last test window is truncated at
    ``horizon`` (the spec's ``end``) or the final bar — never dropped, since
    recent history is the most valuable out-of-sample evidence.
    """
    if len(index) <= wf.warmup_bars:
        return []
    pos = wf.warmup_bars
    if origin is not None:
        pos = max(pos, int(index.searchsorted(origin)))
    if pos >= len(index):
        return []
    first = index[pos].normalize()
    last = index[-1].normalize()
    if horizon is not None:
        last = min(last, horizon.normalize())
    windows: list[Window] = []
    train_start = first
    test_start = first + pd.DateOffset(months=wf.train_months)
    i = 0
    while test_start <= last:
        test_end = min(
            test_start + pd.DateOffset(months=wf.test_months) - pd.Timedelta(days=1), last
        )
        if not wf.anchored:
            train_start = test_start - pd.DateOffset(months=wf.train_months)
        windows.append(
            Window(
                index=i,
                train_start=train_start,
                train_end=test_start - pd.Timedelta(days=1),
                test_start=test_start,
                test_end=test_end,
            )
        )
        i += 1
        test_start = test_start + pd.DateOffset(months=wf.test_months)
    return windows


def make_walk_forward_id(base: RunSpec, grid: Grid, wf: WalkForwardSpec) -> str:
    """Deterministic id of a walk-forward study (strategy + base + grid + settings)."""
    payload = json.dumps(
        {
            "base": json.loads(base.model_copy(update={"sweep_id": None}).canonical_json()),
            "grid": {k: list(v) for k, v in grid.items()},
            "wf": wf.model_dump(mode="json"),
        },
        sort_keys=True,
        default=str,
    )
    return f"wf_{base.strategy}_{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


def slice_with_warmup(
    bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, warmup_bars: int
) -> pd.DataFrame:
    """Bars from ``warmup_bars`` sessions before ``start`` through ``end`` (inclusive)."""
    sessions = pd.DatetimeIndex(bars["timestamp"].unique()).sort_values()
    pos = int(sessions.searchsorted(start))
    lo = sessions[max(pos - warmup_bars, 0)]
    hi = end + pd.Timedelta(days=1)
    out = bars[(bars["timestamp"] >= lo) & (bars["timestamp"] < hi)].reset_index(drop=True)
    out.attrs.update(bars.attrs)
    return out


def select_best(rows: pd.DataFrame, objective: str, min_trades: int) -> pd.Series:
    """The in-sample winner: max ``m_<objective>`` among cells with enough trades.

    Falls back to every cell when none clears ``min_trades`` (a sparse strategy is
    still a strategy; the OOS test will judge it). Ties/NaNs: first non-NaN max.
    """
    col = f"m_{objective}"
    if col not in rows.columns:
        raise KeyError(f"objective {objective!r} not among sweep metrics")
    eligible = rows
    if "m_n_trades" in rows.columns:
        enough = rows[rows["m_n_trades"] >= min_trades]
        if not enough.empty:
            eligible = enough
    eligible = eligible[eligible[col].notna()]
    if eligible.empty:
        raise ValueError("every sweep cell produced a NaN objective")
    return cast(pd.Series, eligible.loc[eligible[col].idxmax()])


# --- the harness --------------------------------------------------------------------


def walk_forward(
    base_spec: RunSpec,
    grid: Grid,
    bars: pd.DataFrame,
    wf: WalkForwardSpec | None = None,
    *,
    registry: ExperimentRegistry | None = None,
    n_workers: int | None = None,
) -> WalkForwardResult:
    """Run the full optimize -> validate study and log every piece to the registry."""
    wf = wf or WalkForwardSpec()
    registry = registry if registry is not None else ExperimentRegistry()
    tf = Timeframe(str(bars.attrs.get("timeframe", Timeframe.DAY.value)))
    sessions = pd.DatetimeIndex(bars["timestamp"].unique()).sort_values()
    windows = make_windows(
        sessions,
        wf,
        origin=pd.Timestamp(base_spec.start, tz="UTC"),
        horizon=pd.Timestamp(base_spec.end, tz="UTC"),
    )
    if not windows:
        raise ValueError(
            f"not enough history for walk-forward: {len(sessions)} bars, need warm-up "
            f"{wf.warmup_bars} + {wf.train_months} train months + a test window"
        )
    wf_id = make_walk_forward_id(base_spec, grid, wf)

    rows: list[dict[str, Any]] = []
    oos_parts: list[pd.Series] = []
    trade_parts: list[pd.DataFrame] = []
    stress_parts: dict[float, list[pd.Series]] = {m: [] for m in wf.stress_cost_multipliers}
    trial_sharpes: list[float] = []
    train_sweep_ids: list[str] = []

    for w in windows:
        # 1. optimize on the train slice (parallel sweep; every cell logged)
        train_sid = f"{wf_id}:w{w.index:02d}:train"
        train_spec = base_spec.model_copy(
            update={"start": _iso(w.train_start), "end": _iso(w.train_end)}
        )
        train_bars = slice_with_warmup(bars, w.train_start, w.train_end, wf.warmup_bars)
        cells = run_sweep(
            train_spec, grid, train_bars, registry=registry, n_workers=n_workers, sweep_id=train_sid
        )
        train_sweep_ids.append(train_sid)
        trial_sharpes.extend(float(x) for x in cells["m_sharpe"].dropna())
        best = select_best(cells, wf.objective, wf.min_trades)
        best_params: dict[str, Any] = json.loads(best["params_json"])

        # 2. validate the winner on the untouched test slice
        test_spec = base_spec.model_copy(
            update={
                "params": best_params,
                "start": _iso(w.test_start),
                "end": _iso(w.test_end),
                "sweep_id": f"{wf_id}:oos",
            }
        )
        test_bars = slice_with_warmup(bars, w.test_start, w.test_end, wf.warmup_bars)
        test = run_backtest(test_spec, test_bars)
        test_hash = registry.log_run(test_spec, test, parent_hash=str(best["config_hash"]))
        oos_parts.append(test.equity.set_index("timestamp")["net_return"])
        trade_parts.append(test.trades)

        # 3. the same winner under harsher costs (cheap: one backtest per multiplier)
        for mult in wf.stress_cost_multipliers:
            stressed = test_spec.model_copy(
                update={
                    "costs": test_spec.costs.model_copy(
                        update={"slippage_bps": test_spec.costs.slippage_bps * mult}
                    ),
                    "sweep_id": f"{wf_id}:oos:x{mult:g}",
                }
            )
            res = run_backtest(stressed, test_bars)
            registry.log_run(stressed, res, parent_hash=test_hash)
            stress_parts[mult].append(res.equity.set_index("timestamp")["net_return"])

        rows.append(
            {
                **w.as_dict(),
                "params": best_params,
                "is_sharpe": float(best["m_sharpe"]),
                f"is_{wf.objective}": float(best[f"m_{wf.objective}"]),
                "oos_sharpe": test.metrics.get("sharpe", float("nan")),
                "oos_cagr": test.metrics.get("cagr", float("nan")),
                "oos_max_drawdown": test.metrics.get("max_drawdown", float("nan")),
                "oos_n_trades": test.metrics.get("n_trades", 0.0),
                "n_cells": int(len(cells)),
                "train_sweep_id": train_sid,
                "train_hash": str(best["config_hash"]),
                "test_hash": test_hash,
            }
        )

    # 4. stitch the out-of-sample curve and log it as one reproducible run
    oos = pd.concat(oos_parts).sort_index()
    trades = pd.concat(trade_parts, ignore_index=True)
    equity = equity_from_returns(oos, base_spec.initial_capital)
    oos_metrics = core_metrics(oos, equity["equity"], trades, tf.periods_per_year)
    windows_df = pd.DataFrame(rows)
    stress_metrics: dict[float, dict[str, float]] = {}
    for mult, parts in stress_parts.items():
        s = pd.concat(parts).sort_index()
        eq = equity_from_returns(s, base_spec.initial_capital)
        stress_metrics[mult] = core_metrics(s, eq["equity"], trades, tf.periods_per_year)
        oos_metrics[f"sharpe_cost_x{mult:g}"] = stress_metrics[mult]["sharpe"]
        oos_metrics[f"cagr_cost_x{mult:g}"] = stress_metrics[mult]["cagr"]
    oos_metrics["n_windows"] = float(len(windows))
    oos_metrics["mean_is_sharpe"] = float(windows_df["is_sharpe"].mean())
    is_sr = oos_metrics["mean_is_sharpe"]
    oos_metrics["wf_efficiency"] = (
        oos_metrics["sharpe"] / is_sr if math.isfinite(is_sr) and is_sr > 0 else float("nan")
    )
    oos_metrics["share_windows_positive"] = float((windows_df["oos_sharpe"] > 0).mean())

    stitched_spec = base_spec.model_copy(
        update={
            "params": {
                "walk_forward": wf.model_dump(mode="json"),
                "grid": {k: list(v) for k, v in grid.items()},
                "base_params": base_spec.params,
            },
            "start": _iso(windows[0].test_start),
            "end": _iso(windows[-1].test_end),
            "sweep_id": wf_id,
        }
    )
    stitched_hash = registry.log_run(
        stitched_spec,
        RunResult(metrics=oos_metrics, equity=equity, trades=trades, status="walk_forward"),
    )

    return WalkForwardResult(
        wf_id=wf_id,
        base_spec=base_spec,
        grid=grid,
        wf_spec=wf,
        windows=windows_df,
        oos_returns=oos,
        oos_metrics=oos_metrics,
        stitched_hash=stitched_hash,
        trial_sharpes=np.asarray(trial_sharpes, dtype=float),
        train_sweep_ids=train_sweep_ids,
        stress=stress_metrics,
    )


def _iso(ts: pd.Timestamp) -> str:
    return str(ts.date().isoformat())
