"""Walk-forward harness + gauntlet: windows tile correctly, nothing leaks, verdicts are honest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ehf.research.analysis.gauntlet import (
    Check,
    GauntletConfig,
    market_proxy_returns,
    run_gauntlet,
)
from ehf.research.backtest.engine import run_backtest
from ehf.research.backtest.walkforward import (
    WalkForwardSpec,
    make_walk_forward_id,
    make_windows,
    select_best,
    slice_with_warmup,
    walk_forward,
)
from ehf.research.config.spec import RunSpec, UniverseSpec
from ehf.research.results.registry import ExperimentRegistry
from ehf.research.results.store import LocalResultStore
from ehf.tests.conftest import make_daily_bars


def _bars(n: int = 1100, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return make_daily_bars(
        {
            "SPY": list(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))),
            "TLT": list(90 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, n)))),
            "GLD": list(120 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, n)))),
        },
        start="2018-01-02",
    )


def _base() -> RunSpec:
    return RunSpec(
        strategy="tsmom",
        params={"vol_window": 21},
        universe=UniverseSpec(symbols=("SPY", "TLT", "GLD")),
        start="2018-01-01",
        end="2030-01-01",
    )


_WF = WalkForwardSpec(train_months=12, test_months=6, warmup_bars=60, min_trades=1)
_GRID = {"lookback": [42, 63], "target_vol": [0.1, 0.2]}


# --- pure pieces ----------------------------------------------------------------------------


def test_windows_tile_without_overlap_and_cover_the_tail() -> None:
    idx = pd.DatetimeIndex(_bars()["timestamp"].unique())
    windows = make_windows(idx, _WF)
    assert len(windows) >= 4
    for a, b in zip(windows, windows[1:], strict=False):
        assert a.test_end < b.test_start  # non-overlapping OOS
        assert b.test_start == a.test_end + pd.Timedelta(days=1)  # contiguous
        assert a.train_end < a.test_start  # train strictly before test
    assert windows[-1].test_end == idx[-1].normalize()  # last window truncated, not dropped
    rolling = make_windows(idx, _WF.model_copy(update={"anchored": False}))
    anchored = make_windows(idx, _WF.model_copy(update={"anchored": True}))
    assert all(w.train_start == anchored[0].train_start for w in anchored)
    assert rolling[-1].train_start > rolling[0].train_start


def test_windows_anchor_to_the_spec_origin_when_warmup_is_available() -> None:
    idx = pd.DatetimeIndex(_bars()["timestamp"].unique())
    origin = idx[300].normalize()
    anchored = make_windows(idx, _WF, origin=origin)
    assert anchored[0].train_start == origin  # 60 warm-up bars exist before it
    early = make_windows(idx, _WF, origin=idx[10].normalize())
    assert early[0].train_start == idx[_WF.warmup_bars].normalize()  # not enough warm-up: fallback
    capped = make_windows(idx, _WF, origin=origin, horizon=idx[800].normalize())
    assert capped[-1].test_end == idx[800].normalize()
    assert make_windows(idx, _WF, origin=idx[-1].normalize() + pd.Timedelta(days=5)) == []


def test_windows_empty_when_history_too_short() -> None:
    idx = pd.DatetimeIndex(_bars(n=100)["timestamp"].unique())
    assert make_windows(idx, _WF) == []


def test_slice_with_warmup_prepends_exactly_n_sessions() -> None:
    bars = _bars()
    sessions = pd.DatetimeIndex(bars["timestamp"].unique())
    start, end = sessions[300].normalize(), sessions[400].normalize()
    out = slice_with_warmup(bars, start, end, warmup_bars=50)
    got = pd.DatetimeIndex(out["timestamp"].unique())
    assert got[0] == sessions[250]
    assert got[-1] == sessions[400]
    assert out.attrs["timeframe"] == bars.attrs["timeframe"]


def test_select_best_prefers_cells_with_enough_trades() -> None:
    rows = pd.DataFrame(
        {
            "config_hash": ["a", "b", "c"],
            "params_json": ["{}", "{}", "{}"],
            "m_sharpe": [3.0, 1.0, 2.0],
            "m_n_trades": [2.0, 50.0, 40.0],
        }
    )
    assert select_best(rows, "sharpe", min_trades=10)["config_hash"] == "c"
    assert select_best(rows, "sharpe", min_trades=1000)["config_hash"] == "a"  # fallback: all
    with pytest.raises(KeyError):
        select_best(rows, "nope", min_trades=1)


def test_walk_forward_id_is_deterministic_and_ignores_sweep_id() -> None:
    a = make_walk_forward_id(_base(), _GRID, _WF)
    b = make_walk_forward_id(_base().model_copy(update={"sweep_id": "x"}), _GRID, _WF)
    assert a == b and a.startswith("wf_tsmom_")
    assert make_walk_forward_id(_base(), {"lookback": [1]}, _WF) != a


# --- engine evaluation window (what makes warm-up leak-free) ------------------------------------


def test_engine_evaluation_window_matches_a_fresh_run_on_the_same_bars() -> None:
    bars = _bars(n=400)
    sessions = pd.DatetimeIndex(bars["timestamp"].unique())
    start, end = sessions[200].normalize(), sessions[-1].normalize()
    spec = _base().model_copy(
        update={
            "params": {"lookback": 42, "vol_window": 21},
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
        }
    )
    windowed = run_backtest(spec, bars)  # full bars, accounting restricted
    assert windowed.equity["timestamp"].iloc[0] == sessions[200]
    assert len(windowed.equity) == 200
    # The opening position is a real trade: the first in-window bar carries entries.
    first = windowed.trades[windowed.trades["timestamp"] == sessions[200]]
    assert not first.empty
    with pytest.raises(ValueError, match="no bars inside"):
        run_backtest(spec.model_copy(update={"start": "2031-01-01", "end": "2032-01-01"}), bars)


# --- the harness end to end -----------------------------------------------------------------------


def test_walk_forward_logs_everything_and_stitches_oos(tmp_path) -> None:
    reg = ExperimentRegistry(store=LocalResultStore(root=tmp_path))
    bars = _bars()
    wf = walk_forward(_base(), _GRID, bars, _WF, registry=reg, n_workers=2)

    n = wf.n_windows
    assert n >= 4
    # one train sweep per window, 4 cells each, distinct sweep ids
    assert len(set(wf.train_sweep_ids)) == n
    for sid in wf.train_sweep_ids:
        assert len(reg.query("WHERE sweep_id = ?", (sid,))) == 4
    assert len(wf.trial_sharpes) == 4 * n
    # one OOS run per window + the stress twin + the stitched run
    assert len(reg.query("WHERE sweep_id = ?", (f"{wf.wf_id}:oos",))) == n
    assert len(reg.query("WHERE sweep_id = ?", (f"{wf.wf_id}:oos:x2",))) == n
    stitched = reg.query("WHERE sweep_id = ? AND status = 'walk_forward'", (wf.wf_id,))
    assert len(stitched) == 1 and stitched["config_hash"].iloc[0] == wf.stitched_hash
    # OOS curve = concatenation of the test windows, strictly increasing, no overlap
    assert wf.oos_returns.index.is_monotonic_increasing
    assert not wf.oos_returns.index.has_duplicates
    assert len(wf.oos_returns) == sum(
        len(reg.get_run(h)["equity"]) for h in wf.windows["test_hash"]
    )
    # test runs point at the train winner they came from
    parents = reg.query("WHERE sweep_id = ?", (f"{wf.wf_id}:oos",))["parent_hash"]
    assert set(parents) == set(wf.windows["train_hash"])
    # metrics carry the stress + efficiency numbers the gauntlet reads
    for key in ("sharpe_cost_x2", "wf_efficiency", "share_windows_positive", "n_windows"):
        assert key in wf.oos_metrics
    assert wf.oos_metrics["n_windows"] == n
    # re-running is idempotent (same hashes, nothing duplicated)
    again = walk_forward(_base(), _GRID, bars, _WF, registry=reg, n_workers=2)
    assert again.stitched_hash == wf.stitched_hash
    assert len(reg.query("WHERE sweep_id = ?", (f"{wf.wf_id}:oos",))) == n

    # --- gauntlet on top: every check present, a fixed config, markdown renders
    verdict = run_gauntlet(wf, registry=reg, market_returns=market_proxy_returns(bars))
    names = [c.name for c in verdict.checks]
    assert names == [
        "windows", "oos_sharpe", "psr", "dsr", "efficiency", "pbo",
        "cost_stress", "trades", "windows_pos", "regimes",
    ]  # fmt: skip
    assert verdict.status in {"promoted", "killed"}
    md = verdict.to_markdown()
    assert "Verdict:" in md and "| pbo |" in md
    assert verdict.as_dict()["wf_id"] == wf.wf_id
    # random-walk data should not be promoted by a strict config
    strict = run_gauntlet(wf, registry=reg, cfg=GauntletConfig(min_oos_sharpe=5.0))
    assert not strict.passed and "oos_sharpe" in strict.failed


def test_check_semantics() -> None:
    assert Check("x", 1.0, 0.5, ">=").passed
    assert not Check("x", 1.0, 0.5, "<=").passed
    assert not Check("x", float("nan"), 0.5, ">=").passed  # NaN never passes
    assert Check("x", float("nan"), 0.5, ">=").as_dict()["value"] is None


def test_market_proxy_returns() -> None:
    bars = _bars(n=50)
    s = market_proxy_returns(bars, "SPY")
    assert s is not None and len(s) == 50 and np.isnan(s.iloc[0])
    assert market_proxy_returns(bars, "NOPE") is None
