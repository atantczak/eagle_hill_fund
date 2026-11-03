"""The leverage study pieces: synthetic fund, the pair dataset, the rules, the event study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ehf.research.analysis.events import event_study, excess_path, forward_returns, turn_events
from ehf.research.backtest.engine import run_backtest
from ehf.research.backtest.panel import Panel
from ehf.research.config.spec import RunSpec, UniverseSpec
from ehf.research.data.leverage import (
    PAIRS,
    Pair,
    assemble_pair_bars,
    fit_drag,
    pair_universe,
    real_symbol,
    select_symbols,
    synthetic_leveraged,
)
from ehf.research.data.vix import assemble_vix_bars
from ehf.research.results.registry import ExperimentRegistry
from ehf.research.results.store import LocalResultStore
from ehf.research.signals.dial import enter_exit_gate
from ehf.research.strategies.base import build_strategy
from ehf.research.strategies.leverage import GRIDS, LeverageDial
from ehf.research.studies.leverage import run_study
from ehf.tests.conftest import make_daily_bars

PAIR = Pair("QQQ", "TQQQ", 3.0, "test pair")


def _paths(n: int = 1500, seed: int = 3) -> tuple[dict[str, list[float]], np.ndarray]:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.013, n)
    qqq = 100 * np.exp(np.cumsum(r))
    v = np.clip(
        20
        + 300 * pd.Series(r).rolling(10).std().fillna(0.013).to_numpy()
        - 4
        + rng.normal(0, 1, n),
        9,
        80,
    )
    irx = np.full(n, 2.0)
    return {
        "QQQ": list(qqq),
        "^VIX": list(v),
        "^IRX": list(irx),
        "^VIX3M": list(v * 1.05),
        "^VVIX": list(90 + 0 * v),
        "^VIX9D": list(v),
    }, r


@pytest.fixture(scope="module")
def world() -> dict[str, object]:
    paths, r = _paths()
    raw = make_daily_bars(paths, start="2015-01-02")
    asset = raw[raw["symbol"] == "QQQ"].reset_index(drop=True)
    asset.attrs.update(raw.attrs)
    idx_bars = raw[raw["symbol"] != "QQQ"].reset_index(drop=True)
    vix_bars = assemble_vix_bars(asset, idx_bars, None, asset="QQQ")
    # a "real" 3x fund born half-way through, built as exactly 2.95x daily minus a known drag
    base = vix_bars[vix_bars["symbol"] == "QQQ"].set_index("timestamp")
    irx = vix_bars[vix_bars["symbol"] == "^IRX"].set_index("timestamp")["close"]
    syn = synthetic_leveraged(
        base, irx, leverage=2.95, symbol="TQQQ", expense_ratio=0.01, spread=0.002
    )
    born = syn["timestamp"].iloc[700]
    real = syn[syn["timestamp"] >= born].reset_index(drop=True)
    bars, facts = assemble_pair_bars(vix_bars, real, PAIR, extend=True)
    return {"bars": bars, "facts": facts, "vix_bars": vix_bars, "real": real, "r": r}


def test_synthetic_close_is_leverage_times_base_minus_drag() -> None:
    raw = make_daily_bars({"B": [100.0, 101.0, 99.0, 102.0]}, start="2020-01-02")
    base = raw.set_index("timestamp")
    irx = pd.Series(0.0, index=base.index)
    syn = synthetic_leveraged(base, irx, leverage=3.0, symbol="S", expense_ratio=0.0).set_index(
        "timestamp"
    )
    expected = 100 * np.cumprod([1, 1 + 3 * 0.01, 1 + 3 * (99 / 101 - 1), 1 + 3 * (102 / 99 - 1)])
    np.testing.assert_allclose(syn["close"].to_numpy(), expected, rtol=1e-12)
    assert set(syn["symbol"]) == {"S"}


def test_fit_recovers_the_multiple_and_the_drag(world: dict[str, object]) -> None:
    bars = world["vix_bars"]
    base = bars[bars["symbol"] == "QQQ"].set_index("timestamp")
    irx = bars[bars["symbol"] == "^IRX"].set_index("timestamp")["close"]
    real = world["real"].set_index("timestamp")
    fit = fit_drag(real, base, irx, leverage=3.0)
    assert abs(fit["beta"] - 2.95) < 0.01
    assert fit["corr"] > 0.999
    # drag = fee 1% + (beta-1) * (2% bill + 0.2% spread)
    assert abs(fit["drag_annual"] - (0.01 + 1.95 * 0.022)) < 0.004


def test_extension_splices_and_matches_the_real_fund(world: dict[str, object]) -> None:
    bars, facts = world["bars"], world["facts"]
    syms = set(bars["symbol"])
    assert {"QQQ", "TQQQ", "TQQQ.X", "CASH", "^VIX"} <= syms
    w = bars.pivot(index="timestamp", columns="symbol", values="close")
    real = w["TQQQ"].dropna()
    ext = w["TQQQ.X"]
    assert ext.notna().all()  # extended covers every session
    assert len(real) < len(ext)
    np.testing.assert_allclose(
        ext.loc[real.index].to_numpy(), real.to_numpy()
    )  # real where it exists
    fit = facts["fit"]
    assert abs(fit["growth_synthetic"] / fit["growth_real"] - 1) < 0.05
    assert facts["extended"] == "TQQQ.X" and real_symbol("TQQQ.X") == "TQQQ"


def test_pair_universe_and_select(world: dict[str, object]) -> None:
    uni = pair_universe(PAIR, extended=True)
    assert "TQQQ.X" in uni and "TQQQ" not in uni and "QQQ" in uni
    sub = select_symbols(world["bars"], uni)
    assert set(sub["symbol"]) == set(uni)
    assert sub.attrs["timeframe"] == world["bars"].attrs["timeframe"]


def test_enter_exit_gate() -> None:
    enter = pd.Series([0, 1, 0, 0, 1, 0, np.nan], dtype=float)
    exit_ = pd.Series([0, 0, 0, 1, 1, 0, 0], dtype=float)
    out = enter_exit_gate(enter, exit_)
    assert out.tolist() == [0, 1, 1, 0, 0, 0, 0]  # exit wins a tie; NaN reads 0


@pytest.mark.parametrize("key", list(GRIDS))
def test_every_rule_runs_and_splits_between_fund_and_base(
    world: dict[str, object], key: str
) -> None:
    uni = pair_universe(PAIR, extended=True)
    bars = select_symbols(world["bars"], uni)
    panel = Panel.from_bars(bars)
    strat = build_strategy(key, {"asset": "TQQQ.X", "base": "QQQ", "benchmark": "QQQ"})
    assert isinstance(strat, LeverageDial)
    w = strat.target_weights(panel)
    np.testing.assert_allclose((w["TQQQ.X"] + w["QQQ"]).to_numpy(), 1.0)
    assert (w.drop(columns=["TQQQ.X", "QQQ"]).abs().sum().sum()) == 0
    assert w["TQQQ.X"].between(0, 1).all()
    spec = RunSpec(
        strategy=key,
        params=strat.params,
        universe=UniverseSpec(symbols=uni),
        start="2017-01-01",
        end="2020-12-31",
    )
    res = run_backtest(spec, bars)
    assert np.isfinite(res.metrics["sharpe"])


def test_hold_is_always_leveraged_and_sma_switches(world: dict[str, object]) -> None:
    panel = Panel.from_bars(select_symbols(world["bars"], pair_universe(PAIR, extended=True)))
    hold = build_strategy("lev_hold", {"asset": "TQQQ.X", "base": "QQQ"}).target_weights(panel)
    assert (hold["TQQQ.X"] == 1.0).all()
    sma = build_strategy(
        "lev_sma", {"asset": "TQQQ.X", "base": "QQQ", "window": 50}
    ).target_weights(panel)
    assert 0 < sma["TQQQ.X"].mean() < 1
    assert (sma["TQQQ.X"].diff().abs() > 0).sum() > 2


def test_vixfade_fires_once_per_storm() -> None:
    n = 200
    vix = np.full(n, 15.0)
    vix[50:60] = 35.0  # a scare
    vix[60:70] = 25.0  # settling
    vix[70:] = 15.0  # calm again: the first calm close is bar 70
    paths = {
        "QQQ": list(100 + np.arange(n) * 0.1),
        "TQQQ": list(100 + np.arange(n) * 0.3),
        "^VIX": list(vix),
    }
    bars = make_daily_bars(paths, start="2021-01-04")
    panel = Panel.from_bars(bars)
    strat = build_strategy(
        "lev_vixfade",
        {"asset": "TQQQ", "base": "QQQ", "panic": 30, "calm": 20, "lookback": 63, "hold": 10},
    )
    e = strat.exposure(panel).to_numpy()
    assert e[:70].sum() == 0
    assert e[70:80].sum() == 10 and e[80:].sum() == 0


def test_event_study_counts_turns_and_reads_forward_returns() -> None:
    idx = pd.bdate_range("2020-01-01", periods=40, tz="UTC")
    exposure = pd.Series([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10, index=idx, dtype=float)
    events = turn_events(exposure)
    assert list(events) == [idx[10], idx[30]]
    lev = pd.Series(np.linspace(100, 139, 40), index=idx)  # +1 a day
    base = pd.Series(100.0, index=idx)
    fr = forward_returns(lev, events, 5)
    # entry at the next open (idx[11] = 111), exit 5 sessions later (idx[16] = 116)
    assert fr.index[0] == idx[10] and abs(fr.iloc[0] - (116 / 111 - 1)) < 1e-12
    assert len(fr) == 2  # entry 31, exit 36: still inside the series
    assert len(forward_returns(lev, events, 10)) == 1  # entry 31, exit 41: no room, dropped
    study = event_study(exposure, lev, base, horizons=(5, 10))
    assert study["n_events"] == 2
    row = study["rows"][0]
    assert row["h"] == 5 and row["hit_rate"] == 1.0 and row["n"] >= 1
    assert study["best_h"] is None  # fewer than 5 events: no recommendation
    path = excess_path(exposure, lev, base, max_h=5)
    assert list(path.index) == [1, 2, 3, 4, 5] and (path["median"] > 0).all()


def test_study_runs_end_to_end_and_writes_linked_scorecards(
    world: dict[str, object], tmp_path: Path
) -> None:
    store = LocalResultStore(tmp_path / "results")
    reg = ExperimentRegistry(store)
    cards = tmp_path / "cards"
    bars, facts = world["bars"], world["facts"]
    res = run_study(
        lambda pair: (bars, facts),
        pairs=("QQQ/TQQQ",),
        start="2016-01-04",
        end="2020-11-30",
        variants=(True,),
        strategies=("lev_hold", "lev_sma"),
        train_months=12,
        test_months=6,
        quick=True,
        registry=reg,
        out_dir=cards,
        tag="test",
    )
    assert not res.failures, res.failures
    assert {o.strategy for o in res.outcomes} == {"lev_hold", "lev_sma"}
    assert "QQQ/TQQQ:extended" in res.holds and store.has_run(
        res.holds["QQQ/TQQQ:extended"]["hash"]
    )
    for o in res.outcomes:
        card = json.loads((cards / f"{o.wf.wf_id}.json").read_text())
        assert card["study"] == "leverage"
        assert card["facts"]["pair"] == "QQQ/TQQQ" and card["facts"]["full_hash"] == o.full_hash
        assert "events" in card["facts"] and store.has_run(o.wf.stitched_hash)
    summary = json.loads(res.summary_json.read_text())
    assert summary["study"] == "leverage" and len(summary["outcomes"]) == 2
    assert summary["outcomes"][0]["excess_path"] is not None  # QQQ/TQQQ extended carries the curve
    assert res.summary_md.read_text().startswith("# Leverage study")
    assert PAIRS["QQQ/TQQQ"].leverage == 3.0
