"""Parallel sweep: every cell logged, sweep registered, results == serial."""

from __future__ import annotations

import numpy as np
from ehf.server.research.backtest.engine import run_backtest
from ehf.server.research.backtest.sweep import expand_grid, make_sweep_id, run_sweep
from ehf.server.research.config.spec import RunSpec, UniverseSpec
from ehf.server.research.results.registry import ExperimentRegistry
from ehf.server.research.results.store import LocalResultStore

from tests.conftest import make_daily_bars


def _bars():
    rng = np.random.default_rng(3)
    return make_daily_bars(
        {
            "A": list(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 250)))),
            "B": list(90 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 250)))),
        }
    )


def _base() -> RunSpec:
    return RunSpec(
        strategy="rise_fall",
        params={"trend_window": 50, "atr_window": 10, "max_hold": 10},
        universe=UniverseSpec(symbols=("A", "B")),
        start="2020-01-01",
        end="2025-01-01",
    )


def test_expand_grid_cartesian_and_tagged() -> None:
    specs = expand_grid(_base(), {"drop_pct": [0.005, 0.01], "rise_pct": [0.02, 0.04]}, "sid1")
    assert len(specs) == 4
    assert all(s.sweep_id == "sid1" for s in specs)
    assert {(s.params["drop_pct"], s.params["rise_pct"]) for s in specs} == {
        (0.005, 0.02),
        (0.005, 0.04),
        (0.01, 0.02),
        (0.01, 0.04),
    }


def test_sweep_id_is_deterministic() -> None:
    grid = {"drop_pct": [0.005, 0.01]}
    assert make_sweep_id(_base(), grid) == make_sweep_id(_base(), grid)


def test_run_sweep_logs_all_cells_and_registers(tmp_path) -> None:
    reg = ExperimentRegistry(store=LocalResultStore(root=tmp_path))
    bars = _bars()
    grid = {"drop_pct": [0.005, 0.01], "rise_pct": [0.02, 0.04]}
    out = run_sweep(_base(), grid, bars, registry=reg, n_workers=2)

    assert len(out) == 4  # one row per cell
    sid = make_sweep_id(_base(), grid)
    assert (out["sweep_id"] == sid).all()
    assert "m_sharpe" in out.columns


def test_parallel_matches_serial(tmp_path) -> None:
    """A swept cell's metrics equal a direct serial run of the same spec."""
    reg = ExperimentRegistry(store=LocalResultStore(root=tmp_path))
    bars = _bars()
    grid = {"drop_pct": [0.007]}
    run_sweep(_base(), grid, bars, registry=reg, n_workers=1)

    spec = expand_grid(_base(), grid, make_sweep_id(_base(), grid))[0]
    serial = run_backtest(spec, bars)
    stored = reg.get_run(spec.config_hash)["metrics"]
    assert stored["sharpe"] == serial.metrics["sharpe"]
    assert stored["total_return"] == serial.metrics["total_return"]
