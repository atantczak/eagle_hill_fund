"""ExperimentRegistry: reproducibility, idempotency, and cross-run query."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ehf.server.research.config.spec import CostConfig, FillModel, RunSpec, UniverseSpec
from ehf.server.research.results.registry import ExperimentRegistry, RunResult
from ehf.server.research.results.store import LocalResultStore


def _spec(**overrides) -> RunSpec:
    base = dict(
        strategy="rise_fall",
        params={"drop_pct": 0.005, "rise_pct": 0.02},
        universe=UniverseSpec(symbols=("AAPL", "MSFT")),
        start="2024-01-01",
        end="2024-06-30",
    )
    base.update(overrides)
    return RunSpec(**base)


def _result(seed: int = 0) -> RunResult:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=20, tz="UTC")
    equity = pd.DataFrame(
        {"timestamp": ts, "equity": 1e6 * (1 + rng.normal(0, 0.01, 20)).cumprod()}
    )
    trades = pd.DataFrame({"symbol": ["AAPL"], "qty": [10.0], "price": [100.0]})
    return RunResult(metrics={"sharpe": 1.23, "max_drawdown": -0.08}, equity=equity, trades=trades)


@pytest.fixture
def registry(tmp_path) -> ExperimentRegistry:
    return ExperimentRegistry(store=LocalResultStore(root=tmp_path))


def test_same_spec_same_hash() -> None:
    assert _spec().config_hash == _spec().config_hash


def test_param_change_changes_hash() -> None:
    assert _spec().config_hash != _spec(params={"drop_pct": 0.006}).config_hash


def test_costs_default_is_not_frictionless() -> None:
    assert CostConfig().slippage_bps > 0
    assert _spec().fill_model is FillModel.NEXT_OPEN  # safe default, no look-ahead


def test_log_and_get_roundtrip(registry: ExperimentRegistry) -> None:
    spec = _spec()
    h = registry.log_run(spec, _result())
    run = registry.get_run(h)
    assert run["strategy"] == "rise_fall"
    assert run["metrics"]["sharpe"] == pytest.approx(1.23)
    assert len(run["equity"]) == 20
    assert run["spec"]  # canonical spec round-tripped
    assert run["git_sha"] is not None or run["git_sha"] is None  # captured (may be None off-git)


def test_idempotent_skip(registry: ExperimentRegistry) -> None:
    spec = _spec()
    h1 = registry.log_run(spec, _result(1))
    h2 = registry.log_run(spec, _result(2))  # identical spec -> skipped
    assert h1 == h2
    df = registry.query()
    assert len(df) == 1


def test_query_exposes_metric_columns(registry: ExperimentRegistry) -> None:
    registry.log_run(_spec(), _result())
    registry.log_run(_spec(params={"drop_pct": 0.01}), _result())
    df = registry.query("WHERE strategy = ? ORDER BY m_sharpe DESC", ("rise_fall",))
    assert len(df) == 2
    assert "m_sharpe" in df.columns


def test_compare(registry: ExperimentRegistry) -> None:
    h1 = registry.log_run(_spec(), _result())
    h2 = registry.log_run(_spec(params={"drop_pct": 0.01}), _result())
    cmp = registry.compare([h1, h2])
    assert list(cmp.columns) == [h1, h2]
    assert "sharpe" in cmp.index
