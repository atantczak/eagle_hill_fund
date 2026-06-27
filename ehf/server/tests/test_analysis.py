"""Analysis suite: robustness metrics validated against known properties."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless

from ehf.server.research.analysis import metrics, regime, runs
from ehf.server.research.analysis.tearsheet import comparison_tearsheet, tearsheet
from ehf.server.research.config.spec import RunSpec, UniverseSpec
from ehf.server.research.results.registry import ExperimentRegistry, RunResult
from ehf.server.research.results.store import LocalResultStore

# ---- pure metrics --------------------------------------------------------


def test_sharpe_basic() -> None:
    assert metrics.sharpe_ratio(pd.Series([0.01])) == 0.0  # too short -> 0 by guard
    rng = np.random.default_rng(0)
    assert metrics.sharpe_ratio(rng.normal(0.001, 0.01, 2000)) > 0  # positive drift
    assert metrics.sharpe_ratio(rng.normal(-0.001, 0.01, 2000)) < 0  # negative drift


def test_psr_bounds_and_monotonic() -> None:
    rng = np.random.default_rng(1)
    flat = rng.normal(0.0, 0.01, 2000)  # zero-mean -> PSR ~ 0.5
    strong = rng.normal(0.002, 0.01, 2000)  # clear edge -> PSR ~ 1
    psr_flat = metrics.probabilistic_sharpe_ratio(flat, 0.0)
    psr_strong = metrics.probabilistic_sharpe_ratio(strong, 0.0)
    assert 0.0 <= psr_flat <= 1.0 and 0.0 <= psr_strong <= 1.0
    assert abs(psr_flat - 0.5) < 0.25
    assert psr_strong > psr_flat
    assert psr_strong > 0.9


def test_expected_max_sharpe_grows_with_trials() -> None:
    rng = np.random.default_rng(2)
    sharpes = rng.normal(0, 0.5, 100)
    few = metrics.expected_max_sharpe(sharpes[:5])
    many = metrics.expected_max_sharpe(sharpes)
    assert many > few > 0


def test_deflated_sharpe_below_psr() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(0.0015, 0.01, 2000)
    trial_sharpes = rng.normal(1.0, 0.8, 50)  # 50 darts thrown
    psr = metrics.probabilistic_sharpe_ratio(r, 0.0)
    dsr = metrics.deflated_sharpe_ratio(r, trial_sharpes)
    assert dsr <= psr  # deflation can only lower confidence


def test_pbo_dominant_vs_noise() -> None:
    rng = np.random.default_rng(4)
    t, n = 2000, 20
    # one genuinely-better config -> low overfit probability
    noise = rng.normal(0.0, 0.01, (t, n))
    dominant = noise.copy()
    dominant[:, 0] += 0.0015  # config 0 truly best everywhere
    pbo_real = metrics.probability_of_backtest_overfitting(dominant, n_splits=10)
    assert 0.0 <= pbo_real <= 1.0
    assert pbo_real < 0.2
    # identical noise configs -> selection is chance -> PBO around 0.5
    pbo_noise = metrics.probability_of_backtest_overfitting(noise, n_splits=10)
    assert 0.25 < pbo_noise < 0.75


def test_monte_carlo_keys_and_centering() -> None:
    rng = np.random.default_rng(5)
    r = rng.normal(0.0008, 0.01, 1500)
    mc = metrics.monte_carlo_paths(r, n_sims=500, seed=0)
    for k in ("sharpe_median", "sharpe_p05", "sharpe_p95", "maxdd_worst", "prob_sharpe_positive"):
        assert k in mc
    assert mc["sharpe_p05"] < mc["sharpe_median"] < mc["sharpe_p95"]
    assert abs(mc["sharpe_median"] - metrics.sharpe_ratio(r)) < 0.3


def test_regime_breakdown_partitions() -> None:
    rng = np.random.default_rng(6)
    idx = pd.date_range("2020-01-01", periods=1000, tz="UTC")
    mkt = pd.Series(rng.normal(0.0003, 0.012, 1000), index=idx)
    strat = pd.Series(rng.normal(0.0005, 0.01, 1000), index=idx)
    bd = regime.regime_breakdown(strat, mkt)
    assert any(i.startswith("trend:") for i in bd.index)
    assert any(i.startswith("vol:") for i in bd.index)
    assert (bd["time_share"] >= 0).all()


# ---- registry integration ------------------------------------------------


def _logged_sweep(tmp_path) -> tuple[ExperimentRegistry, list[str], str]:
    reg = ExperimentRegistry(store=LocalResultStore(root=tmp_path))
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=600, tz="UTC")
    hashes = []
    sweep_id = "sweep_test"
    for drift in [0.0002, 0.0006, 0.0010, 0.0004]:
        ret = rng.normal(drift, 0.01, 600)
        eq = 1e6 * np.cumprod(1 + ret)
        equity = pd.DataFrame({"timestamp": idx, "equity": eq, "net_return": ret})
        res = RunResult(
            metrics={"sharpe": metrics.sharpe_ratio(ret), "cagr": 0.1, "max_drawdown": -0.1},
            equity=equity,
            trades=pd.DataFrame({"symbol": ["A"], "qty": [1.0], "price": [10.0]}),
        )
        spec = RunSpec(
            strategy="demo",
            params={"drift": drift},
            universe=UniverseSpec(symbols=("A",)),
            start="2020-01-01",
            end="2022-06-30",
            sweep_id=sweep_id,
        )
        hashes.append(reg.log_run(spec, res))
    return reg, hashes, sweep_id


def test_run_returns_and_robustness_report(tmp_path) -> None:
    reg, hashes, sweep_id = _logged_sweep(tmp_path)
    r = runs.run_returns(reg, hashes[2])
    assert len(r) == 600

    rep = runs.robustness_report(reg, hashes[2], sweep_id=sweep_id, n_sims=200)
    assert "psr_vs_zero" in rep and "monte_carlo" in rep
    assert rep["n_trials"] == 4
    assert "deflated_sharpe" in rep and "pbo" in rep


def test_parameter_sensitivity_and_matrix(tmp_path) -> None:
    reg, _, sweep_id = _logged_sweep(tmp_path)
    table = runs.parameter_sensitivity(reg, sweep_id, metric="sharpe")
    assert "drift" in table.columns and "sharpe" in table.columns and len(table) == 4
    matrix, cols = runs.sweep_return_matrix(reg, sweep_id)
    assert matrix.shape == (600, 4) and len(cols) == 4


def test_tearsheet_renders(tmp_path) -> None:
    reg, hashes, sweep_id = _logged_sweep(tmp_path)
    fig = tearsheet(reg, hashes[2], sweep_id=sweep_id)
    assert fig is not None and len(fig.axes) >= 6
    assert comparison_tearsheet(reg, hashes[:2]) is not None
    import matplotlib.pyplot as plt

    plt.close("all")
