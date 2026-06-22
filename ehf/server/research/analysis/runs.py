"""Bridge between the results store and the robustness metrics.

Everything here reads *stored* runs by hash (or a whole sweep by id) and feeds
the pure functions in ``metrics`` — so any historical run can be interrogated
without rerunning it.
"""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pandas as pd

from ehf.server.research.analysis import metrics
from ehf.server.research.results.registry import ExperimentRegistry


def run_returns(registry: ExperimentRegistry, config_hash: str) -> pd.Series:
    """Per-bar net returns of a stored run, indexed by timestamp."""
    eq = registry.get_run(config_hash)["equity"]
    return cast(pd.Series, eq.set_index("timestamp")["net_return"])


def sweep_return_matrix(
    registry: ExperimentRegistry, sweep_id: str
) -> tuple[np.ndarray, list[str]]:
    """(T x N) per-bar return matrix across every cell of a sweep, time-aligned.

    Returns (matrix, hashes). Cells that don't share the common timestamp index
    are dropped via an inner join so PBO sees one rectangular panel.
    """
    rows = registry.query("WHERE sweep_id = ?", (sweep_id,))
    series = {}
    for h in rows["config_hash"]:
        series[h] = run_returns(registry, h)
    if not series:
        return np.empty((0, 0)), []
    panel = pd.DataFrame(series).dropna()
    return panel.to_numpy(), list(panel.columns)


def parameter_sensitivity(
    registry: ExperimentRegistry,
    sweep_id: str,
    metric: str = "sharpe",
    by: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Metric across a sweep's params — the "plateau vs. needle" view.

    Returns a long table (params + metric), or a 2-D pivot when ``by`` names two
    parameters. A broad high-metric plateau is robust; a lone high cell is a
    needle (likely overfit).
    """
    rows = registry.query("WHERE sweep_id = ?", (sweep_id,))
    params = pd.json_normalize([json.loads(p) for p in rows["params_json"]])
    table = pd.concat(
        [params.reset_index(drop=True), rows[[f"m_{metric}"]].reset_index(drop=True)], axis=1
    )
    table = table.rename(columns={f"m_{metric}": metric})
    if by is not None:
        return table.pivot_table(index=by[0], columns=by[1], values=metric, aggfunc="mean")
    return table


def robustness_report(
    registry: ExperimentRegistry,
    config_hash: str,
    *,
    sweep_id: str | None = None,
    periods_per_year: float = 252.0,
    n_sims: int = 1000,
) -> dict[str, Any]:
    """One-call honesty scorecard for a stored run.

    Always: Probabilistic Sharpe + Monte-Carlo spread. If ``sweep_id`` is given
    (the set of trials this run was selected from): Deflated Sharpe + PBO, which
    correct for how many darts were thrown.
    """
    r = run_returns(registry, config_hash)
    report: dict[str, Any] = {
        "config_hash": config_hash,
        "psr_vs_zero": metrics.probabilistic_sharpe_ratio(r, 0.0),
        "monte_carlo": metrics.monte_carlo_paths(
            r, n_sims=n_sims, periods_per_year=periods_per_year, seed=0
        ),
    }
    if sweep_id is not None:
        rows = registry.query("WHERE sweep_id = ?", (sweep_id,))
        trial_sharpes = rows["m_sharpe"].to_numpy()
        report["n_trials"] = int(len(trial_sharpes))
        report["deflated_sharpe"] = metrics.deflated_sharpe_ratio(r, trial_sharpes)
        matrix, _ = sweep_return_matrix(registry, sweep_id)
        if matrix.size:
            report["pbo"] = metrics.probability_of_backtest_overfitting(matrix)
    return report
