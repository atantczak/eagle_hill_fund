"""Parallel parameter sweeps — saturate every core for backtests.

A sweep runs one strategy across a grid of params over the *same* bars. Cells are
embarrassingly parallel, so we fan them across a process pool (CPU-bound work
needs processes, not threads — the GIL). Design:

  * **Workers compute, parent writes.** Each worker runs ``run_backtest`` and
    returns a ``RunResult``; the parent logs them through the single
    ``ExperimentRegistry`` (SQLite is single-writer — no DB contention). Compute
    is the expensive part; logging is cheap.
  * **Bars shipped once per worker** via the pool initializer (not once per cell).
  * **BLAS pinned to 1 thread/worker** so N processes don't each spawn N math
    threads and thrash the cores.

Every cell is tagged with the same ``sweep_id`` and the sweep is registered, so
the whole set is one query: ``registry.query("WHERE sweep_id = ?", (sid,))``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from itertools import product
from typing import Any

import pandas as pd

from ehf.research.config.spec import RunSpec
from ehf.research.results.config import result_store_from_descriptor
from ehf.research.results.registry import ExperimentRegistry
from ehf.research.results.store import ResultStore

# Set on each worker by the initializer. Bars are shared via a parquet file the
# workers read (OS page-cached) rather than pickling the whole frame to each — a
# big win when bars are large and there are many workers. Each worker also gets a
# store handle (rebuilt from a backend-agnostic descriptor) so it can write its
# OWN equity/trades arrays in parallel — local parquet or S3, no contention — and
# return only tiny metrics to the parent.
_WORKER_BARS: pd.DataFrame | None = None
_WORKER_STORE: ResultStore | None = None

_BLAS_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _init_worker(
    bars_path: str, timeframe: str, adjusted: bool, store_descriptor: dict[str, Any]
) -> None:
    global _WORKER_BARS, _WORKER_STORE
    bars = pd.read_parquet(bars_path)
    bars.attrs["timeframe"] = timeframe
    bars.attrs["adjusted"] = adjusted
    _WORKER_BARS = bars
    _WORKER_STORE = result_store_from_descriptor(store_descriptor)


def _run_cell(spec: RunSpec) -> tuple[RunSpec, dict[str, float], str, str | None]:
    """Worker: run a backtest, write its arrays, return only lightweight metrics."""
    from ehf.research.backtest.engine import run_backtest

    assert _WORKER_BARS is not None and _WORKER_STORE is not None, "worker not initialized"
    result = run_backtest(spec, _WORKER_BARS)
    _WORKER_STORE.write_arrays(spec.config_hash, result.equity, result.trades)
    return spec, result.metrics, result.status, result.kill_reason


def make_sweep_id(base: RunSpec, grid: dict[str, Sequence[Any]]) -> str:
    """Deterministic sweep id from strategy + grid, so re-running is stable."""
    payload = json.dumps(
        {"strategy": base.strategy, "grid": {k: list(v) for k, v in grid.items()}},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{base.strategy}_{digest}"


def expand_grid(base: RunSpec, grid: dict[str, Sequence[Any]], sweep_id: str) -> list[RunSpec]:
    """Cartesian product of ``grid`` -> one RunSpec per cell (tagged with sweep_id)."""
    keys = list(grid)
    specs: list[RunSpec] = []
    for combo in product(*(grid[k] for k in keys)):
        cell = dict(zip(keys, combo, strict=True))
        specs.append(
            base.model_copy(update={"params": {**base.params, **cell}, "sweep_id": sweep_id})
        )
    return specs


def run_sweep(
    base_spec: RunSpec,
    grid: dict[str, Sequence[Any]],
    bars: pd.DataFrame,
    *,
    registry: ExperimentRegistry | None = None,
    n_workers: int | None = None,
    sweep_id: str | None = None,
) -> pd.DataFrame:
    """Run a parameter sweep in parallel and log every cell to the registry.

    Args:
        base_spec: the strategy/universe/dates/costs shared by every cell.
        grid: ``{param_name: [values...]}`` — the cartesian product is swept.
        bars: canonical bars (fetched once, shared to all workers).
        registry: where results are logged (defaults to a local registry).
        n_workers: process count (defaults to all cores — saturate the machine).
        sweep_id: override the auto/deterministic id.

    Returns:
        The registry query for this sweep (one row per cell, metrics as ``m_*``).
    """
    registry = registry if registry is not None else ExperimentRegistry()
    store = registry.store

    sid = sweep_id or make_sweep_id(base_spec, grid)
    specs = expand_grid(base_spec, grid, sid)
    workers = n_workers or os.cpu_count() or 1
    workers = min(workers, len(specs))

    store.register_sweep(
        sid,
        base_spec.strategy,
        {k: list(v) for k, v in grid.items()},
        len(specs),
        datetime.now(UTC).isoformat(),
    )

    tf = str(bars.attrs.get("timeframe", "1day"))
    adjusted = bool(bars.attrs.get("adjusted", True))
    store_descriptor = store.to_worker_descriptor()
    saved = _pin_blas_single_thread()
    tmpdir = tempfile.mkdtemp(prefix="ehf_sweep_")
    bars_path = os.path.join(tmpdir, "bars.parquet")
    bars.to_parquet(bars_path, index=False)  # written once; workers read it
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(bars_path, tf, adjusted, store_descriptor),
        ) as pool:
            # Workers compute + write their own arrays in parallel; collect the
            # tiny metrics payloads, then the parent bulk-logs metadata in ONE
            # transaction with a single git-identity call.
            items = list(pool.map(_run_cell, specs))
    finally:
        _restore_env(saved)
        shutil.rmtree(tmpdir, ignore_errors=True)

    registry.log_runs_bulk(items)
    return registry.query("WHERE sweep_id = ?", (sid,))


def _pin_blas_single_thread() -> dict[str, str | None]:
    """Pin BLAS to 1 thread/worker (children inherit env on spawn). Returns prior."""
    saved = {v: os.environ.get(v) for v in _BLAS_VARS}
    for v in _BLAS_VARS:
        os.environ[v] = "1"
    return saved


def _restore_env(saved: dict[str, str | None]) -> None:
    for v, val in saved.items():
        if val is None:
            os.environ.pop(v, None)
        else:
            os.environ[v] = val
