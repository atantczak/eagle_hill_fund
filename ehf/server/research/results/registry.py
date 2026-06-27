"""``ExperimentRegistry`` — the only public API to the results store.

Computes the config hash, captures code identity (git SHA + dirty flag),
enforces idempotency, and writes through to a ``ResultStore``. Strategies and
the walk-forward harness call ``log_run``; analysis calls ``get_run`` / ``query``
/ ``compare``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ehf.server.research.config.spec import RunSpec
from ehf.server.research.results.store import LocalResultStore, ResultStore


@dataclass
class RunResult:
    """The derived output of a backtest, ready to persist.

    ``equity`` is a frame with columns [timestamp, equity] (+ optional returns).
    ``trades`` is the trade ledger. ``metrics`` is the full metrics suite.
    """

    metrics: dict[str, float]
    equity: pd.DataFrame
    trades: pd.DataFrame
    status: str = "backtest"
    kill_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ExperimentRegistry:
    """Write-through, idempotent registry over a ``ResultStore``."""

    def __init__(self, store: ResultStore | None = None):
        self.store = store if store is not None else LocalResultStore()

    def log_run(
        self,
        spec: RunSpec,
        result: RunResult,
        *,
        versioned: bool = False,
        parent_hash: str | None = None,
    ) -> str:
        """Persist a run. Returns its ``config_hash``.

        Idempotent: if a run with this hash already exists it is skipped (the
        existing hash is returned) unless ``versioned=True``, which re-stores it
        with ``parent_hash`` set so history is preserved rather than overwritten.
        """
        config_hash = spec.config_hash
        if self.store.has_run(config_hash) and not versioned:
            return config_hash

        git_sha, dirty = _git_identity()
        record = self._build_record(
            spec, result.status, result.kill_reason, git_sha, dirty, parent_hash
        )
        if dirty and result.status in {"paper", "live"}:
            import warnings

            warnings.warn(
                f"Logging a {result.status!r} run from a DIRTY working tree "
                f"(uncommitted changes); reproducibility is not guaranteed.",
                stacklevel=2,
            )
        self.store.write_run(record, result.metrics, result.equity, result.trades)
        return config_hash

    def _build_record(
        self,
        spec: RunSpec,
        status: str,
        kill_reason: str | None,
        git_sha: str | None,
        dirty: bool,
        parent_hash: str | None = None,
    ) -> dict[str, Any]:
        return {
            "config_hash": spec.config_hash,
            "strategy": spec.strategy,
            "params_json": _json(spec.params),
            "universe": spec.universe.name or ",".join(spec.universe.symbols),
            "data_start": spec.start,
            "data_end": spec.end,
            "timeframe": spec.timeframe,
            "initial_capital": spec.initial_capital,
            "cost_model_json": spec.costs.model_dump_json(),
            "fill_model": spec.fill_model.value,
            "git_sha": git_sha,
            "dirty": int(dirty),
            "spec_json": spec.canonical_json(),
            "status": status,
            "kill_reason": kill_reason,
            "sweep_id": spec.sweep_id,
            "parent_hash": parent_hash,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def log_runs_bulk(
        self, items: list[tuple[RunSpec, dict[str, float], str, str | None]]
    ) -> list[str]:
        """Log many runs efficiently — arrays already written by the caller/workers.

        Used by the parallel sweep: computes git identity ONCE, skips runs that
        already exist (idempotent), and writes all metadata + metrics in one
        SQLite transaction. ``items`` are ``(spec, metrics, status, kill_reason)``.
        Returns every cell's ``config_hash``.
        """
        from ehf.server.research.results.store import LocalResultStore

        if not isinstance(self.store, LocalResultStore):
            raise TypeError("log_runs_bulk requires a LocalResultStore")

        git_sha, dirty = _git_identity()
        records: list[dict[str, Any]] = []
        metrics_map: dict[str, dict[str, float]] = {}
        hashes: list[str] = []
        for spec, metrics, status, kill_reason in items:
            h = spec.config_hash
            hashes.append(h)
            if self.store.has_run(h):
                continue
            records.append(self._build_record(spec, status, kill_reason, git_sha, dirty))
            metrics_map[h] = metrics
        if records:
            self.store.write_metadata_bulk(records, metrics_map)
        return hashes

    def get_run(self, config_hash: str) -> dict[str, Any]:
        """Rehydrate a stored run (spec, metrics, equity curve, trade ledger)."""
        return self.store.read_run(config_hash)

    def query(self, clause: str = "", params: tuple[Any, ...] = ()) -> pd.DataFrame:
        """Cross-run query. Metric columns are exposed as ``m_<name>``.

        ``clause`` is trailing SQL after ``SELECT * FROM joined``, e.g.
        ``query("WHERE strategy = ? ORDER BY m_sharpe DESC", ("rise_fall",))``.
        """
        return self.store.query(clause, params)

    def compare(self, hashes: list[str]) -> pd.DataFrame:
        """Side-by-side metrics table for several runs (one column per run)."""
        cols = {}
        for h in hashes:
            run = self.store.read_run(h)
            cols[h] = run["metrics"]
        return pd.DataFrame(cols)


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, default=str)


def _git_identity() -> tuple[str | None, bool]:
    """Return (short git SHA, dirty?) for the current working tree."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        return sha, bool(status.strip())
    except Exception:
        return None, False
