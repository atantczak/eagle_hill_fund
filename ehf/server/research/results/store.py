"""Result-store backends. Default: local SQLite (queryable) + parquet (arrays).

See ADR-0002 for the rationale. The relational core (runs, metrics, sweeps,
theses) lives in SQLite for cheap cross-run SQL; the heavy per-run arrays
(equity curve, trade ledger) live in parquet keyed by ``config_hash``. Raw price
data is never stored per run — only references + derived results.

This module holds the ``ResultStore`` contract and the local backend. The cloud
backend (``PostgresResultStore`` — RDS Postgres + S3) implements the same
contract in ``store_postgres`` and is selected via ``results.config`` when the
environment is configured; nothing in the registry or strategy code changes.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from ehf.server.research.paths import RESULTS_DIR as DEFAULT_RESULTS_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    config_hash   TEXT PRIMARY KEY,
    strategy      TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    universe      TEXT NOT NULL,
    data_start    TEXT NOT NULL,
    data_end      TEXT NOT NULL,
    timeframe     TEXT NOT NULL,
    initial_capital REAL NOT NULL,
    cost_model_json TEXT NOT NULL,
    fill_model    TEXT NOT NULL,
    git_sha       TEXT,
    dirty         INTEGER NOT NULL DEFAULT 0,
    spec_json     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'backtest',
    kill_reason   TEXT,
    sweep_id      TEXT,
    parent_hash   TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_runs_strategy ON runs(strategy);
CREATE INDEX IF NOT EXISTS ix_runs_sweep ON runs(sweep_id);
CREATE INDEX IF NOT EXISTS ix_runs_status ON runs(status);

CREATE TABLE IF NOT EXISTS metrics (
    config_hash TEXT NOT NULL REFERENCES runs(config_hash) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    value       REAL,
    PRIMARY KEY (config_hash, name)
);
CREATE INDEX IF NOT EXISTS ix_metrics_name ON metrics(name);

CREATE TABLE IF NOT EXISTS sweeps (
    sweep_id   TEXT PRIMARY KEY,
    strategy   TEXT NOT NULL,
    grid_json  TEXT NOT NULL,
    n_cells    INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS theses (
    thesis_id   TEXT PRIMARY KEY,
    symbol      TEXT,
    hypothesis  TEXT NOT NULL,
    conditions_json TEXT,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  TEXT NOT NULL
);
"""


class ResultStore(ABC):
    """Backend interface. ``ExperimentRegistry`` is the only caller.

    Two concrete backends implement this: ``LocalResultStore`` (SQLite + local
    parquet, the dev default) and ``PostgresResultStore`` (RDS Postgres + S3, the
    cloud backend). The contract is split into a metadata path and an array path
    so a parallel sweep can have workers write their own arrays concurrently
    (``write_arrays``) while the parent does one bulk metadata transaction
    (``write_metadata_bulk``) — both backends honor that split.
    """

    @abstractmethod
    def has_run(self, config_hash: str) -> bool: ...

    @abstractmethod
    def write_run(
        self,
        record: dict[str, Any],
        metrics: dict[str, float],
        equity: pd.DataFrame,
        trades: pd.DataFrame,
    ) -> None: ...

    @abstractmethod
    def write_arrays(self, config_hash: str, equity: pd.DataFrame, trades: pd.DataFrame) -> None:
        """Persist a run's heavy arrays (equity curve, trade ledger)."""

    @abstractmethod
    def write_metadata_bulk(
        self, records: list[dict[str, Any]], metrics: dict[str, dict[str, float]]
    ) -> None:
        """Insert many runs' metadata + metrics in one transaction."""

    @abstractmethod
    def register_sweep(
        self, sweep_id: str, strategy: str, grid: dict[str, Any], n_cells: int, created_at: str
    ) -> None:
        """Record a sweep's grid so its cells can be explored as a set."""

    @abstractmethod
    def read_run(self, config_hash: str) -> dict[str, Any]: ...

    @abstractmethod
    def query(self, clause: str = "", params: tuple[Any, ...] = ()) -> pd.DataFrame: ...

    @abstractmethod
    def to_worker_descriptor(self) -> dict[str, Any]:
        """A picklable descriptor a sweep worker uses to rebuild this store.

        Workers run in separate processes (spawn) and only call ``write_arrays``;
        the descriptor carries just enough config to reconstruct the store there.
        Rebuilt via ``results.config.result_store_from_descriptor``.
        """


class LocalResultStore(ResultStore):
    """SQLite (metadata + metrics) + parquet (equity + trades) on local disk."""

    def __init__(self, root: str | Path = DEFAULT_RESULTS_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "ehf.db"
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _run_dir(self, config_hash: str) -> Path:
        d = self.root / config_hash
        d.mkdir(parents=True, exist_ok=True)
        return d

    def has_run(self, config_hash: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("SELECT 1 FROM runs WHERE config_hash = ?", (config_hash,))
            return cur.fetchone() is not None

    def write_arrays(self, config_hash: str, equity: pd.DataFrame, trades: pd.DataFrame) -> None:
        """Write a run's heavy arrays (equity curve, trade ledger) to parquet.

        Split out so a parallel sweep can have *workers* write their own arrays
        (different files, no contention) while the parent only does metadata.
        """
        run_dir = self._run_dir(config_hash)
        equity.to_parquet(run_dir / "equity.parquet", index=False)
        trades.to_parquet(run_dir / "trades.parquet", index=False)

    def write_metadata_bulk(
        self, records: list[dict[str, Any]], metrics: dict[str, dict[str, float]]
    ) -> None:
        """Insert many runs' metadata + metrics in a single transaction."""
        with self._connect() as conn:
            for record in records:
                cols = ", ".join(record.keys())
                placeholders = ", ".join("?" for _ in record)
                conn.execute(
                    f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({placeholders})",
                    tuple(record.values()),
                )
            for config_hash, m in metrics.items():
                conn.execute("DELETE FROM metrics WHERE config_hash = ?", (config_hash,))
                conn.executemany(
                    "INSERT INTO metrics (config_hash, name, value) VALUES (?, ?, ?)",
                    [(config_hash, k, _as_float(v)) for k, v in m.items()],
                )
            conn.commit()

    def write_run(
        self,
        record: dict[str, Any],
        metrics: dict[str, float],
        equity: pd.DataFrame,
        trades: pd.DataFrame,
    ) -> None:
        self.write_arrays(record["config_hash"], equity, trades)
        self.write_metadata_bulk([record], {record["config_hash"]: metrics})

    def read_run(self, config_hash: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE config_hash = ?", (config_hash,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no run with config_hash={config_hash!r}")
            metrics = {
                r["name"]: r["value"]
                for r in conn.execute(
                    "SELECT name, value FROM metrics WHERE config_hash = ?",
                    (config_hash,),
                )
            }
        run_dir = self.root / config_hash
        equity = pd.read_parquet(run_dir / "equity.parquet")
        trades = pd.read_parquet(run_dir / "trades.parquet")
        out = dict(row)
        out["spec"] = json.loads(out["spec_json"])
        out["params"] = json.loads(out["params_json"])
        out["metrics"] = metrics
        out["equity"] = equity
        out["trades"] = trades
        return out

    def query(self, clause: str = "", params: tuple[Any, ...] = ()) -> pd.DataFrame:
        """Cross-run query. Metrics are pivoted into ``m_<name>`` columns.

        The pivot is wrapped in a CTE so ``clause`` can filter and/or order on
        both run columns and metric columns uniformly. ``clause`` is the trailing
        SQL appended after ``SELECT * FROM joined``, e.g.::

            query("WHERE strategy = ? ORDER BY m_sharpe DESC", ("rise_fall",))
        """
        with self._connect() as conn:
            metric_names = [r["name"] for r in conn.execute("SELECT DISTINCT name FROM metrics")]
            pivots = ", ".join(
                f"MAX(CASE WHEN m.name = '{n}' THEN m.value END) AS m_{n}" for n in metric_names
            )
            inner_select = "r.*" + (f", {pivots}" if pivots else "")
            sql = (
                f"WITH joined AS ("
                f"  SELECT {inner_select} FROM runs r"
                f"  LEFT JOIN metrics m ON m.config_hash = r.config_hash"
                f"  GROUP BY r.config_hash"
                f") SELECT * FROM joined {clause}".strip()
            )
            return pd.read_sql_query(sql, conn, params=params)

    def register_sweep(
        self, sweep_id: str, strategy: str, grid: dict[str, Any], n_cells: int, created_at: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sweeps (sweep_id, strategy, grid_json, n_cells, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (sweep_id, strategy, json.dumps(grid, sort_keys=True), n_cells, created_at),
            )
            conn.commit()

    def to_worker_descriptor(self) -> dict[str, Any]:
        return {"kind": "local", "root": str(self.root)}


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


# The cloud backend (RDS Postgres + S3) lives in ``store_postgres`` so that
# psycopg/boto3 are only imported when it is actually used. Import it from there
# or via ``results.PostgresResultStore`` (lazy) — see ADR-0002.
