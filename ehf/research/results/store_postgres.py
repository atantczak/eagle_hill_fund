"""Cloud result store — RDS Postgres (relational core) + S3 (heavy arrays).

The cloud sibling of ``LocalResultStore``, behind the same ``ResultStore``
contract (ADR-0002). The split mirrors local one-for-one:

  * **Postgres (RDS)** holds the queryable relational core — ``runs``,
    ``metrics``, ``sweeps``, ``theses`` — the analog of the local SQLite file.
  * **S3** holds the bulky per-run arrays — ``equity.parquet`` /
    ``trades.parquet`` keyed by ``config_hash`` — the analog of the local
    ``var/results/<hash>/`` parquet directory. Same parquet bytes, different
    home, so nothing about serialization changes.

Point ``ExperimentRegistry`` at this and no strategy, engine, or analysis code
changes. ``psycopg`` (v3) and ``boto3`` are imported lazily inside the methods
that need them, so local-only users never pay for (or need to install) them.

Idempotency is preserved: ``write_metadata_bulk`` upserts via
``INSERT ... ON CONFLICT (config_hash) DO UPDATE``, the Postgres equivalent of
SQLite's ``INSERT OR REPLACE``.
"""

from __future__ import annotations

import io
import json
from functools import cached_property
from typing import TYPE_CHECKING, Any

import pandas as pd

from ehf.research.results.store import _SCHEMA, ResultStore, _as_float

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3 import S3Client

# SQLite DDL is almost portable to Postgres; only the two scalar types differ
# (REAL -> DOUBLE PRECISION). Reusing ``_SCHEMA`` keeps the two backends'
# table shapes provably identical instead of letting them drift.
_PG_SCHEMA = _SCHEMA.replace(" REAL", " DOUBLE PRECISION")


def qmark_to_pyformat(sql: str) -> str:
    """Translate SQLite ``?`` placeholders to psycopg's ``%s``.

    The ``query()`` API takes ``?``-style clauses (the LocalResultStore
    convention) so callers are backend-agnostic; psycopg uses ``%s``. Clauses are
    internal and never contain ``?`` literals, so a straight swap is safe. We also
    escape any literal ``%`` so psycopg's own param substitution leaves it alone.
    """
    return sql.replace("%", "%%").replace("?", "%s")


def upsert_sql(table: str, columns: list[str], conflict_key: str) -> str:
    """Build an ``INSERT ... ON CONFLICT (key) DO UPDATE`` for the given columns.

    Every non-key column is overwritten from the proposed row (``EXCLUDED``), so a
    re-logged run updates in place — the Postgres analog of ``INSERT OR REPLACE``.
    """
    cols = ", ".join(columns)
    placeholders = ", ".join("%s" for _ in columns)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c != conflict_key)
    return (
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_key}) DO UPDATE SET {updates}"
    )


class PostgresResultStore(ResultStore):
    """RDS Postgres (metadata + metrics) + S3 (equity + trades parquet)."""

    def __init__(
        self,
        dsn: str,
        s3_bucket: str,
        *,
        s3_prefix: str = "ehf-results",
        region: str | None = None,
        ensure_schema: bool = True,
    ):
        self._dsn = dsn
        self._bucket = s3_bucket
        self._s3_prefix = s3_prefix.strip("/")
        self._region = region
        # Connecting (and creating the schema) is deferred until first use so the
        # store is cheap to construct and pickle for sweep workers.
        self._schema_ready = not ensure_schema

    # ---- connections (lazy; opened per operation) -------------------------- #
    def _connect(self) -> Any:
        import psycopg

        conn = psycopg.connect(self._dsn)
        if not self._schema_ready:
            with conn.cursor() as cur:
                cur.execute(_PG_SCHEMA)
            conn.commit()
            self._schema_ready = True
        return conn

    @cached_property
    def _s3(self) -> S3Client:
        import boto3

        return boto3.client("s3", region_name=self._region)

    # ---- S3 array path ----------------------------------------------------- #
    def _array_key(self, config_hash: str, name: str) -> str:
        return f"{self._s3_prefix}/{config_hash}/{name}.parquet"

    def _put_parquet(self, key: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=buf.getvalue())

    def _get_parquet(self, key: str) -> pd.DataFrame:
        obj = self._s3.get_object(Bucket=self._bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    def write_arrays(self, config_hash: str, equity: pd.DataFrame, trades: pd.DataFrame) -> None:
        """Upload a run's equity curve + trade ledger to S3 as parquet.

        Split out so sweep *workers* upload their own arrays in parallel (distinct
        S3 keys, no contention) while the parent only does the metadata transaction.
        """
        self._put_parquet(self._array_key(config_hash, "equity"), equity)
        self._put_parquet(self._array_key(config_hash, "trades"), trades)

    # ---- Postgres metadata path ------------------------------------------- #
    def has_run(self, config_hash: str) -> bool:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM runs WHERE config_hash = %s", (config_hash,))
            return cur.fetchone() is not None

    def write_metadata_bulk(
        self, records: list[dict[str, Any]], metrics: dict[str, dict[str, float]]
    ) -> None:
        if not records and not metrics:
            return
        with self._connect() as conn, conn.cursor() as cur:
            for record in records:
                cols = list(record.keys())
                cur.execute(upsert_sql("runs", cols, "config_hash"), tuple(record.values()))
            for config_hash, m in metrics.items():
                cur.execute("DELETE FROM metrics WHERE config_hash = %s", (config_hash,))
                cur.executemany(
                    "INSERT INTO metrics (config_hash, name, value) VALUES (%s, %s, %s)",
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
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM runs WHERE config_hash = %s", (config_hash,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"no run with config_hash={config_hash!r}")
            colnames = [d.name for d in cur.description]
            out = dict(zip(colnames, row, strict=True))
            cur.execute("SELECT name, value FROM metrics WHERE config_hash = %s", (config_hash,))
            metrics = {name: value for name, value in cur.fetchall()}
        out["spec"] = json.loads(out["spec_json"])
        out["params"] = json.loads(out["params_json"])
        out["metrics"] = metrics
        out["equity"] = self._get_parquet(self._array_key(config_hash, "equity"))
        out["trades"] = self._get_parquet(self._array_key(config_hash, "trades"))
        return out

    def query(self, clause: str = "", params: tuple[Any, ...] = ()) -> pd.DataFrame:
        """Cross-run query; metrics pivoted into ``m_<name>`` columns (as Local).

        Same shape as ``LocalResultStore.query``: a CTE pivots the long ``metrics``
        table to wide ``m_*`` columns so ``clause`` can filter/order on both run
        and metric columns uniformly. ``clause`` uses ``?`` placeholders, which we
        translate to psycopg's ``%s``.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT name FROM metrics")
            metric_names = [r[0] for r in cur.fetchall()]
            pivots = ", ".join(
                f"MAX(CASE WHEN m.name = '{n}' THEN m.value END) AS m_{n}" for n in metric_names
            )
            inner_select = "r.*" + (f", {pivots}" if pivots else "")
            sql = (
                f"WITH joined AS ("
                f"  SELECT {inner_select} FROM runs r"
                f"  LEFT JOIN metrics m ON m.config_hash = r.config_hash"
                f"  GROUP BY r.config_hash"
                f") SELECT * FROM joined {qmark_to_pyformat(clause)}".strip()
            )
            cur.execute(sql, params)
            colnames = [d.name for d in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=colnames)

    def register_sweep(
        self, sweep_id: str, strategy: str, grid: dict[str, Any], n_cells: int, created_at: str
    ) -> None:
        cols = ["sweep_id", "strategy", "grid_json", "n_cells", "created_at"]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                upsert_sql("sweeps", cols, "sweep_id"),
                (sweep_id, strategy, json.dumps(grid, sort_keys=True), n_cells, created_at),
            )
            conn.commit()

    def to_worker_descriptor(self) -> dict[str, Any]:
        # Workers only call write_arrays (-> S3); they never open the DB. The DSN
        # rides along (in-memory, via the pool's spawn pickling) so the rebuilt
        # store is a faithful copy. ensure_schema is off — the parent already ran it.
        return {
            "kind": "postgres",
            "dsn": self._dsn,
            "s3_bucket": self._bucket,
            "s3_prefix": self._s3_prefix,
            "region": self._region,
            "ensure_schema": False,
        }
