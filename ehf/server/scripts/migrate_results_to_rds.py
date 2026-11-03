"""Backfill the local results store (SQLite + parquet) into RDS Postgres + S3.

Reads the local ``var/results`` store and replays every run, sweep, and metric
into the cloud backend so RDS+S3 reflects your local environment. Idempotent: the
metadata upserts on ``config_hash`` and array uploads overwrite, so re-running is
safe (handy if you migrate, keep working locally, then top up).

Usage::

    # Configure the cloud backend (same env the registry reads):
    export EHF_DATABASE_URL='postgresql://user:pass@host:5432/ehf'
    export EHF_RESULTS_S3_BUCKET='my-ehf-bucket'
    # ...or EHF_DB_SECRET_ARN=<secrets-manager-arn> instead of the URL.

    python ehf/server/scripts/migrate_results_to_rds.py [--local-root PATH] [--dry-run]

The destination is built from the environment via ``result_store_from_env`` — the
same switch the registry uses — so a misconfigured env fails loudly here, not in
production logging.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from ehf.server.research.results.config import result_store_from_env
from ehf.server.research.results.store import DEFAULT_RESULTS_DIR, LocalResultStore
from ehf.server.research.results.store_postgres import PostgresResultStore

_BATCH = 200  # runs per metadata transaction


def _rows(conn: sqlite3.Connection, sql: str) -> list[dict]:
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql)]


def migrate(local_root: Path, dest: PostgresResultStore, *, dry_run: bool) -> None:
    local = LocalResultStore(local_root)
    db_path = local.db_path
    if not db_path.exists():
        sys.exit(f"no local results DB at {db_path} — nothing to migrate")

    with sqlite3.connect(db_path) as conn:
        runs = _rows(conn, "SELECT * FROM runs")
        metrics_rows = _rows(conn, "SELECT config_hash, name, value FROM metrics")
        sweeps = _rows(conn, "SELECT * FROM sweeps")

    metrics: dict[str, dict[str, float]] = {}
    for r in metrics_rows:
        metrics.setdefault(r["config_hash"], {})[r["name"]] = r["value"]

    print(f"local store: {db_path}")
    print(f"  runs={len(runs)}  metrics={len(metrics_rows)}  sweeps={len(sweeps)}")
    if dry_run:
        print("dry-run: nothing written")
        return

    # 1. Sweeps (so runs' sweep_id FKs have somewhere to point conceptually).
    for s in sweeps:
        dest.register_sweep(
            s["sweep_id"], s["strategy"], json.loads(s["grid_json"]), s["n_cells"], s["created_at"]
        )
    print(f"  -> {len(sweeps)} sweeps written")

    # 2. Per-run arrays to S3 (skip runs missing parquet — metadata still migrates).
    uploaded = 0
    for i, r in enumerate(runs, 1):
        h = r["config_hash"]
        run_dir = local_root / h
        eq, tr = run_dir / "equity.parquet", run_dir / "trades.parquet"
        if eq.exists() and tr.exists():
            dest.write_arrays(h, pd.read_parquet(eq), pd.read_parquet(tr))
            uploaded += 1
        if i % 50 == 0 or i == len(runs):
            print(f"  -> arrays {i}/{len(runs)} ({uploaded} uploaded)")

    # 3. Metadata + metrics in batched transactions.
    for start in range(0, len(runs), _BATCH):
        batch = runs[start : start + _BATCH]
        batch_metrics = {r["config_hash"]: metrics.get(r["config_hash"], {}) for r in batch}
        dest.write_metadata_bulk(batch, batch_metrics)
    print(f"  -> {len(runs)} run records + metrics written")
    print("done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-root", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = ap.parse_args()

    dest = result_store_from_env()
    if not isinstance(dest, PostgresResultStore):
        sys.exit(
            "destination is not the cloud backend — set EHF_DATABASE_URL (or "
            "EHF_DB_SECRET_ARN) and EHF_RESULTS_S3_BUCKET before migrating."
        )
    migrate(args.local_root, dest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
