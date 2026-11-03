"""Cloud results backend (RDS Postgres + S3): config, SQL translation, wiring.

The pure logic — placeholder translation, upsert SQL, secret parsing, env-based
backend selection, worker-descriptor round-trips — is tested without any AWS or
database (constructing ``PostgresResultStore`` never connects; that is deferred to
first use). A live round-trip test is included but skipped unless
``EHF_TEST_DATABASE_URL`` + ``EHF_TEST_S3_BUCKET`` are set.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ehf.server.research.config.spec import RunSpec, UniverseSpec
from ehf.server.research.results.config import (
    dsn_from_secret_payload,
    result_store_from_descriptor,
    result_store_from_env,
)
from ehf.server.research.results.store import LocalResultStore
from ehf.server.research.results.store_postgres import (
    PostgresResultStore,
    qmark_to_pyformat,
    upsert_sql,
)

_PG_ENV = {"EHF_DATABASE_URL": "postgresql://u:p@h:5432/ehf", "EHF_RESULTS_S3_BUCKET": "bkt"}


# --------------------------------------------------------------------------- #
# SQL translation helpers                                                     #
# --------------------------------------------------------------------------- #
def test_qmark_to_pyformat_translates_placeholders() -> None:
    assert qmark_to_pyformat("WHERE strategy = ? ORDER BY m_sharpe DESC") == (
        "WHERE strategy = %s ORDER BY m_sharpe DESC"
    )
    assert qmark_to_pyformat("") == ""


def test_qmark_to_pyformat_escapes_literal_percent() -> None:
    # A literal % (e.g. a LIKE pattern) must survive psycopg's own substitution.
    assert qmark_to_pyformat("WHERE x LIKE 'a%' AND y = ?") == "WHERE x LIKE 'a%%' AND y = %s"


def test_upsert_sql_updates_non_key_columns() -> None:
    sql = upsert_sql("runs", ["config_hash", "strategy", "status"], "config_hash")
    assert sql == (
        "INSERT INTO runs (config_hash, strategy, status) VALUES (%s, %s, %s) "
        "ON CONFLICT (config_hash) DO UPDATE SET strategy = EXCLUDED.strategy, "
        "status = EXCLUDED.status"
    )


# --------------------------------------------------------------------------- #
# Secrets Manager parsing (pure)                                              #
# --------------------------------------------------------------------------- #
def test_dsn_from_secret_payload_full() -> None:
    payload = (
        '{"username": "ehf", "password": "s3cr3t", "host": "db.abc.rds.amazonaws.com", '
        '"port": 5432, "dbname": "research"}'
    )
    assert dsn_from_secret_payload(payload) == (
        "postgresql://ehf:s3cr3t@db.abc.rds.amazonaws.com:5432/research"
    )


def test_dsn_from_secret_payload_defaults_dbname() -> None:
    payload = '{"username": "ehf", "password": "pw", "host": "h"}'
    dsn = dsn_from_secret_payload(payload, default_dbname="ehf")
    assert dsn == "postgresql://ehf:pw@h:5432/ehf"  # port + dbname defaulted


# --------------------------------------------------------------------------- #
# Backend selection from environment                                          #
# --------------------------------------------------------------------------- #
def test_env_without_db_is_local() -> None:
    assert isinstance(result_store_from_env({}), LocalResultStore)


def test_env_with_db_is_postgres() -> None:
    store = result_store_from_env(_PG_ENV)
    assert isinstance(store, PostgresResultStore)
    assert store._bucket == "bkt"
    assert store._dsn == "postgresql://u:p@h:5432/ehf"


def test_env_db_without_bucket_raises() -> None:
    with pytest.raises(RuntimeError, match="EHF_RESULTS_S3_BUCKET"):
        result_store_from_env({"EHF_DATABASE_URL": "postgresql://u:p@h/ehf"})


def test_env_prefix_and_region_threaded_through() -> None:
    env = {**_PG_ENV, "EHF_RESULTS_S3_PREFIX": "custom/pfx", "AWS_REGION": "us-east-2"}
    store = result_store_from_env(env)
    assert store._s3_prefix == "custom/pfx"
    assert store._region == "us-east-2"


# --------------------------------------------------------------------------- #
# Worker-descriptor round-trips (sweep parallelism)                           #
# --------------------------------------------------------------------------- #
def test_local_descriptor_round_trip(tmp_path) -> None:
    original = LocalResultStore(tmp_path)
    rebuilt = result_store_from_descriptor(original.to_worker_descriptor())
    assert isinstance(rebuilt, LocalResultStore)
    assert rebuilt.root == original.root


def test_postgres_descriptor_round_trip() -> None:
    original = result_store_from_env(_PG_ENV)
    desc = original.to_worker_descriptor()
    assert desc["kind"] == "postgres"
    assert desc["ensure_schema"] is False  # parent already ensured the schema
    rebuilt = result_store_from_descriptor(desc)
    assert isinstance(rebuilt, PostgresResultStore)
    assert (rebuilt._dsn, rebuilt._bucket) == (original._dsn, original._bucket)


def test_unknown_descriptor_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown store descriptor"):
        result_store_from_descriptor({"kind": "bogus"})


# --------------------------------------------------------------------------- #
# S3 key layout                                                               #
# --------------------------------------------------------------------------- #
def test_array_key_layout() -> None:
    store = PostgresResultStore(dsn="x", s3_bucket="b", s3_prefix="ehf-results")
    assert store._array_key("abc123", "equity") == "ehf-results/abc123/equity.parquet"
    assert store._array_key("abc123", "trades") == "ehf-results/abc123/trades.parquet"


# --------------------------------------------------------------------------- #
# Live round-trip (skipped unless a test DB + bucket are configured)          #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (os.getenv("EHF_TEST_DATABASE_URL") and os.getenv("EHF_TEST_S3_BUCKET")),
    reason="set EHF_TEST_DATABASE_URL + EHF_TEST_S3_BUCKET for the live RDS+S3 round-trip",
)
def test_live_write_read_round_trip() -> None:
    from ehf.server.research.backtest.engine import run_backtest
    from ehf.server.research.results.registry import ExperimentRegistry
    from ehf.server.tests.conftest import make_daily_bars

    store = PostgresResultStore(
        dsn=os.environ["EHF_TEST_DATABASE_URL"],
        s3_bucket=os.environ["EHF_TEST_S3_BUCKET"],
        s3_prefix="ehf-test",
        region=os.getenv("AWS_REGION"),
    )
    reg = ExperimentRegistry(store=store)

    path = list(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.001, 0.01, 120))))
    bars = make_daily_bars({"A": path})
    spec = RunSpec(
        strategy="tsmom",
        params={"lookback": 30, "vol_window": 10},
        universe=UniverseSpec(symbols=("A",)),
        start="2020-01-01",
        end="2025-01-01",
    )
    h = reg.log_run(spec, run_backtest(spec, bars))

    assert store.has_run(h)
    rehydrated = reg.get_run(h)
    assert rehydrated["strategy"] == "tsmom"
    assert "sharpe" in rehydrated["metrics"]
    assert not rehydrated["equity"].empty
    assert isinstance(reg.query("WHERE strategy = ?", ("tsmom",)), pd.DataFrame)
