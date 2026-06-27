"""Result-store selection from environment — the one place that decides backend.

``result_store_from_env()`` returns the cloud backend (RDS Postgres + S3) when a
database connection is configured, and otherwise the local SQLite+parquet store.
This is the single switch that makes RDS canonical: set the env, every
``ExperimentRegistry()`` writes to RDS+S3; unset it, behavior is exactly as before.

Connection sources, in priority order:
  1. ``EHF_DATABASE_URL`` (or ``DATABASE_URL``) — a full ``postgresql://`` DSN.
  2. ``EHF_DB_SECRET_ARN`` — an AWS Secrets Manager secret holding the RDS
     credentials (the standard ``{username, password, host, port, dbname}`` JSON
     that RDS-managed secrets use); resolved via boto3 at call time.

S3 (required whenever a DB is configured):
  * ``EHF_RESULTS_S3_BUCKET`` — bucket for the equity/trade parquet arrays.
  * ``EHF_RESULTS_S3_PREFIX`` — key prefix (default ``ehf-results``).

Region falls back to ``AWS_REGION`` / ``AWS_DEFAULT_REGION``.

AWS auth (S3, Secrets Manager) uses boto3's standard credential chain — env vars,
shared profile, or instance role — nothing EHF-specific. When ``env`` is not
supplied we best-effort load a project ``.env`` first (as the rest of the repo
does), so a bare ``ExperimentRegistry()`` picks up both the EHF settings and the
AWS credentials without the caller wiring anything.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from ehf.server.research.results.store import LocalResultStore, ResultStore

Env = Mapping[str, str]

_DOTENV_LOADED = False


def _ensure_dotenv() -> None:
    """Load a project ``.env`` into ``os.environ`` once (best effort, optional dep)."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv

        load_dotenv()  # does not override already-set vars
    except Exception:  # python-dotenv absent or no .env — fine, use os.environ as-is
        pass


def result_store_from_env(env: Env | None = None) -> ResultStore:
    """Pick the backend from the environment (RDS+S3 if configured, else local)."""
    if env is None:
        _ensure_dotenv()
        env = os.environ
    dsn = _resolve_dsn(env)
    if not dsn:
        return LocalResultStore()  # unchanged local default

    bucket = env.get("EHF_RESULTS_S3_BUCKET")
    if not bucket:
        raise RuntimeError(
            "A database is configured (EHF_DATABASE_URL / EHF_DB_SECRET_ARN) but "
            "EHF_RESULTS_S3_BUCKET is unset — the cloud backend needs a bucket for "
            "the equity/trade arrays."
        )
    # Imported here so psycopg/boto3 stay optional for local-only users.
    from ehf.server.research.results.store_postgres import PostgresResultStore

    return PostgresResultStore(
        dsn=dsn,
        s3_bucket=bucket,
        s3_prefix=env.get("EHF_RESULTS_S3_PREFIX", "ehf-results"),
        region=_region(env),
    )


def result_store_from_descriptor(descriptor: dict[str, Any]) -> ResultStore:
    """Rebuild a store from ``to_worker_descriptor()`` (used by sweep workers)."""
    kind = descriptor.get("kind")
    if kind == "local":
        return LocalResultStore(descriptor["root"])
    if kind == "postgres":
        from ehf.server.research.results.store_postgres import PostgresResultStore

        return PostgresResultStore(
            dsn=descriptor["dsn"],
            s3_bucket=descriptor["s3_bucket"],
            s3_prefix=descriptor.get("s3_prefix", "ehf-results"),
            region=descriptor.get("region"),
            ensure_schema=descriptor.get("ensure_schema", True),
        )
    raise ValueError(f"unknown store descriptor kind {kind!r}")


def _resolve_dsn(env: Env) -> str | None:
    if env.get("EHF_DATABASE_URL"):
        return env["EHF_DATABASE_URL"]
    if env.get("DATABASE_URL"):
        return env["DATABASE_URL"]
    secret_id = env.get("EHF_DB_SECRET_ARN")
    if secret_id:
        return _dsn_from_secret(secret_id, env)
    return None


def _dsn_from_secret(secret_id: str, env: Env) -> str:
    import boto3

    sm = boto3.client("secretsmanager", region_name=_region(env))
    payload = sm.get_secret_value(SecretId=secret_id)["SecretString"]
    return dsn_from_secret_payload(payload, default_dbname=env.get("EHF_DB_NAME", "ehf"))


def dsn_from_secret_payload(payload: str, *, default_dbname: str = "ehf") -> str:
    """Build a ``postgresql://`` DSN from an RDS-style Secrets Manager JSON blob.

    Pure (no AWS) so it is unit-testable: RDS-managed secrets store
    ``{username, password, host, port, dbname}``; ``dbname`` falls back to
    ``default_dbname`` when the secret omits it (common for cluster-level secrets).
    """
    d = json.loads(payload)
    user = d["username"]
    pw = d["password"]
    host = d["host"]
    port = d.get("port", 5432)
    dbname = d.get("dbname") or default_dbname
    return f"postgresql://{user}:{pw}@{host}:{port}/{dbname}"


def _region(env: Env) -> str | None:
    return env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
