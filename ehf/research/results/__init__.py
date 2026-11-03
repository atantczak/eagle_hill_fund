"""Reproducible results storage.

``ExperimentRegistry`` is the only public API. It writes through to a
``ResultStore`` backend and guarantees every run can be regenerated from its
``config_hash``. Two backends implement the contract (ADR-0002):

  * ``LocalResultStore``    — SQLite + local parquet (the dev default).
  * ``PostgresResultStore`` — RDS Postgres + S3 (the cloud backend).

Which one a bare ``ExperimentRegistry()`` uses is decided by the environment via
``result_store_from_env`` (see ``results.config``): set the DB env and every
registry writes to RDS+S3; unset it and it stays local. Strategies and the
walk-forward harness log through the registry — nothing touches the raw store.

``PostgresResultStore`` / ``result_store_from_env`` / ``result_store_from_descriptor``
are exposed lazily so psycopg/boto3 are imported only when the cloud backend is used.
"""

from typing import TYPE_CHECKING, Any

from ehf.research.results.registry import ExperimentRegistry, RunResult
from ehf.research.results.store import LocalResultStore, ResultStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ehf.research.results.config import (
        result_store_from_descriptor,
        result_store_from_env,
    )
    from ehf.research.results.store_postgres import PostgresResultStore

__all__ = [
    "ExperimentRegistry",
    "RunResult",
    "ResultStore",
    "LocalResultStore",
    "PostgresResultStore",
    "result_store_from_env",
    "result_store_from_descriptor",
]

_LAZY = {
    "PostgresResultStore": ("store_postgres", "PostgresResultStore"),
    "result_store_from_env": ("config", "result_store_from_env"),
    "result_store_from_descriptor": ("config", "result_store_from_descriptor"),
}


def __getattr__(name: str) -> Any:
    """Lazily import the cloud-backend symbols on first access (PEP 562)."""
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        mod = importlib.import_module(f"ehf.research.results.{module}")
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
