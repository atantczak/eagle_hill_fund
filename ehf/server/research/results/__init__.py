"""Reproducible results storage.

``ExperimentRegistry`` is the only public API. It writes through to a
``ResultStore`` backend (local SQLite + parquet by default; see ADR-0002) and
guarantees every run can be regenerated from its ``config_hash``.

Strategies and the walk-forward harness log through the registry — nothing
touches the raw store directly.
"""

from ehf.server.research.results.registry import ExperimentRegistry, RunResult
from ehf.server.research.results.store import LocalResultStore, ResultStore

__all__ = ["ExperimentRegistry", "RunResult", "ResultStore", "LocalResultStore"]
