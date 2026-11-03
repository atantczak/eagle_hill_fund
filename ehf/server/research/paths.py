"""Where generated runtime artifacts live — one anchor for every ``var/`` path.

Everything the platform *generates* (price cache, results DB + parquet, universe
snapshots, job logs, lock files) lives under ``ehf/server/var/`` (the Unix
``/var`` convention), git-ignored, anchored to the package so the paths hold
regardless of the caller's working directory. New artifact kinds get a constant
here — never a hardcoded path elsewhere.
"""

from __future__ import annotations

from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]  # ehf/server
VAR_DIR = SERVER_DIR / "var"

CACHE_DIR = VAR_DIR / "cache"  # parquet price cache (BarCache)
RESULTS_DIR = VAR_DIR / "results"  # SQLite results DB + per-run parquet (LocalResultStore)
UNIVERSE_DIR = VAR_DIR / "universe"  # daily point-in-time universe snapshots
LOG_DIR = VAR_DIR / "log"  # scheduled-job logs
RUN_DIR = VAR_DIR / "run"  # lock files + last-run summaries
SCORECARD_DIR = VAR_DIR / "scorecards"  # walk-forward / gauntlet scorecards (markdown + json)


def ensure_dirs() -> None:
    """Create every runtime directory (idempotent; called by jobs at startup)."""
    for d in (CACHE_DIR, RESULTS_DIR, UNIVERSE_DIR, LOG_DIR, RUN_DIR, SCORECARD_DIR):
        d.mkdir(parents=True, exist_ok=True)
