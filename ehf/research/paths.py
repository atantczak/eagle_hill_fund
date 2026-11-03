"""Where generated runtime artifacts live — one anchor for every ``var/`` path.

Everything the platform *generates* (price cache, results DB + parquet, universe
snapshots, job logs, lock files) lives under ``var/`` (the Unix
``/var`` convention), git-ignored, anchored to the repository root so the paths hold
regardless of the caller's working directory. New artifact kinds get a constant
here — never a hardcoded path elsewhere.

``EHF_VAR_DIR`` relocates the whole tree (the site container sets it to its
writable scratch volume, ``/var/ehf``, and mirrors the Studio's ``var/`` into it
from S3 — see ``ehf.web.lab.mirror`` and ADR-0009). Unset, it is ``<repo>/var``.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # repository root (ehf/research/paths.py)
VAR_DIR = Path(os.getenv("EHF_VAR_DIR") or ROOT_DIR / "var").expanduser()

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
