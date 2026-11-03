"""Schedulable entrypoints (one module per job) + the runtime they share.

A *job* is a thin CLI that the Studio's launchd (``infra/studio``) or any other
scheduler runs as ``python -m ehf.research.jobs.<name>``. Jobs parse
arguments, take a lock, configure logging, call a library function, and write a
summary — nothing else. All research logic lives in the libraries they call.

  refresh_data     nightly price-cache + universe-snapshot refresh (data.refresh)
  vix_study        weekly VIX overlay study (walk-forward + gauntlet -> scorecards)
  leverage_study   monthly leverage study (3x vs plain fund when the market turns bullish)
  sync_var         nightly copy of var/ up to S3 for the deployed lab (ADR-0009)
  (next)           research_batch: backlog -> walk-forward -> gauntlet -> scorecards

Shared runtime (this module): ``single_instance`` (no overlapping runs),
``configure_logging`` (stdout + rotating file under ``var/log``), ``write_summary``
(``var/run/<job>.last.json`` — what the morning email and a status command read).
"""

from __future__ import annotations

import fcntl
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import IO, Any

from ehf.research.paths import LOG_DIR, RUN_DIR, ensure_dirs


class AlreadyRunning(RuntimeError):
    """Another instance of the job holds the lock."""


@contextmanager
def single_instance(job: str) -> Iterator[None]:
    """Hold ``var/run/<job>.lock`` for the duration; raise if someone else does."""
    ensure_dirs()
    path = RUN_DIR / f"{job}.lock"
    fh: IO[str] = path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        fh.close()
        raise AlreadyRunning(f"{job}: lock held ({path})") from exc
    try:
        fh.write(f"{datetime.now(UTC).isoformat()}\n")
        fh.flush()
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def configure_logging(job: str, *, verbose: bool = False) -> logging.Logger:
    """Log to stdout and to ``var/log/<job>.log`` (rotated at 10 MB x 5)."""
    ensure_dirs()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        fh = RotatingFileHandler(LOG_DIR / f"{job}.log", maxBytes=10_000_000, backupCount=5)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    if not any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stdout for h in root.handlers
    ):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    for noisy in ("urllib3", "botocore", "boto3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(f"ehf.jobs.{job}")


def write_summary(job: str, payload: dict[str, Any]) -> Path:
    """Persist the job's last-run summary as JSON (overwrites)."""
    ensure_dirs()
    path = RUN_DIR / f"{job}.last.json"
    body = {"job": job, "finished_at": datetime.now(UTC).isoformat(), **payload}
    path.write_text(json.dumps(body, indent=1, default=str))
    return path


def read_summary(job: str) -> dict[str, Any] | None:
    path = RUN_DIR / f"{job}.last.json"
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def load_dotenv_quietly() -> None:
    """Load the project ``.env`` if python-dotenv is available (never overrides env)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - optional dependency
        pass
