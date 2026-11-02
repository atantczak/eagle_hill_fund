"""Nightly data refresh — keep the local price cache and universe snapshots current.

The library half of the job (``ehf.server.research.jobs.refresh_data`` is the
schedulable CLI). Pure decisions (which symbols are stale, which session should
be present) are separate functions so they're tested without a network.

What a refresh does, per universe:
  1. resolve membership as of today (``resolve_universe``) and **snapshot** it to
     ``var/universe/<name>/<YYYY-MM-DD>.parquet`` — EHF's own point-in-time record
     grows by one day every night;
  2. for each requested timeframe, find symbols whose cached tail is older than
     the last completed session (or missing entirely) and fetch them in bounded
     batches via ``FMPProvider.fetch_and_cache`` (write-through cache). Cached
     symbols get a cheap **tail top-up** (from ``TAIL_OVERLAP_DAYS`` before their
     last bar); missing symbols get full history from ``start``.
  3. ``backfill=True`` re-fetches *every* member from ``start`` — run it once when
     ``start`` moves earlier or when history is suspected incomplete, and monthly
     so back-adjusted prices (splits/dividends rewrite the past) stay current.

Holidays make every symbol look stale for a day and trigger one harmless
tail top-up; the cache merge is idempotent.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ehf.server.research.data.cache import BarCache
from ehf.server.research.data.fmp_provider import FMPProvider
from ehf.server.research.data.provider import Universe
from ehf.server.research.data.schema import Timeframe
from ehf.server.research.data.universes import resolve_universe
from ehf.server.research.paths import UNIVERSE_DIR

log = logging.getLogger(__name__)

DEFAULT_START = "2000-01-01"
SESSION_CLOSE_HOUR_ET = 17  # vendor EOD bars are complete comfortably after this
TAIL_OVERLAP_DAYS = 30  # tail top-ups start this far before the last cached bar


@dataclass
class RefreshReport:
    as_of: str
    through: str
    universes: dict[str, int] = field(default_factory=dict)
    symbols: int = 0
    refreshed: dict[str, int] = field(default_factory=dict)  # timeframe -> symbols fetched
    failed: dict[str, list[str]] = field(default_factory=dict)  # timeframe -> symbols
    snapshots: list[str] = field(default_factory=list)
    seconds: float = 0.0
    dry_run: bool = False
    backfill: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "through": self.through,
            "universes": self.universes,
            "symbols": self.symbols,
            "refreshed": self.refreshed,
            "failed": {k: len(v) for k, v in self.failed.items()},
            "failed_symbols": self.failed,
            "snapshots": self.snapshots,
            "seconds": round(self.seconds, 1),
            "dry_run": self.dry_run,
            "backfill": self.backfill,
        }


# --- pure decisions -------------------------------------------------------------------


def last_completed_session(now: datetime | None = None) -> date:
    """The most recent US session whose daily bar should exist by ``now``.

    Today (New York) if it's a weekday past the close-settlement hour, otherwise
    the previous weekday. Exchange holidays are not modelled (see module doc).
    """
    now = now or datetime.now(UTC)
    ny = pd.Timestamp(now).tz_convert("America/New_York")
    d = ny.normalize()
    if ny.weekday() >= 5 or ny.hour < SESSION_CLOSE_HOUR_ET:
        d = d - pd.offsets.BDay(1)
    return date(d.year, d.month, d.day)


def stale_symbols(
    cache: BarCache,
    symbols: Iterable[str],
    timeframe: Timeframe,
    *,
    through: date,
    adjusted: bool = True,
) -> list[str]:
    """Symbols whose cached history is missing or ends before ``through``."""
    cutoff = pd.Timestamp(through, tz="UTC")
    out: list[str] = []
    for sym in symbols:
        covered = cache.covered_range(sym, timeframe, adjusted)
        if covered is None or covered[1].normalize() < cutoff:
            out.append(sym)
    return out


def plan_fetches(
    cache: BarCache,
    symbols: Sequence[str],
    timeframe: Timeframe,
    *,
    start: str,
    batch_size: int,
    backfill: bool = False,
    adjusted: bool = True,
) -> list[tuple[str, list[str]]]:
    """Group symbols into ``(from_date, batch)`` fetches (pure given the cache).

    Missing symbols (or everything, when ``backfill``) fetch from ``start``.
    Cached symbols fetch only from ``TAIL_OVERLAP_DAYS`` before the earliest
    last-bar in their batch — one small request per symbol instead of a full
    history every night.
    """
    full: list[str] = []
    tails: list[tuple[pd.Timestamp, str]] = []
    for sym in symbols:
        covered = None if backfill else cache.covered_range(sym, timeframe, adjusted)
        if covered is None:
            full.append(sym)
        else:
            tails.append((covered[1], sym))
    plan: list[tuple[str, list[str]]] = []
    for i in range(0, len(full), batch_size):
        plan.append((start, full[i : i + batch_size]))
    tails.sort()
    for i in range(0, len(tails), batch_size):
        chunk = tails[i : i + batch_size]
        frm = (chunk[0][0] - pd.Timedelta(days=TAIL_OVERLAP_DAYS)).date().isoformat()
        plan.append((max(frm, start), [sym for _, sym in chunk]))
    return plan


def snapshot_path(name: str, as_of: date, root: Path = UNIVERSE_DIR) -> Path:
    return root / name / f"{as_of.isoformat()}.parquet"


def write_snapshot(universe: Universe, root: Path = UNIVERSE_DIR) -> Path:
    """Persist a universe's membership for its ``as_of`` date (one parquet per day)."""
    path = snapshot_path(universe.name, universe.as_of, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "symbol": sorted(universe.symbols),
            "as_of": universe.as_of.isoformat(),
            "universe": universe.name,
            "survivorship_safe": universe.survivorship_safe,
        }
    )
    frame.to_parquet(path, index=False)
    return path


def read_snapshot(name: str, as_of: date, root: Path = UNIVERSE_DIR) -> list[str] | None:
    path = snapshot_path(name, as_of, root)
    if not path.exists():
        return None
    return [str(s) for s in pd.read_parquet(path, columns=["symbol"])["symbol"]]


# --- the refresh --------------------------------------------------------------------------


def refresh(
    provider: FMPProvider,
    *,
    universes: Sequence[str] = ("etf_core", "us_liquid"),
    timeframes: Sequence[Timeframe | str] = (Timeframe.DAY,),
    start: str = DEFAULT_START,
    as_of: date | None = None,
    through: date | None = None,
    batch_size: int = 100,
    snapshot_root: Path = UNIVERSE_DIR,
    dry_run: bool = False,
    backfill: bool = False,
) -> RefreshReport:
    """Snapshot each universe, then bring every member's cache up to ``through``."""
    t0 = time.perf_counter()
    as_of = as_of or last_completed_session()
    through = through or as_of
    report = RefreshReport(
        as_of=as_of.isoformat(), through=through.isoformat(), dry_run=dry_run, backfill=backfill
    )

    members: dict[str, None] = {}
    for name in universes:
        uni = resolve_universe(name, provider, as_of)
        report.universes[uni.name] = len(uni.symbols)
        if not dry_run:
            report.snapshots.append(str(write_snapshot(uni, snapshot_root)))
        for s in uni.symbols:
            members.setdefault(s.upper(), None)
    symbols = list(members)
    report.symbols = len(symbols)
    log.info("universes %s -> %d symbols", report.universes, len(symbols))

    for tf_in in timeframes:
        tf = Timeframe(tf_in) if not isinstance(tf_in, Timeframe) else tf_in
        stale = symbols if backfill else stale_symbols(provider.cache, symbols, tf, through=through)
        log.info(
            "%s: %d/%d symbols to fetch through %s", tf.value, len(stale), len(symbols), through
        )
        report.refreshed[tf.value] = 0
        report.failed[tf.value] = []
        if dry_run:
            report.refreshed[tf.value] = len(stale)
            continue
        plan = plan_fetches(
            provider.cache, stale, tf, start=start, batch_size=batch_size, backfill=backfill
        )
        done = 0
        for frm, batch in plan:
            try:
                fetched = provider.fetch_and_cache(batch, frm, through.isoformat(), tf)
                got = set(fetched["symbol"].unique()) if not fetched.empty else set()
                report.refreshed[tf.value] += len(got)
                report.failed[tf.value].extend(s for s in batch if s not in got)
            except Exception as exc:  # noqa: BLE001 - one bad batch must not sink the night
                log.exception("batch from %s (%d symbols) failed: %s", frm, len(batch), exc)
                report.failed[tf.value].extend(batch)
            done += len(batch)
            log.info("%s: %d/%d fetched (from %s)", tf.value, done, len(stale), frm)
    report.seconds = time.perf_counter() - t0
    return report
