"""Data refresh: session logic, staleness detection, universe snapshots (no network)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from ehf.server.research.data.cache import BarCache
from ehf.server.research.data.provider import Universe
from ehf.server.research.data.refresh import (
    last_completed_session,
    plan_fetches,
    read_snapshot,
    refresh,
    stale_symbols,
    write_snapshot,
)
from ehf.server.research.data.schema import Timeframe
from ehf.server.research.data.universes import STATIC_UNIVERSES
from ehf.server.tests.conftest import make_daily_bars


def test_last_completed_session_respects_close_and_weekends() -> None:
    # Wednesday 2026-10-28 20:00 ET (= 00:00 UTC Thu) -> Wednesday
    assert last_completed_session(datetime(2026, 10, 29, 0, 0, tzinfo=UTC)) == date(2026, 10, 28)
    # Wednesday 10:00 ET -> Tuesday (today's bar isn't final yet)
    assert last_completed_session(datetime(2026, 10, 28, 14, 0, tzinfo=UTC)) == date(2026, 10, 27)
    # Saturday -> Friday; Sunday -> Friday
    assert last_completed_session(datetime(2026, 10, 31, 15, 0, tzinfo=UTC)) == date(2026, 10, 30)
    assert last_completed_session(datetime(2026, 11, 1, 15, 0, tzinfo=UTC)) == date(2026, 10, 30)


def test_stale_symbols_flags_missing_and_old_tails(tmp_path) -> None:
    cache = BarCache(vendor="fmp", cache_dir=tmp_path)
    bars = make_daily_bars({"AAA": [1.0] * 10, "BBB": [2.0] * 10}, start="2026-10-12")
    cache.write(bars, Timeframe.DAY, adjusted=True)  # last bar 2026-10-23 (Fri)
    assert stale_symbols(
        cache, ["AAA", "BBB", "CCC"], Timeframe.DAY, through=date(2026, 10, 23)
    ) == ["CCC"]
    assert stale_symbols(cache, ["AAA", "CCC"], Timeframe.DAY, through=date(2026, 10, 26)) == [
        "AAA",
        "CCC",
    ]


def test_snapshot_roundtrip(tmp_path) -> None:
    u = Universe("etf_core", date(2026, 11, 1), ["SPY", "TLT"], survivorship_safe=True)
    path = write_snapshot(u, tmp_path)
    assert path == tmp_path / "etf_core" / "2026-11-01.parquet"
    assert read_snapshot("etf_core", date(2026, 11, 1), tmp_path) == ["SPY", "TLT"]
    assert read_snapshot("etf_core", date(2026, 11, 2), tmp_path) is None


class _FakeProvider:
    """Only what refresh() touches: a cache + fetch_and_cache + get_universe."""

    def __init__(self, cache: BarCache) -> None:
        self.cache = cache
        self.calls: list[list[str]] = []
        self.starts: list[str] = []

    def fetch_and_cache(self, symbols, start, end, timeframe=Timeframe.DAY, *, adjusted=True):
        self.calls.append(list(symbols))
        self.starts.append(start)
        bars = make_daily_bars({s: [1.0] * 5 for s in symbols}, start="2026-10-26")
        self.cache.write(bars, Timeframe.DAY, adjusted=True)
        return bars

    def get_universe(self, name, as_of):  # pragma: no cover - static universes only here
        raise AssertionError("not used")


def test_refresh_snapshots_then_fetches_only_stale_in_batches(tmp_path) -> None:
    cache = BarCache(vendor="fmp", cache_dir=tmp_path / "cache")
    provider = _FakeProvider(cache)
    fresh = make_daily_bars({"XLK": [1.0] * 5}, start="2026-10-26")  # already current
    cache.write(fresh, Timeframe.DAY, adjusted=True)

    report = refresh(
        provider,  # type: ignore[arg-type]
        universes=("etf_sectors",),
        as_of=date(2026, 10, 30),
        batch_size=4,
        snapshot_root=tmp_path / "universe",
    )
    assert report.universes == {"etf_sectors": 11}
    assert report.symbols == 11
    assert report.refreshed["1day"] == 10  # everything but XLK
    assert report.failed["1day"] == []
    assert [len(c) for c in provider.calls] == [4, 4, 2]
    assert "XLK" not in {s for c in provider.calls for s in c}
    assert (tmp_path / "universe" / "etf_sectors" / "2026-10-30.parquet").exists()
    # second pass: nothing stale, no vendor calls, snapshot rewritten idempotently
    provider.calls.clear()
    again = refresh(
        provider,
        universes=("etf_sectors",),
        as_of=date(2026, 10, 30),  # type: ignore[arg-type]
        snapshot_root=tmp_path / "universe",
    )
    assert again.refreshed["1day"] == 0 and provider.calls == []


def test_refresh_dry_run_counts_without_fetching(tmp_path) -> None:
    provider = _FakeProvider(BarCache(vendor="fmp", cache_dir=tmp_path))
    report = refresh(
        provider,
        universes=("etf_sectors",),
        as_of=date(2026, 10, 30),  # type: ignore[arg-type]
        snapshot_root=tmp_path / "u",
        dry_run=True,
    )
    assert report.dry_run and report.refreshed["1day"] == 11 and provider.calls == []
    assert not (tmp_path / "u").exists()
    assert pd.Timestamp(report.through) == pd.Timestamp("2026-10-30")


def test_plan_fetches_full_for_missing_and_tail_topup_for_cached(tmp_path) -> None:
    cache = BarCache(vendor="fmp", cache_dir=tmp_path)
    old = make_daily_bars({"AAA": [1.0] * 10, "BBB": [1.0] * 10}, start="2026-09-01")  # ends 09-14
    cache.write(old, Timeframe.DAY, adjusted=True)
    plan = plan_fetches(
        cache, ["AAA", "BBB", "NEW"], Timeframe.DAY, start="2000-01-01", batch_size=10
    )
    assert plan[0] == ("2000-01-01", ["NEW"])  # missing -> full history
    frm, syms = plan[1]
    assert syms == ["AAA", "BBB"] and frm == "2026-08-15"  # 30 days before the last bar
    full = plan_fetches(
        cache, ["AAA", "NEW"], Timeframe.DAY, start="2000-01-01", batch_size=1, backfill=True
    )
    assert full == [("2000-01-01", ["AAA"]), ("2000-01-01", ["NEW"])]
    # the tail start never precedes --start
    late = plan_fetches(cache, ["AAA"], Timeframe.DAY, start="2026-09-01", batch_size=10)
    assert late == [("2026-09-01", ["AAA"])]


def test_refresh_backfill_refetches_everyone_from_start(tmp_path) -> None:
    cache = BarCache(vendor="fmp", cache_dir=tmp_path / "cache")
    provider = _FakeProvider(cache)
    cache.write(
        make_daily_bars({"XLK": [1.0] * 5}, start="2026-10-26"), Timeframe.DAY, adjusted=True
    )
    report = refresh(
        provider,
        universes=("etf_sectors",),
        as_of=date(2026, 10, 30),  # type: ignore[arg-type]
        start="2000-01-01",
        batch_size=100,
        snapshot_root=tmp_path / "u",
        backfill=True,
    )
    assert report.backfill and report.refreshed["1day"] == 11
    assert len(provider.calls) == 1 and sorted(provider.calls[0]) == sorted(
        STATIC_UNIVERSES["etf_sectors"]
    )
    assert provider.starts == ["2000-01-01"]
