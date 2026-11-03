"""Parallel intraday fetch flattens (symbol x chunk) and stamps symbols. No network."""

from __future__ import annotations

import pandas as pd

from ehf.research.data.cache import BarCache
from ehf.research.data.fmp_provider import FMPProvider


class _FakePriceTool:
    """Returns one canned bar per (symbol, chunk) request."""

    def get_intraday_prices(self, symbol, interval, from_date, to_date):  # noqa: ANN001
        return [
            {
                "date": f"{from_date} 09:30:00",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            }
        ]


class _FakeParallel:
    """Runs tasks inline (deterministic) instead of threads — same contract."""

    def parallel_map_with_retry(self, func, items, **_):  # noqa: ANN001
        return [func(i) for i in items]


class _FakeCompiler:
    def __init__(self):
        self.price_data_tool = _FakePriceTool()
        self.parallel_tool = _FakeParallel()

    def _chunk_intraday_date_ranges(self, start, end, interval):  # noqa: ANN001
        return [("2021-01-01", "2021-02-01"), ("2021-02-02", "2021-03-01")]


def test_intraday_parallel_flattens_and_stamps(tmp_path) -> None:
    prov = FMPProvider(cache=BarCache(vendor="t", cache_dir=tmp_path))
    prov._compiler = _FakeCompiler()  # inject fake; no network, no real chunker

    raw = prov._fetch_intraday_parallel(["AAPL", "MSFT"], "2021-01-01", "2021-03-01", "1hour")

    # 2 symbols x 2 chunks = 4 rows, each stamped with its own symbol.
    assert len(raw) == 4
    assert set(raw["symbol"]) == {"AAPL", "MSFT"}
    assert (raw.groupby("symbol").size() == 2).all()


def test_intraday_parallel_empty_tasks(tmp_path) -> None:
    prov = FMPProvider(cache=BarCache(vendor="t", cache_dir=tmp_path))

    class _NoChunks(_FakeCompiler):
        def _chunk_intraday_date_ranges(self, start, end, interval):  # noqa: ANN001
            return []

    prov._compiler = _NoChunks()
    assert prov._fetch_intraday_parallel([], "2021-01-01", "2021-03-01", "1hour").empty


def test_daily_chunks_cover_range_without_gaps_or_overlap() -> None:
    from ehf.research.data.fmp_provider import daily_chunks

    chunks = daily_chunks("2000-01-01", "2026-10-30", years=15)
    assert chunks == [("2000-01-01", "2014-12-31"), ("2015-01-01", "2026-10-30")]
    assert daily_chunks("2024-01-01", "2024-06-30") == [("2024-01-01", "2024-06-30")]
    assert daily_chunks("2024-01-01", "2023-01-01") == []
    three = daily_chunks("1990-03-15", "2026-10-30", years=15)
    assert three[0] == ("1990-03-15", "2005-03-14") and three[-1][1] == "2026-10-30"
    for a, b in zip(three, three[1:], strict=False):
        assert pd.Timestamp(b[0]) == pd.Timestamp(a[1]) + pd.Timedelta(days=1)


def _daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    from ehf.research.data.schema import Timeframe, validate_bars

    idx = pd.bdate_range(start, end, tz="UTC") + pd.Timedelta(hours=5)
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": idx,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    out.attrs["timeframe"] = Timeframe.DAY.value
    out.attrs["adjusted"] = True
    return validate_bars(out)


def test_refresh_tail_pulls_from_the_end_of_the_cache_forward(tmp_path, monkeypatch) -> None:
    """A tail stitched onto a stale cache would hide the hole; the pull must close it."""
    from ehf.research.data.schema import Timeframe

    prov = FMPProvider(cache=BarCache(vendor="t", cache_dir=tmp_path))
    prov.cache.write(_daily("OLD", "2024-01-01", "2024-12-31"), Timeframe.DAY, True)
    calls: list[tuple[list[str], str, str]] = []

    def fake_fetch(symbols, start, end, tf, adjusted):  # type: ignore[no-untyped-def]
        calls.append((list(symbols), start, end))
        frames = [_daily(s, start, end) for s in symbols]
        out = pd.concat(frames, ignore_index=True)
        out.attrs.update(frames[0].attrs)
        return out

    monkeypatch.setattr(prov, "_fetch", fake_fetch)
    out = prov.refresh_tail(["OLD", "NEW"], days=10, end="2026-11-03")
    assert sorted(calls) == [
        (["NEW"], "2026-10-24", "2026-11-03"),  # nothing cached: just the tail
        (["OLD"], "2024-12-30", "2026-11-03"),  # cached through 2024: from there forward
    ]
    assert set(out["symbol"]) == {"OLD", "NEW"}
    lo, hi = prov.cache.covered_range("OLD", Timeframe.DAY, True)
    cached = prov.cache.read("OLD", Timeframe.DAY, True)
    assert lo.year == 2024 and hi.date().isoformat() == "2026-11-03"
    assert cached["timestamp"].dt.year.value_counts().get(2025, 0) > 250  # the hole is filled
