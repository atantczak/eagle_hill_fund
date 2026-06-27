"""Intraday requests are chunked under FMP's per-request cap and stitched.

Guards the fix for "ask for years of hourly, get only ~3 months": the legacy
chunker must split long intraday ranges into contiguous capped windows. Pure
(no network) — exercises only the date-window math.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from ehf.server.tools.financial.data.tool import FinancialDataCompiler


def _chunks(frm: str, to: str, interval: str) -> list[tuple]:
    return FinancialDataCompiler()._chunk_intraday_date_ranges(frm, to, interval)


def _d(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def test_daily_is_a_single_request() -> None:
    assert _chunks("2006-01-01", "2024-12-31", "1day") == [("2006-01-01", "2024-12-31")]


def test_hourly_long_range_is_chunked_contiguously() -> None:
    chunks = _chunks("2021-01-01", "2023-01-01", "1hour")
    assert len(chunks) > 1  # a 2-year hourly range must be split
    assert chunks[0][0] == "2021-01-01" and chunks[-1][1] == "2023-01-01"
    for start, end in chunks:
        assert (_d(end) - _d(start)).days <= 60  # under the ~430-bar/1hour cap
    for (_, end0), (start1, _) in zip(chunks, chunks[1:], strict=False):
        assert _d(start1) == _d(end0) + timedelta(days=1)  # contiguous, no gaps/overlap


def test_minute_chunks_stay_tiny() -> None:
    for start, end in _chunks("2021-06-01", "2021-06-30", "1min"):
        assert (_d(end) - _d(start)).days <= 3  # ~1170-minute cap
