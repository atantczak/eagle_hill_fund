"""Parquet cache: round-trip, merge, and slicing without re-fetching."""

from __future__ import annotations

import pandas as pd

from ehf.server.research.data.cache import BarCache
from ehf.server.research.data.schema import Timeframe, validate_bars


def test_write_then_read_roundtrip(tmp_path, canonical_bars: pd.DataFrame) -> None:
    cache = BarCache(vendor="test", cache_dir=tmp_path)
    cache.write(canonical_bars, Timeframe.DAY, adjusted=True)
    got = cache.read("AAPL", Timeframe.DAY, adjusted=True)
    assert got is not None
    validate_bars(got)
    assert set(got["symbol"]) == {"AAPL"}
    assert len(got) == (canonical_bars["symbol"] == "AAPL").sum()


def test_read_missing_returns_none(tmp_path) -> None:
    cache = BarCache(vendor="test", cache_dir=tmp_path)
    assert cache.read("NOPE", Timeframe.DAY, adjusted=True) is None


def test_merge_is_idempotent(tmp_path, canonical_bars: pd.DataFrame) -> None:
    cache = BarCache(vendor="test", cache_dir=tmp_path)
    cache.write(canonical_bars, Timeframe.DAY, adjusted=True)
    cache.write(canonical_bars, Timeframe.DAY, adjusted=True)  # again
    got = cache.read("MSFT", Timeframe.DAY, adjusted=True)
    assert got is not None
    assert not got.duplicated(subset=["symbol", "timestamp"]).any()


def test_adjusted_and_raw_are_separate(tmp_path, canonical_bars: pd.DataFrame) -> None:
    cache = BarCache(vendor="test", cache_dir=tmp_path)
    cache.write(canonical_bars, Timeframe.DAY, adjusted=True)
    assert cache.read("AAPL", Timeframe.DAY, adjusted=False) is None


def test_covered_range(tmp_path, canonical_bars: pd.DataFrame) -> None:
    cache = BarCache(vendor="test", cache_dir=tmp_path)
    cache.write(canonical_bars, Timeframe.DAY, adjusted=True)
    rng = cache.covered_range("AAPL", Timeframe.DAY, adjusted=True)
    assert rng is not None
    lo, hi = rng
    assert lo < hi
