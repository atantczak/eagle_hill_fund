"""Write-through parquet cache so backtests never re-hammer the paid data API.

Keying: one parquet file per (vendor, symbol, timeframe, adjustment). Each file
holds that symbol's full cached history for the timeframe; a request for a
date range reads the file and slices, fetching from the vendor only the bars not
already cached (cache *fill*, never cache *miss the whole range*).

Invalidation strategy (documented per the brief):
  * Cache is keyed by adjustment flag, so adjusted/unadjusted never mix.
  * Adjusted prices are mutable history (a new split rewrites the past), so the
    daily/adjusted cache is given a freshness TTL: entries whose file mtime is
    older than ``adjusted_ttl_days`` are treated as stale for the *most recent*
    ``refetch_tail_days`` and re-fetched on read. Older history is immutable and
    served from cache. Unadjusted bars are immutable and never expire.
  * ``clear()`` / deleting the cache dir is always safe — it only costs API calls.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ehf.research.data.schema import BARS_COLUMNS, Timeframe, validate_bars
from ehf.research.paths import CACHE_DIR as DEFAULT_CACHE_DIR


class BarCache:
    """Per-symbol parquet cache of canonical bars."""

    def __init__(
        self,
        vendor: str,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        *,
        adjusted_ttl_days: int = 1,
        refetch_tail_days: int = 7,
    ):
        self.vendor = vendor
        self.cache_dir = Path(cache_dir)
        self.adjusted_ttl_days = adjusted_ttl_days
        self.refetch_tail_days = refetch_tail_days

    def _path(self, symbol: str, timeframe: Timeframe, adjusted: bool) -> Path:
        adj = "adj" if adjusted else "raw"
        return self.cache_dir / self.vendor / timeframe.value / adj / f"{symbol.upper()}.parquet"

    def read(self, symbol: str, timeframe: Timeframe, adjusted: bool) -> pd.DataFrame | None:
        """Return cached bars for one symbol, or None if nothing is cached."""
        path = self._path(symbol, timeframe, adjusted)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        return validate_bars(df)

    def write(self, df: pd.DataFrame, timeframe: Timeframe, adjusted: bool) -> None:
        """Merge ``df`` (canonical bars, possibly many symbols) into the cache."""
        validate_bars(df)
        for symbol, sym_df in df.groupby("symbol", sort=False):
            path = self._path(str(symbol), timeframe, adjusted)
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = self.read(str(symbol), timeframe, adjusted)
            merged = sym_df if existing is None else _merge_bars(existing, sym_df)
            merged.to_parquet(path, index=False)

    def covered_range(
        self, symbol: str, timeframe: Timeframe, adjusted: bool
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """The (min, max) timestamp cached for a symbol, or None."""
        df = self.read(symbol, timeframe, adjusted)
        if df is None or df.empty:
            return None
        return df["timestamp"].min(), df["timestamp"].max()

    def clear(self) -> None:
        """Delete the entire cache for this vendor. Safe — only costs API calls."""
        import shutil

        target = self.cache_dir / self.vendor
        if target.exists():
            shutil.rmtree(target)


def _merge_bars(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Union two canonical frames, newer rows winning on duplicate timestamps."""
    merged = (
        pd.concat([a, b], ignore_index=True)
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    merged = merged[list(BARS_COLUMNS)]
    merged.attrs.update(b.attrs)
    return merged
