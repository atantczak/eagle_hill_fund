"""Cboe's free daily index histories (SKEW today; VIX futures later).

FMP carries the VIX family but not the SKEW index, and we have no option-chain
history, so the put-demand proxy behind the ``vix_skew`` rule comes straight from
Cboe's public CSV. Cached as parquet under ``var/cache/cboe/`` and refreshed only
when the cached copy is older than ``max_age_days``; offline runs read the cache.

Emits a plain ``pd.Series`` indexed by *session date* (naive, America/New_York
calendar date). ``data.vix`` lifts it into canonical bars aligned to SPY sessions.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ehf.research.paths import CACHE_DIR

log = logging.getLogger(__name__)

CBOE_BASE = "https://cdn.cboe.com/api/global/us_indices/daily_prices"
SKEW_URL = f"{CBOE_BASE}/SKEW_History.csv"
CBOE_CACHE = CACHE_DIR / "cboe"


def _cache_path(name: str) -> Path:
    return CBOE_CACHE / f"{name}.parquet"


def _is_fresh(path: Path, max_age_days: float) -> bool:
    if not path.exists():
        return False
    age = datetime.now(UTC) - datetime.fromtimestamp(path.stat().st_mtime, UTC)
    return age.days < max_age_days


def parse_cboe_csv(text: str, value_col: str) -> pd.Series:
    """Parse a Cboe ``DATE,<INDEX>`` CSV into a float Series indexed by session date."""
    import io

    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    if "DATE" not in df.columns or value_col.upper() not in df.columns:
        raise ValueError(f"unexpected Cboe columns {list(df.columns)}")
    dates = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce")
    values = pd.to_numeric(df[value_col.upper()], errors="coerce")
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=value_col.upper())
    s = s.loc[s.index.notna()].dropna().sort_index()
    return s.loc[~s.index.duplicated(keep="last")]


def skew_history(*, max_age_days: float = 1.0, offline: bool = False) -> pd.Series:
    """Cboe SKEW index, daily, back to 1990. Cached; network only when stale."""
    path = _cache_path("SKEW")
    if offline or _is_fresh(path, max_age_days):
        if path.exists():
            cached = pd.read_parquet(path)["SKEW"]
            cached.index = pd.DatetimeIndex(cached.index)
            return cached
        if offline:
            raise FileNotFoundError(f"no cached SKEW history at {path} and offline=True")
    try:
        import requests

        resp = requests.get(SKEW_URL, timeout=30)
        resp.raise_for_status()
        s = parse_cboe_csv(resp.text, "SKEW")
    except Exception as exc:  # network down -> fall back to whatever we have
        if path.exists():
            log.warning("SKEW refresh failed (%s); using cached copy", exc)
            cached = pd.read_parquet(path)["SKEW"]
            cached.index = pd.DatetimeIndex(cached.index)
            return cached
        raise
    CBOE_CACHE.mkdir(parents=True, exist_ok=True)
    s.rename("SKEW").to_frame().to_parquet(path)
    log.info("SKEW history refreshed: %d rows through %s", len(s), s.index[-1].date())
    return s
