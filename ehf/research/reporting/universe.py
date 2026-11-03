"""The screening universe: liquid US common stocks (screener) ∪ S&P 500.

Pure helpers build the frame from JSON rows so they are testable; ``build_universe``
is the only function that calls FMP.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ehf.research.reporting.config import ReportConfig
from ehf.research.reporting.fmp import FMPReportClient

UNIVERSE_COLUMNS = ["symbol", "name", "sector", "industry", "market_cap", "avg_volume", "in_sp500"]


def _clean(sym: object) -> str | None:
    """Keep plain common-stock tickers (drop share classes / warrants with '.' or '-')."""
    if not isinstance(sym, str) or not sym:
        return None
    s = sym.strip().upper()
    if "." in s or "-" in s or " " in s:
        return None
    return s


def universe_from_rows(
    screener_rows: list[dict[str, Any]],
    sp500_rows: list[dict[str, Any]],
    *,
    exchanges: tuple[str, ...],
    include_sp500: bool = True,
) -> pd.DataFrame:
    """Pure: merge screener + S&P 500 rows into the universe frame (sorted by symbol)."""
    allowed = {e.upper() for e in exchanges}
    rows: dict[str, dict[str, Any]] = {}
    for r in screener_rows:
        sym = _clean(r.get("symbol"))
        if sym is None:
            continue
        if allowed and str(r.get("exchangeShortName", "")).upper() not in allowed:
            continue
        rows[sym] = {
            "symbol": sym,
            "name": r.get("companyName"),
            "sector": r.get("sector"),
            "industry": r.get("industry"),
            "market_cap": r.get("marketCap"),
            "avg_volume": r.get("avgVolume"),
            "in_sp500": False,
        }
    if include_sp500:
        for r in sp500_rows:
            sym = _clean(r.get("symbol"))
            if sym is None:
                continue
            if sym in rows:
                rows[sym]["in_sp500"] = True
            else:
                rows[sym] = {
                    "symbol": sym,
                    "name": r.get("name"),
                    "sector": r.get("sector"),
                    "industry": r.get("subSector"),
                    "market_cap": None,
                    "avg_volume": None,
                    "in_sp500": True,
                }
    df = pd.DataFrame(list(rows.values()), columns=UNIVERSE_COLUMNS)
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["avg_volume"] = pd.to_numeric(df["avg_volume"], errors="coerce")
    df["in_sp500"] = df["in_sp500"].astype(bool)
    return df.sort_values("symbol").reset_index(drop=True)


def fill_from_quotes(universe: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Pure: fill missing name / market cap (S&P names absent from the screener) from quotes."""
    if quotes.empty or "symbol" not in quotes.columns:
        return universe
    q = quotes.set_index("symbol")
    out = universe.copy()
    if "name" in q.columns:
        out["name"] = out["name"].fillna(out["symbol"].map(q["name"]))
    if "marketCap" in q.columns:
        out["market_cap"] = out["market_cap"].fillna(out["symbol"].map(q["marketCap"]))
    if "avgVolume" in q.columns:
        out["avg_volume"] = out["avg_volume"].fillna(out["symbol"].map(q["avgVolume"]))
    return out


def build_universe(client: FMPReportClient, cfg: ReportConfig) -> pd.DataFrame:
    """Network: screener ∪ S&P 500, optionally capped for quick local runs."""
    screener = client.screener(
        market_cap_more_than=cfg.min_market_cap,
        volume_more_than=cfg.min_avg_volume,
        exchanges=cfg.exchanges,
    )
    sp500 = client.sp500_constituents() if cfg.include_sp500 else []
    uni = universe_from_rows(
        screener, sp500, exchanges=cfg.exchanges, include_sp500=cfg.include_sp500
    )
    if cfg.max_symbols is not None and len(uni) > cfg.max_symbols:
        # Deterministic cap: largest caps first so a small run still looks like the market.
        uni = (
            uni.sort_values("market_cap", ascending=False, na_position="last")
            .head(cfg.max_symbols)
            .sort_values("symbol")
            .reset_index(drop=True)
        )
    return uni
