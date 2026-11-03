"""Pure column mapping of FMP's bulk TTM fundamentals into one tidy frame.

The bulk CSVs cover the whole market (tens of thousands of rows). We parse only
the columns the screens use and keep only the universe.
"""

from __future__ import annotations

import pandas as pd

KEY_METRICS_COLS = [
    "symbol",
    "marketCap",
    "evToEBITDATTM",
    "netDebtToEBITDATTM",
    "returnOnInvestedCapitalTTM",
    "returnOnEquityTTM",
    "earningsYieldTTM",
    "freeCashFlowYieldTTM",
    "incomeQualityTTM",
]
RATIO_COLS = [
    "symbol",
    "grossProfitMarginTTM",
    "operatingProfitMarginTTM",
    "netProfitMarginTTM",
    "priceToEarningsRatioTTM",
    "priceToSalesRatioTTM",
    "priceToBookRatioTTM",
    "debtToEquityRatioTTM",
    "interestCoverageRatioTTM",
    "dividendYieldTTM",
]
SCORE_COLS = ["symbol", "altmanZScore", "piotroskiScore"]

_RENAME = {
    "marketCap": "market_cap_ttm",
    "evToEBITDATTM": "ev_ebitda",
    "netDebtToEBITDATTM": "net_debt_ebitda",
    "returnOnInvestedCapitalTTM": "roic",
    "returnOnEquityTTM": "roe",
    "earningsYieldTTM": "earnings_yield",
    "freeCashFlowYieldTTM": "fcf_yield",
    "incomeQualityTTM": "income_quality",
    "grossProfitMarginTTM": "gross_margin",
    "operatingProfitMarginTTM": "op_margin",
    "netProfitMarginTTM": "net_margin",
    "priceToEarningsRatioTTM": "pe",
    "priceToSalesRatioTTM": "ps",
    "priceToBookRatioTTM": "pb",
    "debtToEquityRatioTTM": "debt_to_equity",
    "interestCoverageRatioTTM": "interest_coverage",
    "dividendYieldTTM": "dividend_yield",
    "altmanZScore": "altman_z",
    "piotroskiScore": "piotroski",
    "strongBuy": "rating_strong_buy",
    "buy": "rating_buy",
    "hold": "rating_hold",
    "sell": "rating_sell",
    "strongSell": "rating_strong_sell",
    "consensus": "rating_consensus",
}

FUNDAMENTAL_COLUMNS = [v for v in _RENAME.values()]


# Plausible ranges; anything outside is a vendor artefact (e.g. Altman Z of 4,000) -> NaN.
_BOUNDS: dict[str, tuple[float, float]] = {
    "altman_z": (-20.0, 60.0),
    "piotroski": (0.0, 9.0),
    "fcf_yield": (-1.0, 1.0),
    "earnings_yield": (-1.0, 1.0),
    "roic": (-2.0, 2.0),
    "roe": (-5.0, 5.0),
    "net_debt_ebitda": (-50.0, 50.0),
    "ev_ebitda": (-500.0, 500.0),
    "pe": (-1000.0, 1000.0),
    "gross_margin": (-5.0, 1.0),
    "op_margin": (-5.0, 1.0),
    "net_margin": (-5.0, 1.0),
}


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Pure: null out values outside plausible ranges (keeps a bad print from ranking first)."""
    out = df.copy()
    for col, (lo, hi) in _BOUNDS.items():
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = s.where((s >= lo) & (s <= hi))
    return out


def merge_fundamentals(
    symbols: list[str],
    key_metrics: pd.DataFrame,
    ratios: pd.DataFrame,
    scores: pd.DataFrame,
    consensus: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pure: outer-join the bulk frames on symbol, restricted to ``symbols``."""
    want = set(symbols)
    parts = [key_metrics, ratios, scores] + ([consensus] if consensus is not None else [])
    merged: pd.DataFrame | None = None
    for part in parts:
        if part is None or part.empty or "symbol" not in part.columns:
            continue
        p = part[part["symbol"].isin(want)].drop_duplicates("symbol").set_index("symbol")
        merged = p if merged is None else merged.join(p, how="outer")
    if merged is None:
        return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS).rename_axis("symbol")
    merged = merged.rename(columns=_RENAME)
    for col in FUNDAMENTAL_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    numeric = [c for c in FUNDAMENTAL_COLUMNS if c != "rating_consensus"]
    merged[numeric] = merged[numeric].apply(pd.to_numeric, errors="coerce")
    merged = sanitize(merged)
    merged.index.name = "symbol"
    return merged[FUNDAMENTAL_COLUMNS].reindex(sorted(want))
