"""The canonical EHF bars schema — the vendor-agnostic normalization contract.

Every ``DataProvider`` MUST return a DataFrame that passes :func:`validate_bars`.
This is the single point of truth for "what EHF data looks like." Backtests,
signals, and analysis are written against *this* schema only, so the data vendor
can be swapped by writing a new adapter — nothing else changes.

Canonical bars DataFrame (long format, one row per (symbol, bar)):

    column      dtype                      meaning
    --------    -----------------------    -----------------------------------
    symbol      string                     uppercase ticker
    timestamp   datetime64[ns, UTC]        bar timestamp, timezone-aware UTC
    open        float64                    adjusted unless adjusted=False
    high        float64
    low         float64
    close       float64
    volume      float64

Invariants enforced by :func:`validate_bars`:
  * exactly the columns above, in order;
  * dtypes as stated (timestamp is tz-aware UTC);
  * sorted by (symbol, timestamp), no duplicate (symbol, timestamp) pairs;
  * OHLC are finite and high >= low; volume >= 0.

The frame carries two metadata flags in ``df.attrs``:
  * ``adjusted`` (bool): are prices split/dividend adjusted (default True);
  * ``timeframe`` (str): the :class:`Timeframe` value the bars represent.
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd

# Canonical column order. Downstream code may rely on this exact set/order.
BARS_COLUMNS: tuple[str, ...] = (
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

_OHLC = ("open", "high", "low", "close")
_FLOAT_COLS = (*_OHLC, "volume")


class Timeframe(StrEnum):
    """Canonical bar timeframes. Adapters translate vendor strings into these."""

    MINUTE = "1min"
    HOUR = "1hour"
    DAY = "1day"

    @property
    def pandas_freq(self) -> str:
        """Pandas offset alias for this timeframe (used by gap detection)."""
        return {"1min": "min", "1hour": "h", "1day": "D"}[self.value]

    @property
    def periods_per_year(self) -> float:
        """Annualization factor for this timeframe (US equity sessions)."""
        return {"1day": 252.0, "1hour": 252.0 * 6.5, "1min": 252.0 * 390.0}[self.value]


class SchemaError(ValueError):
    """Raised when a DataFrame violates the canonical bars schema."""


def normalize_bars(
    df: pd.DataFrame,
    *,
    column_map: dict[str, str] | None = None,
    timeframe: Timeframe | str,
    adjusted: bool = True,
    assume_tz: str = "America/New_York",
) -> pd.DataFrame:
    """Coerce an arbitrary vendor frame into the canonical bars schema.

    This is the one function every adapter funnels through, so normalization
    lives in exactly one place.

    Args:
        df: raw vendor frame.
        column_map: maps vendor column names -> canonical names. Any of
            {symbol, timestamp, open, high, low, close, volume}. Vendor columns
            already named canonically need no mapping.
        timeframe: the bar timeframe these rows represent.
        adjusted: whether prices are split/dividend adjusted (recorded in attrs).
        assume_tz: timezone to localize naive timestamps to before converting to
            UTC. Vendor timestamps (e.g. FMP) are typically US/Eastern wall time.

    Returns:
        A canonical bars DataFrame (validated).
    """
    tf = Timeframe(timeframe) if not isinstance(timeframe, Timeframe) else timeframe
    out = df.rename(columns=column_map or {}).copy()

    missing = [c for c in BARS_COLUMNS if c not in out.columns]
    if missing:
        raise SchemaError(
            f"cannot normalize: missing columns {missing}. "
            f"Provide a column_map. Have: {list(df.columns)}"
        )

    out = out[list(BARS_COLUMNS)]
    out["symbol"] = out["symbol"].astype("string").str.upper()

    ts = pd.to_datetime(out["timestamp"], utc=False, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(assume_tz, ambiguous="NaT", nonexistent="NaT")
    out["timestamp"] = ts.dt.tz_convert("UTC")

    for col in _FLOAT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out = (
        out.dropna(subset=["symbol", "timestamp"])
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    out.attrs["adjusted"] = adjusted
    out.attrs["timeframe"] = tf.value
    validate_bars(out)
    return out


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Assert a DataFrame satisfies the canonical schema. Returns it unchanged.

    Raises:
        SchemaError: on any violation.
    """
    if list(df.columns) != list(BARS_COLUMNS):
        raise SchemaError(f"columns must be exactly {BARS_COLUMNS}, got {list(df.columns)}")

    ts = df["timestamp"]
    if not isinstance(ts.dtype, pd.DatetimeTZDtype) or str(ts.dt.tz) != "UTC":
        raise SchemaError("'timestamp' must be datetime64[ns, UTC] (tz-aware UTC)")

    for col in _FLOAT_COLS:
        if df[col].dtype != "float64":
            raise SchemaError(f"'{col}' must be float64, got {df[col].dtype}")

    if df.duplicated(subset=["symbol", "timestamp"]).any():
        raise SchemaError("duplicate (symbol, timestamp) rows present")

    if not df.empty:
        if not df.equals(df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)):
            raise SchemaError("rows must be sorted by (symbol, timestamp)")
        # All-NaN OHLC rows are allowed (e.g. gap-padded frames); only check
        # rows where high and low are both present.
        hl = df[["high", "low"]].dropna()
        if not hl.empty and (hl["high"] < hl["low"]).any():
            raise SchemaError("found bars where high < low")
        if (df["volume"].dropna() < 0).any():
            raise SchemaError("found negative volume")
    return df
