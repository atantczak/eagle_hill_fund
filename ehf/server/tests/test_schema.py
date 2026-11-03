"""The canonical schema is the vendor-agnostic contract — test it hard."""

from __future__ import annotations

import pandas as pd
import pytest

from ehf.server.research.data.schema import (
    BARS_COLUMNS,
    SchemaError,
    Timeframe,
    normalize_bars,
    validate_bars,
)


def test_normalize_produces_canonical_columns_and_dtypes(canonical_bars: pd.DataFrame) -> None:
    assert list(canonical_bars.columns) == list(BARS_COLUMNS)
    assert isinstance(canonical_bars["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(canonical_bars["timestamp"].dt.tz) == "UTC"
    for col in ("open", "high", "low", "close", "volume"):
        assert canonical_bars[col].dtype == "float64"
    # vendor-only columns (e.g. vwap) are dropped
    assert "vwap" not in canonical_bars.columns


def test_normalize_drops_vendor_columns_and_uppercases_symbols(
    fmp_daily_records: pd.DataFrame,
) -> None:
    recs = fmp_daily_records.copy()
    recs["symbol"] = recs["symbol"].str.lower()
    out = normalize_bars(
        recs,
        column_map={"date": "timestamp"},
        timeframe=Timeframe.DAY,
    )
    assert set(out["symbol"]) == {"AAPL", "MSFT"}


def test_timestamps_are_eastern_to_utc() -> None:
    # A US/Eastern naive midnight should convert to 05:00Z (EST, UTC-5).
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "timestamp": ["2024-01-02"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    out = normalize_bars(df, timeframe=Timeframe.DAY)
    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-01-02 05:00", tz="UTC")


def test_validate_rejects_unsorted() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": pd.to_datetime(["2024-01-03", "2024-01-02"]).tz_localize("UTC"),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )
    with pytest.raises(SchemaError):
        validate_bars(df)


def test_validate_rejects_high_below_low(canonical_bars: pd.DataFrame) -> None:
    bad = canonical_bars.copy()
    bad.loc[0, "high"] = bad.loc[0, "low"] - 1.0
    with pytest.raises(SchemaError, match="high < low"):
        validate_bars(bad)


def test_normalize_deduplicates(fmp_daily_records: pd.DataFrame) -> None:
    dupes = pd.concat([fmp_daily_records, fmp_daily_records.iloc[:3]], ignore_index=True)
    out = normalize_bars(dupes, column_map={"date": "timestamp"}, timeframe=Timeframe.DAY)
    assert not out.duplicated(subset=["symbol", "timestamp"]).any()


def test_missing_columns_raises() -> None:
    with pytest.raises(SchemaError, match="missing columns"):
        normalize_bars(pd.DataFrame({"symbol": ["X"]}), timeframe=Timeframe.DAY)


def test_timeframe_pandas_freq() -> None:
    assert Timeframe.DAY.pandas_freq == "D"
    assert Timeframe.MINUTE.pandas_freq == "min"
