"""``Panel`` — canonical long bars pivoted into aligned wide price matrices.

Signals and strategies operate on wide frames (timestamp index, one column per
symbol). ``Panel.from_bars`` is the single conversion from the canonical schema
to that shape, so strategies never re-pivot by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ehf.server.research.data.schema import Timeframe, validate_bars


@dataclass(frozen=True)
class Panel:
    """Aligned OHLCV wide frames + timeframe metadata."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    timeframe: Timeframe

    @property
    def periods_per_year(self) -> float:
        return self.timeframe.periods_per_year

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index  # type: ignore[return-value]

    def slice(self, lo: pd.Timestamp, hi: pd.Timestamp) -> Panel:
        """Rows with ``lo <= timestamp < hi`` (all fields, same symbols)."""
        mask = (self.index >= lo) & (self.index < hi)
        return Panel(
            open=self.open.loc[mask],
            high=self.high.loc[mask],
            low=self.low.loc[mask],
            close=self.close.loc[mask],
            volume=self.volume.loc[mask],
            timeframe=self.timeframe,
        )

    @classmethod
    def from_bars(cls, bars: pd.DataFrame) -> Panel:
        """Pivot canonical bars into aligned wide OHLCV frames."""
        validate_bars(bars)
        tf = Timeframe(bars.attrs.get("timeframe", Timeframe.DAY.value))

        def pivot(field: str) -> pd.DataFrame:
            wide = bars.pivot(index="timestamp", columns="symbol", values=field)
            wide.columns.name = None
            return wide.sort_index()

        return cls(
            open=pivot("open"),
            high=pivot("high"),
            low=pivot("low"),
            close=pivot("close"),
            volume=pivot("volume"),
            timeframe=tf,
        )
