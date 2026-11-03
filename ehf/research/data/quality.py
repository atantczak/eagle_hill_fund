"""Data-quality checks. Surfaces problems loudly instead of silently biasing.

The brief's rule: a backtest on bad/biased data must WARN, not quietly produce a
clean-looking wrong number.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pandas as pd

from ehf.research.data.schema import Timeframe, validate_bars


class DataQualityWarning(UserWarning):
    """Raised (as a warning) when data has gaps, staleness, or bias risk."""


@dataclass
class QualityReport:
    """Per-symbol data-quality summary for a bars frame."""

    n_rows: int
    n_symbols: int
    missing_bars: dict[str, int] = field(default_factory=dict)
    zero_volume: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0

    @property
    def has_issues(self) -> bool:
        return bool(self.missing_bars or self.zero_volume or self.duplicate_rows)


def check_bars(
    df: pd.DataFrame,
    timeframe: Timeframe | str,
    *,
    warn: bool = True,
    calendar: str = "XNYS",
) -> QualityReport:
    """Inspect a canonical bars frame for gaps and anomalies.

    For daily bars, "missing" is measured against US business days (a coarse
    proxy for the exchange calendar — exact holiday handling can plug in later).
    For intraday, gap detection is left coarse pending a session calendar.

    Returns a :class:`QualityReport`; emits :class:`DataQualityWarning` when
    ``warn`` and issues are found.
    """
    validate_bars(df)
    tf = Timeframe(timeframe) if not isinstance(timeframe, Timeframe) else timeframe
    report = QualityReport(n_rows=len(df), n_symbols=int(df["symbol"].nunique()))
    report.duplicate_rows = int(df.duplicated(subset=["symbol", "timestamp"]).sum())

    for symbol, sym_df in df.groupby("symbol", sort=False):
        ts = sym_df["timestamp"]
        if tf is Timeframe.DAY and len(ts) > 1:
            expected = pd.bdate_range(ts.min(), ts.max(), tz="UTC")
            missing = len(expected) - ts.nunique()
            if missing > 0:
                report.missing_bars[str(symbol)] = int(missing)
        zv = int((sym_df["volume"] == 0).sum())
        if zv:
            report.zero_volume[str(symbol)] = zv

    if warn and report.has_issues:
        warnings.warn(
            f"Data quality issues: missing_bars={report.missing_bars}, "
            f"zero_volume={report.zero_volume}, duplicate_rows={report.duplicate_rows} "
            f"(calendar={calendar})",
            DataQualityWarning,
            stacklevel=2,
        )
    return report


def warn_survivorship(universe_name: str, as_of: object) -> None:
    """Emit the standard survivorship-bias warning for a non-PIT universe."""
    warnings.warn(
        f"Universe {universe_name!r} as_of={as_of} reflects CURRENT listings, not "
        f"point-in-time membership. Results may be survivorship-biased. Treat any "
        f"universe-level study (Russell, cross-sectional momentum, dislocation "
        f"screen) with caution until point-in-time membership is wired in.",
        DataQualityWarning,
        stacklevel=2,
    )
