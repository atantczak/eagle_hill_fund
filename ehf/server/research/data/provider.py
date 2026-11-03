"""The ``DataProvider`` interface — the seam between EHF and any data vendor.

The platform depends only on this ABC and the canonical schema. Concrete
adapters (e.g. :class:`ehf.server.research.data.fmp_provider.FMPProvider`) wrap whatever vendor
tooling exists and return canonical bars. Add or swap a vendor → write one new
subclass; backtests and analysis are untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from ehf.server.research.data.schema import Timeframe


class Universe:
    """A point-in-time universe of symbols.

    ``survivorship_safe`` is False when membership reflects *today's* listings
    rather than the listings as of ``as_of`` — the platform warns loudly when a
    study relies on a non-survivorship-safe universe (see ADR / brief).
    """

    def __init__(self, name: str, as_of: date, symbols: list[str], *, survivorship_safe: bool):
        self.name = name
        self.as_of = as_of
        self.symbols = symbols
        self.survivorship_safe = survivorship_safe

    def __repr__(self) -> str:  # pragma: no cover - trivial
        flag = "PIT" if self.survivorship_safe else "SURVIVORSHIP-BIASED"
        return f"Universe({self.name!r}, as_of={self.as_of}, n={len(self.symbols)}, {flag})"


class DataProvider(ABC):
    """Vendor-agnostic data access. All returns conform to the canonical schema."""

    @abstractmethod
    def get_bars(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date,
        timeframe: Timeframe | str = Timeframe.DAY,
        *,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Return canonical bars for ``symbols`` between ``start`` and ``end``.

        The returned frame passes :func:`ehf.server.research.data.schema.validate_bars`. Prices
        are split/dividend adjusted unless ``adjusted=False`` (a debug path).
        """

    @abstractmethod
    def get_universe(self, name: str, as_of: str | date) -> Universe:
        """Return the membership of a named universe as of a date.

        Implementations MUST set ``survivorship_safe=False`` and emit a warning
        when they can only supply *current* membership for a historical ``as_of``.
        """

    def get_fundamentals(self, symbols: list[str], as_of: str | date) -> pd.DataFrame:
        """Point-in-time fundamentals (optional; not all vendors support this).

        Default raises ``NotImplementedError`` so callers can feature-detect.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide fundamentals")
