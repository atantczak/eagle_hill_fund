"""Data layer: the only part of the platform that knows where data comes from.

Everything above this package consumes the **canonical EHF bars schema**
(``ehf.server.research.data.schema``) and never sees a vendor's field names. Swapping
``FMPProvider`` for another vendor means writing one new adapter that emits the
same canonical schema — nothing downstream changes.
"""

from ehf.server.research.data.provider import DataProvider
from ehf.server.research.data.schema import BARS_COLUMNS, Timeframe, normalize_bars, validate_bars

__all__ = [
    "DataProvider",
    "Timeframe",
    "BARS_COLUMNS",
    "normalize_bars",
    "validate_bars",
]
