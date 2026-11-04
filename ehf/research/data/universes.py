"""Named universes — the one place that says what "the ETF set" or "liquid US" means.

Two kinds:

* **Static** (code constants, versioned with the repo): ETF baskets. These have
  no survivorship problem *per instrument* (every ETF listed still trades), but
  the basket itself was chosen with hindsight — a mild selection bias that the
  docs flag and the gauntlet's regime/efficiency checks partly absorb.
* **Dynamic** (resolved through a ``DataProvider`` as of a date): ``us_liquid``
  (current screener floors — survivorship-biased, warned) and ``sp500`` (true
  point-in-time membership rebuilt from the constituent-change log).

Every resolution goes through :func:`resolve_universe`, and the nightly data
job snapshots each dynamic universe to ``var/universe/<name>/<date>.parquet`` —
so from the day the job starts running, EHF owns its own point-in-time record.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

from ehf.research.data.provider import DataProvider, Universe

# --- static baskets --------------------------------------------------------------------

ETF_ASSET_CLASSES: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA",  # US equity: large, growth, small, blue-chip
    "EFA", "EEM", "VGK", "EWJ",  # developed ex-US, emerging, Europe, Japan
    "TLT", "IEF", "SHY", "TIP",  # treasuries: long, intermediate, short, inflation-linked
    "LQD", "HYG", "BND",  # credit: IG, HY, aggregate
    "GLD", "SLV", "DBC", "USO", "UNG",  # metals, broad commodities, oil, gas
    "VNQ",  # REITs
    "UUP", "FXE",  # dollar, euro
)  # fmt: skip

ETF_SECTORS: tuple[str, ...] = (
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC",
)  # fmt: skip

# The VIX overlay dataset (``data.vix``): SPY plus the Cboe index family FMP carries.
# ``^IRX`` is the 13-week bill yield behind the synthetic CASH leg.
VIX_FAMILY: tuple[str, ...] = ("SPY", "^VIX", "^VIX3M", "^VIX9D", "^VVIX", "^IRX")

STATIC_UNIVERSES: dict[str, tuple[str, ...]] = {
    "etf_asset_classes": ETF_ASSET_CLASSES,
    "etf_sectors": ETF_SECTORS,
    "etf_core": tuple(dict.fromkeys(ETF_ASSET_CLASSES + ETF_SECTORS)),
    "vix_family": VIX_FAMILY,
}

DYNAMIC_UNIVERSES: tuple[str, ...] = ("us_liquid", "sp500")


@dataclass(frozen=True)
class LiquidityFloors:
    """The screener floors behind ``us_liquid`` (shared with the daily report's v1)."""

    min_market_cap: float = 2e9
    min_price: float = 5.0
    min_volume: float = 1_000_000.0
    exchanges: tuple[str, ...] = ("NASDAQ", "NYSE")


US_LIQUID = LiquidityFloors()


def universe_names() -> list[str]:
    return sorted([*STATIC_UNIVERSES, *DYNAMIC_UNIVERSES])


def resolve_universe(name: str, provider: DataProvider | None, as_of: str | date) -> Universe:
    """Resolve a universe name to a ``Universe`` (static sets need no provider)."""
    as_of_d = as_of if isinstance(as_of, date) else date.fromisoformat(str(as_of))
    key = name.lower()
    if key in STATIC_UNIVERSES:
        return Universe(
            name=key, as_of=as_of_d, symbols=list(STATIC_UNIVERSES[key]), survivorship_safe=True
        )
    if key in DYNAMIC_UNIVERSES:
        if provider is None:
            raise ValueError(f"universe {name!r} needs a DataProvider to resolve")
        return provider.get_universe(key, as_of_d)
    raise KeyError(f"unknown universe {name!r}; known: {universe_names()}")


# --- point-in-time S&P 500 (pure) ------------------------------------------------------------


def change_date(c: dict[str, Any]) -> str:
    """ISO date of a change-log row. FMP's ``date`` is ISO; its ``dateAdded`` is prose
    ("November 18, 1982"), which must never be compared as a string."""
    for key in ("date", "dateAdded"):
        raw = str(c.get(key) or "").strip()
        if not raw:
            continue
        if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
            return raw[:10]
        try:
            from datetime import datetime as _dt

            for fmt in ("%B %d, %Y", "%b %d, %Y"):
                try:
                    return _dt.strptime(raw, fmt).date().isoformat()
                except ValueError:
                    pass
        except Exception:
            pass
    return ""


def sp500_members_as_of(
    current: Iterable[str], changes: Iterable[dict[str, Any]], as_of: str | date
) -> set[str]:
    """Rebuild S&P 500 membership on ``as_of`` from today's list + the change log.

    ``changes`` are FMP ``historical-sp500-constituent`` rows (``dateAdded``,
    ``symbol`` = the ticker added, ``removedTicker`` = the one it replaced).
    Walking the log backwards from today: for every change *after* ``as_of``,
    undo it — drop the added name, restore the removed one. Pure; tested.
    """
    as_of_s = as_of.isoformat() if isinstance(as_of, date) else str(as_of)[:10]
    members = {s.upper() for s in current if s}
    later = [c for c in changes if change_date(c) > as_of_s]
    later.sort(key=change_date, reverse=True)
    for c in later:
        added = str(c.get("symbol") or "").upper()
        removed = str(c.get("removedTicker") or "").upper()
        if added:
            members.discard(added)
        if removed:
            members.add(removed)
    return {s for s in members if s and " " not in s}
