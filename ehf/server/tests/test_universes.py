"""Named universes: static sets are clean, S&P 500 point-in-time rebuild is exact."""

from __future__ import annotations

from datetime import date

import pytest

from ehf.server.research.data.universes import (
    DYNAMIC_UNIVERSES,
    STATIC_UNIVERSES,
    resolve_universe,
    sp500_members_as_of,
    universe_names,
)


def test_static_universes_are_upper_unique_and_named() -> None:
    for name, syms in STATIC_UNIVERSES.items():
        assert len(set(syms)) == len(syms), name
        # Cboe-style index tickers (^VIX, ^IRX) carry a caret; otherwise plain letters.
        assert all(s == s.upper() and s.lstrip("^").isalnum() for s in syms), name
    assert set(STATIC_UNIVERSES["etf_core"]) == set(STATIC_UNIVERSES["etf_asset_classes"]) | set(
        STATIC_UNIVERSES["etf_sectors"]
    )
    assert set(universe_names()) == set(STATIC_UNIVERSES) | set(DYNAMIC_UNIVERSES)


def test_resolve_static_needs_no_provider_and_is_pit_safe() -> None:
    u = resolve_universe("etf_sectors", None, "2015-06-30")
    assert u.survivorship_safe and u.as_of == date(2015, 6, 30) and "XLK" in u.symbols
    with pytest.raises(ValueError, match="needs a DataProvider"):
        resolve_universe("sp500", None, "2015-06-30")
    with pytest.raises(KeyError):
        resolve_universe("nope", None, "2015-06-30")


def test_sp500_members_as_of_undoes_later_changes_only() -> None:
    current = ["AAPL", "NEW1", "NEW2"]
    changes = [
        {"dateAdded": "2024-03-18", "symbol": "NEW2", "removedTicker": "OLD2"},
        {"dateAdded": "2023-06-20", "symbol": "NEW1", "removedTicker": "OLD1"},
        {"dateAdded": "2020-01-01", "symbol": "AAPL", "removedTicker": "ANCIENT"},
    ]
    # between the two later changes: NEW1 is in, NEW2 isn't yet, OLD2 still is
    assert sp500_members_as_of(current, changes, "2023-12-31") == {"AAPL", "NEW1", "OLD2"}
    # before both: both originals back
    assert sp500_members_as_of(current, changes, date(2022, 1, 1)) == {"AAPL", "OLD1", "OLD2"}
    # as of today: unchanged
    assert sp500_members_as_of(current, changes, "2025-01-01") == set(current)
    # a change dated exactly on as_of already happened (not undone)
    assert "NEW2" in sp500_members_as_of(current, changes, "2024-03-18")
