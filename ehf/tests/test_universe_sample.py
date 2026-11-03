"""Universe sampling: filtering, reproducible draws, and point-in-time (as_of)
survivorship handling — all without network."""

from __future__ import annotations

import types

import pytest

from ehf.research.data.fmp_provider import (
    FMPProvider,
    delisted_alive_on,
    sample_screen_rows,
)


def _rows() -> list[dict]:
    rows = [{"symbol": f"S{i:02d}", "exchangeShortName": "NASDAQ"} for i in range(20)]
    rows += [{"symbol": f"N{i:02d}", "exchangeShortName": "NYSE"} for i in range(20)]
    rows += [
        {"symbol": "OTCJUNK", "exchangeShortName": "OTC"},  # wrong exchange -> dropped
        {"symbol": "BRK.B", "exchangeShortName": "NYSE"},  # share class -> dropped
        {"symbol": "FOO-WT", "exchangeShortName": "NYSE"},  # warrant -> dropped
        {"symbol": None, "exchangeShortName": "NYSE"},  # no symbol -> dropped
    ]
    return rows


def test_filters_exchanges_and_odd_tickers() -> None:
    out = sample_screen_rows(_rows(), 100)  # n > available -> return all kept
    assert "OTCJUNK" not in out and "BRK.B" not in out and "FOO-WT" not in out
    assert len(out) == 40  # 20 NASDAQ + 20 NYSE


def test_sampling_is_seeded_and_reproducible() -> None:
    a = sample_screen_rows(_rows(), 10, seed=42)
    b = sample_screen_rows(_rows(), 10, seed=42)
    c = sample_screen_rows(_rows(), 10, seed=7)
    assert a == b and len(a) == 10
    assert a != c  # different seed -> different draw
    assert a == sorted(a)  # returned sorted


def test_exchange_filter_override() -> None:
    out = sample_screen_rows(_rows(), 100, exchanges=("NYSE",))
    assert all(s.startswith("N") for s in out) and len(out) == 20


def test_non_list_raises() -> None:
    with pytest.raises(TypeError):
        sample_screen_rows({"error": "bad"}, 5)


# --------------------------------------------------------------------------- #
# delisted_alive_on: "was this name listed & trading on date D?"              #
# --------------------------------------------------------------------------- #
def _delisted() -> list[dict]:
    return [
        # listed 2001, died 2020 -> alive across the 2010 window
        {
            "symbol": "DEADCO",
            "exchange": "NYSE",
            "ipoDate": "2001-05-01",
            "delistedDate": "2020-09-15",
        },
        # died in 2008, BEFORE 2010 -> not alive on 2010
        {
            "symbol": "GONEBY08",
            "exchange": "NASDAQ",
            "ipoDate": "1999-01-01",
            "delistedDate": "2008-09-15",
        },
        # IPO'd 2015, AFTER 2010 -> didn't exist yet on 2010
        {
            "symbol": "TOOYOUNG",
            "exchange": "NASDAQ",
            "ipoDate": "2015-01-01",
            "delistedDate": "2022-01-01",
        },
        {
            "symbol": "CHINACO",
            "exchange": "SHH",
            "ipoDate": "2000-01-01",
            "delistedDate": "2019-01-01",
        },  # foreign
        {
            "symbol": "BRK.A",
            "exchange": "NYSE",
            "ipoDate": "1990-01-01",
            "delistedDate": "2019-01-01",
        },  # odd ticker
        {
            "symbol": "NODATE",
            "exchange": "NYSE",
            "ipoDate": "",
            "delistedDate": None,
        },  # missing dates
    ]


def test_delisted_alive_on_window() -> None:
    alive = delisted_alive_on(_delisted(), "2010-06-01")
    assert alive == {"DEADCO"}  # only the name whose [ipo, delisted) brackets 2010-06-01


def test_delisted_alive_on_boundaries_and_filters() -> None:
    rows = _delisted()
    # On its delisting day a name is already gone (interval is half-open: [ipo, delisted)).
    assert "DEADCO" not in delisted_alive_on(rows, "2020-09-15")
    # On its IPO day it is alive.
    assert "DEADCO" in delisted_alive_on(rows, "2001-05-01")
    # A 2007 date catches the name that later died in 2008.
    assert delisted_alive_on(rows, "2007-01-01") == {"DEADCO", "GONEBY08"}
    # Exchange override drops NYSE.
    assert delisted_alive_on(rows, "2007-01-01", exchanges=("NASDAQ",)) == {"GONEBY08"}


# --------------------------------------------------------------------------- #
# random_symbols(as_of=...) composition — no network (fake compiler + memo).  #
# --------------------------------------------------------------------------- #
def _provider_with_fakes() -> FMPProvider:
    prov = FMPProvider()
    # Active survivors returned by the screener (today's liquid names).
    active = [
        {"symbol": "AAA", "exchangeShortName": "NYSE"},
        {"symbol": "BBB", "exchangeShortName": "NASDAQ"},
    ]
    prov._compiler = types.SimpleNamespace(screen_companies=lambda **_: active)
    prov._delisted = _delisted()  # pre-seed the memo so no network call happens
    return prov


def test_random_symbols_as_of_none_is_survivors_only() -> None:
    prov = _provider_with_fakes()
    assert set(prov.random_symbols(100, as_of=None)) == {"AAA", "BBB"}


def test_random_symbols_as_of_adds_delisted_alive_then() -> None:
    prov = _provider_with_fakes()
    got = set(prov.random_symbols(100, as_of="2010-06-01"))
    assert got == {"AAA", "BBB", "DEADCO"}  # survivors + the name alive-then-but-since-dead
    # Toggle the delisted union off -> back to survivors only.
    survivors = prov.random_symbols(100, as_of="2010-06-01", include_delisted=False)
    assert set(survivors) == {"AAA", "BBB"}
