"""sample_screen_rows: filtering + reproducible sampling, no network."""

from __future__ import annotations

import pytest

from ehf.server.research.data.fmp_provider import sample_screen_rows


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
