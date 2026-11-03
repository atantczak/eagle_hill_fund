"""The daily report's market dial: context from bars, the hero tile, the bullet."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ehf.server.research.reporting.config import ReportConfig
from ehf.server.research.reporting.market import dial_context, dial_label
from ehf.server.research.reporting.render import dial_bullet, dial_tile, hero_tiles


def _bars(
    n: int = 300, *, stress: bool = False, seed: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    r = rng.normal(0.0004, 0.01, n)
    spy = 100 * np.exp(np.cumsum(r))
    vix = np.clip(18 - 300 * r + rng.normal(0, 0.5, n), 9, None)
    vix3m = vix * 1.06
    if stress:  # the last fortnight: curve inverted, realized vol far above implied
        spy[-10:] = spy[-11] * np.exp(np.cumsum(rng.normal(-0.02, 0.05, 10)))
        vix[-10:] = 30.0
        vix3m[-10:] = 26.0
    spy_bars = pd.DataFrame({"symbol": "SPY", "date": dates, "close": spy})
    vix_bars = pd.concat(
        [
            pd.DataFrame({"symbol": "^VIX", "date": dates, "close": vix}),
            pd.DataFrame({"symbol": "^VIX3M", "date": dates[::2], "close": vix3m[::2]}),  # gaps
        ],
        ignore_index=True,
    )
    return spy_bars, vix_bars


def test_dial_context_calm_market_is_in() -> None:
    spy, vix = _bars()
    d = dial_context(spy, vix, ReportConfig())
    assert d["label"] in {"IN", "HALF"} and d["exposure"] >= 0.5
    assert d["term"]["exposure"] == 1.0 and d["term"]["ratio"] < 1.0
    assert d["as_of"] == str(spy["date"].iloc[-1].date())
    assert d["days_in_state"] >= 1 and isinstance(d["changed"], bool)


def test_dial_context_stress_is_out() -> None:
    spy, vix = _bars(stress=True)
    d = dial_context(spy, vix, ReportConfig())
    assert d["term"]["exposure"] == 0.0 and d["term"]["ratio"] > 1.0
    assert d["vrp"]["exposure"] == 0.0 and d["vrp"]["points"] < 0  # realized blew through implied
    assert d["label"] in {"OUT", "HALF"} and d["exposure"] <= 0.5


def test_dial_context_degrades_to_empty() -> None:
    spy, vix = _bars()
    assert dial_context(spy, pd.DataFrame(), ReportConfig()) == {}
    assert dial_context(spy, vix[vix["symbol"] == "^VIX"], ReportConfig()) == {}  # no VIX3M


def test_dial_label_bands() -> None:
    assert dial_label(1.0) == "IN" and dial_label(0.5) == "HALF" and dial_label(0.0) == "OUT"
    assert dial_label(None) == "unknown"


def test_dial_tile_and_bullet_render() -> None:
    dial = {
        "exposure": 0.5,
        "label": "HALF",
        "changed": True,
        "days_in_state": 1,
        "term": {"ratio": 0.97, "exposure": 1.0, "out_above": 1.0},
        "resid": {"z": 2.4, "exposure": 0.0, "warn_z": 2.0},
        "vrp": {"points": 3.1, "exposure": 0.5, "out_below": 0.0, "full_above": 4.0},
    }
    tile = dial_tile(dial)
    assert tile is not None and tile.value == "HALF" and tile.tone == ""
    assert "50.0% in" in tile.detail and "changed today" in tile.detail
    assert "curve 0.97" in tile.detail
    b = dial_bullet(dial)
    assert "Market dial: HALF" in b and "(in;" in b and "(out;" in b and "(half;" in b
    tiles = hero_tiles({"dial": dial, "vix": {"level": 20.0, "change_pct": 1.0}})
    assert tiles[0].label == "Market dial" and tiles[1].label == "VIX"
    assert dial_tile({}) is None and dial_tile({"exposure": None}) is None
