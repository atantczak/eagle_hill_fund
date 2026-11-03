"""``DialStrategy`` — one asset, one cash leg, and an exposure dial in [0, 1].

The "should I be in the market?" question, made mechanical. A dial strategy
produces a single exposure series ``e_t`` (1 = fully in, 0.5 = half, 0 = out)
from whatever it watches; the base class turns that into target weights
``asset = e``, ``cash = 1 - e`` (every other column, e.g. the VIX indices that
ride along as signals, gets 0). The engine then does what it always does.

The shared state machines (hysteresis gate, three-state dial, cooldown, dwell,
band rebalance, confirm) live in ``signals.dial`` and are re-exported here, so
every rule uses the same tested logic and the daily report can run it live.

Two benchmarks ship alongside so every VIX rule is judged against the honest
alternatives: ``dial_hold`` (always in) and ``dial_sma`` (the 200-day filter).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, cast

import pandas as pd

from ehf.research import signals
from ehf.research.backtest.panel import Panel
from ehf.research.strategies.base import Strategy, register

DIAL_DEFAULTS: dict[str, Any] = {
    "asset": "SPY",
    "cash": "CASH",
    "min_dwell": 1,  # bars a new dial state must persist (1 = no brake)
}


# The state machines live in ``signals.dial`` (pure, Lambda-bundled); re-exported
# here so strategies and tests keep one import path.
from ehf.research.signals.dial import (  # noqa: E402
    band_rebalance,
    confirm,
    cooldown_gate,
    hysteresis_gate,
    min_dwell,
    three_state,
)

__all__ = [
    "DIAL_DEFAULTS",
    "DialStrategy",
    "band_rebalance",
    "confirm",
    "cooldown_gate",
    "hysteresis_gate",
    "min_dwell",
    "three_state",
]


# --- the base class --------------------------------------------------------------------


class DialStrategy(Strategy):
    """Subclasses implement ``exposure(panel) -> Series in [0, 1]``; nothing else."""

    defaults: dict[str, Any] = dict(DIAL_DEFAULTS)

    @abstractmethod
    def exposure(self, panel: Panel) -> pd.Series:
        """Exposure dial for ``asset`` using data through each bar."""

    def col(self, name: str, panel: Panel) -> pd.Series:
        """A close-price column named by the ``name`` parameter (e.g. ``vix``)."""
        sym = str(self.p(name))
        if sym not in panel.close.columns:
            raise KeyError(f"{type(self).__name__} needs column {sym!r} (param {name!r})")
        return panel.close[sym]

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        e = self.exposure(panel).reindex(panel.index).astype(float).clip(0.0, 1.0).fillna(0.0)
        e = min_dwell(e, int(self.p("min_dwell")))
        w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
        asset, cash = str(self.p("asset")), str(self.p("cash"))
        if asset not in w.columns:
            raise KeyError(f"asset column {asset!r} not in panel {panel.symbols}")
        w[asset] = e
        if cash in w.columns:
            w[cash] = 1.0 - e
        return w

    def size_positions(self, signals_df: pd.DataFrame, panel: Panel) -> pd.DataFrame:
        return signals_df  # the dial already *is* the weight vector


# --- benchmarks ------------------------------------------------------------------------


@register("dial_hold")
class Hold(DialStrategy):
    """Always fully in. The bar every overlay has to clear on return."""

    def exposure(self, panel: Panel) -> pd.Series:
        return pd.Series(1.0, index=panel.index)


@register("dial_sma")
class SMAFilter(DialStrategy):
    """The classic trend filter: in above the N-day average, out below (with a band)."""

    defaults = {**DIAL_DEFAULTS, "window": 200, "band": 0.0}

    def exposure(self, panel: Panel) -> pd.Series:
        px = self.col("asset", panel)
        ma = cast(pd.Series, signals.sma(px, int(self.p("window"))))
        rel = px / ma - 1.0  # distance from the average, as a fraction
        band = float(self.p("band"))
        # out when price sinks below the band under the average; in when it clears above
        return hysteresis_gate(-rel, out_above=band, in_below=-band)
