"""The leverage rules: the 3x fund when the market has turned bullish, the plain fund otherwise.

Dad's question, made mechanical. Each rule is a ``DialStrategy`` whose ``asset`` is
the leveraged fund (``TQQQ``, or the extended ``TQQQ.X``) and whose parking place,
``base``, is the plain fund (``QQQ``) instead of cash: exposure 1 means all the
money is in the 3x fund, 0 means all of it is in the plain fund, 0.5 means half
each. The question "is the market turning bullish?" is answered from the plain
fund's own price (and, for one rule, the VIX), never from the leveraged fund's.

  lev_hold       always the leveraged fund — the "just hold TQQQ" record
  lev_sma        the plain fund above its N-day average (with a band) -> leveraged
  lev_cross      golden cross: the fast average above the slow one -> leveraged
  lev_swing      up X% from the recent low -> leveraged; down Y% from the recent high -> plain
  lev_breakout   a new N-day high -> leveraged; below the M-day low -> plain
  lev_vixfade    the VIX has dropped back below ``calm`` after a scare above ``panic``
                 -> leveraged for ``hold`` sessions (the "after the storm" trade)
  lev_voltarget  not when but how much: leveraged share = target vol / (multiple x realized vol)

``benchmark`` names what the lab draws the grey line against (the plain fund).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

from ehf.research import signals
from ehf.research.backtest.panel import Panel
from ehf.research.signals.dial import (
    band_rebalance,
    cooldown_gate,
    enter_exit_gate,
    hysteresis_gate,
)
from ehf.research.strategies.base import register
from ehf.research.strategies.dial import DIAL_DEFAULTS, DialStrategy

LEV_BASE: dict[str, Any] = {
    **DIAL_DEFAULTS,
    "asset": "TQQQ",  # the leveraged fund (or TQQQ.X, the extended series)
    "base": "QQQ",  # the plain fund: the signal source and the parking place
    "benchmark": "QQQ",  # what the lab charts the run against
    "vix": "^VIX",
}
LEV_BASE.pop("cash")  # the parking place is ``base``; there is no cash leg here


class LeverageDial(DialStrategy):
    """``exposure`` = share in the leveraged fund; ``1 - exposure`` sits in the plain fund."""

    defaults: dict[str, Any] = dict(LEV_BASE)

    def base(self, panel: Panel) -> pd.Series:
        return self.col("base", panel)

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        e = self.exposure(panel).reindex(panel.index).astype(float).clip(0.0, 1.0).fillna(0.0)
        e = signals.min_dwell(e, int(self.p("min_dwell")))
        w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
        asset, base = str(self.p("asset")), str(self.p("base"))
        for name in (asset, base):
            if name not in w.columns:
                raise KeyError(f"column {name!r} not in panel {panel.symbols}")
        w[asset] = e
        w[base] = 1.0 - e
        return w


@register("lev_hold")
class HoldLeveraged(LeverageDial):
    """Always in the leveraged fund. What "just hold TQQQ" would have done."""

    def exposure(self, panel: Panel) -> pd.Series:
        return pd.Series(1.0, index=panel.index)


@register("lev_sma")
class AboveAverage(LeverageDial):
    """Leveraged while the plain fund sits above its N-day average, plain below (with a band)."""

    defaults = {**LEV_BASE, "window": 200, "band": 0.01}

    def exposure(self, panel: Panel) -> pd.Series:
        px = self.base(panel)
        ma = cast(pd.Series, signals.sma(px, int(self.p("window"))))
        rel = px / ma - 1.0
        band = float(self.p("band"))
        # leveraged once price clears the band above the average, plain once it sinks below
        return hysteresis_gate(-rel, out_above=band, in_below=-band).where(ma.notna(), 0.0)


@register("lev_cross")
class GoldenCross(LeverageDial):
    """Leveraged while the fast average sits above the slow one (a golden cross), plain after
    a death cross."""

    defaults = {**LEV_BASE, "fast": 50, "slow": 200, "band": 0.0}

    def exposure(self, panel: Panel) -> pd.Series:
        px = self.base(panel)
        fast = cast(pd.Series, signals.sma(px, int(self.p("fast"))))
        slow = cast(pd.Series, signals.sma(px, int(self.p("slow"))))
        rel = fast / slow - 1.0
        band = float(self.p("band"))
        return hysteresis_gate(-rel, out_above=band, in_below=-band).where(slow.notna(), 0.0)


@register("lev_swing")
class Swing(LeverageDial):
    """A new bull market by the popular definition: up ``up_pct`` from the recent low ->
    leveraged; a bear by the same definition: down ``down_pct`` from the recent high -> plain."""

    defaults = {**LEV_BASE, "up_pct": 0.20, "down_pct": 0.20, "lookback": 252}

    def exposure(self, panel: Panel) -> pd.Series:
        px = self.base(panel)
        n = int(self.p("lookback"))
        from_low = px / px.rolling(n, min_periods=n).min() - 1.0
        from_high = px / px.rolling(n, min_periods=n).max() - 1.0
        enter = from_low >= float(self.p("up_pct"))
        exit_ = from_high <= -float(self.p("down_pct"))
        return enter_exit_gate(
            enter.astype(float).where(from_low.notna()),
            exit_.astype(float).where(from_high.notna()),
        )


@register("lev_breakout")
class Breakout(LeverageDial):
    """A new ``window``-day high -> leveraged; a close under the ``exit_window``-day low
    -> plain."""

    defaults = {**LEV_BASE, "window": 126, "exit_window": 42}

    def exposure(self, panel: Panel) -> pd.Series:
        px = self.base(panel)
        n, m = int(self.p("window")), int(self.p("exit_window"))
        prior_high = px.shift(1).rolling(n, min_periods=n).max()
        prior_low = px.shift(1).rolling(m, min_periods=m).min()
        enter = px >= prior_high
        exit_ = px <= prior_low
        return enter_exit_gate(
            enter.astype(float).where(prior_high.notna()),
            exit_.astype(float).where(prior_high.notna()),
        )


@register("lev_vixfade")
class FearFade(LeverageDial):
    """After a scare (VIX above ``panic`` within the last ``lookback`` sessions) the VIX
    closes back below ``calm`` for the first time: the storm has passed. Leveraged for
    ``hold`` sessions from that day, plain otherwise."""

    defaults = {**LEV_BASE, "panic": 30.0, "calm": 20.0, "lookback": 63, "hold": 63}

    def exposure(self, panel: Panel) -> pd.Series:
        vix = self.col("vix", panel)
        n = int(self.p("lookback"))
        scared = vix.shift(1).rolling(n, min_periods=1).max() > float(self.p("panic"))
        calm = vix < float(self.p("calm"))
        settled = calm & ~calm.shift(1, fill_value=False)  # the first calm close, not every one
        trigger = (scared & settled).astype(float).where(vix.notna())
        return 1.0 - cooldown_gate(trigger, int(self.p("hold")))


@register("lev_voltarget")
class VolTargetLeverage(LeverageDial):
    """How much, not when: the leveraged share that would run the whole position at
    ``target_vol``, given the plain fund's realized volatility and the fund's multiple."""

    defaults = {**LEV_BASE, "target_vol": 0.25, "leverage": 3.0, "rv_window": 21, "band": 0.10}

    def exposure(self, panel: Panel) -> pd.Series:
        rv = cast(pd.Series, signals.realized_vol(self.base(panel), int(self.p("rv_window"))))
        # position vol ~= base vol * (1 + (L - 1) * share); solve for the share that hits target
        lev = float(self.p("leverage"))
        share = ((float(self.p("target_vol")) / rv) - 1.0) / max(lev - 1.0, 1e-9)
        return band_rebalance(share.clip(0.0, 1.0), float(self.p("band"))).where(rv.notna(), 0.0)


#: Walk-forward grids, deliberately small (<= 36 cells) so the Deflated Sharpe stays honest.
GRIDS: dict[str, dict[str, Sequence[Any]]] = {
    "lev_hold": {"min_dwell": [1]},
    "lev_sma": {"window": [100, 150, 200, 250], "band": [0.0, 0.01, 0.03]},
    "lev_cross": {"fast": [20, 50], "slow": [100, 150, 200], "band": [0.0, 0.01]},
    "lev_swing": {
        "up_pct": [0.10, 0.15, 0.20],
        "down_pct": [0.10, 0.15, 0.20],
        "lookback": [126, 252],
    },
    "lev_breakout": {"window": [63, 126, 252], "exit_window": [21, 42, 63]},
    "lev_vixfade": {"panic": [28.0, 32.0], "calm": [18.0, 20.0, 22.0], "hold": [21, 63, 126]},
    "lev_voltarget": {
        "target_vol": [0.20, 0.25, 0.30, 0.35],
        "rv_window": [21, 63],
        "band": [0.10],
    },
}
LEV_RULES: tuple[str, ...] = tuple(k for k in GRIDS if k != "lev_hold")
