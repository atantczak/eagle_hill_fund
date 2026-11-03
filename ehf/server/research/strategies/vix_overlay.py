"""The nine VIX rules + their vote, as dial strategies on SPY vs. cash.

Each class is a thin declaration: one or two signals from ``signals.vix``, one
state-machine helper from ``strategies.dial``, a handful of sweepable knobs.
Numbering follows ``docs/reports/VIX_SPY_Explainer.pdf``:

  1 vix_term      fear-curve flip        VIX / VIX3M above 1 -> out
  2 vix_resid     unexplained fear       VIX moved more than SPY justifies -> out, cooldown
  3 vix_vrp       overpriced insurance   VIX minus realized: negative -> out, large -> in
  4 vix_spike     spike and fade         buy fear: full in after a z-score spike
  5 vix_sized     size by fear           exposure = target vol / forecast vol
  6 vix_surprise  surprise count         share of days beating the implied move
  7 vix_corr      broken link            VIX and SPY stop moving opposite -> out
  8 vix_vvix      fear of fear           VVIX (or VVIX/VIX) stretched -> out, cooldown
  9 vix_skew      insurance mix          Cboe SKEW stretched -> out
 10 vix_vote      mean of rules 1, 2, 3

Every rule reads the VIX family through named columns (``vix``, ``vix3m``,
``vvix``, ``skew``) so a test can feed synthetic data under any names. NaN
inputs resolve to exposure 0 (flat) — see ``DialStrategy``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

from ehf.server.research import signals
from ehf.server.research.backtest.panel import Panel
from ehf.server.research.strategies.base import register
from ehf.server.research.strategies.dial import (
    DIAL_DEFAULTS,
    DialStrategy,
    band_rebalance,
    cooldown_gate,
    three_state,
)

VIX_COLUMNS: dict[str, Any] = {
    "vix": "^VIX",
    "vix3m": "^VIX3M",
    "vvix": "^VVIX",
    "skew": "^SKEW",
}
BASE: dict[str, Any] = {**DIAL_DEFAULTS, **VIX_COLUMNS}


@register("vix_term")
class TermStructure(DialStrategy):
    """Rule 1. Out when spot fear exceeds 3-month fear; back in once the curve re-steepens."""

    defaults = {**BASE, "out_above": 1.0, "band": 0.05, "confirm_bars": 1}

    def exposure(self, panel: Panel) -> pd.Series:
        return signals.exposure_term(
            self.col("vix", panel),
            self.col("vix3m", panel),
            out_above=float(self.p("out_above")),
            band=float(self.p("band")),
            confirm_bars=int(self.p("confirm_bars")),
        )


@register("vix_resid")
class UnexplainedFear(DialStrategy):
    """Rule 2. Hedgers paying up beyond what the tape explains -> step out for a cooldown."""

    defaults = {**BASE, "window": 63, "warn_z": 2.0, "cooldown": 10}

    def exposure(self, panel: Panel) -> pd.Series:
        return signals.exposure_resid(
            self.col("asset", panel),
            self.col("vix", panel),
            window=int(self.p("window")),
            warn_z=float(self.p("warn_z")),
            cooldown=int(self.p("cooldown")),
        )


@register("vix_vrp")
class OverpricedInsurance(DialStrategy):
    """Rule 3. VIX below realized -> out; comfortably above -> fully in; between -> half."""

    defaults = {**BASE, "rv_window": 21, "out_below": 0.0, "full_above": 4.0}

    def exposure(self, panel: Panel) -> pd.Series:
        return signals.exposure_vrp(
            self.col("vix", panel),
            self.col("asset", panel),
            rv_window=int(self.p("rv_window")),
            out_below=float(self.p("out_below")),
            full_above=float(self.p("full_above")),
        )


@register("vix_spike")
class SpikeAndFade(DialStrategy):
    """Rule 4. Hold ``base`` normally; go fully in for ``hold`` bars after a VIX spike."""

    defaults = {**BASE, "window": 63, "spike_z": 2.0, "hold": 21, "base": 0.5}

    def exposure(self, panel: Panel) -> pd.Series:
        z = signals.vix_zscore(self.col("vix", panel), int(self.p("window")))
        spiked = 1.0 - cooldown_gate(z > float(self.p("spike_z")), int(self.p("hold")))
        base = float(self.p("base"))
        return (base + (1.0 - base) * spiked).where(z.notna(), 0.0)


@register("vix_sized")
class SizeByFear(DialStrategy):
    """Rule 5. A dial, not a switch: exposure = target vol / forecast vol, capped at 1."""

    defaults = {**BASE, "target_vol": 0.12, "blend": 1.0, "rv_window": 21, "band": 0.05}

    def exposure(self, panel: Panel) -> pd.Series:
        vix = self.col("vix", panel) / 100.0
        rv = cast(
            pd.Series, signals.realized_vol(self.col("asset", panel), int(self.p("rv_window")))
        )
        b = float(self.p("blend"))
        forecast = b * vix + (1.0 - b) * rv
        target = (float(self.p("target_vol")) / forecast).clip(upper=1.0)
        return band_rebalance(target, float(self.p("band")))


@register("vix_surprise")
class SurpriseCount(DialStrategy):
    """Rule 6. Too many days beating the implied move -> out; rare again -> in."""

    defaults = {**BASE, "window": 21, "out_above": 0.35, "band": 0.10}

    def exposure(self, panel: Panel) -> pd.Series:
        rate = signals.surprise_rate(
            self.col("asset", panel), self.col("vix", panel), int(self.p("window"))
        )
        hi = float(self.p("out_above"))
        return three_state(rate, lo=hi - float(self.p("band")), hi=hi)


@register("vix_corr")
class BrokenLink(DialStrategy):
    """Rule 7. VIX and SPY rising together (correlation toward zero) -> out."""

    defaults = {**BASE, "window": 21, "out_above": -0.3, "band": 0.2}

    def exposure(self, panel: Panel) -> pd.Series:
        corr = signals.vix_spy_correlation(
            self.col("asset", panel), self.col("vix", panel), int(self.p("window"))
        )
        hi = float(self.p("out_above"))
        return three_state(corr, lo=hi - float(self.p("band")), hi=hi)


@register("vix_vvix")
class FearOfFear(DialStrategy):
    """Rule 8. Vol-of-vol stretched (VVIX, or VVIX/VIX) -> out for a cooldown."""

    defaults = {**BASE, "window": 63, "out_z": 2.0, "cooldown": 10, "use_ratio": False}

    def exposure(self, panel: Panel) -> pd.Series:
        vvix = self.col("vvix", panel)
        x = signals.vol_of_vol_ratio(vvix, self.col("vix", panel)) if self.p("use_ratio") else vvix
        z = cast(pd.Series, signals.zscore(x, int(self.p("window"))))
        return cooldown_gate(z > float(self.p("out_z")), int(self.p("cooldown")))


@register("vix_skew")
class InsuranceMix(DialStrategy):
    """Rule 9. Crash-protection demand (Cboe SKEW) surging vs. its norm -> out."""

    defaults = {**BASE, "window": 63, "out_z": 1.5, "band": 1.0}

    def exposure(self, panel: Panel) -> pd.Series:
        z = signals.put_demand_zscore(self.col("skew", panel), int(self.p("window")))
        hi = float(self.p("out_z"))
        return three_state(z, lo=hi - float(self.p("band")), hi=hi)


@register("vix_vote")
class Vote(DialStrategy):
    """Rule 10. Average of rules 1, 2 and 3 — three nearly independent reads of fear."""

    defaults = {
        **BASE,
        "term_out_above": 1.0,
        "term_band": 0.05,
        "resid_window": 63,
        "resid_warn_z": 2.0,
        "resid_cooldown": 10,
        "vrp_rv_window": 21,
        "vrp_out_below": 0.0,
        "vrp_full_above": 4.0,
    }

    def _sub(self, cls: type[DialStrategy], prefix: str) -> DialStrategy:
        shared = {k: self.params[k] for k in BASE}
        own = {
            k[len(prefix) :]: v
            for k, v in self.params.items()
            if k.startswith(prefix) and k[len(prefix) :] in cls.defaults
        }
        return cls(**{**shared, **own, "min_dwell": 1})

    def exposure(self, panel: Panel) -> pd.Series:
        parts = [
            self._sub(TermStructure, "term_").exposure(panel),
            self._sub(UnexplainedFear, "resid_").exposure(panel),
            self._sub(OverpricedInsurance, "vrp_").exposure(panel),
        ]
        return pd.concat(parts, axis=1).mean(axis=1)


#: Parameter grids for the walk-forward study — the "mess with parameters" set.
#: Each is deliberately small (12–54 cells) so the Deflated Sharpe stays honest.
GRIDS: dict[str, dict[str, Sequence[Any]]] = {
    "dial_hold": {"min_dwell": [1]},
    "dial_sma": {"window": [100, 150, 200, 250], "band": [0.0, 0.01, 0.02]},
    "vix_term": {
        "out_above": [0.95, 1.0, 1.05],
        "band": [0.02, 0.05, 0.10],
        "confirm_bars": [1, 3],
    },
    "vix_resid": {"window": [42, 63, 126], "warn_z": [1.5, 2.0, 2.5], "cooldown": [5, 10, 21]},
    "vix_vrp": {
        "rv_window": [10, 21, 42],
        "out_below": [-2.0, 0.0, 2.0],
        "full_above": [2.0, 4.0, 6.0],
    },
    "vix_spike": {
        "window": [42, 63, 126],
        "spike_z": [1.5, 2.0, 2.5],
        "hold": [10, 21, 42],
        "base": [0.0, 0.5],
    },
    "vix_sized": {
        "target_vol": [0.08, 0.10, 0.12, 0.15],
        "blend": [1.0, 0.5],
        "band": [0.05, 0.10],
    },
    "vix_surprise": {"window": [10, 21, 42], "out_above": [0.30, 0.35, 0.40], "band": [0.05, 0.10]},
    "vix_corr": {"window": [21, 42, 63], "out_above": [-0.4, -0.3, -0.2, 0.0], "band": [0.1, 0.2]},
    "vix_vvix": {
        "window": [42, 63, 126],
        "out_z": [1.5, 2.0, 2.5],
        "cooldown": [5, 10, 21],
        "use_ratio": [False, True],
    },
    "vix_skew": {"window": [42, 63, 126], "out_z": [1.0, 1.5, 2.0], "band": [0.5, 1.0]},
    "vix_vote": {
        "term_out_above": [1.0, 1.05],
        "resid_warn_z": [2.0, 2.5],
        "vrp_out_below": [0.0, 2.0],
        "min_dwell": [1, 5],
    },
}
VIX_RULES: tuple[str, ...] = tuple(k for k in GRIDS if k.startswith("vix_"))
BENCHMARKS: tuple[str, ...] = ("dial_hold", "dial_sma")
