"""Regime-gated TSMOM — time-series momentum that trades only in trending regimes.

The first end-to-end complexity result through the engine: vanilla TSMOM with the
permutation-entropy / Hurst regime gate (``overlays.regime``) layered on, so the
book sits in cash whenever the tape turns choppy or mean-reverting. Pure
composition — the momentum signal and vol-target sizing come straight from
``TSMOM``; the gate from the overlay mix-in. No new math, no copied logic.

Hypothesis: momentum's losing stretches cluster in high-entropy / low-persistence
chop; gating those out should lift risk-adjusted return versus plain TSMOM.
"""

from __future__ import annotations

from ehf.research.overlays.regime import RegimeGateMixin
from ehf.research.strategies.base import register
from ehf.research.strategies.tsmom import TSMOM


@register("tsmom_regime")
class RegimeGatedTSMOM(RegimeGateMixin, TSMOM):
    # Inherit TSMOM's signal + sizing knobs and add the gate's, all sweepable.
    # MRO is RegimeGatedTSMOM -> RegimeGateMixin -> TSMOM -> Strategy, so
    # generate_signals/size_positions come from TSMOM while manage_exits comes
    # from the mix-in (which then delegates up to TSMOM/Strategy exits).
    defaults = {**TSMOM.defaults, **RegimeGateMixin.gate_defaults}
