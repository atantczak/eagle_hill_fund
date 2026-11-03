"""Regime-gate overlay — trade a strategy only while the tape is *trending*.

A trend/momentum edge is regime-dependent: it pays in orderly, persistent markets
and bleeds in choppy, mean-reverting ones. This overlay switches a strategy's
exposure *off* outside the trending regime, identified from two cheap,
complementary complexity measures already living in ``ehf.server.research.signals``:

  * **Hurst exponent** (structure-function, ``signals.hurst``) — the persistence
    of *fluctuations*. It is drift-invariant: a random walk and a random walk
    *with* drift read alike, so it cannot tell "trending up" from "flat" (that is
    the strategy's job — the sign of the signal). What it *does* separate is
    persistent/random-walk structure (trend-friendly) from mean-reverting
    structure (trend-hostile). We require ``H >= hurst_min`` to gate OFF the
    mean-reverting case. NOTE on scale: this finite-sample, short-lag estimator
    reads a true random walk near ~0.4 (not the textbook 0.5) over ~150-bar
    windows — matching the repo's own loose Hurst test — so ``hurst_min``
    defaults to 0.40, *not* 0.5. Strong Ornstein-Uhlenbeck reversion reads ~0.13
    and is reliably excluded.
  * **Permutation entropy** (Bandt-Pompe) — ordinal-pattern complexity in [0, 1].
    On noisy price it sits near ~0.97 whether trending or reverting, so it is a
    weak *direction* discriminator; its honest role here is to reject degenerate,
    near-deterministic regimes (it drops sharply for smooth cycles). We require
    ``PE <= pe_max`` as a mild "not maximally disordered" filter.

A bar counts as "trending" only when BOTH agree. The AND is what makes the pair
work where either alone fails: a clean *cycle* is low-entropy (PE would pass it)
yet mean-reverting (H≈0 fails it), so the cycle is correctly gated OFF. We are
deliberately conservative — rather sit out a real trend than hold through chop.
The mask is causal by construction (both signals are trailing-window rollings),
so gating a bar never consults future data; the same look-ahead safety the
signal layer guarantees.

Shipped as a mix-in so it composes onto *any* ``Strategy`` (the seam this package
reserves): list ``RegimeGateMixin`` first in the bases and it filters whatever
weights the inner strategy produced, inside ``manage_exits``.
"""

from __future__ import annotations

import pandas as pd

from ehf.server.research import signals
from ehf.server.research.backtest.panel import Panel

#: Default gate parameters. Kept as a module constant so concrete strategies can
#: splice them into their own ``defaults`` (e.g. ``{**TSMOM.defaults, **GATE_DEFAULTS}``)
#: and so every knob is sweepable like any other strategy parameter.
GATE_DEFAULTS = {
    "entropy_window": 100,  # bars of history for permutation entropy
    "entropy_order": 3,  # ordinal-pattern length (3 = 6 patterns)
    "pe_max": 0.97,  # gate ON only when normalized PE <= this (reject pure disorder)
    "hurst_window": 150,  # bars of history for the Hurst estimate (repo test uses ~200)
    "hurst_min": 0.40,  # gate ON only when Hurst >= this; ~0.4 = random-walk on this estimator
}


def trending_regime(
    close: pd.DataFrame,
    *,
    entropy_window: int = 100,
    entropy_order: int = 3,
    pe_max: float = 0.97,
    hurst_window: int = 150,
    hurst_min: float = 0.40,
) -> pd.DataFrame:
    """Boolean mask: ``True`` where each series sits in a low-entropy, persistent regime.

    A bar is trending only when permutation entropy is low (``<= pe_max``) AND
    the Hurst exponent is high (``>= hurst_min``), both over trailing windows.
    Warm-up bars (insufficient history → NaN measures) resolve to ``False``: the
    strategy stays flat until each estimator has enough data. Same shape as
    ``close``.
    """
    pe = signals.rolling_permutation_entropy(close, entropy_window, order=entropy_order)
    h = signals.hurst(close, hurst_window)
    # NaN <= x and NaN >= x are both False, so warm-up bars gate off automatically;
    # fillna(False) only guards against a stray NaN surviving the AND.
    return ((pe <= pe_max) & (h >= hurst_min)).fillna(False)


class RegimeGateMixin:
    """Mix-in that zeroes a strategy's weights outside the trending regime.

    Place it FIRST in the base list so its ``manage_exits`` runs, then delegates
    up the MRO to the inner strategy's exits::

        @register("tsmom_regime")
        class RegimeGatedTSMOM(RegimeGateMixin, TSMOM):
            defaults = {**TSMOM.defaults, **RegimeGateMixin.gate_defaults}

    Parameters are read from ``self.params`` (so they must be merged into the
    concrete strategy's ``defaults``), which makes the gate sweepable exactly
    like any other strategy knob. This is a mix-in, not a standalone class: it
    relies on the host providing ``p()`` and an upstream ``manage_exits``.
    """

    #: Defaults a concrete strategy splices into its own ``defaults`` map.
    gate_defaults = GATE_DEFAULTS

    def manage_exits(self, weights: pd.DataFrame, panel: Panel) -> pd.DataFrame:
        # Let the inner strategy apply its own exits first, then gate the result.
        weights = super().manage_exits(weights, panel)  # type: ignore[misc]
        mask = trending_regime(
            panel.close,
            entropy_window=self.p("entropy_window"),  # type: ignore[attr-defined]
            entropy_order=self.p("entropy_order"),  # type: ignore[attr-defined]
            pe_max=self.p("pe_max"),  # type: ignore[attr-defined]
            hurst_window=self.p("hurst_window"),  # type: ignore[attr-defined]
            hurst_min=self.p("hurst_min"),  # type: ignore[attr-defined]
        )
        # Hold the inner weights only on trending bars; flat (cash) everywhere else.
        return weights.where(mask, 0.0)
