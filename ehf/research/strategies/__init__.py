"""Strategy declarations — thin subclasses on a shared base.

A strategy is ~30-80 lines: an entry signal from the ``signals`` library, a
sizing choice from ``.sizing``, and exit rules from ``.exits``. Importing this
package registers the built-in strategies:

  * ``rise_fall``           — asymmetric, ATR-anchored, trend-filtered reversion
  * ``tsmom``               — time-series momentum, vol-scaled (Moskowitz et al.)
  * ``tsmom_regime``        — TSMOM gated to trending (low-entropy/high-Hurst) regimes
  * ``bollinger_reversion`` — long-term Bollinger-band mean reversion
  * ``xsmom``               — cross-sectional momentum, dollar-neutral long/short
  * ``dial_hold``, ``dial_sma`` — in/out benchmarks on one asset vs. cash (``dial``)
  * ``vix_term`` … ``vix_vote`` — the nine VIX rules + their vote (``vix_overlay``)

The §10 conviction ideas will use a separate lightweight ``Screen`` base.
"""

# Importing the concrete strategies registers them under their RunSpec keys.
from ehf.research.strategies import (  # noqa: E402,F401  (import for side effect)
    bollinger,
    dial,
    regime_gated,
    rise_fall,
    tsmom,
    vix_overlay,
    xsmom,
)
from ehf.research.strategies.base import Strategy, build_strategy, register

__all__ = ["Strategy", "build_strategy", "register"]
