"""Composition overlays: wrappers that take a strategy and return a modified one.

Built as wrappers/mix-ins so they compose onto any Strategy, after the core
engine. The composition seam reserved here:

  * ``regime`` — permutation-entropy / Hurst regime gate (IMPLEMENTED): trade a
    strategy only while the tape is trending. Used by ``tsmom_regime``.
  * volatility-targeting overlay (planned)
  * HRP allocator across multiple strategies (planned)
"""

from ehf.server.research.overlays.regime import (
    GATE_DEFAULTS,
    RegimeGateMixin,
    trending_regime,
)

__all__ = ["GATE_DEFAULTS", "RegimeGateMixin", "trending_regime"]
