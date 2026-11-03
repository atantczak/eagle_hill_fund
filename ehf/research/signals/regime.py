"""Regime-character signals: rolling wrappers over complexity measures.

These let a strategy consume complexity measures *exactly like any other signal*
(``signals.rolling_permutation_entropy(...)``). The heavy math lives once in
``ehf.research.complexity.measures``; here we only apply it over a rolling window.

Causal by construction (each window uses only past/current data). Note: the
rolling Hurst signal already exists as ``ehf.research.signals.hurst`` (structure-function
estimator); these add the ordinal-complexity and DFA regime signals.
"""

from __future__ import annotations

import numpy as np

from ehf.research.complexity.measures import dfa, permutation_entropy
from ehf.research.signals.core import Frame


def rolling_permutation_entropy(
    prices: Frame, window: int, order: int = 3, delay: int = 1
) -> Frame:
    """Rolling Bandt-Pompe permutation entropy (low = orderly/trending regime)."""

    def _pe(a: np.ndarray) -> float:
        return permutation_entropy(a, order=order, delay=delay)

    return prices.rolling(window).apply(_pe, raw=True)


def rolling_dfa(prices: Frame, window: int, order: int = 1) -> Frame:
    """Rolling DFA exponent (a > 0.5 persistent, < 0.5 mean-reverting)."""

    def _dfa(a: np.ndarray) -> float:
        return dfa(a, order=order, min_scale=4, max_scale=max(8, window // 4))

    return prices.rolling(window).apply(_dfa, raw=True)
