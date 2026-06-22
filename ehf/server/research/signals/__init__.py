"""Pure, vectorized, causal signal primitives (price -> signal). The DRY core.

Strategies compose these; they never reimplement them. All functions accept a
Series or a wide DataFrame and use only past/current data.
"""

from ehf.server.research.signals.core import (
    atr,
    bollinger,
    ema,
    hurst,
    log_returns,
    realized_vol,
    returns,
    rolling_slope,
    rolling_std,
    rsi,
    sma,
    zscore,
)
from ehf.server.research.signals.regime import rolling_dfa, rolling_permutation_entropy

__all__ = [
    "returns",
    "log_returns",
    "sma",
    "ema",
    "rolling_std",
    "zscore",
    "bollinger",
    "realized_vol",
    "atr",
    "rsi",
    "rolling_slope",
    "hurst",
    # regime-character signals (wrappers over ehf.server.research.complexity.measures)
    "rolling_permutation_entropy",
    "rolling_dfa",
]
