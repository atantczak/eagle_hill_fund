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
from ehf.server.research.signals.dial import (
    band_rebalance,
    confirm,
    cooldown_gate,
    hysteresis_gate,
    min_dwell,
    three_state,
)
from ehf.server.research.signals.regime import rolling_dfa, rolling_permutation_entropy
from ehf.server.research.signals.vix import (
    exposure_resid,
    exposure_term,
    exposure_vote,
    exposure_vrp,
    implied_daily_move,
    put_demand_zscore,
    realized_vol_points,
    residual_zscore,
    surprise_rate,
    term_structure_ratio,
    variance_risk_premium,
    vix_beta_residual,
    vix_spy_correlation,
    vix_zscore,
    vol_of_vol_ratio,
)

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
    # VIX-family signals (signals.vix)
    "implied_daily_move",
    "term_structure_ratio",
    "realized_vol_points",
    "variance_risk_premium",
    "vix_zscore",
    "vix_beta_residual",
    "residual_zscore",
    "surprise_rate",
    "vix_spy_correlation",
    "vol_of_vol_ratio",
    "put_demand_zscore",
    "exposure_term",
    "exposure_resid",
    "exposure_vrp",
    "exposure_vote",
    # dial state machines (signals.dial)
    "hysteresis_gate",
    "three_state",
    "cooldown_gate",
    "min_dwell",
    "band_rebalance",
    "confirm",
]
