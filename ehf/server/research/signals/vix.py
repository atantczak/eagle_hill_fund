"""VIX-family signal primitives — the fear gauge read nine different ways.

Every function is pure, vectorized and causal (trailing windows only), takes
aligned ``pd.Series`` (one asset: SPY close, VIX level, ...) and returns a
Series on the same index. The dial strategies in ``strategies.vix_overlay``
compose these; the explainer in ``docs/reports`` describes each in plain words.

Conventions: VIX-family levels are in *vol points* (20.0 = 20%); realized vol
is converted to the same units so differences are directly comparable.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ehf.server.research.signals.core import log_returns, realized_vol, returns, zscore

_SQRT_252 = float(np.sqrt(252.0))


def implied_daily_move(vix: pd.Series) -> pd.Series:
    """The one-sigma daily move the VIX is pricing (fraction): VIX/100/sqrt(252)."""
    return vix / 100.0 / _SQRT_252


def term_structure_ratio(vix: pd.Series, vix_long: pd.Series) -> pd.Series:
    """Spot VIX over a longer-dated VIX (e.g. VIX3M). > 1 = backwardation = stress."""
    return vix / vix_long


def realized_vol_points(close: pd.Series, window: int) -> pd.Series:
    """Trailing realized vol of ``close`` in VIX units (annualized percent)."""
    return cast(pd.Series, realized_vol(close, window) * 100.0)


def variance_risk_premium(vix: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """VIX minus *trailing* realized vol (vol points). Positive = insurance overpriced.

    Causal by construction: uses the realized vol that has already happened, not
    the forward realization the premium is named for.
    """
    return vix - realized_vol_points(close, window)


def vix_zscore(vix: pd.Series, window: int = 63) -> pd.Series:
    """How stretched the VIX is against its own recent history."""
    return cast(pd.Series, zscore(vix, window))


def vix_beta_residual(close: pd.Series, vix: pd.Series, window: int = 63) -> pd.Series:
    """The part of today's VIX move the SPY move does *not* explain (log-VIX units).

    Rolling OLS of dlog(VIX) on SPY returns over ``window`` bars; the residual is
    today's dlog(VIX) minus beta * today's return. Positive = hedgers paid up more
    than the tape justifies; negative = a sell-off nobody bothered to hedge.
    """
    r = cast(pd.Series, returns(close))
    dv = cast(pd.Series, log_returns(vix))
    cov = r.rolling(window).cov(dv)
    var = r.rolling(window).var()
    beta = (cov / var).replace([np.inf, -np.inf], np.nan)
    return dv - beta * r


def residual_zscore(close: pd.Series, vix: pd.Series, window: int = 63) -> pd.Series:
    """``vix_beta_residual`` scaled by its own rolling std (mean ~0 by construction)."""
    resid = vix_beta_residual(close, vix, window)
    sd = resid.rolling(window).std()
    return resid / sd


def surprise_rate(close: pd.Series, vix: pd.Series, window: int = 21) -> pd.Series:
    """Share of recent days whose |SPY move| beat the VIX-implied one-sigma move.

    Under the VIX's own pricing you'd expect ~32% (one sigma); the premium keeps
    the long-run rate nearer 20%. A climb toward 40%+ means the market is being
    surprised repeatedly — the regime is shifting under the insurance price.
    """
    r = cast(pd.Series, returns(close)).abs()
    sigma = implied_daily_move(vix).shift(1)  # yesterday's close priced today's move
    beat = (r > sigma).astype(float).where(r.notna() & sigma.notna())
    return beat.rolling(window).mean()


def vix_spy_correlation(close: pd.Series, vix: pd.Series, window: int = 21) -> pd.Series:
    """Rolling correlation of SPY returns and dlog(VIX); normally deeply negative."""
    r = cast(pd.Series, returns(close))
    dv = cast(pd.Series, log_returns(vix))
    return r.rolling(window).corr(dv)


def vol_of_vol_ratio(vvix: pd.Series, vix: pd.Series) -> pd.Series:
    """VVIX over VIX: how jumpy the fear gauge is relative to its level."""
    return vvix / vix


def put_demand_zscore(skew: pd.Series, window: int = 63) -> pd.Series:
    """Cboe SKEW z-scored: a surge = crash protection getting bid vs. upside bets."""
    return cast(pd.Series, zscore(skew, window))


# --- the three core rules as pure exposure functions -------------------------------
#
# Rules 1-3 are the ones the daily report reads live, so they live here (pure,
# bundled to Lambda) and the strategies in ``strategies.vix_overlay`` wrap them.


def exposure_term(
    vix: pd.Series,
    vix_long: pd.Series,
    *,
    out_above: float = 1.0,
    band: float = 0.05,
    confirm_bars: int = 1,
) -> pd.Series:
    """Rule 1. Out when VIX/VIX3M breaches ``out_above``; in again below ``out_above - band``."""
    from ehf.server.research.signals.dial import confirm, hysteresis_gate

    ratio = term_structure_ratio(vix, vix_long)
    if confirm_bars > 1:
        breached = confirm(ratio > out_above, confirm_bars)
        ratio = ratio.where(~((ratio > out_above) & ~breached), out_above)
    return hysteresis_gate(ratio, out_above=out_above, in_below=out_above - band)


def exposure_resid(
    close: pd.Series,
    vix: pd.Series,
    *,
    window: int = 63,
    warn_z: float = 2.0,
    cooldown: int = 10,
) -> pd.Series:
    """Rule 2. Out for ``cooldown`` bars whenever the unexplained VIX move exceeds ``warn_z``."""
    from ehf.server.research.signals.dial import cooldown_gate

    z = residual_zscore(close, vix, window)
    return cooldown_gate(z > warn_z, cooldown)


def exposure_vrp(
    vix: pd.Series,
    close: pd.Series,
    *,
    rv_window: int = 21,
    out_below: float = 0.0,
    full_above: float = 4.0,
) -> pd.Series:
    """Rule 3. VIX minus realized: below ``out_below`` -> 0, above ``full_above`` -> 1, else 0.5."""
    from ehf.server.research.signals.dial import three_state

    vrp = variance_risk_premium(vix, close, rv_window)
    return three_state(vrp, lo=out_below, hi=full_above, high_is_risk=False)


def exposure_vote(
    close: pd.Series,
    vix: pd.Series,
    vix_long: pd.Series,
    *,
    term: dict[str, float] | None = None,
    resid: dict[str, float] | None = None,
    vrp: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rules 1-3 side by side plus their mean, as columns ``term, resid, vrp, vote``."""
    parts = {
        "term": exposure_term(vix, vix_long, **(term or {})),  # type: ignore[arg-type]
        "resid": exposure_resid(close, vix, **(resid or {})),  # type: ignore[arg-type]
        "vrp": exposure_vrp(vix, close, **(vrp or {})),  # type: ignore[arg-type]
    }
    df = pd.DataFrame(parts)
    df["vote"] = df[["term", "resid", "vrp"]].mean(axis=1)
    return df
