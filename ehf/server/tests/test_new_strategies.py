"""The three new strategies + their reusable building blocks.

Covers the DRY seams each one adds:
  * regime gate overlay (``overlays.regime``)         -> ``tsmom_regime``
  * pure reuse of bollinger + hold_until_target       -> ``bollinger_reversion``
  * cross-sectional rank sizer (``sizing``)           -> ``xsmom``

Plus an end-to-end check that all three build from the registry and produce a
full-length equity curve through the real engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ehf.server.research.backtest.engine import run_backtest
from ehf.server.research.backtest.panel import Panel
from ehf.server.research.config.spec import RunSpec, UniverseSpec
from ehf.server.research.overlays.regime import RegimeGateMixin, trending_regime
from ehf.server.research.strategies import sizing
from ehf.server.research.strategies.base import build_strategy
from ehf.server.research.strategies.tsmom import TSMOM
from ehf.server.tests.conftest import make_daily_bars


# --------------------------------------------------------------------------- #
# Fixtures: a trending series and a choppy (mean-reverting) series.            #
# --------------------------------------------------------------------------- #
def _trend_path(n: int, drift: float = 0.0012, vol: float = 0.015, seed: int = 1) -> list[float]:
    """Geometric Brownian motion with drift — a *noisy* trend (random-walk structure)."""
    rng = np.random.default_rng(seed)
    return list(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))))


def _revert_path(n: int, theta: float = 0.25, sigma: float = 0.02, seed: int = 2) -> list[float]:
    """Ornstein-Uhlenbeck process — a *noisy* mean-reverting series (trend-hostile)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] - theta * x[i - 1] + rng.normal(0, sigma)
    return list(100 * np.exp(x))


def _cycle_path(n: int, amp: float = 0.06, seed: int = 2) -> list[float]:
    """Smooth sine + tiny noise — LOW entropy but mean-reverting (the AND's stress test)."""
    rng = np.random.default_rng(seed)
    base = 100 * (1 + amp * np.sin(2 * np.pi * np.arange(n) / 20))
    return list(base * (1 + rng.normal(0, 0.002, n)))


def _spec(strategy: str, symbols: tuple[str, ...], **params) -> RunSpec:
    return RunSpec(
        strategy=strategy,
        params=params,
        universe=UniverseSpec(symbols=symbols),
        start="2020-01-01",
        end="2025-01-01",
    )


# --------------------------------------------------------------------------- #
# 1. Regime gate overlay + tsmom_regime                                       #
# --------------------------------------------------------------------------- #
def test_trending_regime_is_boolean_and_causal() -> None:
    close = pd.DataFrame({"A": _trend_path(600)}, index=pd.date_range("2020-01-01", periods=600))
    mask = trending_regime(close)

    assert pd.api.types.is_bool_dtype(mask["A"])
    assert mask.shape == close.shape

    # Causality: perturbing only the future must not change a past bar's gate.
    k = 400
    bumped = close.copy()
    bumped.iloc[k + 1 :] *= 1.5
    after = trending_regime(bumped)
    pd.testing.assert_series_equal(mask["A"].iloc[: k + 1], after["A"].iloc[: k + 1])


def test_gate_passes_trend_and_rejects_reversion_and_cycles() -> None:
    """Persistent (random-walk/trend) structure gates ON; mean-reversion and
    smooth cycles gate OFF — the latter is the AND's job (a cycle is low-entropy
    yet mean-reverting, so Hurst must veto what entropy alone would pass through)."""
    idx = pd.date_range("2020-01-01", periods=600)
    trend = trending_regime(pd.DataFrame({"A": _trend_path(600)}, index=idx))["A"].mean()
    revert = trending_regime(pd.DataFrame({"A": _revert_path(600)}, index=idx))["A"].mean()
    cycle = trending_regime(pd.DataFrame({"A": _cycle_path(600)}, index=idx))["A"].mean()

    assert trend > 0.10  # fires a meaningful fraction of the time on trend/RW
    assert trend > revert  # and more than on noisy mean-reversion
    assert revert < 0.10  # strong OU reversion is essentially always gated off
    assert cycle < 0.10  # smooth cycle (low PE) is vetoed by low Hurst


def test_regime_gate_only_removes_exposure() -> None:
    """Gated weights must be a subset of the inner strategy's: same or flat, never new."""
    bars = make_daily_bars({"A": _trend_path(600, seed=3), "B": _revert_path(600, seed=4)})
    panel = Panel.from_bars(bars)
    params = {"lookback": 100, "vol_window": 40, "entropy_window": 80, "hurst_window": 80}

    plain = TSMOM(**{k: v for k, v in params.items() if k in TSMOM.defaults})
    gated = build_strategy("tsmom_regime", params)

    w_plain = plain.target_weights(panel).reindex_like(gated_w := gated.target_weights(panel))
    # Wherever the gate keeps a position it must equal the plain weight; wherever
    # it drops one the weight is exactly zero. So |gated| <= |plain| everywhere.
    assert (gated_w.abs() <= w_plain.abs() + 1e-12).all().all()
    # And the gate actually bites somewhere (chop bars get zeroed).
    assert (gated_w.abs() < w_plain.abs() - 1e-12).any().any()


def test_regime_gate_mixin_is_first_in_mro() -> None:
    # Guards the wiring: manage_exits must resolve to the mix-in, not TSMOM/base.
    from ehf.server.research.strategies.regime_gated import RegimeGatedTSMOM

    assert RegimeGatedTSMOM.manage_exits is RegimeGateMixin.manage_exits


# --------------------------------------------------------------------------- #
# 2. Bollinger reversion                                                      #
# --------------------------------------------------------------------------- #
def test_bollinger_enters_below_band_and_exits_on_reversion() -> None:
    # V-shape: long flat run, a sharp dip well below the band, then recovery.
    path = [100.0] * 80 + [100 - i for i in range(1, 16)] + list(range(85, 130))
    bars = make_daily_bars({"A": [float(p) for p in path]})
    strat = build_strategy("bollinger_reversion", {"window": 40, "n_std": 1.5, "max_hold": 30})
    weights = strat.target_weights(Panel.from_bars(bars))

    held = weights["A"]
    assert held.max() > 0  # took a long at some point
    assert held.iloc[-1] == 0.0  # and exited (reverted/target) by the end


# --------------------------------------------------------------------------- #
# 3. Cross-sectional rank sizer + xsmom                                       #
# --------------------------------------------------------------------------- #
def test_cross_sectional_rank_is_dollar_neutral() -> None:
    # One bar, five symbols with strictly ordered scores.
    scores = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], columns=list("ABCDE"))
    w = sizing.cross_sectional_rank(scores, quantile=0.2, gross=1.0)

    assert w.loc[0, "E"] > 0  # top score -> long
    assert w.loc[0, "A"] < 0  # bottom score -> short
    assert abs(float(w.iloc[0].sum())) < 1e-9  # nets to zero (dollar-neutral)
    assert abs(float(w.iloc[0].abs().sum()) - 1.0) < 1e-9  # gross == 1.0


def test_cross_sectional_rank_flat_when_no_pair() -> None:
    # Two names can't form both a long and a short 20% leg -> sit out.
    scores = pd.DataFrame([[1.0, 2.0]], columns=list("AB"))
    w = sizing.cross_sectional_rank(scores, quantile=0.2, gross=1.0)
    assert (w == 0.0).all().all()


# --------------------------------------------------------------------------- #
# End-to-end: all three run through the registry + engine.                    #
# --------------------------------------------------------------------------- #
def test_new_strategies_run_through_engine() -> None:
    syms = tuple("ABCDE")
    paths = {s: _trend_path(300, drift=0.001 * (i - 2), seed=10 + i) for i, s in enumerate(syms)}
    bars = make_daily_bars(paths)

    cases = (
        ("tsmom_regime", {"lookback": 60, "vol_window": 20, "hurst_window": 50}),
        ("bollinger_reversion", {"window": 40, "max_hold": 20}),
        ("xsmom", {"lookback": 60, "skip": 5, "quantile": 0.4}),
    )
    for strategy, params in cases:
        res = run_backtest(_spec(strategy, syms, **params), bars)
        assert len(res.equity) == 300, strategy
        assert "sharpe" in res.metrics, strategy
        assert res.metrics["n_bars"] > 0, strategy
