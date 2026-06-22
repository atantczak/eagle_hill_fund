"""Signal primitives: correctness, causality (no look-ahead), and properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ehf.server.research import signals


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=300, tz="UTC")
    data = {
        "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))),
        "B": 50 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 300))),
    }
    return pd.DataFrame(data, index=idx)


def test_sma_matches_pandas(prices: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(signals.sma(prices, 10), prices.rolling(10).mean())


def test_zscore_is_standardized(prices: pd.DataFrame) -> None:
    z = signals.zscore(prices, 20).dropna()
    # A z-score should be roughly mean-zero, unit-scale across time.
    assert abs(float(z["A"].mean())) < 0.5
    assert 0.5 < float(z["A"].std()) < 2.0


def test_returns_and_realized_vol_positive(prices: pd.DataFrame) -> None:
    assert signals.returns(prices).shape == prices.shape
    rv = signals.realized_vol(prices, 20).dropna()
    assert (rv >= 0).all().all()


def test_atr_positive(prices: pd.DataFrame) -> None:
    high, low, close = prices + 1, prices - 1, prices
    a = signals.atr(high, low, close, 14).dropna()
    assert (a > 0).all().all()


def test_rolling_slope_sign() -> None:
    up = pd.Series(np.arange(50, dtype=float))
    assert float(signals.rolling_slope(up, 10).dropna().iloc[-1]) > 0


def test_hurst_random_walk_near_half() -> None:
    rng = np.random.default_rng(0)
    rw = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1000))))
    h = float(signals.hurst(rw, 200).dropna().mean())
    assert 0.3 < h < 0.7  # loose: random walk ~ 0.5


def test_signals_are_causal(prices: pd.DataFrame) -> None:
    """A signal at bar t must not change if a FUTURE bar is altered."""
    k = 200
    base = signals.zscore(prices, 20)
    perturbed_prices = prices.copy()
    perturbed_prices.iloc[k + 1 :] *= 1.5  # change only the future
    perturbed = signals.zscore(perturbed_prices, 20)
    pd.testing.assert_series_equal(base["A"].iloc[: k + 1], perturbed["A"].iloc[: k + 1])
