"""Pure, vectorized signal primitives — the DRY core.

Every function is:
  * **pure** — no state, no I/O; same input → same output;
  * **vectorized** — accepts a price Series OR a wide DataFrame (timestamp index,
    one column per symbol) and operates column-wise;
  * **causal** — uses only past/current data (rolling/ewm, never centered), so a
    value at bar *t* never depends on *t+1*. This is what keeps strategies
    look-ahead-safe by construction.

Strategies COMPOSE these; they never reimplement them.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

Frame = pd.Series | pd.DataFrame


def returns(prices: Frame, periods: int = 1) -> Frame:
    """Simple percentage returns over ``periods`` bars."""
    return prices.pct_change(periods)


def log_returns(prices: Frame, periods: int = 1) -> Frame:
    """Log returns over ``periods`` bars."""
    return cast(Frame, np.log(prices)).diff(periods)


def sma(prices: Frame, window: int) -> Frame:
    """Simple moving average."""
    return prices.rolling(window).mean()


def ema(prices: Frame, span: int) -> Frame:
    """Exponential moving average (span parameterization)."""
    return prices.ewm(span=span, adjust=False).mean()


def rolling_std(prices: Frame, window: int) -> Frame:
    """Rolling standard deviation."""
    return prices.rolling(window).std()


def zscore(prices: Frame, window: int) -> Frame:
    """Rolling z-score: (x - SMA) / rolling std. The mean-reversion workhorse."""
    mu = sma(prices, window)
    sd = rolling_std(prices, window)
    return (prices - mu) / sd


def bollinger(prices: Frame, window: int = 20, n_std: float = 2.0) -> tuple[Frame, Frame, Frame]:
    """Bollinger bands. Returns (middle, upper, lower)."""
    mid = sma(prices, window)
    sd = rolling_std(prices, window)
    return mid, mid + n_std * sd, mid - n_std * sd


def realized_vol(prices: Frame, window: int, periods_per_year: float = 252.0) -> Frame:
    """Annualized realized volatility of simple returns over a rolling window."""
    return cast(Frame, returns(prices).rolling(window).std() * np.sqrt(periods_per_year))


def atr(high: Frame, low: Frame, close: Frame, window: int = 14) -> Frame:
    """Average True Range (Wilder). Inputs are aligned high/low/close frames."""
    prev_close = close.shift(1)
    tr = _elementwise_max(high - low, (high - prev_close).abs(), (low - prev_close).abs())
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rsi(prices: Frame, window: int = 14) -> Frame:
    """Relative Strength Index (Wilder)."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def rolling_slope(prices: Frame, window: int) -> Frame:
    """Rolling OLS slope of price vs. time (per ``window``). Trend strength."""
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denom = float((x * x).sum())

    def _slope(y: np.ndarray) -> float:
        return float(np.dot(x, y - y.mean()) / denom)

    return prices.rolling(window).apply(_slope, raw=True)


def hurst(prices: Frame, window: int) -> Frame:
    """Rolling Hurst exponent via the structure function (lag-variance scaling).

    For log-prices, std of τ-lagged differences scales as τ^H, so H is the slope
    of log(std of differences) vs log(τ). H ~ 0.5 random walk; H > 0.5 trending;
    H < 0.5 mean-reverting. Expensive (rolling apply) — use on daily data.
    """
    max_lag = max(2, min(20, window // 2))
    lags = np.arange(2, max_lag)
    log_lags = np.log(lags)

    def _hurst(arr: np.ndarray) -> float:
        ts = np.log(arr)
        taus = []
        for lag in lags:
            diff = ts[lag:] - ts[:-lag]
            taus.append(diff.std())
        taus_arr = np.asarray(taus)
        if np.any(taus_arr <= 0):
            return np.nan
        slope, _ = np.polyfit(log_lags, np.log(taus_arr), 1)
        return float(slope)

    return prices.rolling(window).apply(_hurst, raw=True)


def _elementwise_max(*frames: Frame) -> Frame:
    """Elementwise max across aligned frames (works for Series and DataFrame)."""
    out = frames[0]
    for f in frames[1:]:
        out = cast(Frame, np.maximum(out, f))
    return out
