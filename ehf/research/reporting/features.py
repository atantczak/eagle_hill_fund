"""Pure price features per symbol from a long frame of daily bars.

Input: ``symbol, date, open, high, low, close, volume`` (one row per bar).
Output: one row per symbol, indexed by symbol. Everything is computed on the
wide close/volume panels with the shared ``signals`` primitives, so the numbers
here are the same ones the backtest engine would see.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ehf.research.signals import core as sig

FEATURE_COLUMNS = [
    "last_date",
    "n_bars",
    "close",
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "ret_252d",
    "mom_12_1",
    "dvol_60",
    "vol_60",
    "z_1d",
    "rsi_14",
    "sma_50",
    "sma_200",
    "above_50",
    "above_200",
    "high_252",
    "low_252",
    "pct_from_high_252",
    "pct_from_low_252",
    "new_high_252",
    "new_low_252",
    "dd_from_high_63",
    "vol_ratio_20",
    "dollar_vol_20",
    "gap_1d",
    "golden_cross_5d",
    "death_cross_5d",
]


def _wide(bars: pd.DataFrame, col: str) -> pd.DataFrame:
    return bars.pivot(index="date", columns="symbol", values=col).sort_index()


def _last(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[-1]


def price_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute the per-symbol feature table. Pure; tolerant of short histories (NaN)."""
    if bars.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS).rename_axis("symbol")

    close = _wide(bars, "close")
    open_ = _wide(bars, "open").reindex_like(close)
    volume = _wide(bars, "volume").reindex_like(close)

    rets = cast(pd.DataFrame, sig.returns(close, 1))
    dvol_60 = rets.rolling(60).std()
    sma50 = cast(pd.DataFrame, sig.sma(close, 50))
    sma200 = cast(pd.DataFrame, sig.sma(close, 200))
    rsi14 = cast(pd.DataFrame, sig.rsi(close, 14))

    # 52-week levels on closes (a 'new high' means a closing high, not an intraday wick)
    prior_high_252 = close.shift(1).rolling(252, min_periods=120).max()
    prior_low_252 = close.shift(1).rolling(252, min_periods=120).min()
    high_252 = close.rolling(252, min_periods=120).max()
    low_252 = close.rolling(252, min_periods=120).min()
    high_63 = close.rolling(63, min_periods=40).max()
    vol_avg_20 = volume.shift(1).rolling(20).mean()
    dollar_vol_20 = (close * volume).rolling(20).mean()

    def ret_n(n: int) -> pd.Series:
        return _last(cast(pd.DataFrame, sig.returns(close, n)))

    cross_up = (sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))
    cross_dn = (sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))

    last_close = _last(close)
    out = pd.DataFrame(
        {
            "last_date": close.apply(lambda s: s.last_valid_index()),
            "n_bars": close.notna().sum(),
            "close": last_close,
            "ret_1d": ret_n(1),
            "ret_5d": ret_n(5),
            "ret_21d": ret_n(21),
            "ret_63d": ret_n(63),
            "ret_126d": ret_n(126),
            "ret_252d": ret_n(252),
            # 12-1 momentum: trailing year excluding the most recent month.
            "mom_12_1": (close.shift(21) / close.shift(252) - 1.0).iloc[-1],
            "dvol_60": _last(dvol_60),
            "vol_60": _last(dvol_60) * np.sqrt(252.0),
            "z_1d": (_last(rets) / _last(dvol_60.shift(1))),
            "rsi_14": _last(rsi14),
            "sma_50": _last(sma50),
            "sma_200": _last(sma200),
            "above_50": last_close > _last(sma50),
            "above_200": last_close > _last(sma200),
            "high_252": _last(high_252),
            "low_252": _last(low_252),
            "pct_from_high_252": last_close / _last(high_252) - 1.0,
            "pct_from_low_252": last_close / _last(low_252) - 1.0,
            "new_high_252": _last(close) >= _last(prior_high_252),
            "new_low_252": _last(close) <= _last(prior_low_252),
            "dd_from_high_63": last_close / _last(high_63) - 1.0,
            "vol_ratio_20": _last(volume) / _last(vol_avg_20),
            "dollar_vol_20": _last(dollar_vol_20),
            "gap_1d": _last(open_) / close.shift(1).iloc[-1] - 1.0,
            "golden_cross_5d": cross_up.tail(5).any(),
            "death_cross_5d": cross_dn.tail(5).any(),
        }
    )
    out.index.name = "symbol"
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in (
        "above_50",
        "above_200",
        "new_high_252",
        "new_low_252",
        "golden_cross_5d",
        "death_cross_5d",
    ):
        out[col] = out[col].fillna(False).astype(bool)
    return out[FEATURE_COLUMNS]


def reaction_since(bars: pd.DataFrame, symbol: str, event_date: pd.Timestamp) -> float | None:
    """Return from the last close *before* ``event_date`` to the latest close (pure).

    Used for post-earnings drift: an after-close report on day D is reacted to on
    D+1, a pre-market report on D is reacted to on D itself — anchoring on the
    close strictly before D covers both.
    """
    s = (
        bars.loc[bars["symbol"] == symbol, ["date", "close"]]
        .set_index("date")["close"]
        .sort_index()
    )
    before = s.loc[s.index < event_date]
    if before.empty or s.empty:
        return None
    base = float(before.iloc[-1])
    return float(s.iloc[-1]) / base - 1.0 if base else None
