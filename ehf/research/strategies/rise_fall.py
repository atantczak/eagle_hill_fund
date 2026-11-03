"""Simple Rise/Fall — asymmetric, ATR-anchored, trend-filtered mean reversion.

Buy small dips inside an uptrend, take profit on a larger rise. Asymmetric by
design: a shallow entry trigger (~-0.5%) against a larger target (~+2%). The dip
threshold is ATR-anchored (scales with each name's volatility) and gated by a
long-trend filter. A thin declaration over shared signals + exit components.
"""

from __future__ import annotations

import pandas as pd

from ehf.research import signals
from ehf.research.backtest.panel import Panel
from ehf.research.strategies.base import Strategy, register
from ehf.research.strategies.exits import hold_until_target


@register("rise_fall")
class RiseFall(Strategy):
    defaults = {
        "drop_pct": 0.005,  # entry: a ~0.5% dip (floor on the ATR-anchored trigger)
        "rise_pct": 0.02,  # target: take profit at +2%
        "stop_pct": 0.05,  # protective stop
        "atr_window": 14,  # ATR lookback for the volatility anchor
        "atr_mult": 0.5,  # entry dip = max(drop_pct, atr_mult * ATR%)
        "trend_window": 200,  # only trade above this SMA (trend filter)
        "max_hold": 20,  # time-stop in bars
        "gross_exposure": 1.0,
    }

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        close = panel.close
        trend_up = close > signals.sma(close, self.p("trend_window"))

        atr_pct = signals.atr(panel.high, panel.low, close, self.p("atr_window")) / close
        dip_threshold = atr_pct.clip(lower=self.p("drop_pct") / self.p("atr_mult")) * self.p(
            "atr_mult"
        )
        one_bar_ret = signals.returns(close, 1)
        entries = (one_bar_ret <= -dip_threshold) & trend_up

        # Resolve entries into sustained holdings via the shared exit component.
        return hold_until_target(
            close,
            entries.astype(float),
            target_pct=self.p("rise_pct"),
            stop_pct=self.p("stop_pct"),
            max_hold=self.p("max_hold"),
            gate=trend_up,
        )
