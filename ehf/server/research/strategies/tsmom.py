"""Time-Series Momentum (TSMOM) — 12-month lookback, vol-scaled.

Go long assets with positive trailing return, short those with negative, sizing
each leg toward a target volatility (Moskowitz-Ooi-Pedersen). A thin declaration:
one momentum signal + the shared vol-target sizing sleeve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ehf.server.research import signals
from ehf.server.research.backtest.panel import Panel
from ehf.server.research.strategies import sizing
from ehf.server.research.strategies.base import Strategy, register


@register("tsmom")
class TSMOM(Strategy):
    defaults = {
        "lookback": 252,  # ~12 months of daily bars
        "target_vol": 0.10,  # annualized vol target per leg
        "vol_window": 63,  # ~3 months for the vol estimate
        "max_leverage": 1.0,
    }

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        trailing = signals.returns(panel.close, self.p("lookback"))
        return pd.DataFrame(
            np.sign(trailing.to_numpy()),
            index=trailing.index,
            columns=trailing.columns,
        ).fillna(0.0)

    def size_positions(self, signals_df: pd.DataFrame, panel: Panel) -> pd.DataFrame:
        return sizing.vol_target(
            signals_df,
            panel,
            target_vol=self.p("target_vol"),
            window=self.p("vol_window"),
            max_leverage=self.p("max_leverage"),
        )
