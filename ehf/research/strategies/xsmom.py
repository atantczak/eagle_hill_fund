"""Cross-sectional momentum (XSMOM) — long winners, short losers, dollar-neutral.

The relative-value sibling of TSMOM. Where TSMOM asks "is *each* asset trending
up?" and trades it against its own zero line, XSMOM ranks the *cross-section*
each bar and holds the strongest names long against the weakest short. The bet is
on relative performance, so it is roughly market-neutral by construction.

A thin declaration: the shared ``returns`` view of price for the momentum score +
the shared ``cross_sectional_rank`` sizer. The formation window skips the most
recent bars — the standard "12-1" convention — to sidestep short-term reversal,
which otherwise contaminates the momentum signal.
"""

from __future__ import annotations

import pandas as pd

from ehf.research.backtest.panel import Panel
from ehf.research.strategies import sizing
from ehf.research.strategies.base import Strategy, register


@register("xsmom")
class CrossSectionalMomentum(Strategy):
    defaults = {
        "lookback": 252,  # formation window start (~12 months back)
        "skip": 21,  # skip the most recent ~1 month (short-term reversal)
        "quantile": 0.2,  # long the top 20% / short the bottom 20%
        "gross_exposure": 1.0,
    }

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        close = panel.close
        # Formation return from t-lookback to t-skip: price `skip` bars ago over
        # price `lookback` bars ago. Both legs are shifts of past closes, so the
        # score at bar t uses only data through t (look-ahead-safe).
        return close.shift(self.p("skip")) / close.shift(self.p("lookback")) - 1.0

    def size_positions(self, signals_df: pd.DataFrame, panel: Panel) -> pd.DataFrame:
        # Dollar-neutral long/short on the cross-sectional ranking of the score.
        return sizing.cross_sectional_rank(
            signals_df,
            quantile=self.p("quantile"),
            gross=self.p("gross_exposure"),
        )
