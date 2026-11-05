"""Portfolio benchmarks — the honest alternative every basket strategy must beat.

``ew_hold``: equal weight across every name in the universe that has a price on
the bar (names enter as they list). Pair it with ``rebalance_every`` (e.g. 21 for
monthly) so it does not trade daily on drift-free weights that merely change when
a name lists.
"""

from __future__ import annotations

import pandas as pd

from ehf.research.backtest.panel import Panel
from ehf.research.strategies.base import Strategy, register


@register("ew_hold")
class EqualWeightHold(Strategy):
    defaults = {"gross_exposure": 1.0, "exclude": ()}

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        live = panel.close.notna().astype(float)
        for sym in self.p("exclude") or ():
            if sym in live.columns:
                live[sym] = 0.0
        return live
