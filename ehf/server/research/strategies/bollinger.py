"""Long-term Bollinger-band mean reversion — big bets on extreme deviation.

Buy when price closes a long way below its long-run average (below the lower
Bollinger band), exit on reversion toward the mean (a take-profit), a protective
stop, or a time-out. "Long-term" = a long band window and wide bands, so entries
are rare, deep dislocations rather than day-to-day noise.

A thin declaration: the shared ``bollinger`` signal for entries + the shared
``hold_until_target`` exit component. No bespoke loop, no copied band math — the
only strategy-specific line is "enter on a close below the lower band."
"""

from __future__ import annotations

import pandas as pd

from ehf.server.research import signals
from ehf.server.research.backtest.panel import Panel
from ehf.server.research.strategies.base import Strategy, register
from ehf.server.research.strategies.exits import hold_until_target


@register("bollinger_reversion")
class BollingerReversion(Strategy):
    defaults = {
        "window": 100,  # long-run average length (~5 months of daily bars)
        "n_std": 2.0,  # band half-width in standard deviations
        "target_pct": 0.10,  # take profit at +10% from entry (mean-reversion bounce)
        "stop_pct": 0.10,  # stop at -10%: cap the falling-knife tail
        "max_hold": 60,  # time-stop (~one quarter) if neither target nor stop hits
        "gross_exposure": 1.0,  # consumed by the default normalize_weights sizer
    }

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        close = panel.close
        # Lower band = long-run mean minus n_std rolling stdevs. A close beneath
        # it is a statistically extreme deviation — our entry trigger.
        _mid, _upper, lower = signals.bollinger(close, self.p("window"), self.p("n_std"))
        entries = (close < lower).astype(float)

        # Resolve discrete entry triggers into sustained 0/1 holdings via the
        # shared exit component (target / stop / time-out). Sizing is left to the
        # base class default (gross-normalized equal weight across open names).
        return hold_until_target(
            close,
            entries,
            target_pct=self.p("target_pct"),
            stop_pct=self.p("stop_pct"),
            max_hold=self.p("max_hold"),
        )
