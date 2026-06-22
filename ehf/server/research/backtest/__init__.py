"""Backtest engine, cost/fill models, accounting, walk-forward harness.

Costs and look-ahead protection are ON by default (see ehf.server.research.config.CostConfig and
FillModel). The vectorized engine consumes a strategy's target-weight panel and
produces an equity curve, trade ledger, and core metrics.

Note: ``run_backtest`` lives in ``ehf.server.research.backtest.engine`` and is imported from
there (``from ehf.server.research.backtest.engine import run_backtest``) — this package
``__init__`` deliberately exports only ``Panel`` to avoid an import cycle
(engine -> strategies.base -> backtest.panel).
"""

from ehf.server.research.backtest.panel import Panel

__all__ = ["Panel"]
