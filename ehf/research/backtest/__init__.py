"""Backtest engine, cost/fill models, accounting, walk-forward harness.

Costs and look-ahead protection are ON by default (see ehf.research.config.CostConfig and
FillModel). The vectorized engine consumes a strategy's target-weight panel and
produces an equity curve, trade ledger, and core metrics.

Modules (import from them directly — this ``__init__`` exports only ``Panel`` to
avoid an import cycle: engine -> strategies.base -> backtest.panel):

  engine       ``run_backtest(spec, bars)``: one cost-aware, look-ahead-safe run,
               accounted only inside the spec's evaluation window (extra bars = warm-up)
  metrics      ``core_metrics`` / ``equity_from_returns`` shared by every run producer
  sweep        ``run_sweep``: parallel parameter grid, every cell logged
  walkforward  ``walk_forward``: rolling/anchored optimize -> validate windows, stitched
               OOS curve logged as its own run, cost-stress twins, trial Sharpes for DSR
"""

from ehf.research.backtest.panel import Panel

__all__ = ["Panel"]
