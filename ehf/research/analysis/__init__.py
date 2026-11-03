"""Analysis suite over stored runs: robustness metrics, regime, tearsheets.

Everything renders from the results store by ``config_hash`` — any historical
run is analyzed without rerunning it.

  metrics    pure stats: Probabilistic/Deflated Sharpe, PBO (CSCV), Monte Carlo
  runs       registry bridge: pull returns/sweep matrices, parameter sensitivity,
             one-call ``robustness_report``
  regime     performance conditioned on market trend/volatility state
  tearsheet  one-call ``tearsheet`` / ``comparison_tearsheet`` figures
  gauntlet   fixed-in-advance kill criteria over a walk-forward result -> ``Verdict``
"""

from ehf.research.analysis.gauntlet import (
    GauntletConfig,
    Verdict,
    market_proxy_returns,
    run_gauntlet,
)
from ehf.research.analysis.metrics import (
    deflated_sharpe_ratio,
    monte_carlo_paths,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from ehf.research.analysis.regime import regime_breakdown
from ehf.research.analysis.runs import (
    parameter_sensitivity,
    robustness_report,
    run_returns,
    sweep_return_matrix,
)
from ehf.research.analysis.tearsheet import comparison_tearsheet, tearsheet

__all__ = [
    "GauntletConfig",
    "Verdict",
    "run_gauntlet",
    "market_proxy_returns",
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "monte_carlo_paths",
    "regime_breakdown",
    "run_returns",
    "sweep_return_matrix",
    "parameter_sensitivity",
    "robustness_report",
    "tearsheet",
    "comparison_tearsheet",
]
