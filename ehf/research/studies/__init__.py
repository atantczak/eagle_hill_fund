"""Studies — a *set* of walk-forward runs that answer one research question.

A study is thicker than a job and thinner than the harness: it names the
strategies, grids, dataset and walk-forward settings for one question, runs each
strategy through ``walk_forward`` -> ``run_gauntlet`` -> ``write_scorecard``, adds
whatever question-specific metrics matter, and writes one summary. Jobs in
``research.jobs`` only parse arguments and call these.

  vix_overlay   the nine VIX rules + vote vs. hold and the 200-day filter on SPY
  leverage      the 3x fund when the market turns bullish vs. the plain fund (8 pairs)
"""
