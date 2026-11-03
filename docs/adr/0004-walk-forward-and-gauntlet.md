# ADR-0004 — Evaluation windows, the walk-forward harness, and the gauntlet

- **Status:** Accepted
- **Date:** 2026-11-01
- **Deciders:** Andrew (brief: "research every angle, find strategies I can run") / Claude Code

## Context

Every result in the registry was in-sample: `run_backtest` ignored `spec.start/end`
and scored whatever bars it was handed. There was no out-of-sample machinery and
no fixed definition of "good enough to trade", so nothing could run unattended.
The Studio can grind thousands of backtests a night; without a referee that only
produces more overfit curves faster.

## Decisions

### 1. The engine honors the spec's evaluation window
`run_backtest` computes weights on every bar passed in, then accounts P&L, costs,
trades and metrics only inside `[spec.start, spec.end]`. Bars before `start` are
signal warm-up. Inside the window the book starts **flat**: the opening position is
a real, cost-bearing trade (`_position_deltas` sets the first bar's delta to the
position itself). This makes `RunSpec` truthful and gives walk-forward test
windows lookback history without leakage. Behavioural change for full-range runs:
the very first entry is now charged slippage (previously free) — tiny, and more
honest. `Panel.slice` and `backtest/metrics.py` (`core_metrics`,
`equity_from_returns`) were extracted so the engine and the harness share one
accounting implementation.

### 2. Walk-forward = sweep + backtest, nothing new
`backtest/walkforward.py` adds *no* execution code. Each window's optimize step is
`run_sweep` on the train slice; validation is `run_backtest` on the test slice
with the in-sample winner's params. Windows tile non-overlapping test periods
(rolling or anchored train), the last one truncated at the final bar rather than
dropped. Everything is logged under deterministic ids derived from the inputs:

| what | sweep_id | status |
|---|---|---|
| window *i* optimize cells | `<wf_id>:w0i:train` | backtest |
| window *i* OOS run (parent = train winner hash) | `<wf_id>:oos` | backtest |
| same, slippage × m | `<wf_id>:oos:x<m>` | backtest |
| stitched OOS curve | `<wf_id>` | **walk_forward** |

The stitched run's `RunSpec.params` carries the grid, the base params and the
`WalkForwardSpec`, so `registry.get_run(hash)` regenerates the whole study. We
did **not** add fields to `RunSpec`: a new field changes the canonical JSON and
therefore every existing `config_hash`.

`make_sweep_id` keys only on strategy + grid, so two sweeps of the same grid over
different dates would share an id; the harness always passes explicit ids.

### 3. The gauntlet is configuration, not judgement
`analysis/gauntlet.py` turns a `WalkForwardResult` into a `Verdict` of named
`Check`s against a frozen `GauntletConfig`. Defaults (change deliberately, with a
reason, in one place):

| check | bar | why |
|---|---|---|
| windows | ≥ 3 | one OOS year is an anecdote |
| oos_sharpe | ≥ 0.3 | below this costs and slippage estimates dominate |
| psr | ≥ 0.90 | length/skew/kurtosis-aware confidence the OOS Sharpe > 0 |
| dsr | ≥ 0.95 | corrected for *every* train cell across all windows |
| efficiency | ≥ 0.5 | OOS / mean IS Sharpe — how much of the backtest survived |
| pbo | ≤ 0.30 | CSCV on the last train sweep |
| cost_stress | > 0 at 2× slippage | capacity / execution margin |
| trades | ≥ 100 OOS | statistics about trading, not luck |
| windows_pos | ≥ 50 % of test windows positive | not one lucky year |
| regimes | ≥ 60 % of trend regimes positive | breadth, or label it regime-specific |

A NaN never passes. The verdict renders to markdown (the scorecard) and JSON.

## Consequences

- The nightly research loop can pass/kill unattended, and a human reviewing a
  scorecard sees which bar was missed and by how much.
- `spec.start/end` now matter: a spec whose window lies outside the bars raises.
- Two sweeps per window per strategy is the cost of honesty; on the Studio a
  12-cell grid × 20 windows × 23 ETFs runs in well under a minute.
