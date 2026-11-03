# ADR-0003 — Backtest engine: build vs. buy

- **Status:** Accepted — custom vectorized engine (decided after the Rise/Fall + TSMOM prototype)
- **Date:** 2026-06-21
- **Deciders:** Andrew (review) / Claude Code (proposal)

## Context

The brief required prototyping the engine on Simple Rise/Fall, then committing:
**(a)** a thin custom vectorized engine vs. **(b)** adopting `vectorbt` with our
metrics/store bolted on. Andrew's lean: enough custom control that the results
store and walk-forward harness are first-class.

I built **(a)** — a target-weight vectorized engine (`ehf/backtest/engine.py`) —
and expressed both reference strategies on it: **Rise/Fall** (`rise_fall.py`,
~45 lines) and **TSMOM** (`tsmom.py`, ~35 lines). Both are thin declarations, so
the abstraction holds. I evaluated `vectorbt` against the hard requirements
rather than building a full second prototype, because (a) already satisfied every
non-negotiable cleanly (below).

## Decision

**Adopt the custom vectorized engine (a).** The engine consumes a strategy's
**target-weight panel** and produces equity curve + trade ledger + core metrics;
look-ahead safety, fill model, and costs are enforced in the engine, not the
strategy.

| Criterion | Weight | (a) Custom | (b) vectorbt |
|---|---|---|---|
| Look-ahead safety provable by test | high | **Yes** — `position = w.shift(1)`; `test_no_lookahead_next_open` proves a future price can't change prior P&L, and `SAME_CLOSE` leak is demonstrated as the contrast | Possible but harder to assert as a unit test through its signal API |
| Cost/fill fidelity (½-spread, borrow, t+1 open) | high | **Native** — `CostConfig` + `FillModel` applied directly in `_holding_returns`/`_costs` | Requires fitting our cost+fill model onto its order primitives |
| Results-store / walk-forward integration | high | **First-class** — engine returns `RunResult`; `ExperimentRegistry.log_run` is one call | Bolt-on; its stats objects ≠ our schema |
| Thin-strategy expression | high | **Yes** — Rise/Fall ~45 LOC, TSMOM ~35 LOC | Comparable |
| Sweep throughput | med | Good — `ParallelizationTool` fans out runs; vectorized per-run math | Faster per-run, but we don't yet need it |
| Dependency / maintenance surface | med | Small, owned code | Heavy dependency, version risk |
| Built-in metrics/plots | low | We build them (analysis suite) | Comes for free |

### The one real tension — path-dependent exits
Rise/Fall's asymmetric target/stop is path-dependent and does **not** vectorize
purely. Resolved by isolating it in a shared, reusable exit component
(`ehf/strategies/exits.py::hold_until_target`) that scans per symbol (O(T) per
column — exact and clear for daily data). The engine stays vectorized; only the
weight *construction* scans. This keeps strategies thin and the seam clean, and
is the main thing a pure-vectorbt approach would have fought.

## Consequences

- **+** Full control: every hard requirement (look-ahead test-provability, cost
  fidelity, registry integration) is met natively and covered by tests.
- **+** The `RunSpec → run_backtest → RunResult` seam is engine-agnostic, so
  vectorbt (or any engine) can be added later as an alternative behind the same
  interface without touching strategies or the registry.
- **−** We own the metrics/plotting (coming in `ehf/analysis/`) rather than
  getting them free. Acceptable — the brief specifies a bespoke analysis suite
  (deflated Sharpe, PBO, Monte Carlo, regime breakdowns) beyond vectorbt's stock
  output anyway.
- **−** Per-run speed is lower than vectorbt's compiled core. Not a constraint
  yet; **revisit vectorbt only if sweep throughput becomes a bottleneck**, adding
  it as a second engine behind the existing seam.

### Follow-ups
- The engine currently does cost-aware weight-based accounting. Wiring it to
  *drive the salvaged legacy `PortfolioTool`* for share-level trade accounting is
  a later refinement (the current trade ledger is weight-delta based).
- NEXT_OPEN models P&L open-to-open; an intrabar fill model can plug in later
  behind the same `FillModel` enum.
