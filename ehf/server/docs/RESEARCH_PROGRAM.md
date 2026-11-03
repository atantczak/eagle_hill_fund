# The EHF Research Programme

> **Mandate (Andrew, 2026-11-01):** research every angle and find investment
> strategies I can actually run, instead of buying random stocks. Run it on the
> Mac Studio, every day, with little effort from me.

This is the standing plan: what gets researched, in what order, under which kill
criteria, how the Studio runs it nightly, and where everything lives. It changes
only deliberately. `ARCHITECTURE.md` is the map of the machine; this is the map of
the work.

---

## 1. The loop

```
 backlog (theses)  ─►  walk-forward  ─►  gauntlet  ─►  scorecard  ─►  promoted / killed
      ▲                 (sweep per           (fixed          (markdown +         │
      │                  window, OOS          criteria)       json, by hash)     │
      │                  stitched)                                               ▼
 weekly 30-min review  ◄──────────────────  morning digest  ◄───────────  registry (SQLite)
```

* **Backlog**: a hypothesis = a thin strategy + a parameter grid + a universe +
  a timeframe. Lives in the registry's `theses` table; each row names the
  strategy key, the grid, the universe and a one-line rationale.
* **Walk-forward** (`backtest/walkforward.py`): rolling optimize → validate
  windows; every train cell, OOS run and 2×-cost twin logged.
* **Gauntlet** (`analysis/gauntlet.py`): ten fixed checks → `Verdict`.
* **Scorecard**: the verdict table, the window table, the stitched OOS metrics,
  under `var/scorecards/<wf_id>.md|json`.
* **Promotion path**: `killed` → stays as evidence · `promoted` → candidate for
  the portfolio of survivors → paper (Alpaca seam) → live, small.

Human effort: one 30-minute weekly review of the scorecards; add ideas to the
backlog; decide which promoted survivors move to paper.

## 2. Kill criteria (fixed in advance)

`GauntletConfig` defaults, all required (see ADR-0004 for the reasoning):

| check | bar |
|---|---|
| OOS windows | ≥ 3 |
| OOS Sharpe (stitched) | ≥ 0.3 |
| Probabilistic Sharpe | ≥ 0.90 |
| Deflated Sharpe vs. every trial | ≥ 0.95 |
| Walk-forward efficiency (OOS / mean IS Sharpe) | ≥ 0.5 |
| Probability of backtest overfitting | ≤ 0.30 |
| Sharpe at 2× slippage | > 0 |
| OOS trades | ≥ 100 |
| Test windows positive | ≥ 50 % |
| Trend regimes positive | ≥ 60 % |

Changing a bar is a commit with a reason, not a per-strategy exception.

## 3. The agenda (ordered by how much the data can be trusted)

### Tier A — ETF / index level (no per-instrument survivorship)
Universe `etf_asset_classes` / `etf_sectors` / `etf_core` (static, versioned).

1. Time-series momentum (`tsmom`) and regime-gated TSMOM — lookback × vol target.
2. Dual momentum / tactical asset allocation (absolute + relative, 10-month SMA
   filters) — new thin strategy.
3. Sector rotation (cross-sectional momentum on `etf_sectors`) — `xsmom`.
4. Risk parity / HRP weighting as an **overlay** on 1–3.
5. Index mean reversion (short-horizon, `rise_fall` / `bollinger_reversion`).
6. Overnight vs. intraday return split; seasonality (turn-of-month, day-of-week).
7. **VIX overlay dial on SPY** (`studies/vix_overlay`): nine fear-gauge rules +
   their vote as in/half/out dials vs. `CASH` (bill accrual), judged against hold
   and the 200-day filter on drawdown as much as Sharpe. Weekly on the Studio.

### Tier B — Liquid US single names (bias mitigated)
Universe `sp500` (true point-in-time) first, `us_liquid` (current, warned) second.

7. Cross-sectional momentum, 12-1 and 6-1, with and without short leg.
8. Low volatility / quality tilts from FMP fundamentals.
9. Short-term reversal (1-week) and 52-week breakouts.
10. Event studies for the three daily-report screens already storing hits:
    post-earnings drift, insider clusters, analyst revisions — scored by
    `reporting/evaluate.py` forward returns, then backtested as strategies.

### Tier C — Hourly bars (2023+, exploratory)
11. Intraday momentum / reversal, opening range, VWAP reversion on `etf_core`.
12. Hourly lead-lag transfer entropy (the study `COMPLEXITY.md` calls for).

### Tier D — Complexity module as regime gates
13. Permutation entropy / DFA / Hurst as on-off switches for Tier A survivors.

### Meta
14. Correlation of survivors' OOS returns; portfolio of survivors; capacity.

## 4. Data

| universe | kind | survivorship | source |
|---|---|---|---|
| `etf_asset_classes`, `etf_sectors`, `etf_core` | static | safe per instrument (basket chosen with hindsight — flagged) | `data/universes.py` |
| `sp500` | dynamic, **point-in-time** | safe (rebuilt from the constituent-change log) | FMP `sp500-constituent` + `historical-sp500-constituent` |
| `us_liquid` | dynamic, current | **biased** (warned); EHF's own daily snapshots start 2026-11-01 | FMP screener, floors in `US_LIQUID` |

The nightly `refresh_data` job snapshots every universe to
`var/universe/<name>/<date>.parquet` and brings daily bars current. FMP (Ultimate)
serves ~20 years of daily history per request (back-adjusted only) and ~2 years of
hourly; see `data/refresh.py`.

## 5. The nightly schedule (Studio, local time)

| when | job | what |
|---|---|---|
| 02:00 | `refresh_data` | universes snapshot + daily bars current |
| 02:30 (next) | `research_batch` | pop N theses → walk-forward → gauntlet → scorecards |
| Sun 04:00 | `vix_study` | the VIX overlay study: 12 dials × walk-forward → gauntlet → scorecards + summary |
| 06:30 (AWS) | daily market report | screens + narrative, emailed |
| Sat 03:00 | `refresh_hourly` | hourly bars for `etf_core` |

Everything on the Studio runs `nice -n 10`, under a lock, with rotating logs and
a `var/run/<job>.last.json` summary. Operator surface:
`python infra/studio/jobs.py install | status | run <job>`.

Measured throughput (2026-11-01, `scripts/bench_sweep.py`, 24 symbols × 15 y
daily): 42 ms per backtest cell in parallel across 28 cores — roughly 85,000
small cells per hour, or 1,000–1,500 full-universe cells per hour.

## 6. Conventions (design for the future)

* **Strategies are thin declarations** (30–80 lines) on shared signals, sizing
  and exits. New idea = new file in `strategies/`, registered under its key.
* **No new execution paths.** Walk-forward is sweep + backtest; the gauntlet is
  configuration over stored runs. Anything new must emit a `RunResult` through
  the `ExperimentRegistry` and be regenerable from its hash.
* **Jobs are thin, libraries are thick.** A job parses args, locks, logs, calls a
  library, writes a summary. Logic lives where a notebook and a test can reach it.
* **One anchor per concern:** `paths.py` for `var/`, `universes.py` for universe
  names, `GauntletConfig` for kill criteria, `infra/studio/jobs.py` for schedules.
* **Costs on, look-ahead off, by default.** Frictionless or same-close fills exist
  only as labelled debug toggles.
* **Never add fields to `RunSpec` casually** — it changes every config hash.
* **Report what the data says.** A killed thesis is a result; the scorecard stays.

## 7. Status log

* **2026-11-03** — Results write-up `docs/reports/VIX_SPY_Backtest_Results.pdf`
  (`scripts/build_vix_results_report.py`, reads `var/scorecards` + `var/results`,
  13 pages, same plain-English register as the explainer). Findings that only
  showed up with the equity curves in hand: `vix_term` was *out* for the whole
  Aug-2008..Jun-2009 window (0% DD there, but invested only 25% of 2009, which is
  most of its return gap to `vix_surprise`); `vix_surprise` was invested 91% of
  2009; calendar-year OOS: surprise beat SPY 6/19 years, term 4/19 — by design.
  Recommendation in the report: surprise as the dial, term as the confirmer,
  200-day line for slow grinds (next study).
* **2026-11-02** — VIX overlay study (`studies/vix_overlay`, 12 dials × 19 OOS
  windows 2008-08..2026-10, train 24m / test 12m, Sharpe and Calmar objectives;
  summaries `var/scorecards/vix_overlay_{sharpe,calmar}_2026-11-02.md`). Hold: OOS
  Sharpe 0.60, CAGR 10.1%, max DD −48%. Best dials: `vix_surprise` (0.80–0.84,
  CAGR ~10%, DD −21.5%, in the market 80%), `vix_term` (0.73–0.84, DD −24…−27%,
  2025 DD −5%), `vix_sized` (0.76–0.79, DD −14…−15% but CAGR 6.2%), `vix_vote`
  (0.74, DD −21.7%). All beat the 200-day filter (0.58, CAGR 6.1%). Every dial is
  *killed* by the gauntlet on `dsr`/`pbo`: DSR pools train cells across windows so
  even a 1-cell grid (hold) fails, and PBO is NaN below 2 cells — the bars were
  written for multi-asset sweeps and need a deliberate revisit for single-asset
  dials. Full-sample 2008 (defaults, in-sample): the 200-day filter held 2008 to
  −10% while the VIX rules only trimmed it to −25…−35%; the VIX rules win 2020 and
  2025. Obvious next thesis: term/surprise rule AND 200-day filter.
* **2026-11-01** — engine evaluation window, walk-forward harness, gauntlet,
  named universes (PIT S&P 500), nightly refresh job, Studio scheduler, infra
  split into `aws/` + `studio/`. First real study `wf_tsmom_d07fbfca`: TSMOM on
  `etf_asset_classes`, 2000–2026, 48/12-month rolling windows, 22 OOS windows,
  264 trials, 44 s on the Studio — **killed** (OOS Sharpe 0.12, efficiency 0.29,
  PBO 0.69, DSR 0, bull-only). Lesson: daily vol-targeting rebalances every bar
  (~5,800 trades/yr on 23 ETFs); the first backlog item is a rebalance-frequency /
  threshold overlay, then dual momentum. Scorecard: `var/scorecards/wf_tsmom_d07fbfca.md`.
