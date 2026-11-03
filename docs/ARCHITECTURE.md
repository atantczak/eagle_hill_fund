# EHF Research Platform — Architecture & Walkthrough

This is the map of the whole machine: how data comes in, how it's normalized,
how backtests run, where strategies live, how every run is stored so it can be
reproduced, and how the local SQLite store relates to AWS. Read top to bottom
once; after that, jump to the section you need.

> **Status legend:** ✅ built & tested · 🟡 next increment · ⚪ planned

---

## 1. Philosophy (why it's shaped this way)

Four rules drive every design choice:

1. **DRY, ruthlessly.** A strategy is a *thin declaration of intent* (~30–80 lines).
   All real machinery — signals, sizing, exits, costs, accounting — is shared.
2. **Build the platform, not the strategies.** The engine, data interface, results
   store, and analysis suite are the load-bearing parts. The 63 tracker ideas
   become trivial once those are right.
3. **Never trust a backtest you can't reproduce.** Every run is persisted with its
   full config; any result regenerates from its hash alone.
4. **Costs and look-ahead bias are ON by default.** A zero-cost or future-peeking
   backtest is a *bug*, available only behind an explicit, labeled toggle.

---

## 2. The package map

One package, `ehf/`. The quant platform lives under `ehf/research/`; the website
that publishes its write-ups under `ehf/web/`; the legacy FMP tooling the data
layer wraps under `ehf/{integrations,tools}`. The platform reaches that legacy
tooling through exactly one adapter (`research/data/fmp_provider.py`). Docs,
notebooks, scripts and generated output sit beside the package at the repo root.

```
ehf/
  research/                     ← THE QUANT PLATFORM
    config/      ✅ typed, hashable run specs (RunSpec, CostConfig, FillModel, UniverseSpec)
    data/        ✅ DataProvider + FMPProvider + canonical schema + parquet cache + quality
                    + named universes (ETF baskets, us_liquid, point-in-time sp500) + nightly refresh
    results/     ✅ ExperimentRegistry + LocalResultStore (SQLite + parquet); Postgres stub
    signals/     ✅ pure vectorized primitives (sma, ema, bollinger, atr, zscore, hurst, ...)
    strategies/  ✅ Strategy ABC + thin subclasses (rise_fall, tsmom, ..., the vix_* dials); sizing/exits
    backtest/    ✅ cost-aware look-ahead-safe engine + fill models + parallel sweep + walk-forward
    complexity/  ✅ measures (entropy/memory/information) + reference systems
    analysis/    ✅ robustness (PSR/DSR/PBO/Monte-Carlo) + regime + tearsheets + the gauntlet
                    + dial metrics (time in market, switches, crisis drawdowns) + scorecard writer
    studies/     ✅ one research question = one module (strategies + grids + dataset -> scorecards)
    overlays/    ⚪ vol-target / regime-gate / HRP wrappers (Strategy -> Strategy)
    execution/   ⚪ Alpaca paper/live behind one interface (seam only)
    reporting/   ✅ daily market report (screens + Claude narrative), deployed on Lambda
    jobs/        ✅ schedulable entrypoints (refresh_data, ...) + shared job runtime
    paths.py     ✅ the one anchor for every var/ path
  web/           ✅ eaglehillfund.com: FastAPI + Jinja over the reports bucket, Cognito at the ALB
    lab/         ✅ backtest browser, playground, any-ticker charts (lightweight-charts; local-first, ADR-0008)
  integrations/  Financial Modeling Prep client (the data source research/data wraps)
  tools/         legacy utilities the FMP client depends on (parallelization, HTTP client, portfolio)
  tests/         ✅ unit tests, mirror research/ and web/ (no network)
docs/            this file, the ADRs, the research programme
notebooks/       research playgrounds
scripts/         report builders (build_vix_*.py), benchmarks, one-off tools
var/             (git-ignored) generated runtime artifacts (paths in research/paths.py):
  cache/         local parquet price cache (var/cache/fmp/<tf>/<adj>/<SYM>.parquet)
  results/       SQLite results DB + per-run parquet (var/results/ehf.db, <hash>/*.parquet)
  universe/      daily point-in-time universe snapshots (<name>/<date>.parquet)
  log/  run/     job logs, lock files, last-run summaries
  scorecards/    walk-forward + gauntlet scorecards
  reports/       built PDFs, published to the site with infra/aws/site/ops.py
infra/
  aws/             Terraform + build + ops for everything that runs in AWS (daily report Lambda, the site)
  studio/          the Mac Studio as research server: launchd jobs from one table (jobs.py)
```

Import paths follow the tree, e.g. `from ehf.research.backtest.engine import run_backtest`.
The dependency arrow points **one way**: `strategies → signals → data`,
`strategies → backtest → results`. Only `…/research/data/fmp_provider.py` knows the
vendor is FMP, and it's the sole bridge from `research` into `server/{tools,integrations}`.

---

## 3. How we pull data  ✅

### 3.1 The one rule: everything is canonical

`ehf/research/data/schema.py` defines the **canonical EHF bars schema** — the single,
vendor-agnostic format that *all* EHF data takes. Backtests and analysis are
written against this schema only. **Switching data vendors = writing one new
adapter that emits this schema. Nothing downstream changes.** This is exactly
the normalization guarantee you asked for.

Canonical bars (long format, one row per `(symbol, bar)`):

| column     | dtype                | notes                              |
|------------|----------------------|------------------------------------|
| `symbol`   | string               | uppercase ticker                   |
| `timestamp`| datetime64[ns, **UTC**] | tz-aware; vendors' local time is converted |
| `open`     | float64              | split/dividend **adjusted** by default |
| `high`     | float64              |                                    |
| `low`      | float64              |                                    |
| `close`    | float64              |                                    |
| `volume`   | float64              |                                    |

Plus two metadata flags on `df.attrs`: `adjusted` (bool) and `timeframe`.
`validate_bars()` enforces every invariant (exact columns, UTC tz, dtypes,
sorted, deduped, `high >= low`, `volume >= 0`). Any provider's output must pass
it — that's the contract.

### 3.2 The provider interface

`ehf/research/data/provider.py` — `DataProvider` ABC. The whole platform codes against
this, never against FMP:

```python
provider.get_bars(symbols, start, end, timeframe="1day", adjusted=True) -> canonical bars
provider.get_universe(name, as_of) -> Universe   # point-in-time membership
provider.get_fundamentals(symbols, as_of)        # optional; feature-detected
```

### 3.3 The FMP adapter (the only vendor-aware code)

`ehf/research/data/fmp_provider.py` — `FMPProvider` implements `DataProvider` by wrapping
the **existing, unmodified** `FinancialDataCompiler` (which handles the FMP API
key, rate limits, intraday chunking, and parallel fetch). It then funnels raw
FMP rows through `normalize_fmp_records()` → canonical bars.

```
symbols ─► FMPProvider.get_bars
              │  1. check parquet cache (per symbol/timeframe/adjustment)
              │  2. fetch only the uncached symbols via FinancialDataCompiler  (FMP)
              │  3. normalize_fmp_records()  ── maps FMP fields → canonical, US/Eastern → UTC
              │  4. write-through to cache
              ▼
        canonical bars  (validated)
```

To add a vendor later (e.g. Polygon, Tiingo): write `PolygonProvider(DataProvider)`
with its own `normalize_*` mapping into `normalize_bars(...)`. Done — backtests
don't notice.

### 3.4 The cache  ✅

`ehf/research/data/cache.py` — `BarCache`. Write-through parquet, one file per
`(vendor, symbol, timeframe, adjustment)` under `var/cache/`. Backtests read
from disk; the paid API is hit only for genuinely-uncached bars. Invalidation
strategy is documented in the module header (adjusted prices get a freshness TTL
on the recent tail; unadjusted history is immutable; `clear()` is always safe).

### 3.5 Data quality & survivorship  ✅

`ehf/research/data/quality.py` — `check_bars()` detects missing bars (vs the business-day
calendar), zero-volume, and duplicates, returning a `QualityReport` and emitting
`DataQualityWarning`. `warn_survivorship()` fires whenever a universe reflects
*today's* listings instead of point-in-time membership (FMP only gives current
membership) — so any Russell / cross-sectional / dislocation study **warns
loudly** rather than silently producing a biased number.

---

## 4. How backtests work  ✅ (walk-forward harness ✅)

`ehf/research/backtest/engine.py::run_backtest(spec, bars)` is the entry point. A strategy
emits a **target-weight panel**; the engine enforces the fill model, charges
costs, and returns a `RunResult` (equity curve, trade ledger, core metrics).

- **Strategy seam:** `generate_signals(panel) → size_positions(...) → manage_exits(...)`,
  composed by `Strategy.target_weights`.
- **Costs default-on** via `CostConfig` (commission, slippage as ½-spread bps,
  optional impact, short borrow). `frictionless=True` exists only as a labeled
  debug toggle — and is forced to zero everything else.
- **Look-ahead protection is structural:** the position held over the interval
  starting at bar *t* is `weights.shift(1)` — the most recent decision made
  *before* that interval. Fills default to *t+1* open (`FillModel.NEXT_OPEN`).
  `ehf/tests/test_engine.py::test_no_lookahead_next_open` **proves** a future price
  move can't change a prior bar's P&L; the `SAME_CLOSE` debug fill is shown
  leaking as the contrast.
- **Decision:** custom vectorized engine chosen over `vectorbt` — see
  `adr/0003` (Accepted). The `RunSpec → run_backtest → RunResult` seam is
  engine-agnostic, so another engine can slot in later.
- **Evaluation window:** the engine computes weights on every bar it is given
  but accounts P&L/costs/trades only inside `[spec.start, spec.end]`; bars before
  `start` are signal warm-up. The window starts flat, so the opening position is a
  real, cost-bearing trade. This is what lets walk-forward test windows carry
  lookback history without leaking.
- **Walk-forward harness** (`backtest/walkforward.py`) ✅ — `WalkForwardSpec`
  (train/test months, rolling or anchored, warm-up bars, objective, cost-stress
  multipliers) tiles non-overlapping test windows; each window's optimize step is
  a `run_sweep`, the winner is validated with `run_backtest` on the untouched test
  slice, and the stitched out-of-sample curve is logged as its own run
  (`status='walk_forward'`, `sweep_id=<wf_id>`). Every train cell, OOS run and
  2x-cost twin is in the registry under deterministic ids (`<wf_id>:w03:train`,
  `<wf_id>:oos`, `<wf_id>:oos:x2`). See ADR-0004.
- **The gauntlet** (`analysis/gauntlet.py`) ✅ — fixed kill criteria
  (`GauntletConfig`: OOS Sharpe, PSR, Deflated Sharpe vs. every trial, walk-forward
  efficiency, PBO, 2x-cost survival, trades, share of positive windows, regime
  breadth) turn a walk-forward result into a `Verdict` with one row per check.
- **Accounting refinement:** wiring the engine to drive the salvaged legacy
  `PortfolioTool` for share-level accounting is a later step; today's ledger is
  weight-delta based.

---

## 5. Where & how strategies are built  ✅

`ehf/research/strategies/` — one `Strategy` ABC; everything else is a thin subclass that
*declares intent* and composes `ehf.research.signals` primitives + shared
`sizing`/`exits` components. The engine consumes only `target_weights`.

```python
@register("rise_fall")
class RiseFall(Strategy):
    defaults = {"drop_pct": 0.005, "rise_pct": 0.02, "trend_window": 200, ...}
    def generate_signals(self, panel):
        trend_up = panel.close > signals.sma(panel.close, self.p("trend_window"))
        atr_pct  = signals.atr(panel.high, panel.low, panel.close, self.p("atr_window")) / panel.close
        entries  = (signals.returns(panel.close, 1) <= -atr_pct*self.p("atr_mult")) & trend_up
        return hold_until_target(panel.close, entries.astype(float),       # shared exit component
                                 target_pct=self.p("rise_pct"), stop_pct=self.p("stop_pct"),
                                 max_hold=self.p("max_hold"), gate=trend_up)
```

Both reference strategies are live and thin: **Simple Rise/Fall** (`rise_fall.py`,
~45 LOC) and **Time-Series Momentum** (`tsmom.py`, ~35 LOC, vol-scaled via the
shared `sizing.vol_target` sleeve). Strategies self-register under their `RunSpec`
key, so `run_backtest(RunSpec(strategy="tsmom", ...), bars)` just works.

**Dial strategies** (`dial.py`, `vix_overlay.py`) answer "should I be in the
market?": one asset (SPY) against a synthetic T-bill `CASH` leg, with an exposure
dial in [0, 1] produced from the VIX family (`data/vix.py` builds the dataset;
`signals/vix.py` + `signals/dial.py` hold the pure rules and state machines). Nine
rules + a vote, plus `dial_hold` and `dial_sma` (200-day filter) as the honest
benchmarks. Because the rules are pure signal functions, the daily report computes
the same dial live (`reporting/market.dial_context`). The plain-English version is
the two write-ups in the site library (https://eaglehillfund.com/library): the explainer
(`scripts/build_vix_explainer.py`) and the results report (`scripts/build_vix_results_report.py`,
from the scorecards and stored equity curves). PDFs are published with
`infra/aws/site/ops.py publish-report`, never committed (ADR-0006); the study is `studies/vix_overlay.py`.

Sizing (`sizing.py`: `normalize_weights`, `vol_target`) and exits (`exits.py`:
`hold_until_target`) are **shared, pluggable components** — picked from a menu,
never copy-pasted. The §10 "conviction" ideas will use a separate lightweight
`Screen` base (screens + event-study + thesis registry) on a slower cadence. 🟡

---

## 6. Where & how everything is stored (reproducibility)  ✅

This is the part that makes the platform trustworthy. **Every run is persisted
with enough to regenerate it from its hash alone.**

### 6.1 The flow

```
RunSpec  ──.config_hash──►  ExperimentRegistry.log_run(spec, result)
   │                              │
   │ (typed, frozen, pydantic)    ├─► SQLite  var/results/ehf.db   (queryable metadata + metrics)
   │                              └─► parquet var/results/<hash>/equity.parquet, trades.parquet
   ▼
config_hash = sha256(canonical_json(spec))[:16]   ← the unit of reproducibility
```

- **`RunSpec`** (`ehf/research/config/spec.py`) fully describes a backtest: strategy,
  params, universe, date range, timeframe, capital, **cost model**, **fill
  model**, optional `sweep_id`. It's frozen and hashable; identical spec →
  identical `config_hash`.
- **`ExperimentRegistry`** (`ehf/research/results/registry.py`) is the *only* public API.
  It computes the hash, captures **git SHA + dirty flag** (code identity is
  metadata, deliberately *not* part of the hash), enforces **idempotency**
  (same hash → skip, or `versioned=True` to keep history), and writes through.
- **`LocalResultStore`** (`ehf/research/results/store.py`) is the default backend.

### 6.2 What's in SQLite vs parquet

| Where | What | Why |
|---|---|---|
| **SQLite** `var/results/ehf.db` | `runs` (hash PK, strategy, params, universe, date range, git SHA, cost/fill model, status, sweep_id, parent_hash, spec_json, timestamp), `metrics` (long form), `sweeps`, `theses` | relational + queryable; tiny |
| **parquet** `var/results/<hash>/` | `equity.parquet` (equity curve), `trades.parquet` (trade ledger) | big arrays; columnar, compressed, cheap |

Raw price data is **never** stored per run — only references + derived results.

### 6.3 Querying across runs (the whole point)

You don't run-tweak-rerun-and-lose-history. You ask questions across all runs:

```python
reg = ExperimentRegistry()
# "all Rise/Fall runs, best deflated-Sharpe first"
reg.query("WHERE strategy = ? ORDER BY m_deflated_sharpe DESC", ("rise_fall",))
# every walk-forward window of a sweep
reg.query("WHERE sweep_id = ?", ("sweep_2026_06_21_a",))
# every killed strategy and why
reg.query("WHERE status = 'killed'")
# regenerate any run from its hash — no rerun needed
run = reg.get_run("a1b2c3d4e5f6a7b8")   # -> spec, metrics, equity curve, trade ledger
reg.compare([h1, h2, h3])               # side-by-side metrics table
```

Metrics are pivoted into `m_<name>` columns automatically, so you can sort/filter
on any metric. **Parameter sweeps are first-class:** every cell carries its
`sweep_id`; explore the set with one `WHERE`.

---

## 7. How SQL works locally and how it reaches AWS  ✅ local / ⚪ sync

**Locally** the store is plain SQLite in **WAL mode** — zero-latency, offline,
free SQL over your full run history. No server, no network, no cost per write.
This fits a single researcher running thousands of sweep cells on one machine.
Heavy arrays sit beside the DB as parquet.

**AWS** is for durability and the future cloud-resident execution/reporting:

```
LOCAL (research)                         AWS (durability + live, later)
─────────────────                        ──────────────────────────────
var/results/ehf.db   (SQLite WAL) ─ sync ►  s3://<bucket>/ehf-results/   (parquet + db snapshot)
var/results/<hash>/*.parquet     (explicit registry.sync(),    │
var/cache/*.parquet               never on the write path)      ▼
                                                       Postgres/RDS  ⟵ promotion target for
                                                       live/paper run logging (multi-writer,
                                                       always-on). Same ResultStore interface,
                                                       so no strategy/harness code changes.
```

- **Why local-first SQLite, not RDS now:** the workload is single-writer, bursty
  (a 5,000-cell sweep would be 5,000 network round-trips to Postgres), and
  interactive. SQLite wins on latency, cost, and offline use. (Full reasoning:
  `adr/0002`.)
- **The seam to Postgres exists today:** `ResultStore` is an ABC;
  `PostgresResultStore` is a typed stub. When live/paper logging needs an
  always-on multi-writer store, it drops in behind `ExperimentRegistry` — the
  registry API and every strategy stay untouched.
- **S3 sync** reuses the (to-be-fixed) legacy `s3Instance`; it's an explicit
  `registry.sync()`, so backtests never block on the network.

Secrets (FMP key, AWS, Alpaca) come from environment variables locally
(`.env`, git-ignored — see `.env.example`) and from **AWS Secrets Manager** for
deployed components. Nothing is hardcoded or committed.

---

## 8. Dev workflow & tooling  ✅

The `ehf` package targets **Python 3.11+** (the legacy package stays on 3.9 and
is quarantined — they meet only at the FMP adapter). One-time setup:

```bash
python3.11 -m venv .venv311
.venv311/bin/pip install -e ".[dev]"
.venv311/bin/pre-commit install        # ruff + mypy + hygiene on every commit
```

Day-to-day gates (all currently green):

```bash
.venv311/bin/pytest -q          # tests
.venv311/bin/ruff check ehf    # lint  (tests live under ehf/tests)
.venv311/bin/ruff format ehf   # format
.venv311/bin/mypy               # strict type-check on ehf/
```

`pyproject.toml` is the single source of truth (deps, ruff, mypy-strict, pytest).
Enforcement is via **pre-commit hooks**, not GitHub Actions — by choice, for a
solo local workflow. CI can be added later if it earns its keep; the hooks give
the same checks at commit time.

---

## 9. Quickstart — prove the spine

```python
from ehf.research.config import RunSpec, UniverseSpec
from ehf.research.results import ExperimentRegistry, RunResult
import pandas as pd

reg = ExperimentRegistry()                       # -> var/results/ehf.db (SQLite) + parquet
spec = RunSpec(strategy="demo",
               universe=UniverseSpec(symbols=("AAPL", "MSFT")),
               start="2024-01-01", end="2024-06-30")

result = RunResult(
    metrics={"sharpe": 1.4, "max_drawdown": -0.07},
    equity=pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, tz="UTC"),
                         "equity": [1e6, 1.01e6, 1.02e6]}),
    trades=pd.DataFrame({"symbol": ["AAPL"], "qty": [10.0], "price": [100.0]}),
)
h = reg.log_run(spec, result)                    # reproducible, idempotent
print(reg.query("ORDER BY m_sharpe DESC"))       # cross-run query
print(reg.get_run(h)["metrics"])                 # regenerate from hash
```

Running a real backtest through the engine + logging it:

```python
from ehf.research.config.spec import RunSpec, UniverseSpec, CostConfig
from ehf.research.backtest.engine import run_backtest
from ehf.research.results.registry import ExperimentRegistry

spec = RunSpec(strategy="tsmom", params={"lookback": 252, "vol_window": 63},
               universe=UniverseSpec(symbols=("SPY", "QQQ", "IWM")),
               start="2015-01-01", end="2024-12-31",
               costs=CostConfig(slippage_bps=1.0))   # costs ON by default
result = run_backtest(spec, bars)                     # -> RunResult(metrics, equity, trades)
ExperimentRegistry().log_run(spec, result)            # reproducible + queryable
```

Pulling real data (needs `FMP_API_KEY` in `.env`):

```python
from ehf.research.data import Timeframe
from ehf.research.data.fmp_provider import FMPProvider
bars = FMPProvider().get_bars(["AAPL"], "2024-01-01", "2024-03-31", Timeframe.DAY)
# -> canonical, validated, cached to var/cache/fmp/1day/adj/AAPL.parquet
```

---

## 10. What's next

Built and tested: the data spine, the results spine, signals, the Strategy
framework (5 strategies), the engine, parallel sweeps, the analysis suite, the
**walk-forward harness and gauntlet**, named universes with point-in-time S&P 500,
the nightly data refresh, the daily market report (AWS), and the Studio scheduler.

The standing programme — what gets researched, in what order, under which kill
criteria, and how the Studio runs it nightly — lives in
[`RESEARCH_PROGRAM.md`](RESEARCH_PROGRAM.md). Next increments, in order:

1. **Research batch job** (`jobs/research_batch.py`): a hypothesis backlog
   (`theses` table) → walk-forward → gauntlet → scorecard markdown under
   `var/scorecards/` → thesis marked promoted/killed → morning digest.
2. **Overlays** (`overlays/`): vol-target, regime gate, HRP as Strategy → Strategy
   wrappers, so every survivor can be risk-managed the same way.
3. **Portfolio of survivors** + the Alpaca execution seam (`execution/`) for paper.
4. One FMP client (fold the report's light client and the legacy compiler into a
   single transport behind `FMPProvider`).
