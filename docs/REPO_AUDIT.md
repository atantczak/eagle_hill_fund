# EHF Research Platform — Repo Audit (Phase 0)

**Date:** 2026-06-21
**Branch:** `dev`
**Scope:** Audit the `eagle_hill_fund` repo prior to building the research/trading platform described in the Build Brief. No feature code written yet.

---

## 1. TL;DR

- **The data tooling exists, works, and is clean.** It's a Financial Modeling Prep (FMP) wrapper reachable via `FinancialDataCompiler.compile_price_data(...)`. This is the thing the brief tells me to *wrap, not replace*. It has no `payfusion` cross-repo dependencies. ✅
- **A strategy framework already exists** under `server/tools/financial/strategies/` — but it is the **wrong shape** for the brief: an event-loop backtester fused to a 1,578-line `PortfolioTool`, with no cost model, no look-ahead guard, no walk-forward, no results store, and DRY violations (indicator math baked into the base class; `generate_signals` duplicated as `_generate_single_symbol_signals`). **Salvage the ideas and the portfolio accounting; rebuild the framework.**
- **Most AWS integration tools are non-functional in this repo.** `rds`, `s3`, `ddb`, `secrets_manager`, `ses`, `ssm`, `ecs`, `event_bridge`, `iac`, `cloudwatch`, `textract` all `import from payfusion.server.payfusion.apps...` — a *sibling codebase that is not present here*. They were copy-pasted and will `ImportError` on import. The patterns are good references but none run as-is. This directly shapes the storage decision (see ADR-0002).
- **No tests, no CI, no linting, no type-checking, no `pyproject.toml`.** Two ~600 MB virtualenvs committed-adjacent (`venv/`, `.venv/`); only `venv/` is git-ignored. Python 3.9, `pandas==2.1.4`.
- **Secrets hygiene is OK.** `.env` is git-ignored and **not** in history. `access_keys.py` is tracked but only reads `os.getenv(...)` — no hardcoded secrets.

**Net:** the load-bearing data layer is solid and the portfolio math is reusable. Everything above the data layer (engine, signals lib, results store, analysis, walk-forward) is greenfield. That matches the brief's "build the platform, not the strategies."

---

## 2. What exists

### 2.1 Repo shape
```
eagle_hill_fund/                      # package root (single top-level package per egg-info)
  client/                             # Django + React demo site (out of scope for research platform)
  server/
    integrations/
      financial/financial_modeling_prep/   # ← THE DATA VENDOR (FMP). CLEAN. WRAP THIS.
        tool.py                       # FinancialModelingPrepTool: stock list, screener, exchanges/sectors/industries
        price_data/tool.py            # FMPPriceDataTool: get_daily_prices, get_intraday_prices, get_stock_price
        financials/tool.py            # FMPFinancialsTool (55 lines — fundamentals accessor)
      aws/                            # rds, s3, ddb, secrets_manager, ses, ssm, ecs, ... ← BROKEN (payfusion imports)
      artifical_intelligence/llms/    # OpenAI/LangChain wrappers
      medical_research/, social_media/ # arxiv/pubmed/twitter/etc. (out of scope)
    tools/
      data/
        transmission/api/tool.py      # APIClient (requests wrapper, used by FMP) — CLEAN
        parallelization/tool.py       # ParallelizationTool (hardware-aware thread pools) — CLEAN, reusable
      financial/
        data/tool.py                  # FinancialDataCompiler ← THE DATA ENTRY POINT. CLEAN. WRAP THIS.
        portfolio/tool.py             # PortfolioTool (1,578 lines) — accounting engine. SALVAGE.
        strategies/                   # BaseStrategyTool + mean_reversion/trend impls. REBUILD shape, salvage ideas.
        notebooks/                    # strat_ex_run.ipynb, price_data_sandbox.ipynb, portfolio_sandbox.ipynb
```

### 2.2 The data layer (KEEP — this is what I wrap)

**Entry point:** `eagle_hill_fund/server/tools/financial/data/tool.py` → `FinancialDataCompiler`

```python
FinancialDataCompiler().compile_price_data(
    symbols: List[str],
    from_date: str,   # "YYYY-MM-DD"
    to_date: str,     # "YYYY-MM-DD"
    interval: str = "1min",   # also "1hour", "1day"
) -> pd.DataFrame    # long format, one row per (symbol, bar)
```

- Subclasses `FinancialModelingPrepTool` (which holds the FMP API key from `FMP_API_KEY`) and composes `ParallelizationTool`.
- Handles FMP-specific concerns: 3,000 calls/min rate limit with an 0.8 safety factor, intraday chunking (FMP caps 1-min data at ~1,170 minutes ≈ 3 trading days per call), and parallel per-symbol fetch.
- Daily prices come from FMP `/historical-price-eod/full`; intraday from `/historical-chart/{interval}`.
- **Returned columns** (FMP EOD full): `symbol, date, open, high, low, close, volume, change, changePercent, vwap` — note this is the exact schema the old `BaseStrategyTool.set_data()` validates against.

**Strengths:** clean separation (vendor tool → compiler → caller), rate-limit aware, parallel, no broken imports.

**Gaps the wrapper must close (per brief's Data Layer section):**
- **No caching** — every backtest re-hits the paid API. Brief requires a write-through parquet cache keyed by symbol/timeframe/date-range/adjustment.
- **No point-in-time universe.** FMP `screen_companies` / `get_stock_list` return *today's* listings → **survivorship bias**. There is no historical index membership. The wrapper must `get_universe(name, as_of)` and **WARN loudly** when only current membership is available.
- **Adjustment semantics are implicit.** `get_daily_prices` has no `adjusted` flag; intraday has `nonadjusted`. The `DataProvider.get_bars(..., adjusted=True)` contract must make this explicit and default to adjusted.
- **No missing-bar detection / data-quality flags.**
- **No typed return.** Returns a raw long DataFrame with vendor column names; the `DataProvider` boundary should normalize to a documented OHLCV schema so nothing downstream knows it's FMP.

### 2.3 The portfolio engine (SALVAGE)

`portfolio/tool.py` → `PortfolioTool` (1,578 lines). Real double-entry-ish accounting: `Trade` / `CompletedTrade` / `Position` / `PortfolioSnapshot` dataclasses, `execute_trade`, FIFO-ish position updates, snapshots, `calculate_returns` (with benchmark), trade metrics, performance attribution, VaR / expected shortfall, risk metrics, state save/load. **This is reusable as the portfolio-accounting core under the new engine** — the new engine should drive it rather than reimplement P&L. Caveats: commission-only cost model (no slippage/borrow), and some methods assume the event-loop driver.

### 2.4 The existing strategy framework (REBUILD shape, salvage ideas)

`strategies/tool.py` → `BaseStrategyTool(ABC)` + `StrategyConfig`, `Signal` dataclasses. Concrete: `BollingerBandMeanReversionTool` (mean reversion), `DipBuyingTool` (trend). READMEs sketch 6+ strategy families.

**Why it can't be the brief's `Strategy` base:**
1. **DRY violations.** Indicator math (`sma`, `ema`, MACD, RSI, Bollinger, ATR, momentum, vol) is hardcoded inside `BaseStrategyTool._calculate_basic_indicators` and partially recomputed in each subclass. The brief mandates a **pure, vectorized, tested `ehf.signals` library** that strategies *compose*. Signal logic is also duplicated between `generate_signals(date)` and `_generate_single_symbol_signals(...)`.
2. **No cost/look-ahead discipline.** Costs are a flat per-trade commission; there is no slippage, no borrow, no `t+1` fill model, and nothing prevents a signal at bar *t* from reading bar *t*. The brief makes costs + look-ahead protection **on by default**.
3. **Event-loop coupling.** `run_backtest()` iterates dates and mutates `PortfolioTool` in place — fine as one engine option, but the framework needs a clean `generate_signals → size → exits` seam and a results store, neither of which exists.
4. **No persistence/registry/walk-forward/analysis.** Results are an in-memory dict + ad-hoc JSON export. No config hash, no git SHA, no queryable history.

**What to salvage:** the *strategy intent* (BB mean-reversion thresholds, dip-buy logic, ATR/vol filters), the `StrategyConfig` notion of typed config, and the `Signal` metadata/reason pattern. Re-express them as thin subclasses on the new base.

### 2.5 Reusable utilities (KEEP)
- `ParallelizationTool` — hardware-aware (`psutil`) thread pools, rate-limited map, retry map. Useful for cache-warming and sweep fan-out. Clean.
- `APIClient` — thin `requests` wrapper with `to_dataframe`. Clean. (Note: `make_api_request` swallows errors by returning a dict instead of raising — the data wrapper should treat non-`Response` returns as failures.)

### 2.6 `ehf_finance` (deprecated)
Referenced in the brief as deprecated. **It is not present in this repo** (no files, no imports). Treated as an external/archived codebase. If specific logic from it needs porting, point me at it and I'll evaluate; nothing here depends on it.

---

## 3. What's broken / risky

| Item | Severity | Detail | Action |
|---|---|---|---|
| AWS tools import `payfusion.*` | High (for infra) | `rds/s3/ddb/secrets_manager/ses/ssm/ecs/event_bridge/iac/cloudwatch/textract` all import a sibling package not in this repo → `ImportError`. | Don't depend on them. New platform gets its own thin AWS access via the local `credentials/access_keys.py` pattern (env-driven) + `boto3` directly. Port `RDS`/`s3Instance` deliberately only if/when needed, fixing the credentials import. |
| No tests / CI / lint / types | High (for brief) | Brief mandates pytest, ruff, mypy, pre-commit, GitHub Actions. None exist. No `pyproject.toml`. | Lay down tooling on Phase 1 kickoff (configs drafted, see §5). |
| Two committed-adjacent venvs (~1.2 GB) | Medium | `venv/` ignored; `.venv/` **not** ignored. | Add `.venv/` to `.gitignore`; standardize on one. |
| `pandas==2.1.4`, Python 3.9 | Medium | 3.9 is near EOL; some modern typing (`X | Y`) unavailable. Duplicate pins in `requirements.txt`. | Recommend Python 3.11+ for the `ehf` package; pin deps in `pyproject`. Decide in ADR-0001. |
| `APIClient.make_api_request` returns error dicts instead of raising | Medium | Silent failures become empty DataFrames downstream → a backtest on missing data looks "fine." | Data wrapper must detect and raise/flag. |
| Commission-only cost model | Medium | No slippage/borrow → optimistic backtests. | New cost model is mandatory + default-on (ADR-0003). |
| `access_keys.py` tracked | Low | Tracked but only reads env vars — no secret leak. | Leave; ensure all new secret access is env / AWS Secrets Manager. |

---

## 4. Dependencies (current)

`requirements.txt` (deduped): `boto3, botocore, Django, Jupyter, numpy, openpyxl, pandas==2.1.4, python-dotenv, requests, requests-mock, langchain, langchain-experimental, langchain-openai, lxml, networkx, openai, plotly, Requests==2.31.0, scipy, tweepy, -e .`

**For the research platform we'll add (Phase 1+):** `pyarrow` (parquet cache + results), `pydantic` (typed config/run specs), `pytest`, `ruff`, `mypy`, `pre-commit`. Engine/analysis may add `vectorbt` (pending ADR-0003), `statsmodels` (Markov-switching, regressions), `matplotlib`/`quantstats` or a custom tearsheet renderer, `alpaca-py` (execution, later). Django/social/medical/LLM deps are unrelated to the platform and can be isolated via optional extras.

---

## 5. Tooling plan (ready to lay down on green-light)

Not created yet (the layout is under your review), but planned for Phase 1 kickoff:
- `pyproject.toml` — single source of truth: package metadata, deps, and config for **ruff** (lint+format), **mypy** (strict on `ehf/`), **pytest** (with coverage), targeting Python 3.11+.
- `.pre-commit-config.yaml` — ruff, ruff-format, mypy, trailing-whitespace, end-of-file, no-commit-to-`main`.
- `.github/workflows/ci.yml` — matrix on Python 3.11/3.12: ruff → mypy → pytest. (Repo currently has **no** Actions despite the brief noting an existing pattern; this establishes it.)
- `.gitignore` — add `.venv/`, `data/cache/`, `results/` artifacts, `*.parquet` scratch.
- `.env.example` — **created now** (see repo root) documenting required env vars.

---

## 6. Confirmation: the data tooling I will wrap

I have read it end-to-end and understand it:
- **Vendor:** Financial Modeling Prep (FMP), key via `FMP_API_KEY`.
- **Entry point I'll wrap:** `FinancialDataCompiler.compile_price_data(symbols, from_date, to_date, interval)` → long OHLCV DataFrame.
- **Underlying accessors:** `FMPPriceDataTool.get_daily_prices` / `get_intraday_prices` / `get_stock_price`; `FinancialModelingPrepTool.screen_companies` / `get_stock_list` / `get_available_{exchanges,sectors,industries}`; `FMPFinancialsTool` for fundamentals.
- **Plan:** a `DataProvider` Protocol/ABC with an `FMPProvider` adapter that delegates to `FinancialDataCompiler`, normalizes the column schema, defaults to adjusted prices, adds a write-through parquet cache, and exposes `get_universe(name, as_of)` that **WARNs** on survivorship bias where point-in-time membership is unavailable. **The existing tooling is not modified or replaced — only wrapped.**

---

## 7. Proposed package layout & key decisions

See:
- **ADR-0001 — Package layout** (`docs/adr/0001-package-layout.md`)
- **ADR-0002 — Results storage** (`docs/adr/0002-results-storage.md`) ← *the decision you most want my judgment on*
- **ADR-0003 — Backtest engine build-vs-buy** (`docs/adr/0003-backtest-engine-build-vs-buy.md`) — stub, to be finalized after the Rise/Fall prototype.

**Stop point:** awaiting your review + green-light before Phase 1.
