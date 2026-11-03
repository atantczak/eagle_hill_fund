# ADR-0001 — Package layout for the `ehf` research platform

- **Status:** Proposed (awaiting green-light)
- **Date:** 2026-06-21
- **Deciders:** Andrew (review) / Claude Code (proposal)

## Context

The brief proposes a package layout and says "adjust and justify in an ADR." The existing research code lives deep under `eagle_hill_fund/server/tools/financial/...`, mixed with an unrelated Django site, social/medical integrations, and broken AWS tools. The new platform needs a clean, importable home with a tight dependency surface, while still being able to call the existing (clean) FMP data tooling.

## Decision

Create a **new top-level package `ehf/`** at the repo root, sibling to the legacy `eagle_hill_fund/` package — not nested inside it. The legacy package keeps the data tooling we wrap; `ehf` imports it across the boundary but nothing legacy imports `ehf`.

```
ehf/
  config/        # typed run specs & config objects (pydantic): RunSpec, CostConfig, UniverseSpec, SweepSpec
  data/          # DataProvider protocol + FMPProvider adapter (wraps FinancialDataCompiler) + parquet cache + quality/universe
  signals/       # pure, vectorized, tested fns: sma, ema, bollinger, atr, zscore, rolling_slope, hurst, realized_vol, ...
  strategies/    # Strategy ABC + thin subclasses (rise_fall, tsmom, ...) ; Screen base for §10 conviction ideas
  backtest/      # engine(s), fill model, cost models, portfolio adapter (drives legacy PortfolioTool), walk-forward harness
  analysis/      # metrics suite, robustness (DSR/PBO/MC), regime, attribution, tearsheets, comparison
  results/       # ResultStore backend + ExperimentRegistry API (log_run/get_run/query/compare) ; see ADR-0002
  execution/     # ExecutionProvider interface + Alpaca paper/live (stub the interface now, impl later)
  reporting/     # daily/periodic reports, Notion/email sync (later phase)
  overlays/      # composition wrappers: vol-target, regime-gate, HRP allocator (take a Strategy -> Strategy)
tests/           # mirrors ehf/ ; unit + property tests for signals; look-ahead & cost regression tests
docs/adr/
data/cache/      # local parquet cache (git-ignored)
results/         # local SQLite + parquet results store (git-ignored)
```

### Changes vs. the brief's starting layout
1. **Added `ehf/overlays/`** as a first-class package. The brief lists overlays (vol-target, regime gate, HRP) as a hard requirement ("compose onto any strategy") but folded them under `strategies/`. They are *wrappers over* strategies, not strategies — separating them keeps the composition seam explicit and prevents circular intent.
2. **`Screen` base lives in `strategies/`**, not its own top-level package. The §10 conviction ideas (screens + event-study + thesis registry) share the data layer and results store but run on a slower cadence. They're a sibling base class, not a separate subsystem — co-locating keeps the "declare intent" story in one place. The event-study harness itself lives in `analysis/` (it's analysis machinery), and the thesis registry is a table in `results/`.
3. **`config/` uses pydantic models**, not bare dataclasses, so run specs validate at construction and serialize deterministically (needed for the config hash in ADR-0002).
4. **New top-level `ehf/` package** rather than extending `eagle_hill_fund.server.tools.financial`. Rationale: isolate the research platform's dependency surface and tooling (strict mypy, ruff, 3.11+) from the legacy Django/social/medical/broken-AWS code, which would otherwise pollute type-checking and CI. The legacy FMP tooling is reached via a single adapter in `ehf/data/`.

### Tooling (enforced from day one)
- **Python 3.11+** for `ehf` (legacy stays 3.9; they only meet at the data adapter, which uses plain stdlib typing). Justification: modern typing, `match`, better perf; 3.9 is near EOL. **(Confirmed by Andrew 2026-06-21: 3.11+ for `ehf`, isolate legacy.)**
- `pyproject.toml` as single source of truth (ruff, mypy strict-on-`ehf`, pytest+coverage, deps).
- `.pre-commit-config.yaml` (ruff, ruff-format, mypy, hygiene hooks).
- `.github/workflows/ci.yml` (ruff → mypy → pytest on 3.11/3.12).

## Consequences

- **+** Clean dependency boundary; the research platform can be type-checked strictly without fighting legacy code.
- **+** The "wrap, don't replace" rule is structurally enforced — only `ehf/data/` knows FMP exists.
- **+** Composition seam (overlays) and the screen/strategy split are explicit before any strategy is written.
- **−** Two package roots in one repo. Acceptable: the brief explicitly anticipates a possible future repo split; this is a clean seam for it.
- **−** Cross-version boundary (3.11 ehf ↔ 3.9 legacy) requires the data adapter to avoid 3.10+ typing in any code legacy might import. Mitigated: the adapter is small and one-directional.

## Resolved
- **Python 3.11+ for `ehf`, legacy quarantined at 3.9.** Confirmed by Andrew 2026-06-21.
