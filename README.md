# The Eagle Hill Fund

This repository holds all code for The Eagle Hill Fund: the quantitative research
& trading platform, plus the supporting integrations and (eventually) the website.

![plot](ehf/client/images/ehf_logo.png)

## Layout

Everything lives in one Python package, `ehf`, split into two halves:

```
ehf/
  client/        front-end (Django/React — placeholder for now)
  server/
    research/      ← THE QUANT PLATFORM (data → signals → strategies → backtest → results → analysis)
    integrations/  external services we wrap: financial (FMP), aws, ...
    tools/         shared utilities (parallelization, HTTP client, portfolio)
    tests/         unit + property tests (no network)
    docs/          architecture, ADRs, the research programme
    notebooks/     research playgrounds
    scripts/       benchmarks & one-off tools
    var/           (git-ignored) generated runtime artifacts: cache, results, universes, logs
infra/
  aws/             Terraform + build + ops for what runs in AWS (the daily report Lambda)
  studio/          the Mac Studio as an always-on research server (launchd jobs, one table)
```

## Start here

**To understand how the whole machine loops together — data in, normalization,
backtests, strategies, the reproducible results store, and how it reaches AWS —
read [`ehf/server/docs/ARCHITECTURE.md`](ehf/server/docs/ARCHITECTURE.md).** It's
the top-to-bottom map; everything else hangs off it.

| Doc | What it covers |
|-----|----------------|
| [`docs/ARCHITECTURE.md`](ehf/server/docs/ARCHITECTURE.md) | The full walkthrough — read this first |
| [`docs/RESEARCH_PROGRAM.md`](ehf/server/docs/RESEARCH_PROGRAM.md) | **The standing research programme**: agenda, kill criteria, nightly loop, conventions |
| [`docs/COMPLEXITY.md`](ehf/server/docs/COMPLEXITY.md) | Complexity measures & their acceptance gauntlet |
| [`infra/README.md`](infra/README.md) | Where things run: AWS (daily report) and the Studio (research jobs) |
| [`docs/REPO_AUDIT.md`](ehf/server/docs/REPO_AUDIT.md) | Original audit of the legacy code (point-in-time) |
| [`docs/adr/`](ehf/server/docs/adr) | Architecture Decision Records (package layout, results storage, engine build-vs-buy) |

## Quickstart

```bash
python3.11 -m venv .venv311
.venv311/bin/pip install -e ".[dev]"
.venv311/bin/pre-commit install     # ruff + mypy + hygiene on every commit
.venv311/bin/pytest -q              # run the suite
```

Pulling real data needs `FMP_API_KEY` in `.env` (see `.env.example`). See the
Quickstart section of `ARCHITECTURE.md` for runnable snippets that prove the spine.
