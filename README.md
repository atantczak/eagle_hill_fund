# The Eagle Hill Fund

This repository holds all code for The Eagle Hill Fund: the quantitative research
& trading platform, the infrastructure it runs on, and the website that publishes its write-ups.

![plot](docs/ehf_logo.png)

## Layout

One Python package, `ehf`, with the code; everything else at the top level is
docs, infra, or generated output.

```
ehf/
  research/      THE QUANT PLATFORM: data → signals → strategies → backtest → results → analysis
  web/           eaglehillfund.com: FastAPI over the reports bucket (daily reports + research library)
  integrations/  Financial Modeling Prep client that research/data wraps (legacy, kept on purpose)
  tools/         legacy utilities the FMP client depends on (parallelization, HTTP client)
  tests/         unit + property tests (no network)
docs/            architecture, ADRs, the research programme
infra/
  aws/           Terraform + build + ops for what runs in AWS: the daily report Lambda, the site (ECS)
  studio/        the Mac Studio as an always-on research server (launchd jobs, one table)
notebooks/       research playgrounds
scripts/         report builders, benchmarks, one-off tools
var/             (git-ignored) generated output: price cache, results, scorecards, built PDFs, logs
```

Rule of thumb: if it is imported, it lives under `ehf/`; if it is run, it lives
under `scripts/` or `infra/`; if it is generated, it lives under `var/`.

## Start here

**To understand how the whole machine loops together — data in, normalization,
backtests, strategies, the reproducible results store, and how it reaches AWS —
read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).** It's
the top-to-bottom map; everything else hangs off it.

| Doc | What it covers |
|-----|----------------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The full walkthrough — read this first |
| [`docs/RESEARCH_PROGRAM.md`](docs/RESEARCH_PROGRAM.md) | **The standing research programme**: agenda, kill criteria, nightly loop, conventions |
| [`docs/COMPLEXITY.md`](docs/COMPLEXITY.md) | Complexity measures & their acceptance gauntlet |
| [eaglehillfund.com/library](https://eaglehillfund.com/library) | The research library (login). Write-ups are PDFs built by `scripts/build_*.py` into `var/reports/` and published with `python infra/aws/site/ops.py publish-report ...`; never committed. Current: *The Fear Gauge and the Market* (explainer) and *The Fear Gauge, Put to the Test* (results). |
| [`infra/README.md`](infra/README.md) | Where things run: AWS (daily report) and the Studio (research jobs) |
| [`docs/REPO_AUDIT.md`](docs/REPO_AUDIT.md) | Original audit of the legacy code (point-in-time) |
| [`docs/adr/`](docs/adr) | Architecture Decision Records; 0007 is the current layout |

## Quickstart

```bash
python3.11 -m venv .venv311
.venv311/bin/pip install -e ".[dev]"
.venv311/bin/pre-commit install     # ruff + mypy + hygiene on every commit
.venv311/bin/pytest -q              # run the suite
```

Pulling real data needs `FMP_API_KEY` in `.env` (see `.env.example`). See the
Quickstart section of `ARCHITECTURE.md` for runnable snippets that prove the spine.
