# EHF docs

Documentation for the Eagle Hill Fund research platform.

## Read in this order

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** — the map of the whole machine: how data
   comes in, how it's normalized, how backtests run, where strategies live, how
   every run is stored and reproduced, and how the local store reaches AWS. Start here.
2. **[COMPLEXITY.md](COMPLEXITY.md)** — the complexity-measures module (entropy /
   memory / information) and its reproducible acceptance gauntlet.
3. **[REPO_AUDIT.md](REPO_AUDIT.md)** — the original point-in-time audit of the
   legacy code the platform wraps. Historical; not kept in sync with the current tree.

## Decisions

Architecture Decision Records live in [`adr/`](adr) — point-in-time records of why
things are the way they are (not living docs; they capture the decision at the time):

- [0001 — Package layout](adr/0001-package-layout.md)
- [0002 — Results storage](adr/0002-results-storage.md)
- [0003 — Backtest engine build-vs-buy](adr/0003-backtest-engine-build-vs-buy.md)

> **Note on paths:** generated runtime artifacts (price cache, results DB/parquet)
> live under `ehf/server/var/` (`var/cache/`, `var/results/`). The ADRs and the
> audit predate that move and may still mention top-level `data/` or `results/`.
