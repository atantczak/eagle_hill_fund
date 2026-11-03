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
- [0004 — Walk-forward and the gauntlet](adr/0004-walk-forward-and-gauntlet.md)
- [0005 — Infra and jobs layout](adr/0005-infra-and-jobs-layout.md)
- [0006 — Reports library on the site](adr/0006-reports-library-on-the-site.md)
- [0007 — Flat package layout](adr/0007-flat-package-layout.md)
- [0008 — The research lab in the web app, local first](adr/0008-research-lab-in-the-web-app.md)
- [0009 — The deployed lab reads a read-only S3 mirror of the Studio's var/](adr/0009-lab-deployed-over-an-s3-mirror.md)
- [0010 — The playbook page and the ideas box](adr/0010-playbook-and-ideas-box.md)

> **Note on paths:** the package was flattened on 2026-11-03 (ADR-0007): there is no
> `ehf/server/` or `ehf/client/` any more, and `docs/`, `notebooks/`, `scripts/` and
> `var/` sit at the repository root. ADR-0001 to 0005 and the audit predate that and
> still name the old paths.
