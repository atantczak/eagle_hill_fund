# ADR-0007 — Flatten the package: no `server/`, no `client/`

- **Status:** Accepted
- **Date:** 2026-11-03
- **Supersedes the layout half of:** [ADR-0001](0001-package-layout.md)

## Context

ADR-0001 split `ehf/` into `client/` and `server/` on the assumption that a
front-end would grow beside the research platform. Two years on, `client/` held a
Django to-do demo and a logo, and everything real sat three levels deep under
`ehf/server/research/`. The website (ADR-0006) made the split actively misleading:
a FastAPI app that serves HTML is "the client side" by instinct, yet it was filed
under `server/site`, next to `research/`, with docs, notebooks, scripts and `var/`
also tucked under `server/`.

## Decision

One package, one level:

```
ehf/
  research/      the quant platform (unchanged inside)
  web/           eaglehillfund.com (was server/site)
  integrations/  FMP client the data layer wraps (was server/integrations/financial)
  tools/         legacy utilities the FMP client needs (was server/tools)
  tests/
docs/  notebooks/  scripts/  var/     at the repository root
infra/                                unchanged
```

- `client/` is deleted. If a real front-end ever appears it gets its own top-level
  directory (`web/` or `app/`), not a placeholder in the package.
- The dead integrations (`aws`, `medical_research`, `social_media`,
  `artifical_intelligence`) and the stale top-level `requirements.txt` are deleted.
  `integrations/financial` and `tools/` stay only because
  `research/data/fmp_provider.py` still imports them.
- `var/` moves to the repo root. `research/paths.py` anchors it as
  `Path(__file__).parents[2] / "var"`; nothing else computes a path.
- Ruff and mypy now cover `ehf/web`, `scripts/` and `infra/` as well as
  `ehf/research`; notebooks are excluded as playgrounds; the two report builders
  skip the line-length rule for their prose strings.

## Consequences

- Imports shorten (`ehf.research.…`, `ehf.web.…`). The Lambda handler path and
  the site's uvicorn target change, so both stacks redeploy once: the daily-report
  stack via `terraform apply` (handler + zip), the site via the next merge to `main`.
- The repo reads top-down: code in `ehf/`, decisions in `docs/`, things that run in
  `infra/` and `scripts/`, things that are generated in `var/`.
- ADR-0001 through 0005 and `docs/REPO_AUDIT.md` still name the old paths; they are
  point-in-time records and are left as written.
