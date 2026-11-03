# ADR-0005 — Two execution targets, one job table

- **Status:** Accepted
- **Date:** 2026-11-01

## Context

`infra/` held only the AWS daily-report stack (Terraform + build + ops at the top
level). The Mac Studio — 28 cores, 96 GB, always on — is now the research server
and needs its own scheduling. Mixing launchd plists into a Terraform directory, or
hard-coding paths into plists, would not survive the next job.

## Decision

```
infra/aws/      Terraform, build_lambda.sh, ops.py      (moved; state key unchanged)
infra/studio/   jobs.py (the table) -> launchd/ (rendered, git-ignored)
ehf/research/jobs/   schedulable entrypoints + shared runtime
ehf/research/paths.py   every var/ path, once
```

- **Jobs are modules, infra only schedules them.** A job parses args, takes a
  lock, configures logging, calls a library function, writes a summary. Logic
  stays in the library (`data/refresh.py`, later `program/…`), so the same code
  runs from a notebook, a test, launchd or a Lambda.
- **One table.** `infra/studio/jobs.py` renders plists from `JOBS`; adding a job
  is one row. Plists are generated artifacts, never edited.
- **Runtime artifacts have one anchor.** `paths.py` replaces the per-module
  `Path(__file__).parents[2] / "var"` math in the cache and the store, and adds
  `universe/`, `log/`, `run/`, `scorecards/`.
- Terraform state key and resource names are unchanged by the move; the only
  diff on the next `apply` is the `Repo` tag.

## Consequences

- `python infra/studio/jobs.py install|status|run <job>` is the whole operator
  surface on the Studio; `python infra/aws/ops.py …` the whole surface for AWS.
- The committed `tfplan` binary was removed from git and ignored.
