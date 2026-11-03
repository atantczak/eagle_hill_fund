# ADR-0009 — The deployed lab reads a read-only S3 mirror of the Studio's var/

- **Status:** Accepted
- **Date:** 2026-11-03
- **Amends:** ADR-0008 (the "local first" paragraph)

## Context

ADR-0008 shipped the lab locally only: the container copied `ehf/web` alone and
the playground needs the price cache and the results store on disk. Andrew wants
the lab on eaglehillfund.com, for the cheapest setup that is not a hack. The
options were a shared file system (EFS: ~$0.30/GB-month plus mount targets, and a
second copy of the data to keep in sync anyway), the cloud results backend
(RDS: ~$15/month idle, and the price cache still has to live somewhere), or S3
plus a scratch disk in the task.

## Decision

**S3 is the system of record for `var/`; the container mirrors it on demand.**

- The Studio copies `var/{results,scorecards,cache/fmp/1day,cache/cboe}` to
  `s3://ehf-results-<account>/var/` every night (`ehf.research.jobs.sync_var`,
  06:00 in `infra/studio/jobs.py`). One direction, never deletes, uploads a file
  when the bucket lacks it, has a different size, or is older than the local
  copy. The SQLite results database is checkpointed first so the one file is
  complete. First sync: 27,725 files, 828 MB, about $0.15 of PUTs and ~$0.02 a
  month to store.
- The task gets a scratch volume at `/var/ehf` (part of Fargate's free 20 GiB of
  ephemeral storage); `EHF_VAR_DIR` points the whole platform at it
  (`ehf.research.paths`). `ehf.web.lab.mirror` fills it: the database, the
  scorecards and the Cboe cache at startup, the daily bar cache in a background
  thread, and anything else (a run's parquet, a symbol's bars) on first touch.
  The database and the scorecards are re-checked against the bucket once an hour
  (one HEAD; a re-download only when the ETag moved), so overnight research is
  on the site by morning.
- **Read-only.** Playground runs on the site are never logged; the "log" box is
  hidden; the Studio is the only writer of record. The one upload the container
  makes is bars it had to fetch from FMP because neither disk nor bucket had
  them, so the next task does not fetch them again. The FMP key reaches the task
  from the existing `ehf/fmp` secret.
- The task grows from 0.25 vCPU / 512 MiB to 0.5 vCPU / 2 GiB (a playground run
  is a real backtest; a big universe loads hundreds of parquet files). About
  +$12/month. A free S3 gateway endpoint keeps the mirror traffic inside AWS.
- The image now carries the whole `ehf` package and the research dependencies
  (pandas, numpy, scipy, pyarrow, pydantic; not matplotlib or the report stack);
  the deploy workflow runs the lab and mirror tests with exactly those packages,
  and triggers on any change under `ehf/`.

## Consequences

- A cold task serves the library immediately and the lab within seconds; the
  first open of a run or a new symbol costs one S3 round trip. Nothing is lost
  when a task is replaced: it refills from the bucket.
- Two copies of `var/` exist (Studio, bucket); the Studio's is canonical. If the
  Studio is off for a night the site just shows yesterday's research.
- The results bucket also holds the Terraform state; the `var/` prefix is
  separate and the task role is scoped to it.
- Locally nothing changes: `var/` on disk, `EHF_LAB=auto`, logging allowed.
  Setting `EHF_VAR_S3_BUCKET` locally runs the mirror mode against the bucket
  into whatever `EHF_VAR_DIR` points at (handy for checking what the site sees).
