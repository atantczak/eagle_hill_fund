# infra/studio — the Mac Studio as research server

Apple M3 Ultra, 28 cores (20P + 8E), 96 GB unified memory, 1 TB SSD, always on
(`pmset -c sleep 0`). The Studio runs the compute-heavy, local-data half of EHF:
nightly data refresh and (next) the research batch — sweeps, walk-forward, the
gauntlet. Benefit Cloud work on the same box is protected: every job runs under
`nice -n 10` with low-priority I/O.

## The job table

`jobs.py` is the single source of truth. One `Job` per scheduled task: the module
to run (`python -m ehf.research.jobs.<name>`), its arguments, and its
`StartCalendarInterval`. Everything else (venv path, working directory, logs,
environment, niceness) is derived.

```bash
python infra/studio/jobs.py install     # render plists + load them into launchd (user domain)
python infra/studio/jobs.py status      # loaded? last exit code? last run summary?
python infra/studio/jobs.py run refresh_data   # run one job now, foreground, same command line
python infra/studio/jobs.py uninstall
```

| Job | When (local) | What |
|---|---|---|
| `refresh_data` | daily 02:00 | snapshot `etf_core`, `us_liquid`, `sp500`, `vix_family` membership to `var/universe/`; bring every member's daily bars up to the last completed session |
| `refresh_hourly` | Sat 03:00 | hourly bars for the ETF set (intraday studies) |
| `vix_study` | Sun 04:00 | the VIX overlay study (`studies/vix_overlay`): nine rules + vote vs hold / 200dma through walk-forward + gauntlet; scorecards to `var/scorecards/` |

launchd re-fires a missed slot when the machine wakes, so a night asleep means a
late run, not a lost one. Jobs take a lock (`var/run/<job>.lock`) so overlapping
fires are impossible, and write `var/run/<job>.last.json` for `status` and for
the morning digest.

## Adding a job

1. Write the entrypoint under `ehf/research/jobs/` (parse args → lock →
   logging → call a library function → `write_summary`). Logic lives in the
   library, never in the job.
2. Add a `Job(...)` row to `JOBS` in `jobs.py`.
3. `python infra/studio/jobs.py install`.

## Where things land

```
var/
  cache/fmp/<tf>/adj/<SYM>.parquet    price cache (BarCache)
  universe/<name>/<YYYY-MM-DD>.parquet point-in-time membership, one file per night
  results/                            SQLite registry + per-run parquet
  scorecards/                         walk-forward + gauntlet scorecards
  log/<job>.log, <job>.launchd.log    rotating job log, launchd stdout/stderr
  run/<job>.lock, <job>.last.json     lock + last-run summary
```
