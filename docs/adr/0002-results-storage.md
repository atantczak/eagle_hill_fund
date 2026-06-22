# ADR-0002 — Results storage backend

- **Status:** Accepted (Andrew confirmed 2026-06-21) — *the decision Andrew most wanted my judgment on*
- **Date:** 2026-06-21
- **Deciders:** Andrew (review) / Claude Code (proposal)

## Context

Every backtest, walk-forward window, and sweep cell must persist enough to **regenerate the run from its hash alone**, and the store must be **queryable across runs** ("all Rise/Fall runs sorted by deflated Sharpe", "every walk-forward window for strategy X", "show me every killed strategy and why"). Requirements from the brief:

- Persist per run: **config hash, full param set, data range + universe, code/git SHA, cost assumptions, full metrics suite, equity curve, trade ledger, timestamp**.
- **Queryable, not just storable.** Cross-run analytical queries are the whole point.
- An **`ExperimentRegistry`** API wraps the store: `log_run(spec, results)`, `get_run(hash)`, `query(filters)`, `compare(hashes)`. Nothing touches the raw store directly.
- **Parameter sweeps first-class:** a sweep id tags every cell; explore results as a set.
- **Idempotent + cheap:** identical config (same hash) is detected and skipped or explicitly versioned. Don't store raw price data per run — store references + derived results.

### Constraints discovered in the audit (ADR context)
- The repo *has* a Postgres/RDS helper (`integrations/aws/rds/tool.py`) **but it's broken here** — it imports `payfusion.*` (absent) and would need its credentials import fixed before it runs. There is no live, always-on RDS instance wired to this repo today.
- The S3 helper (`s3Instance`) is likewise broken (same `payfusion` import) but is a small, fixable port.
- The workload is **single-writer, local, research-time**: one researcher running backtests/sweeps on a powerful local machine (the codebase even special-cases an "M3 Ultra" tier). Writes are bursty (a sweep = thousands of cells), reads are analytical.

## Decision

**Local-first: SQLite for metadata/metrics/registry + parquet for big arrays, with optional S3 sync for durability.** Concretely:

- **SQLite** (`results/ehf.db`) holds the queryable relational core:
  - `runs` — `config_hash` (PK), `strategy`, `params_json`, `universe`, `data_start`, `data_end`, `timeframe`, `git_sha`, `dirty` (uncommitted-changes flag), `cost_model_json`, `fill_model`, `created_at`, `sweep_id` (nullable, indexed), `status` (idea/backtest/walkforward/paper/live/killed), `kill_reason`, `parent_hash` (for walk-forward windows / versioned re-runs).
  - `metrics` — `(config_hash, name, value)` long form, indexed on `(name)` and `(config_hash)`, so "sort all Rise/Fall runs by `deflated_sharpe`" is one join. (A denormalized wide `metrics_wide` view can be materialized for convenience.)
  - `sweeps` — `sweep_id` (PK), `strategy`, `grid_json`, `created_at`, `n_cells`.
  - `theses` — the §10 conviction thesis registry (thesis id, symbol, hypothesis, entry conditions, event refs, status), sharing the same DB.
- **Parquet** under `results/<config_hash>/` holds the heavy per-run arrays that don't belong in a relational DB: `equity.parquet` (equity curve / returns), `trades.parquet` (trade ledger), optional `positions.parquet`. Columnar, compressed, cheap; referenced from `runs` by hash, never duplicating raw price data.
- **`ExperimentRegistry`** is the only public API. It computes the config hash, writes SQLite + parquet atomically, enforces idempotency, and exposes `log_run / get_run / query / compare`. The SQLite/parquet split is an implementation detail behind it.
- **Idempotency:** `config_hash = sha256(canonical_json(RunSpec))`. `log_run` does `INSERT ... ON CONFLICT(config_hash)`: by default **skip** (return the existing run) unless `versioned=True`, which stores a new row with `parent_hash` set and a salted hash. The git SHA + `dirty` flag are recorded but *excluded* from the config hash (code identity is metadata, not part of "same experiment") — with a loud warning if `dirty` when promoting toward paper/live.
- **Optional S3 sync** (via a fixed port of `s3Instance`): `results/` mirrors to `s3://<bucket>/ehf-results/` for durability, sharing, and to feed the future AWS-resident execution/reporting components. Sync is explicit (`registry.sync()`), not on the write path, so backtests never block on network.

## Alternatives considered

1. **Postgres/RDS as the primary store.** *Rejected for the research store.* It's an always-on paid instance with per-write network latency; the existing helper is currently broken here; and it's overkill for a single-writer, bursty, local workload (a 5,000-cell sweep = 5,000 network round-trips). **But it's the right promotion target for *live/paper* run logging** (multi-writer, always-on, shared dashboards) — the `ResultStore` interface is written so a `PostgresResultStore` can back it later without touching the registry API or any strategy code.
2. **S3 + Athena.** *Rejected.* Great for scale we don't have; adds query latency, cost, and ceremony for interactive research. Re-evaluate only if the run history outgrows local SQLite (millions of rows) or goes multi-user.
3. **MLflow / W&B.** *Rejected.* Heavy, opinionated, and the cross-run *quant* queries we need (deflated-Sharpe sorts, walk-forward window sets, parameter-sensitivity grids) are awkward in their schemas. A plain SQL table we own is simpler and more queryable for this use case.
4. **Everything in SQLite (equity curves as BLOBs).** *Rejected.* Bloats the DB, kills `VACUUM`/backup ergonomics, and parquet is strictly better for the array payloads. Keep SQLite lean and relational.

## Consequences

- **+** Zero-latency, offline, free interactive queries over full run history; trivial reproducibility (`get_run(hash)` rehydrates the `RunSpec`).
- **+** Sweeps are first-class (`sweep_id` column + `sweeps` table); "explore a sweep as a set" is a `WHERE sweep_id = ?`.
- **+** Cheap: no raw price per run; parquet compression on the arrays; SQLite file is tiny.
- **+** Clean promotion path to Postgres for live without API churn.
- **−** Single-writer by design — concurrent sweep processes need either one writer process fanning out, or SQLite WAL mode + short write txns (we'll use WAL; the parallel layer fans writes through one registry instance).
- **−** S3 durability is opt-in, so a lost laptop loses unsynced runs. Mitigated by making `sync()` a one-liner and running it post-sweep / in CI.

## Implementation notes (for Phase 3 — built before any strategy, per build order)
- `ehf/results/store.py` — `ResultStore` protocol; `LocalResultStore` (SQLite+parquet) impl; `PostgresResultStore` left as a typed stub.
- `ehf/results/registry.py` — `ExperimentRegistry` (the public API).
- `ehf/config/spec.py` — `RunSpec` pydantic model with deterministic `canonical_json()` + `config_hash` property.
- SQLite in **WAL mode**; all writes through one `ExperimentRegistry` instance; parquet via `pyarrow`.
