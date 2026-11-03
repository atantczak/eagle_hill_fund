# Results storage on AWS (RDS Postgres + S3)

The results store has two interchangeable backends behind one `ResultStore`
contract (ADR-0002):

| Backend | Relational core | Heavy arrays | When |
|---|---|---|---|
| `LocalResultStore` (default) | SQLite `var/results/ehf.db` | parquet `var/results/<hash>/` | local dev |
| `PostgresResultStore` | **RDS Postgres** | **S3** (parquet, keyed by `config_hash`) | shared / durable |

`equity.parquet` + `trades.parquet` are the only "heavy" payloads; they go to S3.
Everything queryable — `runs`, `metrics`, `sweeps`, `theses` — goes to Postgres.
No strategy, engine, or analysis code knows which backend is active.

## How the backend is chosen

A bare `ExperimentRegistry()` calls `result_store_from_env()`. Set the database
env and **every** registry (single runs, parallel sweeps, the migration script)
writes to RDS+S3; unset it and behavior is exactly as before (local).

```bash
pip install -e '.[rds]'          # psycopg + boto3 (only needed for the cloud backend)

# Option A — direct DSN
export EHF_DATABASE_URL='postgresql://user:pass@your-instance.rds.amazonaws.com:5432/ehf'

# Option B — RDS-managed credentials in Secrets Manager (preferred; no secrets in env)
export EHF_DB_SECRET_ARN='arn:aws:secretsmanager:us-east-1:...:secret:ehf/rds-XXXX'
export EHF_DB_NAME='ehf'         # used if the secret omits dbname

# Required whenever a DB is configured:
export EHF_RESULTS_S3_BUCKET='my-ehf-results'
export EHF_RESULTS_S3_PREFIX='ehf-results'   # optional, this is the default
export AWS_REGION='us-east-1'
```

AWS auth for S3/Secrets Manager uses the standard boto3 chain (env creds, profile,
or instance role) — nothing EHF-specific.

## Migrating existing local runs

One-time backfill of everything already in `var/results` (idempotent — safe to
re-run):

```bash
python scripts/migrate_results_to_rds.py --dry-run   # report counts only
python scripts/migrate_results_to_rds.py             # runs + metrics + sweeps + arrays
```

## Provisioning (if no instance exists yet)

Minimum viable setup:
1. **RDS Postgres** (a small `db.t4g.micro` is plenty for this workload). Put it
   in a private subnet; allow inbound 5432 only from your IP / the app SG.
2. **Secrets Manager** secret with the RDS credentials (RDS can create/rotate this
   for you) → use its ARN as `EHF_DB_SECRET_ARN`.
3. **S3 bucket** (versioning optional) for the arrays; grant the running identity
   `s3:GetObject`/`s3:PutObject` on `arn:aws:s3:::<bucket>/<prefix>/*`.

The schema is created automatically on first write (`CREATE TABLE IF NOT EXISTS`),
so there is no separate migration/DDL step. Tables match the local SQLite schema
one-for-one (the DDL is shared; only `REAL → DOUBLE PRECISION` differs).

## Tests

Pure logic (placeholder translation, upsert SQL, secret parsing, env selection,
worker-descriptor round-trips) runs with no infra. A live round-trip test in
`tests/test_store_postgres.py` is skipped unless you point it at a throwaway DB +
bucket:

```bash
export EHF_TEST_DATABASE_URL='postgresql://...'
export EHF_TEST_S3_BUCKET='my-test-bucket'
pytest ehf/tests/test_store_postgres.py
```
