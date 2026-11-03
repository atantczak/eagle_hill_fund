# PLAN FROM HERE — Market Data Sync & Backtest Platform

> Working scratch-doc to capture where we landed in a design conversation (2026-06-28)
> so it can be fed back into a future session. Not final architecture — a decision log
> + the empirical findings that should drive the build.

## The goal (what Andrew wants)

A robust system to run **many strategies × many stocks × many timeframes × many time
periods** and store all the results — on a big local box (M3 Ultra, 28 cores, 96GB).
"Store it all," study broadly, keep it reproducible.

## How we got here (conversation summary)

1. The `run_strategy.ipynb` backtest run **didn't actually succeed** — its output was a
   wall of FMP `429`s + `RemoteDisconnected` + `OSError(49) Can't assign requested
   address`. Root cause: **168 fetch workers** (tuned to CPU cores) hammering a
   rate-limited API → ports exhausted, requests throttled. No metrics table rendered.

2. **Key principle established:** separate **fetch** (I/O-bound, externally rate-limited,
   wants concurrency ~4–16) from **backtest** (CPU-bound, embarrassingly parallel, wants
   all 28 cores). The notebook fuses them — that's the bug. The big machine helps the
   sweep, *not* the fetch.

3. **Philosophical debate** — Andrew initially resisted building a local "reflection" of
   FMP, not wanting a second source of truth / a sync burden. Resolution reached:
   - There is no "no cache," only **TTL**. Hitting the live API every run = TTL 0, which
     is the wrong point for immutable historical data.
   - Live-only **breaks reproducibility** — same `config_hash` would yield different
     results if FMP restates/re-adjusts. For a fund, "what data did we see when we
     decided X" is a first-class (audit) requirement. A frozen snapshot is *more*
     trustworthy, not less.
   - A cache of immutable facts is **memoization**, not a forked source of truth.

4. Andrew came around and asked for: **a complete sync of FMP + a weekly update** that
   appends new data and reconciles changes like splits ("old price data DOES change").

5. We refined "old data changes": the **raw print never changes; only the *adjusted*
   series does.** Ideal design = store immutable raw + corporate-actions feed, **derive
   adjusted at read time**. Then a split is just one appended action row, not a history
   rewrite. The weekly job watches the **corporate-actions feed** (the *cause*) instead
   of diffing prices (the *effect*).

6. **We ran a probe** to confirm whether that ideal design is even possible with FMP.
   It is **not** (see findings) — so the design pivoted (see "Revised architecture").

## PROBE RESULTS (empirical, 2026-06-28) — the important part

Script: `scratchpad/fmp_probe.py` (read-only GETs). Endpoints are on
`https://financialmodelingprep.com/stable`.

1. **Intraday history is deep but only via narrow windows.**
   - 1h, one wide `2020→2026` request → only the most recent **~3 months** came back.
   - 1h, narrow **1-week** windows → data present at **2025, 2024, 2022, 2020, 2018, 2015**.
   - 1min → effectively only the **most recent day**.
   - ⇒ "Years of hourly" exists, but ONLY by **chunking into many small requests** →
     request-count-bound → **rate-limit-bound**. Drop 1-minute for history.

2. **FMP serves ONLY a back-adjusted series — there is no raw intraday.** (the big one)
   - NVDA 10:1 split (2024-06-10): the prior week's hourly closes come back at **~$120**,
     not the ~$1,200 that actually traded. Fully split-adjusted, retroactively.
     AAPL confirms (June 2020 ~$80, June 2015 ~$32 — back-adjusted across its splits).
   - **`nonadjusted=True` is a NO-OP** — adjusted vs "nonadjusted" returned *identical*
     numbers. The flag at `integrations/.../price_data/tool.py:55` does nothing on this tier.
   - Daily `close` is also split-adjusted, with **no `adjClose`/raw field** at all.
   - ⇒ The "store immutable raw + derive" plan is **impossible at this source**. FMP
     silently rewrites pre-split history on the next split. This is exactly the
     "old data changes" problem — and it's the default, not avoidable.

3. **Splits & dividends feeds are clean and per-symbol** (the linchpin):
   - `/splits?symbol=NVDA` → `{date, numerator, denominator, splitType}` back to 2000.
   - `/dividends?symbol=NVDA` → `{date, dividend, adjDividend, ...}`.
   - `/historical/stock_split` → 404 (dead v3 path). Calendars hard-cap at 4000 rows →
     use per-symbol, not the calendar.

## Revised architecture (the synthesis)

Can't keep an immutable raw mirror, but CAN keep a series whose **only legal mutations
are split roll-forwards we apply ourselves from the splits feed** — never blind re-fetch:

- Store the **adjusted bars**, tagged with an **adjustment basis** (which splits are
  baked in / the date through which splits are reflected).
- **Weekly job** pulls `/splits?symbol=` as the **change signal**. On a NEW split:
  **roll the stored series forward** — multiply pre-ex-date prices by
  `denominator/numerator` (÷10 for a 10:1), scale volume inversely. New right-edge bars
  already arrive on the new basis. Series stays continuous. Update is O(one factor),
  NOT O(re-fetch the symbol).
- **Reproducibility holds:** every mutation is a *known, invertible function* of the
  splits feed. A backtest records the basis it ran on; any prior basis is re-derivable.
- **Weekly job, full shape:** (1) extend right edge per symbol; (2) ingest new splits →
  roll forward; (3) re-fetch a **trailing window** (~7–30d) to catch silent corrections;
  (4) refresh symbol master (new listings backfill, delistings kept for survivorship).
- **Storage:** adjusted bars as partitioned Parquet under `var/` (per the
  var/ convention — generated artifacts live there); coverage **manifest** + corporate
  actions in Postgres; S3 as durable backup. Manifest = `(symbol, grain, start, end,
  last_synced_at, basis, n_rows, status)` → makes initial + weekly sync resumable/idempotent.
- **Fetch concurrency** bounded to the FMP plan (~4–16), decoupled from the 28 cores.
- Job is a **standalone CLI** (`sync_market_data.py`), cron-able weekly (Studio or AWS).

## OPEN DECISIONS (need Andrew's input)

1. **Grain to anchor on** — *the key fork*:
   - **Daily-first** (cheap: ~1 request/symbol-decade, deep history, robust) + hourly as
     opt-in for a smaller focus list, **OR**
   - **Hourly-first** across the broad universe, accepting a **multi-day** initial sync
     (hourly = ~weeks-per-request × thousands of symbols × years).
2. **FMP plan tier / rate limit** — sets fetch concurrency and initial-sync duration.
3. **Universe scope of "complete"** — all FMP symbols incl. delisted/ETF/intl, or US
   equities only? (Only the incl-delisted version fixes survivorship bias; note
   `FMPProvider.random_symbols(as_of=...)` already pulls delisted-alive-on logic.)

## OPEN QUESTIONS (probe didn't settle)

- **Dividend adjustment:** daily had no `adjClose`, intraday looked split-adjusted only.
  Whether dividends are baked in is **unknown** → probe a high-yield name (REIT around
  an ex-date) before committing. Decides if total-return needs separate handling.
- **Plan-dependence of `nonadjusted`:** the no-op may be a tier limit. If a higher tier
  returns true raw intraday, the cleaner store-raw-and-derive design **reopens** — worth
  a quick check with FMP docs/support given how much it'd simplify things.
- **Silent corrections** (fixed bad prints, late backfills): not captured by the splits
  feed → handled by trailing-window re-fetch + a periodic deep audit.

## BUGS / CLEANUP found along the way

- **`adjusted` flag is a fiction.** `FMPProvider.get_bars(adjusted=False)` fetches the
  *same* adjusted data but caches+labels it `adjusted=False`
  (`research/data/fmp_provider.py:115,127,144`); `nonadjusted` is never passed and is a
  no-op anyway. Asking for unadjusted silently returns adjusted bars mislabeled as raw.
  → Make the flag honest or remove it.
- **`run_strategy.ipynb` fetch fused into the run** with CPU-tuned worker counts → the
  429 storm. Fetch must move out to the sync job; backtest reads from the local store.

## IMMEDIATE NEXT STEPS (when we resume)

1. Andrew answers the 3 OPEN DECISIONS (esp. grain).
2. Run the **dividend probe** to close that gap.
3. Scaffold `sync_market_data.py`: chunked + bounded-concurrency + resumable fetcher →
   Parquet under `var/` + Postgres manifest; then the weekly extend/roll-forward/trailing
   job.
4. Decouple the notebook/backtest path to read from the local store (no fetch at run time).
5. Fix the `adjusted`-flag bug.

## Key file references

- `ehf/research/data/fmp_provider.py` — provider, caching, `random_symbols`
  (survivorship-aware), `_fetch_intraday_parallel` (bounded pool).
- `ehf/integrations/financial/financial_modeling_prep/price_data/tool.py` —
  raw FMP endpoints (`/historical-chart/{interval}`, `/historical-price-eod/full`,
  the no-op `nonadjusted`).
- `ehf/tools/data/transmission/api/tool.py` — base API client (retry/backoff,
  retryable {429,500,502,503,504}).
- `ehf/research/config/spec.py` — `RunSpec` + `config_hash` (reproducibility ID).
- `ehf/research/results/` — `ExperimentRegistry` (Postgres + S3).
- `scratchpad/fmp_probe.py` — the probe (regenerate if scratchpad is cleared).
