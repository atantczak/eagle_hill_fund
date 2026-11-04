# ADR-0008 — The research lab lives in the web app, local first

- **Status:** Accepted (the "local first" paragraph is amended by ADR-0009)
- **Date:** 2026-11-03

## Context

Backtests were only reachable through notebooks and scorecard markdown. Andrew
wanted a TradingView-like way to browse every stored run, drag a strategy's
knobs and watch the curve move, and chart any ticker with the fear-gauge dial on
top — usable by him and his dad, not just by whoever has the repo open.

## Decision

The lab is a sub-package of the site, ``ehf/web/lab``, mounted under ``/lab`` by
the same FastAPI app, behind the same Cognito login when deployed:

- ``data.py`` reads the local results store and ``var/scorecards`` and builds
  one JSON payload per run: equity vs. buy-and-hold (SPY, or the timed asset for a dial), drawdowns, exposure
  rebuilt from the trade ledger (with resets at walk-forward window starts),
  candles of the primary asset with trade markers, yearly returns, windows.
- ``explain.py`` writes the short version the run page opens with: the rule in
  plain words with the numbers that actually ran, the equation, the setup
  (dates, costs, the grey line it is judged against, the walk-forward grid) and
  what happened, with the gauntlet's verdict spelled out check by check. One
  entry per strategy; an unknown strategy gets its docstring and settings.
  ``blurb()`` is the one-or-two-sentence form of the same words, carried on every
  study and run row and shown as a hover card on the backtests list.
- ``playground.py`` turns the strategy registry into a form (ranges from the
  study grids, otherwise widened around the default), runs the real engine on
  demand, caches by config hash, and logs to the registry only when asked.
  Dial strategies take any asset: the VIX dataset is assembled around it.
- ``charts.py`` serves candles, moving averages, the VIX, and the exposure the
  ``vix_vote`` dial would have held in that symbol.
- Charts are TradingView's open-source ``lightweight-charts`` (v5, Apache-2.0,
  pinned on unpkg), one chart with panes per screen, window and out-of-market
  shading drawn as an overlay layer. The whole site got a light/dark theme
  (system preference, a toggle, ``?theme=`` for links).

**Local first.** The lab mounts only when ``ehf.research`` is importable
(``EHF_LAB=auto``), which is true on the Studio and false in the production
container, whose Dockerfile copies ``ehf/web`` alone. The playground needs the
price cache and the results store on local disk; shipping them to ECS (cache in
S3, read-only store snapshot, a bigger image) is a separate decision for when
the local version has earned it. *(Done the same day: ADR-0009 — the container
mirrors the Studio's ``var/`` from S3, read-only.)*

### Freshness (added 2026-11-03)

Bars are cache-first, and the provider counts a span as covered when the cache ends
within four days of the requested end, so a chart could lag by up to three sessions.
The chart page and the playbook therefore re-pull the last ten days per symbol at
most every 15 minutes, or on demand through **Refresh** (`?refresh=1`, throttled to
once a minute), and show a live quote with its timestamp. Measured against FMP:
the quote is 1–10 s behind the tape, 1-minute bars about a minute, and the
end-of-day row for the current day updates a few minutes behind, so an intraday
reading is provisional until the 4 pm close. `refresh_tail` always pulls from the
end of the cache forward, never a bare ten-day tail, so a stale cache cannot be
left with an invisible hole.

The chart page also has a **Today** tab: the latest regular session in one-minute
candles with the VIX alongside, pulled straight from FMP (never the daily cache,
which would count a partial day as covered), kept for 60 s, and re-pulled every
minute by the browser while the market is open. Chart times are Eastern
wall-clock read as UTC so the axis says 9:30 to 4:00 for every viewer.

Indicators on the daily chart are configurable (`sma=`, `ema=`, `bb=window,n_std`,
`rsi=` on the bars API; an indicator bar on the page, remembered in the URL and
the browser). They are computed server-side with the same `signals` functions the
strategies use, over a warm-up long enough for the longest window, so the lines
match what a backtest would see. Drawing is TradingView's open-source Lightweight
Charts (Apache 2.0, attribution logo kept); it draws, we compute.

## Consequences

- ``ehf.web`` now has an optional dependency on ``ehf.research``; the import is
  guarded, the site tests still run without pandas, and a new ``test_lab.py``
  exercises the lab on synthetic data.
- ``lightweight-charts`` and Google Fonts are the only third-party assets; the
  CSP still allows them (it only restricts framing).
- The Studio can run the whole thing with one command (see
  ``infra/aws/site/README.md``). The same URL scheme will work unchanged when
  the lab is deployed.
