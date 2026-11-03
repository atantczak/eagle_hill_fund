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

## Consequences

- ``ehf.web`` now has an optional dependency on ``ehf.research``; the import is
  guarded, the site tests still run without pandas, and a new ``test_lab.py``
  exercises the lab on synthetic data.
- ``lightweight-charts`` and Google Fonts are the only third-party assets; the
  CSP still allows them (it only restricts framing).
- The Studio can run the whole thing with one command (see
  ``infra/aws/site/README.md``). The same URL scheme will work unchanged when
  the lab is deployed.
