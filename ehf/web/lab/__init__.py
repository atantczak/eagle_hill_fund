"""The lab: browse stored backtests, run new ones from sliders, chart any ticker.

Mounted under ``/lab`` only when the research platform is importable (the Studio,
a laptop) — the production container ships ``ehf.web`` without ``ehf.research``
and never sees these routes. See ``ehf.web.app.create_app`` and ADR-0008.

Modules:

    data        read-only view over the results store + scorecards; payload builder
    playground  run a backtest on demand (strategy catalog, parameter ranges, cache)
    charts      candles + overlays for any symbol, with the fear-gauge dial on top
    routes      the FastAPI router (pages + JSON API)
"""
