"""The lab: browse stored backtests, run new ones from sliders, chart any ticker.

Mounted under ``/lab`` wherever the research platform is importable: on the
Studio straight over ``var/``; in the production container over a read-only S3
mirror of the Studio's ``var/`` (ADR-0008, ADR-0009). See
``ehf.web.app.create_app``.

Modules:

    data        read-only view over the results store + scorecards; payload builder
    playground  run a backtest on demand (strategy catalog, parameter ranges, cache)
    charts      candles + overlays for any symbol, with the fear-gauge dial on top
    mirror      the S3-backed store, cache and scorecards the container uses
    routes      the FastAPI router (pages + JSON API)
"""
