"""``/lab`` — pages and the JSON API behind them.

GET  /lab                        the browser: walk-forward studies + stored runs
GET  /lab/run/{hash}             one run, charted
GET  /lab/play                   the playground (sliders -> run)
GET  /lab/chart?symbol=AAPL      any ticker with the fear-gauge dial on top

GET  /lab/api/studies
GET  /lab/api/runs?strategy=&standalone=&sweep_id=&sort=&limit=
GET  /lab/api/run/{hash}
GET  /lab/api/catalog
POST /lab/api/play               {strategy, params, start, end, asset|symbols, ...}
GET  /lab/api/bars/{symbol}?start=&end=&refresh=1&sma=50,200&ema=21&bb=20,2&rsi=14
GET  /lab/api/intraday/{symbol}?refresh=1          the latest session, minute by minute
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ehf.web.config import SiteConfig
from ehf.web.lab.charts import Charts, parse_indicators
from ehf.web.lab.data import LabData
from ehf.web.lab.playground import Playground, PlayRequest, catalog, universe_presets

log = logging.getLogger(__name__)

Page = Callable[..., HTMLResponse]


def default_components(cfg: SiteConfig | None = None) -> tuple[LabData, Playground, Charts]:
    """The real thing: local store, local scorecards, FMP provider over the cache.

    With ``cfg.var_bucket`` set (the container), the store and the cache are
    read-only mirrors of that bucket's ``var/`` (``ehf.web.lab.mirror``)."""
    if cfg is not None and cfg.var_bucket:
        from ehf.web.lab.mirror import mirrored_components

        return mirrored_components(bucket=cfg.var_bucket, prefix=cfg.var_prefix, region=cfg.region)

    from ehf.research.data.fmp_provider import FMPProvider
    from ehf.research.results.registry import ExperimentRegistry
    from ehf.research.results.store import LocalResultStore

    store = LocalResultStore()
    provider = FMPProvider()
    data = LabData(store=store, provider=provider)
    play = Playground(provider, registry=ExperimentRegistry(store))
    return data, play, Charts(provider)


def mount_lab(
    app: FastAPI,
    page: Page,
    *,
    data: LabData,
    playground: Playground,
    charts: Charts,
) -> None:
    r = APIRouter(prefix="/lab")

    # -- pages -------------------------------------------------------------------------

    @r.get("", response_class=HTMLResponse)
    @r.get("/", response_class=HTMLResponse)
    def lab_index(request: Request) -> HTMLResponse:
        studies = data.studies()
        return page(
            request,
            "lab/index.html",
            studies=studies,
            strategies=data.strategies(),
            runs=data.runs(limit=100),
        )

    @r.get("/run/{config_hash}", response_class=HTMLResponse)
    def lab_run(request: Request, config_hash: str) -> HTMLResponse:
        if not data.store.has_run(config_hash):
            raise HTTPException(404, "no such run")
        return page(request, "lab/run.html", config_hash=config_hash)

    @r.get("/play", response_class=HTMLResponse)
    def lab_play(request: Request) -> HTMLResponse:
        return page(
            request,
            "lab/play.html",
            catalog=catalog(),
            presets=universe_presets(),
            today=date.today().isoformat(),
        )

    @r.get("/chart", response_class=HTMLResponse)
    def lab_chart(request: Request, symbol: str = "SPY") -> HTMLResponse:
        return page(
            request, "lab/chart.html", symbol=symbol.upper(), today=date.today().isoformat()
        )

    # -- api ---------------------------------------------------------------------------

    @r.get("/api/studies")
    def api_studies() -> JSONResponse:
        return JSONResponse({"studies": data.studies()})

    @r.get("/api/runs")
    def api_runs(
        strategy: str | None = None,
        standalone: bool = True,
        sweep_id: str | None = None,
        sort: str = "created_at",
        desc: bool = True,
        limit: int = Query(200, ge=1, le=2000),
    ) -> JSONResponse:
        rows = data.runs(
            strategy=strategy,
            standalone=standalone,
            sweep_id=sweep_id,
            sort=sort,
            descending=desc,
            limit=limit,
        )
        return JSONResponse({"runs": rows})

    @r.get("/api/run/{config_hash}")
    def api_run(config_hash: str) -> JSONResponse:
        try:
            return JSONResponse(data.run(config_hash))
        except KeyError:
            raise HTTPException(404, "no such run") from None
        except FileNotFoundError:
            raise HTTPException(404, "run arrays are missing on disk") from None

    @r.get("/api/catalog")
    def api_catalog() -> JSONResponse:
        return JSONResponse({"strategies": catalog(), "presets": universe_presets()})

    @r.post("/api/play")
    async def api_play(request: Request) -> JSONResponse:
        body: dict[str, Any] = await request.json()
        try:
            req = PlayRequest.parse(body)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from None
        try:
            import anyio

            payload = await anyio.to_thread.run_sync(playground.run, req)
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from None
        except Exception as exc:  # the engine said no; show why
            log.exception("playground run failed")
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from None
        return JSONResponse(payload)

    @r.get("/api/bars/{symbol}")
    def api_bars(
        symbol: str,
        start: str = "2015-01-01",
        end: str | None = None,
        refresh: bool = False,
        sma: str | None = None,
        ema: str | None = None,
        bb: str | None = None,
        rsi: str | None = None,
    ) -> JSONResponse:
        end = end or date.today().isoformat()
        try:
            ind = parse_indicators(sma=sma, ema=ema, bb=bb, rsi=rsi)
            return JSONResponse(charts.symbol(symbol, start, end, refresh=refresh, indicators=ind))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from None

    @r.get("/api/intraday/{symbol}")
    def api_intraday(symbol: str, refresh: bool = False) -> JSONResponse:
        try:
            return JSONResponse(charts.intraday(symbol, refresh=refresh))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from None

    app.include_router(r)
