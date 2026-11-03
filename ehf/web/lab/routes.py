"""``/lab`` — pages and the JSON API behind them.

GET  /lab                        the browser: walk-forward studies + stored runs
GET  /lab/run/{hash}             one run, charted
GET  /lab/play                   the playground (sliders -> run)
GET  /lab/chart?symbol=AAPL      any ticker with the fear-gauge dial on top

GET  /lab/api/studies
GET  /lab/api/runs?strategy=&standalone=&sweep_id=&sort=&limit=
GET  /lab/api/run/{hash}             ... plus ``writeups``: the library posts behind it (ADR-0011)
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
from ehf.web.lab.playground import Playground, PlayRequest, catalog, pair_presets, universe_presets

log = logging.getLogger(__name__)

Page = Callable[..., HTMLResponse]


def writeups_for(
    library: Any, config_hash: str, study: str, ideas: Any = None
) -> list[dict[str, Any]]:
    """The write-ups behind a run: those naming it directly, then those about its study.
    Each carries the ideas that prompted it, resolved to titles when the box is at hand."""
    if library is None:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        direct = library.posts_for_run(config_hash)
        via_study = library.posts_for_study(study) if study else []
    except Exception:  # the library is S3; a hiccup there must not break the run page
        log.exception("lab: could not read the library for run %s", config_hash)
        return []
    for post, how in [(p, "run") for p in direct] + [(p, "study") for p in via_study]:
        if post.slug in seen:
            continue
        seen.add(post.slug)
        label = next((r.label for r in post.runs if r.hash == config_hash and r.label), "")
        prompted = []
        for iid in post.ideas:
            idea = None
            if ideas is not None:
                try:
                    idea = ideas.get(iid)
                except Exception:
                    idea = None
            prompted.append(
                {
                    "id": iid,
                    "title": idea.title if idea else iid,
                    "author": idea.author if idea else "",
                }
            )
        out.append(
            {
                "slug": post.slug,
                "title": post.title,
                "date": post.date.isoformat(),
                "summary": post.summary,
                "how": how,
                "label": label,
                "ideas": prompted,
            }
        )
    return out


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
    library: Any = None,
    ideas: Any = None,
) -> None:
    """``library`` (``ehf.web.library.Library``) and ``ideas`` (``IdeaBox``) are optional: with
    them, a run page shows the write-up built on it and the idea that prompted it."""
    r = APIRouter(prefix="/lab")

    def study_writeups() -> dict[str, list[dict[str, str]]]:
        """study key -> the posts about it (for the Studies table)."""
        if library is None:
            return {}
        out: dict[str, list[dict[str, str]]] = {}
        try:
            for key in data.study_keys():
                posts = library.posts_for_study(key)
                if posts:
                    out[key] = [{"slug": p.slug, "title": p.title} for p in posts]
        except Exception:
            log.exception("lab: could not read the library")
        return out

    # -- pages -------------------------------------------------------------------------

    @r.get("", response_class=HTMLResponse)
    @r.get("/", response_class=HTMLResponse)
    def lab_index(request: Request) -> HTMLResponse:
        studies = data.studies()
        return page(
            request,
            "lab/index.html",
            studies=studies,
            study_writeups=study_writeups(),
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
            pairs=pair_presets(),
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
            payload = data.run(config_hash)
        except KeyError:
            raise HTTPException(404, "no such run") from None
        except FileNotFoundError:
            raise HTTPException(404, "run arrays are missing on disk") from None
        study = (payload.get("study") or {}).get("study") or ""
        if not study:  # a full-sample run of a study is not the stitched one its card names
            study = next((s.study for s in data.scorecards() if config_hash in s.hashes), "")
        payload["writeups"] = writeups_for(library, config_hash, study, ideas)
        return JSONResponse(payload)

    @r.get("/api/catalog")
    def api_catalog() -> JSONResponse:
        return JSONResponse(
            {"strategies": catalog(), "presets": universe_presets(), "pairs": pair_presets()}
        )

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
