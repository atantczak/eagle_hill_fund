"""FastAPI application for eaglehillfund.com.

Routes (the load balancer enforces a Cognito login on every path except
``/healthz``; see infra/aws/site/terraform/alb.tf):

    GET /                          home
    GET /healthz                   ALB health check
    GET /library                   write-ups + recent daily reports
    GET /library/{slug}            one write-up (summary, body, embedded PDF)
    GET /library/{slug}/file/{name}
    GET /daily                     daily reports by date
    GET /daily/{date}              the stored report.html, as-is
    GET /daily/{date}/{name}       report.pdf / report.md / ...
    GET /playbook                  the vetted VIX rules read from the latest close (web.playbook)
    GET|POST /ideas                the ideas box: propose research or features (ehf.web.ideas)
    GET /logout                    expire the ALB session, bounce through Cognito's logout
    GET /me                        who the ALB says you are
    /lab/...                       the research lab (ehf.web.lab): on the Studio over var/,
                                   in the container over a read-only S3 mirror of it (ADR-0009)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import markdown
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ehf.web.auth import alb_cookie_names, logout_url, user_from_headers
from ehf.web.config import SiteConfig
from ehf.web.library import Library, Post

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"
SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": "frame-ancestors 'self'",
    "Cache-Control": "private, no-store",
}


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response


def render_body(md: str) -> str:
    """Markdown we wrote ourselves (ops.py publish-report) -> HTML."""
    return markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists", "smarty"])


def _stream(body: Any, chunk: int = 256 * 1024) -> Iterator[bytes]:
    while True:
        data = body.read(chunk)
        if not data:
            break
        yield data


def _file_response(
    opened: tuple[Any, str, int | None] | None, name: str, *, inline: bool
) -> Response:
    if opened is None:
        raise HTTPException(404, "file not found")
    body, content_type, length = opened
    disposition = "inline" if inline else "attachment"
    headers = {"Content-Disposition": f'{disposition}; filename="{name}"'}
    if length:
        headers["Content-Length"] = str(length)
    return StreamingResponse(_stream(body), media_type=content_type, headers=headers)


def lab_available(mode: str) -> bool:
    """Decide whether /lab mounts: 'on', 'off', or 'auto' (= research importable)."""
    if mode == "off":
        return False
    try:
        import ehf.web.lab.routes  # noqa: F401
    except ImportError:
        if mode == "on":
            raise
        return False
    return True


def create_app(
    cfg: SiteConfig | None = None,
    library: Library | None = None,
    lab: Any | None = None,
    ideas: Any | None = None,
) -> FastAPI:
    """Build the app. ``lab`` is an optional ``(data, playground, charts)`` triple
    and ``ideas`` an optional ``IdeaBox`` (tests inject fakes); by default the
    lab mounts when the research platform imports, with the local results store
    and the FMP provider, and the ideas box shares the library's S3 client."""
    cfg = cfg or SiteConfig.from_env()
    if library is None:
        import boto3  # local import: tests inject a fake and never need boto3

        library = Library(
            boto3.client("s3", region_name=cfg.region),
            cfg.reports_bucket,
            library_prefix=cfg.library_prefix,
            daily_prefix=cfg.daily_prefix,
            ttl_s=cfg.cache_ttl_s,
        )
    lib: Library = library

    app = FastAPI(title=cfg.org_name, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SecurityHeaders)
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    with_lab = lab is not None or lab_available(cfg.lab)
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.globals.update(
        org=cfg.org_name,
        site_url=cfg.site_url,
        lab=with_lab,
        lab_readonly=bool(cfg.var_bucket),  # the mirror never writes back: no "log" box
        playbook=False,  # flipped below once the lab's provider is up
    )

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        user = user_from_headers(request.headers)
        return templates.TemplateResponse(
            request, name, {"user": user, "path": request.url.path, **ctx}
        )

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return page(request, "index.html")

    @app.get("/library", response_class=HTMLResponse)
    def library_index(request: Request) -> HTMLResponse:
        return page(request, "library.html", posts=lib.posts(), daily=lib.daily(limit=7))

    @app.get("/library/{slug}", response_class=HTMLResponse)
    def library_post(request: Request, slug: str) -> HTMLResponse:
        post = lib.post(slug)
        if post is None:
            raise HTTPException(404, "no such write-up")
        return page(request, "post.html", post=post, body_html=render_body(post.body_md))

    @app.get("/library/{slug}/file/{name}")
    def library_file(slug: str, name: str, download: bool = False) -> Response:
        post: Post | None = lib.post(slug)
        key = lib.post_file_key(post, name) if post else None
        if key is None:
            raise HTTPException(404, "file not found")
        return _file_response(lib.open(key), name, inline=not download)

    @app.get("/daily", response_class=HTMLResponse)
    def daily_index(request: Request) -> HTMLResponse:
        return page(request, "daily.html", daily=lib.daily(limit=120))

    @app.get("/daily/{day}", response_class=HTMLResponse)
    def daily_page(day: str) -> Response:
        report = lib.daily_report(day)
        key = lib.daily_file_key(report, "report.html") if report else None
        if key is None:
            raise HTTPException(404, "no report for that date")
        raw = lib.read(key)
        if raw is None:
            raise HTTPException(404, "no report for that date")
        return HTMLResponse(raw.decode("utf-8", errors="replace"))

    @app.get("/daily/{day}/{name}")
    def daily_file(day: str, name: str, download: bool = False) -> Response:
        report = lib.daily_report(day)
        key = lib.daily_file_key(report, name) if report else None
        if key is None:
            raise HTTPException(404, "file not found")
        return _file_response(lib.open(key), name, inline=not download)

    @app.get("/logout")
    def logout(request: Request) -> Response:
        resp = RedirectResponse(logout_url(cfg), status_code=302)
        for cookie in alb_cookie_names(request.headers.get("cookie", "")):
            resp.set_cookie(cookie, "", max_age=0, expires=0, path="/", secure=True, httponly=True)
        return resp

    @app.get("/me")
    def me(request: Request) -> JSONResponse:
        user = user_from_headers(request.headers)
        return JSONResponse(
            {"email": user.email, "name": user.name} if user else {"email": None, "name": None}
        )

    if with_lab:
        from ehf.web.lab.routes import default_components, mount_lab

        try:
            data, playground, charts = lab if lab is not None else default_components(cfg)
        except Exception:
            if cfg.lab == "on" or lab is not None:
                raise
            log.exception("lab disabled: could not build its components")
            templates.env.globals["lab"] = False
        else:
            mount_lab(app, page, data=data, playground=playground, charts=charts)
            from ehf.web.playbook import Playbook, mount_playbook

            if data.provider is not None:
                mount_playbook(app, page, Playbook(data.provider, data))
                templates.env.globals["playbook"] = True

    from ehf.web.ideas import IdeaBox, mount_ideas

    box = ideas if ideas is not None else IdeaBox(lib.s3, lib.bucket, prefix=cfg.ideas_prefix)
    mount_ideas(app, page, box)
    return app


app = create_app()
