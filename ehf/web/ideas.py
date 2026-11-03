"""The ideas box: anyone with a login can propose a research question or a site feature.

Ideas are JSON files in the reports bucket under ``ideas/<id>.json`` (no database,
no third service). The site lists them and takes new ones; Andrew moves them
through ``new -> planned -> building -> done`` (or ``declined``) with
``python infra/aws/site/ops.py ideas --set <id> --status planned --note "..."``,
and reads the queue the same way when planning with Claude. When an idea turns into
a write-up, ``--link /library/<slug>`` puts "Read the result" under it, and the
write-up's ``meta.json`` names the idea back (ADR-0011).

    GET  /ideas            the list + the form
    POST /ideas            submit one (plain HTML form, no JavaScript needed)
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ehf.web.auth import user_from_headers
from ehf.web.lab.routes import Page

log = logging.getLogger(__name__)

STATUSES: tuple[str, ...] = ("new", "planned", "building", "done", "declined")
KINDS: dict[str, str] = {
    "research": "Research question",
    "feature": "Site feature",
    "other": "Something else",
}
STATUS_LABEL: dict[str, str] = {
    "new": "New",
    "planned": "Planned",
    "building": "Building",
    "done": "Done",
    "declined": "Not now",
}
TITLE_MAX, DETAILS_MAX, AUTHOR_MAX = 120, 4000, 80


@dataclass(frozen=True)
class Idea:
    id: str
    title: str
    kind: str
    details: str
    author: str
    email: str
    created_at: str
    status: str = "new"
    note: str = ""
    updated_at: str = ""
    link: str = ""  # where the result lives on the site once there is one (/library/<slug>)

    @property
    def kind_label(self) -> str:
        return KINDS.get(self.kind, self.kind)

    @property
    def status_label(self) -> str:
        return STATUS_LABEL.get(self.status, self.status)

    @property
    def created(self) -> str:
        return self.created_at[:10]


def validate(title: str, kind: str, details: str, author: str) -> tuple[str, str, str, str]:
    """Trim and check a submission; raise ``ValueError`` with a sentence for the form."""
    title = " ".join(title.split())
    details = details.strip()
    author = " ".join(author.split())
    if len(title) < 3:
        raise ValueError("Give the idea a title (a few words).")
    if len(title) > TITLE_MAX:
        raise ValueError(f"Keep the title under {TITLE_MAX} characters.")
    if kind not in KINDS:
        raise ValueError("Pick what kind of idea this is.")
    if len(details) > DETAILS_MAX:
        raise ValueError(f"Keep the details under {DETAILS_MAX} characters.")
    if len(author) > AUTHOR_MAX:
        raise ValueError("That name is too long.")
    return title, kind, details, author


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _is_missing(exc: Exception) -> bool:
    code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
    return code in {"NoSuchKey", "404", "NotFound"}


class IdeaBox:
    """``ideas/<id>.json`` objects in a bucket, newest first, with a short list cache."""

    def __init__(self, client: Any, bucket: str, prefix: str = "ideas", ttl_s: float = 20.0):
        self.s3 = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.ttl_s = ttl_s
        self._cache: tuple[float, list[Idea]] | None = None
        self._lock = threading.Lock()

    def key(self, idea_id: str) -> str:
        return f"{self.prefix}/{idea_id}.json"

    def list(self) -> list[Idea]:
        with self._lock:
            if self._cache and time.time() - self._cache[0] < self.ttl_s:
                return list(self._cache[1])
        ideas: list[Idea] = []
        kw: dict[str, Any] = {"Bucket": self.bucket, "Prefix": self.prefix + "/"}
        while True:
            resp = self.s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".json"):
                    continue
                idea = self._read(key)
                if idea is not None:
                    ideas.append(idea)
            if not resp.get("IsTruncated"):
                break
            kw["ContinuationToken"] = resp["NextContinuationToken"]
        ideas.sort(key=lambda i: i.created_at, reverse=True)
        with self._lock:
            self._cache = (time.time(), ideas)
        return list(ideas)

    def get(self, idea_id: str) -> Idea | None:
        return self._read(self.key(idea_id))

    def submit(self, *, title: str, kind: str, details: str, author: str, email: str) -> Idea:
        title, kind, details, author = validate(title, kind, details, author)
        stamp = datetime.now(UTC)
        idea = Idea(
            id=f"{stamp:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}",
            title=title,
            kind=kind,
            details=details,
            author=author or (email.split("@")[0] if email else "someone"),
            email=email,
            created_at=stamp.isoformat(timespec="microseconds"),  # sorts newest-first exactly
        )
        self._write(idea)
        return idea

    def set_status(
        self, idea_id: str, status: str, note: str | None = None, link: str | None = None
    ) -> Idea:
        """Move an idea along; ``link`` is the site path of its result (a write-up, a run)."""
        if status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        if link is not None and link and not link.startswith("/"):
            raise ValueError("link must be a site path, e.g. /library/<slug>")
        idea = self.get(idea_id)
        if idea is None:
            raise KeyError(idea_id)
        updated = replace(
            idea,
            status=status,
            note=idea.note if note is None else note.strip(),
            link=idea.link if link is None else link.strip(),
            updated_at=_now(),
        )
        self._write(updated)
        return updated

    def _read(self, key: str) -> Idea | None:
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            raw = json.loads(body)
            fields = {k: raw.get(k, "") for k in Idea.__dataclass_fields__}
            fields["status"] = raw.get("status") or "new"
            return Idea(**fields)
        except Exception as exc:
            if not _is_missing(exc):
                log.warning("ideas: could not read %s: %s", key, exc)
            return None

    def _write(self, idea: Idea) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.key(idea.id),
            Body=json.dumps(asdict(idea), indent=1).encode(),
            ContentType="application/json",
        )
        with self._lock:
            self._cache = None


def _same_site(request: Request) -> bool:
    """Browsers send Sec-Fetch-Site; a cross-site POST is refused. Older clients pass."""
    site = request.headers.get("sec-fetch-site")
    return site in (None, "same-origin", "same-site", "none")


def mount_ideas(app: FastAPI, page: Page, box: IdeaBox) -> None:
    def render(request: Request, **ctx: Any) -> HTMLResponse:
        try:
            ideas = box.list()
            load_error = None
        except Exception as exc:
            log.exception("ideas: list failed")
            ideas, load_error = [], f"{type(exc).__name__}: {exc}"
        user = user_from_headers(request.headers)
        return page(
            request,
            "ideas.html",
            ideas=ideas,
            kinds=KINDS,
            statuses=STATUS_LABEL,
            me=user,
            load_error=load_error,
            **ctx,
        )

    @app.get("/ideas", response_class=HTMLResponse)
    def ideas_index(request: Request, sent: str | None = None) -> HTMLResponse:
        return render(request, sent=sent, form={}, error=None)

    @app.post("/ideas", response_class=HTMLResponse)
    async def ideas_submit(request: Request) -> HTMLResponse:
        if not _same_site(request):
            raise HTTPException(403, "cross-site form post refused")
        body = (await request.body())[: DETAILS_MAX * 4 + 1024]
        fields = {k: v[0] for k, v in parse_qs(body.decode("utf-8", "replace")).items()}
        user = user_from_headers(request.headers)
        form = {
            "title": fields.get("title", ""),
            "kind": fields.get("kind", "research"),
            "details": fields.get("details", ""),
            "author": (user.display if user else fields.get("author", "")),
        }
        try:
            idea = box.submit(
                title=form["title"],
                kind=form["kind"],
                details=form["details"],
                author=form["author"],
                email=user.email if user else "",
            )
        except ValueError as exc:
            resp = render(request, sent=None, form=form, error=str(exc))
            resp.status_code = 400
            return resp
        except Exception as exc:
            log.exception("ideas: submit failed")
            resp = render(request, sent=None, form=form, error=f"Could not save it: {exc}")
            resp.status_code = 500
            return resp
        return RedirectResponse(f"/ideas?sent={idea.id}", status_code=303)  # type: ignore[return-value]
