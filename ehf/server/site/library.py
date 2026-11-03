"""The library: write-ups and daily reports, read from S3.

Layout in the reports bucket::

    library/<slug>/meta.json          slug, title, date, summary, tags, body_md, files[]
    library/<slug>/<file>             the PDF (and anything else listed in meta.files)
    daily-report/YYYY/MM/DD/report.html, report.pdf, report.md, meta.json ...

``parse_meta`` is pure; ``Library`` wraps a boto3 S3 client with a small TTL
cache so a page view costs at most one listing per minute.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".html": "text/html; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".json": "application/json",
    ".csv": "text/csv",
    ".gz": "application/gzip",
    ".png": "image/png",
}


def content_type_for(name: str) -> str:
    for ext, ct in CONTENT_TYPES.items():
        if name.lower().endswith(ext):
            return ct
    return "application/octet-stream"


@dataclass(frozen=True)
class PostFile:
    name: str
    label: str
    content_type: str
    primary: bool = False


@dataclass(frozen=True)
class Post:
    slug: str
    title: str
    date: date
    summary: str
    tags: tuple[str, ...] = ()
    body_md: str = ""
    files: tuple[PostFile, ...] = ()
    author: str = ""

    @property
    def primary(self) -> PostFile | None:
        for f in self.files:
            if f.primary:
                return f
        return self.files[0] if self.files else None

    @property
    def pdf(self) -> PostFile | None:
        for f in self.files:
            if f.content_type == "application/pdf":
                return f
        return None


@dataclass(frozen=True)
class DailyReport:
    date: date
    files: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key_prefix(self) -> str:
        return self.date.strftime("%Y/%m/%d")


def parse_meta(raw: dict[str, Any]) -> Post:
    """Pure: validate and normalise a ``meta.json`` document. Raises ValueError."""
    slug = str(raw.get("slug", "")).strip()
    if not SLUG_RE.match(slug):
        raise ValueError(f"bad slug {slug!r}")
    title = str(raw.get("title", "")).strip()
    if not title:
        raise ValueError("title is required")
    d = date.fromisoformat(str(raw.get("date", "")))
    files = []
    for f in raw.get("files", []) or []:
        name = str(f.get("name", "")).strip()
        if not FILE_RE.match(name):
            raise ValueError(f"bad file name {name!r}")
        files.append(
            PostFile(
                name=name,
                label=str(f.get("label") or name),
                content_type=str(f.get("content_type") or content_type_for(name)),
                primary=bool(f.get("primary", False)),
            )
        )
    tags = tuple(str(t).strip().lower() for t in (raw.get("tags") or []) if str(t).strip())
    return Post(
        slug=slug,
        title=title,
        date=d,
        summary=str(raw.get("summary", "")).strip(),
        tags=tags,
        body_md=str(raw.get("body_md", "") or ""),
        files=tuple(files),
        author=str(raw.get("author", "") or ""),
    )


class S3Like(Protocol):
    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...


def _is_missing(e: Exception) -> bool:
    """botocore ClientError for a missing key, without importing botocore here."""
    response = getattr(e, "response", None)
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    return code in ("NoSuchKey", "404")


class _TTL:
    def __init__(self, ttl_s: float) -> None:
        self.ttl = ttl_s
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        hit = self._data.get(key)
        if hit and time.monotonic() - hit[0] < self.ttl:
            return hit[1]
        return None

    def put(self, key: str, value: Any) -> Any:
        self._data[key] = (time.monotonic(), value)
        return value

    def clear(self) -> None:
        self._data.clear()


class Library:
    def __init__(
        self,
        client: S3Like,
        bucket: str,
        *,
        library_prefix: str = "library",
        daily_prefix: str = "daily-report",
        ttl_s: float = 60.0,
    ) -> None:
        self.s3 = client
        self.bucket = bucket
        self.library_prefix = library_prefix.strip("/")
        self.daily_prefix = daily_prefix.strip("/")
        self.cache = _TTL(ttl_s)

    # -- low level -----------------------------------------------------------
    def _keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        token: str | None = None
        while True:
            kw: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kw["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kw)
            keys.extend(o["Key"] for o in resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                return keys
            token = resp.get("NextContinuationToken")

    def open(self, key: str) -> tuple[Any, str, int | None] | None:
        """Streaming body, content type and length for a key, or ``None`` if missing."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        except Exception as e:
            if _is_missing(e):
                return None
            raise
        return (
            obj["Body"],
            str(obj.get("ContentType") or content_type_for(key)),
            obj.get("ContentLength"),
        )

    def read(self, key: str) -> bytes | None:
        opened = self.open(key)
        return None if opened is None else bytes(opened[0].read())

    # -- write-ups -----------------------------------------------------------
    def posts(self) -> list[Post]:
        cached = self.cache.get("posts")
        if cached is not None:
            return list(cached)
        posts: list[Post] = []
        for key in self._keys(f"{self.library_prefix}/"):
            if not key.endswith("/meta.json"):
                continue
            raw = self.read(key)
            if raw is None:
                continue
            try:
                posts.append(parse_meta(json.loads(raw)))
            except (ValueError, json.JSONDecodeError):
                continue  # a broken meta.json hides that post, never the whole library
        posts.sort(key=lambda p: (p.date, p.slug), reverse=True)
        return list(self.cache.put("posts", posts))

    def post(self, slug: str) -> Post | None:
        if not SLUG_RE.match(slug):
            return None
        for p in self.posts():
            if p.slug == slug:
                return p
        return None

    def post_file_key(self, post: Post, name: str) -> str | None:
        if not FILE_RE.match(name) or not any(f.name == name for f in post.files):
            return None
        return f"{self.library_prefix}/{post.slug}/{name}"

    # -- daily reports -------------------------------------------------------
    def daily(self, limit: int = 60) -> list[DailyReport]:
        cached = self.cache.get("daily")
        if cached is None:
            by_day: dict[date, set[str]] = {}
            pat = re.compile(
                rf"^{re.escape(self.daily_prefix)}/(\d{{4}})/(\d{{2}})/(\d{{2}})/([^/]+)$"
            )
            for key in self._keys(f"{self.daily_prefix}/"):
                m = pat.match(key)
                if not m:
                    continue
                try:
                    d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    continue
                by_day.setdefault(d, set()).add(m.group(4))
            reports = [
                DailyReport(d, tuple(sorted(fs))) for d, fs in by_day.items() if "report.html" in fs
            ]
            reports.sort(key=lambda r: r.date, reverse=True)
            cached = self.cache.put("daily", reports)
        return list(cached)[:limit]

    def daily_report(self, day: str) -> DailyReport | None:
        if not DATE_RE.match(day):
            return None
        try:
            d = date.fromisoformat(day)
        except ValueError:
            return None
        for r in self.daily(limit=10_000):
            if r.date == d:
                return r
        return None

    def daily_file_key(self, report: DailyReport, name: str) -> str | None:
        if not FILE_RE.match(name) or name not in report.files:
            return None
        return f"{self.daily_prefix}/{report.key_prefix}/{name}"
