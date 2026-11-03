"""The site: meta parsing, routing, auth header decoding. No AWS, no network."""

from __future__ import annotations

import base64
import io
import json
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ehf.server.site.app import create_app
from ehf.server.site.auth import alb_cookie_names, logout_url, user_from_headers
from ehf.server.site.config import SiteConfig
from ehf.server.site.library import Library, parse_meta

META = {
    "slug": "vix-results",
    "title": "The Fear Gauge, Put to the Test",
    "date": "2026-11-03",
    "summary": "Nine VIX rules, walk-forward.",
    "tags": ["VIX", " spy "],
    "body_md": "## Short version\n\nTwo rules **earned** a place.",
    "files": [
        {"name": "results.pdf", "label": "Read the PDF", "primary": True},
        {"name": "notes.md"},
    ],
    "author": "Andrew",
}


class FakeS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.calls = 0

    def list_objects_v2(self, **kw: Any) -> dict[str, Any]:
        self.calls += 1
        keys = sorted(k for k in self.objects if k.startswith(kw["Prefix"]))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def get_object(self, **kw: Any) -> dict[str, Any]:
        key = kw["Key"]
        if key not in self.objects:
            err: Any = type("ClientError", (Exception,), {})()
            err.response = {"Error": {"Code": "NoSuchKey"}}
            raise err
        data = self.objects[key]
        return {"Body": io.BytesIO(data), "ContentLength": len(data)}


@pytest.fixture
def bucket() -> FakeS3:
    return FakeS3(
        {
            "library/vix-results/meta.json": json.dumps(META).encode(),
            "library/vix-results/results.pdf": b"%PDF-1.4 fake",
            "library/vix-results/notes.md": b"# notes",
            "library/broken/meta.json": b"{not json",
            "daily-report/2026/11/03/report.html": b"<html>today</html>",
            "daily-report/2026/11/03/report.pdf": b"%PDF daily",
            "daily-report/2026/11/02/report.html": b"<html>yesterday</html>",
            "daily-report/latest/report.html": b"<html>latest</html>",
        }
    )


@pytest.fixture
def client(bucket: FakeS3) -> TestClient:
    cfg = SiteConfig(
        site_url="https://eaglehillfund.com",
        cognito_domain="auth.example.com",
        cognito_client_id="abc",
    )
    lib = Library(bucket, "test-bucket", ttl_s=60)
    return TestClient(create_app(cfg, lib))


def test_parse_meta_normalises() -> None:
    p = parse_meta(META)
    assert p.slug == "vix-results"
    assert p.date == date(2026, 11, 3)
    assert p.tags == ("vix", "spy")
    assert p.primary is not None and p.primary.name == "results.pdf"
    assert p.pdf is not None and p.pdf.content_type == "application/pdf"
    assert p.files[1].content_type.startswith("text/markdown")


@pytest.mark.parametrize(
    "bad",
    [
        {"slug": "Bad Slug"},
        {"slug": "ok", "title": ""},
        {"slug": "ok", "title": "t", "date": "nope"},
    ],
)
def test_parse_meta_rejects(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_meta({**META, **bad})


def test_parse_meta_rejects_traversal_file_names() -> None:
    with pytest.raises(ValueError):
        parse_meta({**META, "files": [{"name": "../secret.pdf"}]})


def test_library_lists_posts_and_hides_broken_meta(bucket: FakeS3) -> None:
    lib = Library(bucket, "b")
    posts = lib.posts()
    assert [p.slug for p in posts] == ["vix-results"]
    lib.posts()
    assert bucket.calls == 1  # cached
    assert lib.post("../x") is None


def test_library_daily_ignores_latest(bucket: FakeS3) -> None:
    lib = Library(bucket, "b")
    days = [r.date.isoformat() for r in lib.daily()]
    assert days == ["2026-11-03", "2026-11-02"]
    assert lib.daily_report("2026-11-03") is not None
    assert lib.daily_report("latest") is None
    assert lib.daily_report("2026-13-01") is None


def test_public_pages(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"ok": True}
    r = client.get("/")
    assert r.status_code == 200 and "Open the library" in r.text
    assert r.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert "max-age=31536000" in r.headers["Strict-Transport-Security"]


def test_library_pages(client: TestClient) -> None:
    r = client.get("/library")
    assert r.status_code == 200
    assert "The Fear Gauge, Put to the Test" in r.text
    assert "Tuesday, November 3, 2026" in r.text
    r = client.get("/library/vix-results")
    assert r.status_code == 200
    assert "<strong>earned</strong>" in r.text  # body markdown rendered
    assert '/library/vix-results/file/results.pdf"' in r.text
    assert client.get("/library/nope").status_code == 404


def test_library_files_stream_with_disposition(client: TestClient) -> None:
    r = client.get("/library/vix-results/file/results.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith("inline")
    assert r.content == b"%PDF-1.4 fake"
    r = client.get("/library/vix-results/file/results.pdf?download=1")
    assert r.headers["content-disposition"].startswith("attachment")
    assert (
        client.get("/library/vix-results/file/meta.json").status_code == 404
    )  # not listed in meta
    assert client.get("/library/vix-results/file/..%2Fmeta.json").status_code == 404


def test_daily_pages(client: TestClient) -> None:
    assert "November 3, 2026" in client.get("/daily").text
    r = client.get("/daily/2026-11-03")
    assert r.status_code == 200 and r.text == "<html>today</html>"
    assert client.get("/daily/2026-11-03/report.pdf").content == b"%PDF daily"
    assert client.get("/daily/2026-11-03/meta.json").status_code == 404
    assert client.get("/daily/latest").status_code == 404


def _jwt(claims: dict[str, Any]) -> str:
    seg = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJFUzI1NiJ9.{seg}.sig"


def test_user_from_alb_header() -> None:
    u = user_from_headers({"x-amzn-oidc-data": _jwt({"email": "a@b.com", "name": "Andrew"})})
    assert u is not None and u.display == "Andrew" and u.email == "a@b.com"
    assert user_from_headers({}) is None
    assert user_from_headers({"x-amzn-oidc-data": "garbage"}) is None
    assert user_from_headers({"x-amzn-oidc-data": "a.b.c"}) is None


def test_me_and_logout(client: TestClient) -> None:
    assert client.get("/me").json() == {"email": None, "name": None}
    hdr = {
        "x-amzn-oidc-data": _jwt({"email": "a@b.com"}),
        "cookie": "AWSELBAuthSessionCookie-0=x; other=y",
    }
    assert client.get("/me", headers=hdr).json()["email"] == "a@b.com"
    r = client.get("/logout", headers=hdr, follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"].startswith(
        "https://auth.example.com/logout?client_id=abc&logout_uri=https%3A%2F%2Feaglehillfund.com%2F"
    )
    assert "AWSELBAuthSessionCookie-0=" in r.headers["set-cookie"]
    assert "other=" not in r.headers["set-cookie"]


def test_logout_url_without_cognito() -> None:
    assert logout_url(SiteConfig()) == "/"
    assert alb_cookie_names("AWSELBAuthSessionCookie-0=a; AWSELBAuthSessionCookie-1=b; x=1") == [
        "AWSELBAuthSessionCookie-0",
        "AWSELBAuthSessionCookie-1",
    ]
