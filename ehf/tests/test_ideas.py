"""The ideas box: validation, S3 round trip, the form, status updates. No network."""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from ehf.tests.test_site import FakeS3
from ehf.web.app import create_app
from ehf.web.config import SiteConfig
from ehf.web.ideas import IdeaBox, validate
from ehf.web.library import Library


def _jwt(claims: dict[str, str]) -> str:
    seg = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJFUzI1NiJ9.{seg}.sig"


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3({})


@pytest.fixture
def box(s3: FakeS3) -> IdeaBox:
    return IdeaBox(s3, "b", ttl_s=0)


@pytest.fixture
def client(s3: FakeS3, box: IdeaBox) -> TestClient:
    cfg = SiteConfig(lab="off", reports_bucket="b")
    return TestClient(create_app(cfg, library=Library(s3, "b"), ideas=box))


def test_validate() -> None:
    assert validate("  Try  QQQ ", "research", " x ", "Wes") == ("Try QQQ", "research", "x", "Wes")
    with pytest.raises(ValueError):
        validate("hi", "research", "", "")
    with pytest.raises(ValueError):
        validate("A fine title", "nope", "", "")
    with pytest.raises(ValueError):
        validate("A fine title", "feature", "x" * 4001, "")


def test_submit_list_and_status(box: IdeaBox, s3: FakeS3) -> None:
    a = box.submit(title="First", kind="research", details="why", author="Dad", email="d@x.com")
    b = box.submit(title="Second", kind="feature", details="", author="", email="wes@x.com")
    assert a.id != b.id and s3.objects[f"ideas/{a.id}.json"]
    assert b.author == "wes"  # from the email when no name was given
    assert [i.title for i in box.list()] == ["Second", "First"]  # newest first
    done = box.set_status(a.id, "done", "Shipped on the playbook page.")
    assert done.status == "done" and done.note.startswith("Shipped") and done.updated_at
    assert box.get(a.id) == done
    with pytest.raises(ValueError):
        box.set_status(a.id, "maybe")
    with pytest.raises(KeyError):
        box.set_status("nope", "done")
    assert box.get("nope") is None


def test_page_lists_and_takes_submissions(client: TestClient, box: IdeaBox) -> None:
    r = client.get("/ideas")
    assert r.status_code == 200 and "Nothing yet" in r.text and 'name="author"' in r.text

    hdr = {"x-amzn-oidc-data": _jwt({"email": "wes@x.com", "name": "Wes"})}
    r = client.post(
        "/ideas",
        data={"title": "Test the flip on QQQ", "kind": "research", "details": "line 1\nline 2"},
        headers=hdr,
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/ideas?sent=")
    ideas = box.list()
    assert len(ideas) == 1 and ideas[0].author == "Wes" and ideas[0].email == "wes@x.com"
    assert ideas[0].details == "line 1\nline 2"

    r = client.get(r.headers["location"], headers=hdr)
    assert "Test the flip on QQQ" in r.text and "Thanks" in r.text
    assert 'name="author"' not in r.text  # signed in: no name box
    assert "Submitting as Wes" in r.text

    r = client.post("/ideas", data={"title": "x", "kind": "research"}, headers=hdr)
    assert r.status_code == 400 and "Give the idea a title" in r.text
    assert len(box.list()) == 1

    r = client.post(
        "/ideas",
        data={"title": "Evil", "kind": "research"},
        headers={**hdr, "sec-fetch-site": "cross-site"},
    )
    assert r.status_code == 403


def test_status_shows_on_the_page(client: TestClient, box: IdeaBox) -> None:
    i = box.submit(title="Dark mode charts", kind="feature", details="", author="Dad", email="")
    box.set_status(i.id, "planned", "Next week.")
    r = client.get("/ideas")
    assert "Planned" in r.text and "Next week." in r.text and "Dark mode charts" in r.text
