"""The thread (ADR-0011): a write-up names its runs, study and ideas; the lab and the
ideas box read the same links back. No network."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ehf.tests.test_lab import (
    bars,  # noqa: F401
    world,  # noqa: F401  (fixture: a store with one run + one card)
)
from ehf.tests.test_site import META, FakeS3
from ehf.web.app import create_app
from ehf.web.config import SiteConfig
from ehf.web.ideas import IdeaBox
from ehf.web.lab.data import benchmark_symbol
from ehf.web.library import Library, parse_meta


def test_parse_meta_reads_the_thread() -> None:
    raw = {
        **META,
        "runs": ["abcdef0123456789", {"hash": "0123456789ABCDEF", "label": "the dial"}],
        "study": "Leverage",
        "ideas": ["20261103-214217-9cd512"],
    }
    post = parse_meta(raw)
    assert [r.hash for r in post.runs] == ["abcdef0123456789", "0123456789abcdef"]
    assert post.runs[1].label == "the dial" and post.study == "leverage"
    assert post.ideas == ("20261103-214217-9cd512",) and post.links_run("0123456789abcdef")
    assert parse_meta(META).runs == () and parse_meta(META).study == ""


@pytest.mark.parametrize(
    "bad",
    [
        {"runs": ["nope"]},
        {"runs": [{"hash": "abcdef012345678"}]},
        {"study": "has space"},
        {"ideas": ["../x"]},
    ],
)
def test_parse_meta_rejects_bad_links(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_meta({**META, **bad})


def test_library_reverse_lookups() -> None:
    a = {
        **META,
        "slug": "a",
        "runs": ["abcdef0123456789"],
        "study": "leverage",
        "ideas": ["20261103-214217-9cd512"],
    }
    b = {**META, "slug": "b", "study": "leverage"}
    s3 = FakeS3(
        {
            "library/a/meta.json": json.dumps(a).encode(),
            "library/b/meta.json": json.dumps(b).encode(),
        }
    )
    lib = Library(s3, "bucket")
    assert [p.slug for p in lib.posts_for_run("ABCDEF0123456789")] == ["a"]
    assert {p.slug for p in lib.posts_for_study("leverage")} == {"a", "b"}
    assert lib.posts_for_study("") == [] and lib.posts_for_run("0000000000000000") == []
    assert lib.post_for_idea("20261103-214217-9cd512").slug == "a"  # type: ignore[union-attr]
    assert lib.post_for_idea("20261103-000000-000000") is None


def test_benchmark_prefers_a_declared_benchmark() -> None:
    assert (
        benchmark_symbol({"params": {"asset": "TQQQ.X", "base": "QQQ", "benchmark": "qqq"}})
        == "QQQ"
    )
    assert benchmark_symbol({"params": {"asset": "SPY"}}) == "SPY"


def test_ideas_take_a_result_link() -> None:
    s3 = FakeS3({})
    box = IdeaBox(s3, "b", ttl_s=0)
    idea = box.submit(
        title="Leverage vs plain", kind="research", details="", author="Dad", email=""
    )
    done = box.set_status(idea.id, "done", "Written up.", "/library/leverage")
    assert done.link == "/library/leverage" and box.get(idea.id).link == "/library/leverage"  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        box.set_status(idea.id, "done", link="https://elsewhere.example")
    cfg = SiteConfig(lab="off", reports_bucket="b")
    client = TestClient(create_app(cfg, library=Library(s3, "b"), ideas=box))
    html = client.get("/ideas").text
    assert 'href="/library/leverage"' in html and "Read the result" in html


def _thread_world(world: dict[str, Any]) -> tuple[TestClient, str, IdeaBox]:  # noqa: F811
    h = world["hash"]
    s3 = FakeS3({})
    box = IdeaBox(s3, "b", ttl_s=0)
    idea = box.submit(
        title="Try the dial on QQQ", kind="research", details="", author="Dad", email=""
    )
    meta = {
        **META,
        "slug": "tsmom-note",
        "title": "A note on momentum",
        "runs": [{"hash": h, "label": "the headline run"}],
        "study": "",
        "ideas": [idea.id],
    }
    s3.objects["library/tsmom-note/meta.json"] = json.dumps(meta).encode()
    s3.objects["library/tsmom-note/results.pdf"] = b"%PDF"
    cfg = SiteConfig(lab="on", reports_bucket="b")
    app = create_app(
        cfg,
        library=Library(s3, "b"),
        lab=(world["data"], world["play"], world["charts"]),
        ideas=box,
    )
    return TestClient(app), h, box


def test_run_page_carries_its_writeup_and_idea(world: dict[str, Any]) -> None:  # noqa: F811
    client, h, _ = _thread_world(world)
    p = client.get(f"/lab/api/run/{h}").json()
    assert len(p["writeups"]) == 1
    w = p["writeups"][0]
    assert w["slug"] == "tsmom-note" and w["how"] == "run" and w["label"] == "the headline run"
    assert w["ideas"][0]["title"] == "Try the dial on QQQ" and w["ideas"][0]["author"] == "Dad"
    html = client.get(f"/lab/run/{h}").text
    assert "run-root" in html  # the page shells; the card is drawn from the payload


def test_post_page_shows_the_runs_behind_it(world: dict[str, Any]) -> None:  # noqa: F811
    client, h, _ = _thread_world(world)
    html = client.get("/library/tsmom-note").text
    assert "The backtests behind this write-up" in html
    assert f'href="/lab/run/{h}"' in html and "the headline run" in html
    assert "Prompted by" in html and "Try the dial on QQQ" in html and "(Dad)" in html
    assert "CAGR" in html  # the lab was up, so the headline numbers came along


def test_post_page_without_the_lab_still_links(world: dict[str, Any]) -> None:  # noqa: F811
    h = world["hash"]
    s3 = FakeS3({})
    meta = {**META, "slug": "n", "runs": [h], "study": "leverage"}
    s3.objects["library/n/meta.json"] = json.dumps(meta).encode()
    client = TestClient(
        create_app(
            SiteConfig(lab="off", reports_bucket="b"),
            library=Library(s3, "b"),
            ideas=IdeaBox(s3, "b"),
        )
    )
    html = client.get("/library/n").text
    assert f'href="/lab/run/{h}"' in html and "the lab is not mounted here" in html


def test_studies_table_links_the_study_writeup(world: dict[str, Any]) -> None:  # noqa: F811
    s3 = FakeS3({})
    card = world["data"].scorecard_dir / "wf_tsmom_abcd1234.json"
    payload = json.loads(card.read_text())
    payload["study"] = "momentum"
    card.write_text(json.dumps(payload))
    meta = {**META, "slug": "mom", "title": "Momentum, tested", "study": "momentum"}
    s3.objects["library/mom/meta.json"] = json.dumps(meta).encode()
    client = TestClient(
        create_app(
            SiteConfig(lab="on", reports_bucket="b"),
            library=Library(s3, "b"),
            lab=(world["data"], world["play"], world["charts"]),
            ideas=IdeaBox(s3, "b"),
        )
    )
    html = client.get("/lab").text
    assert "momentum" in html and 'href="/library/mom"' in html
    # and the stitched run resolves its write-up through the study
    p = client.get(f"/lab/api/run/{world['hash']}").json()
    assert p["study"]["study"] == "momentum" and p["writeups"][0]["how"] == "study"
    assert date.fromisoformat(p["writeups"][0]["date"])
