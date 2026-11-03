"""The playbook over synthetic SPY / VIX / VIX3M bars. No network."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ehf.tests.test_lab import FakeProvider, synthetic_bars
from ehf.tests.test_site import FakeS3
from ehf.web.app import create_app
from ehf.web.config import SiteConfig
from ehf.web.lab.charts import Charts
from ehf.web.lab.data import LabData
from ehf.web.lab.playground import Playground
from ehf.web.library import Library
from ehf.web.playbook import RULES, Playbook, combine, dial_label, switches


def test_combine_rules() -> None:
    assert combine("IN", "IN")[0] == "IN"
    assert combine("OUT", "OUT")[0] == "OUT"
    assert combine("IN", "OUT")[0] == "HALF"
    assert combine("HALF", "IN")[0] == "HALF"
    assert combine("OUT", "IN")[0] == "HALF"
    assert combine("unknown", "IN")[0] == "unknown"
    assert "strongest" in combine("OUT", "OUT")[1]


def test_dial_label_and_switches() -> None:
    assert [dial_label(v) for v in (1.0, 0.5, 0.0, None)] == ["IN", "HALF", "OUT", "unknown"]
    times = [f"2026-01-0{i}" for i in range(1, 8)]
    labels = ["IN", "IN", "HALF", "HALF", "OUT", "IN", "IN"]
    sw = switches(times, labels)
    assert [s["date"] for s in sw] == ["2026-01-06", "2026-01-05", "2026-01-03"]
    assert sw[0] == {"date": "2026-01-06", "from": "OUT", "to": "IN"}
    assert switches(times, labels, limit=1) == [sw[0]]


@pytest.fixture(scope="module")
def provider() -> FakeProvider:
    bars = synthetic_bars(("SPY", "^VIX", "^VIX3M"), "2019-01-02", "2023-12-29")
    return FakeProvider(bars)


def test_refresh_rebuilds_after_a_minute(provider: FakeProvider, monkeypatch: Any) -> None:
    pb = Playbook(provider, None, ttl_s=900)
    clock = [5_000_000.0]
    monkeypatch.setattr("ehf.web.playbook.time.time", lambda: clock[0])
    builds = []
    monkeypatch.setattr(pb, "build", lambda end: builds.append(end) or {"n": len(builds)})
    assert pb.today(date(2023, 12, 29)) == {"n": 1}
    assert pb.today(date(2023, 12, 29), refresh=True) == {"n": 1}  # within the minute
    clock[0] += 61
    assert pb.today(date(2023, 12, 29), refresh=True) == {"n": 2}
    clock[0] += 61
    assert pb.today(date(2023, 12, 29)) == {"n": 2}  # plain views honour the 15-minute cache


def test_payload_shape(provider: FakeProvider, tmp_path: Any) -> None:
    from ehf.research.results.store import LocalResultStore

    data = LabData(
        store=LocalResultStore(tmp_path / "r"), provider=provider, scorecard_dir=tmp_path
    )
    pb = Playbook(provider, data, ttl_s=60)
    p = pb.build(date(2023, 12, 29))
    assert p["as_of"] == "2023-12-29" and p["asset"] == "SPY"
    assert p["built_at"].endswith("+00:00") and p["live"] == {}  # FakeProvider has no quotes
    assert p["combined"]["state"] in {"IN", "HALF", "OUT"}
    assert [r["key"] for r in p["rules"]] == [r.key for r in RULES]
    for r in p["rules"]:
        assert r["state"] in {"IN", "HALF", "OUT"}
        assert r["days_in_state"] >= 1 and r["since"] <= p["as_of"]
        n = len(r["series"]["time"])
        assert (
            n
            == len(r["series"]["reading"])
            == len(r["series"]["exposure"])
            == len(p["series"]["time"])
        )
        assert 0 < n <= 3 * 252 + 5
        assert r["study"] is None  # no scorecard on file in this world
        assert r["exposure"] in (0.0, 0.5, 1.0)
    assert p["market"]["vix"] is not None and p["market"]["close"] is not None
    # cached: the second call is free and identical
    first = pb.today(date(2023, 12, 29))
    calls = provider.calls
    assert pb.today(date(2023, 12, 29)) is first and provider.calls == calls


def test_page_and_api(provider: FakeProvider, tmp_path: Any) -> None:
    from ehf.research.results.store import LocalResultStore

    data = LabData(
        store=LocalResultStore(tmp_path / "r"), provider=provider, scorecard_dir=tmp_path
    )
    lab = (data, Playground(provider), Charts(provider))
    cfg = SiteConfig(lab="on", reports_bucket="b")
    app = create_app(cfg, library=Library(FakeS3({}), "b"), lab=lab)
    c = TestClient(app)
    r = c.get("/playbook")
    assert r.status_code == 200
    assert (
        "Surprise count" in r.text and "Fear curve flip" in r.text and "EHF.playbookPage" in r.text
    )
    assert 'href="/playbook"' in c.get("/").text  # nav + card
    j = c.get("/playbook/api/today").json()
    assert j["combined"]["state"] in {"IN", "HALF", "OUT"}


def test_page_without_the_lab_is_absent() -> None:
    app = create_app(SiteConfig(lab="off"), library=Library(FakeS3({}), "b"))
    c = TestClient(app)
    assert c.get("/playbook").status_code == 404
    assert 'href="/playbook"' not in c.get("/").text
    assert c.get("/ideas").status_code == 200  # the ideas box needs no research code
