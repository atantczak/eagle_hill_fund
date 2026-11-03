"""The S3 mirror behind the deployed lab, and the Studio job that fills the bucket.

A dict stands in for S3; nothing touches the network."""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ehf.research.data.schema import Timeframe
from ehf.research.jobs.sync_var import checkpoint_sqlite, sync
from ehf.research.results.store import LocalResultStore
from ehf.web.lab.mirror import (
    ETAG_SUFFIX,
    MirroredBarCache,
    MirroredLabData,
    MirroredResultStore,
    S3Mirror,
    mirrored_components,
    read_etag,
)


class _Missing(Exception):
    def __init__(self, code: str = "NoSuchKey") -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": 404}}


class _Pages:
    def __init__(self, s3: FakeS3) -> None:
        self.s3 = s3

    def paginate(self, **kw: Any) -> list[dict[str, Any]]:
        self.s3.calls["list"] += 1
        keys = sorted(k for k in self.s3.objects if k.startswith(kw["Prefix"]))
        contents = [
            {
                "Key": k,
                "Size": len(self.s3.objects[k]),
                "ETag": self.s3.etag(k),
                "LastModified": self.s3.stamps[k],
            }
            for k in keys
        ]
        return [{"Contents": contents}] if contents else [{}]


class FakeS3:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects: dict[str, bytes] = dict(objects or {})
        self.stamps: dict[str, datetime] = {k: datetime.now(UTC) for k in self.objects}
        self.calls = {"get": 0, "head": 0, "put": 0, "list": 0}

    def etag(self, key: str) -> str:
        return f'"{hash(self.objects[key]) & 0xFFFFFFFF:08x}"'

    def get_object(self, **kw: Any) -> dict[str, Any]:
        self.calls["get"] += 1
        key = kw["Key"]
        if key not in self.objects:
            raise _Missing()
        import io

        return {"Body": io.BytesIO(self.objects[key]), "ETag": self.etag(key)}

    def head_object(self, **kw: Any) -> dict[str, Any]:
        self.calls["head"] += 1
        key = kw["Key"]
        if key not in self.objects:
            raise _Missing("404")
        return {"ETag": self.etag(key), "ContentLength": len(self.objects[key])}

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.calls["put"] += 1
        self.objects[key] = Path(filename).read_bytes()
        self.stamps[key] = datetime.now(UTC)

    def get_paginator(self, name: str) -> _Pages:
        assert name == "list_objects_v2"
        return _Pages(self)


def _age(p: Path, seconds: float) -> None:
    old = time.time() - seconds
    os.utime(p, (old, old))


# -- S3Mirror ---------------------------------------------------------------------------


def test_ensure_pulls_once_and_records_etag(tmp_path: Path) -> None:
    s3 = FakeS3({"var/scorecards/wf_x.json": b"{}"})
    m = S3Mirror(s3, "b", "var", tmp_path)
    p = m.ensure("scorecards/wf_x.json")
    assert p is not None and p.read_bytes() == b"{}"
    assert read_etag(p) == s3.etag("var/scorecards/wf_x.json")
    assert m.ensure("scorecards/wf_x.json") == p
    assert s3.calls["get"] == 1  # second call served from disk
    assert m.ensure("scorecards/nope.json") is None
    assert not list(tmp_path.glob("**/*.part"))


def test_refresh_checks_hourly_and_replaces_changed_files(tmp_path: Path) -> None:
    s3 = FakeS3({"var/results/ehf.db": b"v1"})
    m = S3Mirror(s3, "b", "var", tmp_path, max_age_s=100)
    p = m.refresh("results/ehf.db")
    assert p is not None and p.read_bytes() == b"v1"
    assert m.refresh("results/ehf.db") == p and s3.calls["head"] == 0  # fresh: no HEAD

    _age(p, 1000)
    m.refresh("results/ehf.db")
    assert s3.calls["head"] == 1 and s3.calls["get"] == 1  # unchanged: HEAD only
    assert time.time() - p.stat().st_mtime < 5  # ...and the clock restarts

    s3.objects["var/results/ehf.db"] = b"v2"
    wal = tmp_path / "results" / "ehf.db-wal"
    wal.write_bytes(b"stale journal")
    _age(p, 1000)
    m.refresh("results/ehf.db")
    assert p.read_bytes() == b"v2" and not wal.exists()

    del s3.objects["var/results/ehf.db"]
    _age(p, 1000)
    assert m.refresh("results/ehf.db") == p and p.read_bytes() == b"v2"  # gone upstream: keep


def test_pull_tree_skips_unchanged(tmp_path: Path) -> None:
    s3 = FakeS3({"var/scorecards/a.json": b"a", "var/scorecards/b.json": b"b", "var/x": b"x"})
    m = S3Mirror(s3, "b", "var", tmp_path)
    assert m.pull_tree("scorecards") == 2
    assert sorted(p.name for p in (tmp_path / "scorecards").glob("*.json")) == ["a.json", "b.json"]
    assert m.pull_tree("scorecards") == 0
    s3.objects["var/scorecards/a.json"] = b"aa"
    assert m.pull_tree("scorecards") == 1
    assert (tmp_path / "scorecards" / "a.json").read_bytes() == b"aa"
    assert m.pull_tree("nothing-here") == 0


def test_pull_tree_if_stale_rate_limits(tmp_path: Path) -> None:
    s3 = FakeS3({"var/scorecards/a.json": b"a"})
    m = S3Mirror(s3, "b", "var", tmp_path, max_age_s=100)
    m.pull_tree_if_stale("scorecards")
    m.pull_tree_if_stale("scorecards")
    assert s3.calls["list"] == 1


# -- the mirrored research objects -----------------------------------------------------


def _studio_store(root: Path) -> tuple[LocalResultStore, str]:
    store = LocalResultStore(root)
    record = {
        "config_hash": "abc123",
        "strategy": "vix_vote",
        "params_json": "{}",
        "universe": "SPY",
        "data_start": "2020-01-01",
        "data_end": "2020-12-31",
        "timeframe": "1day",
        "initial_capital": 100000.0,
        "cost_model_json": "{}",
        "fill_model": "next_open",
        "spec_json": '{"strategy": "vix_vote"}',
        "created_at": "2026-11-03T00:00:00",
    }
    equity = pd.DataFrame(
        {"timestamp": pd.date_range("2020-01-01", periods=3), "equity": [1, 2, 3]}
    )
    trades = pd.DataFrame({"symbol": ["SPY"], "qty": [1.0]})
    store.write_run(record, {"sharpe": 1.5}, equity, trades)
    return store, "abc123"


def _upload_tree(s3: FakeS3, root: Path, prefix: str = "var") -> None:
    for p in root.rglob("*"):
        if p.is_file() and not p.name.endswith(("-wal", "-shm")):
            s3.upload_file(str(p), "b", f"{prefix}/{p.relative_to(root).as_posix()}")


def test_mirrored_store_reads_lazily(tmp_path: Path) -> None:
    studio = tmp_path / "studio"
    _studio_store(studio / "results")
    checkpoint_sqlite(studio / "results" / "ehf.db")
    s3 = FakeS3()
    _upload_tree(s3, studio)

    m = S3Mirror(s3, "b", "var", tmp_path / "container")
    store = MirroredResultStore(m)
    assert (tmp_path / "container" / "results" / "ehf.db").exists()
    assert store.has_run("abc123") and not store.has_run("nope")
    assert not (tmp_path / "container" / "results" / "abc123").exists()  # not yet
    run = store.read_run("abc123")
    assert run["metrics"] == {"sharpe": 1.5} and list(run["equity"]["equity"]) == [1, 2, 3]
    assert (tmp_path / "container" / "results" / "abc123" / "trades.parquet").exists()
    assert not store.query("WHERE strategy = ?", ("vix_vote",)).empty


def test_mirrored_store_starts_empty_without_a_bucket_copy(tmp_path: Path) -> None:
    store = MirroredResultStore(S3Mirror(FakeS3(), "b", "var", tmp_path))
    assert store.query().empty


def test_mirrored_cache_pulls_and_pushes(tmp_path: Path, canonical_bars: pd.DataFrame) -> None:
    aapl = canonical_bars[canonical_bars["symbol"] == "AAPL"].reset_index(drop=True)
    msft = canonical_bars[canonical_bars["symbol"] == "MSFT"].reset_index(drop=True)
    studio = tmp_path / "studio" / "cache" / "fmp" / "1day" / "adj"
    studio.mkdir(parents=True)
    aapl.to_parquet(studio / "AAPL.parquet", index=False)
    s3 = FakeS3()
    _upload_tree(s3, tmp_path / "studio")

    m = S3Mirror(s3, "b", "var", tmp_path / "container")
    cache = MirroredBarCache(m)
    got = cache.read("AAPL", Timeframe.DAY, True)
    assert got is not None and len(got) == len(aapl) and s3.calls["get"] == 1
    cache.read("AAPL", Timeframe.DAY, True)
    assert s3.calls["get"] == 1  # on disk now
    assert cache.read("MSFT", Timeframe.DAY, True) is None and s3.calls["get"] == 2

    puts = s3.calls["put"]
    cache.write(msft, Timeframe.DAY, True)  # what the provider does after a vendor fetch
    assert "var/cache/fmp/1day/adj/MSFT.parquet" in s3.objects and s3.calls["put"] == puts + 1
    assert cache.covered_range("MSFT", Timeframe.DAY, True) is not None


def test_mirrored_components_are_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FMP_API_KEY", "test")
    studio = tmp_path / "studio"
    _studio_store(studio / "results")
    checkpoint_sqlite(studio / "results" / "ehf.db")
    (studio / "scorecards").mkdir()
    (studio / "scorecards" / "wf_vix_vote_1.json").write_text(
        '{"wf_id": "wf_vix_vote_1", "strategy": "vix_vote", "stitched_hash": "abc123", '
        '"oos_metrics": {}, "verdict": "pass", "notes": "run `abc123`"}'
    )
    s3 = FakeS3()
    _upload_tree(s3, studio)

    data, play, charts = mirrored_components(
        bucket="b", prefix="var", root=tmp_path / "container", client=s3, warm=False
    )
    assert isinstance(data, MirroredLabData)
    assert [s.wf_id for s in data.scorecards()] == ["wf_vix_vote_1"]
    assert play.registry is None
    assert data.store.has_run("abc123")
    assert charts.provider.cache.cache_dir == tmp_path / "container" / "cache"


# -- the Studio side: sync_var ---------------------------------------------------------


def test_sync_uploads_new_and_changed_files_only(tmp_path: Path) -> None:
    root = tmp_path / "var"
    _studio_store(root / "results")
    (root / "scorecards").mkdir()
    (root / "scorecards" / "wf_a.json").write_text("{}")
    (root / "results" / "ehf.db.part").write_bytes(b"half a download")  # never uploaded
    (root / "cache" / "fmp" / "1hour").mkdir(parents=True)
    (root / "cache" / "fmp" / "1hour" / "SPY.parquet").write_bytes(b"not included")
    s3 = FakeS3()

    dry = sync(s3, "b", root=root, dry_run=True)
    assert dry.uploaded == 4 and s3.calls["put"] == 0  # db + 2 parquet + scorecard

    rep = sync(s3, "b", root=root)
    assert rep.uploaded == 4 and rep.failed == [] and rep.skipped == 0
    assert set(s3.objects) == {
        "var/results/ehf.db",
        "var/results/abc123/equity.parquet",
        "var/results/abc123/trades.parquet",
        "var/scorecards/wf_a.json",
    }
    wal = root / "results" / "ehf.db-wal"
    assert not wal.exists() or wal.stat().st_size == 0  # checkpointed: the db file is whole

    again = sync(s3, "b", root=root)
    assert again.uploaded == 0 and again.skipped == 4

    time.sleep(0.01)
    (root / "scorecards" / "wf_a.json").write_text('{"verdict": "pass"}')
    changed = sync(s3, "b", root=root)
    assert changed.uploaded == 1
    assert s3.objects["var/scorecards/wf_a.json"] == b'{"verdict": "pass"}'


def test_checkpoint_is_safe_on_missing_db(tmp_path: Path) -> None:
    checkpoint_sqlite(tmp_path / "nope.db")
    db = tmp_path / "x.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (x)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()
    checkpoint_sqlite(db)
    assert sqlite3.connect(db).execute("SELECT x FROM t").fetchone() == (1,)


def test_etag_suffix_is_skipped_by_sync(tmp_path: Path) -> None:
    root = tmp_path / "var"
    (root / "scorecards").mkdir(parents=True)
    (root / "scorecards" / "wf_a.json").write_text("{}")
    (root / "scorecards" / f"wf_a.json{ETAG_SUFFIX}").write_text('"x"')
    rep = sync(FakeS3(), "b", root=root, include=("scorecards",))
    assert rep.uploaded == 1
