"""A lazy, read-only S3 mirror of ``var/`` for the deployed lab (ADR-0009).

The Studio owns ``var/`` and copies it to ``s3://<bucket>/<prefix>/`` every
night (``python -m ehf.research.jobs.sync_var``). The container starts with an
empty scratch volume (``EHF_VAR_DIR``) and fills it from the bucket:

* at startup, the small things everything needs — the results database, the
  scorecards, the Cboe cache — and, in a background thread, the daily bar cache;
* on first touch, anything else — a run's equity/trade parquet when its page
  opens, a symbol's bars when the playground asks for them;
* every ``max_age_s`` (an hour), a re-check of the database and the scorecards
  against the bucket, so the Studio's overnight work shows up by morning.

Nothing is ever deleted here and nothing is written back, with one exception:
bars the container had to fetch from FMP because neither disk nor bucket had
them are uploaded, so the next container does not fetch them again. Playground
runs are never logged from the site; the Studio is the only writer of record.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ehf.research.data.cache import BarCache
from ehf.research.data.schema import Timeframe
from ehf.research.results.store import LocalResultStore
from ehf.web.lab.charts import Charts
from ehf.web.lab.data import LabData
from ehf.web.lab.playground import Playground

log = logging.getLogger(__name__)

ETAG_SUFFIX = ".s3etag"  # sidecar next to each mirrored file: the object's ETag when pulled
SQLITE_SIDECARS = ("-wal", "-shm")  # must not survive a replaced database file
STARTUP_TREES = ("scorecards", "cache/cboe")
BACKGROUND_TREES = ("cache/fmp/1day",)


def is_missing(exc: BaseException) -> bool:
    """True for boto3's 'no such key' flavours (NoSuchKey, 404, NotFound)."""
    resp = getattr(exc, "response", None) or {}
    code = str(resp.get("Error", {}).get("Code", ""))
    status = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"NoSuchKey", "NotFound", "404"} or status == 404


class S3Mirror:
    """``root/<rel>`` <- ``s3://bucket/<prefix>/<rel>``, pulled when asked for."""

    def __init__(
        self,
        client: Any,
        bucket: str,
        prefix: str,
        root: str | Path,
        *,
        max_age_s: float = 3600.0,
        workers: int = 16,
    ) -> None:
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.root = Path(root)
        self.max_age_s = max_age_s
        self.workers = workers
        self.root.mkdir(parents=True, exist_ok=True)
        self._tree_checked: dict[str, float] = {}
        self._lock = threading.Lock()
        for part in self.root.rglob("*.part"):  # downloads a previous process left half-done
            part.unlink(missing_ok=True)

    # -- addressing --------------------------------------------------------------------

    def key(self, rel: str) -> str:
        rel = rel.strip("/")
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def path(self, rel: str) -> Path:
        return self.root / rel.strip("/")

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    # -- single files ------------------------------------------------------------------

    def ensure(self, rel: str) -> Path | None:
        """The local file, downloading it first if it is not on disk. None = not in S3."""
        p = self.path(rel)
        if p.exists():
            return p
        return self._download(rel, p)

    def refresh(self, rel: str) -> Path | None:
        """``ensure`` plus: once the local copy is ``max_age_s`` old, re-pull it if the
        object's ETag moved. Cheap when fresh (one ``stat``), one HEAD per hour otherwise."""
        p = self.path(rel)
        if not p.exists():
            return self._download(rel, p)
        if time.time() - p.stat().st_mtime < self.max_age_s:
            return p
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=self.key(rel))
        except Exception as exc:
            if not is_missing(exc):
                log.warning("mirror: HEAD %s failed: %s", rel, exc)
            os.utime(p)  # try again in max_age_s, not on every request
            return p
        if head.get("ETag") == read_etag(p):
            os.utime(p)  # unchanged: wait another max_age_s before looking again
            return p
        return self._download(rel, p)

    def put(self, rel: str) -> None:
        """Upload the local file (bars fetched on demand; see the module docstring)."""
        p = self.path(rel)
        self.client.upload_file(str(p), self.bucket, self.key(rel))

    def _download(self, rel: str, p: Path) -> Path | None:
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=self.key(rel))
        except Exception as exc:
            if is_missing(exc):
                return None
            raise
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".part")
        tmp = Path(tmp_name)
        try:
            body = resp["Body"]
            with os.fdopen(fd, "wb") as fh:
                for chunk in iter(lambda: body.read(1 << 20), b""):
                    fh.write(chunk)
            for side in SQLITE_SIDECARS:
                p.with_name(p.name + side).unlink(missing_ok=True)
            os.replace(tmp, p)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        write_etag(p, str(resp.get("ETag", "")))
        return p

    # -- whole prefixes ----------------------------------------------------------------

    def list_tree(self, rel_prefix: str) -> dict[str, tuple[int, str]]:
        """``rel -> (size, etag)`` for every object under ``rel_prefix/``."""
        key_prefix = self.key(rel_prefix) + "/"
        out: dict[str, tuple[int, str]] = {}
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(self.prefix) + 1 :] if self.prefix else obj["Key"]
                out[rel] = (int(obj.get("Size", 0)), str(obj.get("ETag", "")))
        return out

    def pull_tree(self, rel_prefix: str) -> int:
        """Download everything under ``rel_prefix/`` that is missing or changed.
        Returns how many files were pulled."""
        remote = self.list_tree(rel_prefix)
        todo: list[str] = []
        for rel, (size, etag) in remote.items():
            p = self.path(rel)
            if p.exists() and p.stat().st_size == size and read_etag(p) == etag:
                continue
            todo.append(rel)
        if not todo:
            return 0

        def one(rel: str) -> bool:
            try:
                return self._download(rel, self.path(rel)) is not None
            except Exception as exc:  # keep going; the file stays lazy
                log.warning("mirror: pull %s failed: %s", rel, exc)
                return False

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            pulled = sum(pool.map(one, todo))
        log.info("mirror: pulled %d/%d files under %s/", pulled, len(todo), rel_prefix)
        return pulled

    def pull_tree_if_stale(self, rel_prefix: str) -> None:
        """``pull_tree`` at most once per ``max_age_s`` (the scorecards on each /lab)."""
        with self._lock:
            last = self._tree_checked.get(rel_prefix, 0.0)
            if time.time() - last < self.max_age_s:
                return
            self._tree_checked[rel_prefix] = time.time()
        try:
            self.pull_tree(rel_prefix)
        except Exception as exc:
            log.warning("mirror: refresh of %s/ failed: %s", rel_prefix, exc)

    def pull_in_background(self, rel_prefixes: Iterable[str]) -> threading.Thread:
        prefixes = list(rel_prefixes)

        def run() -> None:
            for rel_prefix in prefixes:
                try:
                    self.pull_tree(rel_prefix)
                except Exception as exc:
                    log.warning("mirror: background pull of %s/ failed: %s", rel_prefix, exc)

        t = threading.Thread(target=run, name="ehf-mirror-warm", daemon=True)
        t.start()
        return t


def read_etag(p: Path) -> str:
    try:
        return p.with_name(p.name + ETAG_SUFFIX).read_text().strip()
    except OSError:
        return ""


def write_etag(p: Path, etag: str) -> None:
    try:
        p.with_name(p.name + ETAG_SUFFIX).write_text(etag)
    except OSError as exc:  # the file itself is fine; it will just be re-pulled later
        log.warning("mirror: could not record ETag for %s: %s", p, exc)


# -- the research objects, mirrored -----------------------------------------------------


class MirroredResultStore(LocalResultStore):
    """The Studio's SQLite + parquet store, pulled from S3 as it is read."""

    def __init__(self, mirror: S3Mirror, rel: str = "results") -> None:
        self.mirror = mirror
        self.rel = rel
        mirror.ensure(f"{rel}/ehf.db")  # before the base class creates an empty one
        super().__init__(mirror.path(rel))

    def _connect(self) -> Any:
        self.mirror.refresh(f"{self.rel}/ehf.db")
        return super()._connect()

    def read_run(self, config_hash: str) -> dict[str, Any]:
        for name in ("equity.parquet", "trades.parquet"):
            self.mirror.ensure(f"{self.rel}/{config_hash}/{name}")
        return super().read_run(config_hash)


class MirroredBarCache(BarCache):
    """``BarCache`` whose misses look in S3 before the vendor, and whose writes go back up."""

    def __init__(
        self, mirror: S3Mirror, vendor: str = "fmp", *, write_through: bool = True, **kw: Any
    ) -> None:
        super().__init__(vendor, cache_dir=mirror.path("cache"), **kw)
        self.mirror = mirror
        self.write_through = write_through

    def read(self, symbol: str, timeframe: Timeframe, adjusted: bool) -> pd.DataFrame | None:
        path = self._path(symbol, timeframe, adjusted)
        if not path.exists():
            self.mirror.ensure(self.mirror.rel(path))
        return super().read(symbol, timeframe, adjusted)

    def write(self, df: pd.DataFrame, timeframe: Timeframe, adjusted: bool) -> None:
        super().write(df, timeframe, adjusted)
        if not self.write_through:
            return
        for symbol in df["symbol"].unique():
            path = self._path(str(symbol), timeframe, adjusted)
            try:
                self.mirror.put(self.mirror.rel(path))
            except Exception as exc:
                log.warning("mirror: upload of %s failed: %s", path.name, exc)


@dataclass
class MirroredLabData(LabData):
    """``LabData`` that re-checks the scorecards against S3 once an hour."""

    mirror: S3Mirror | None = None

    def scorecards(self) -> list[Any]:
        if self.mirror is not None:
            self.mirror.pull_tree_if_stale("scorecards")
        return super().scorecards()


def mirrored_components(
    *,
    bucket: str,
    prefix: str = "var",
    region: str | None = None,
    root: str | Path | None = None,
    client: Any | None = None,
    warm: bool = True,
) -> tuple[LabData, Playground, Charts]:
    """The lab's three components over an S3 mirror of ``var/`` (read-only)."""
    from ehf.research.data.fmp_provider import FMPProvider
    from ehf.research.paths import VAR_DIR

    if client is None:
        import boto3

        client = boto3.client("s3", region_name=region)
    mirror = S3Mirror(client, bucket, prefix, root if root is not None else VAR_DIR)
    for tree in STARTUP_TREES:
        mirror.pull_tree(tree)
    store = MirroredResultStore(mirror)
    provider = FMPProvider(MirroredBarCache(mirror))
    data = MirroredLabData(
        store=store, provider=provider, scorecard_dir=mirror.path("scorecards"), mirror=mirror
    )
    play = Playground(provider, registry=None)  # never logs: the Studio is the writer
    if warm:
        mirror.pull_in_background(BACKGROUND_TREES)
    log.info("lab: mirroring s3://%s/%s into %s", bucket, mirror.prefix, mirror.root)
    return data, play, Charts(provider)
