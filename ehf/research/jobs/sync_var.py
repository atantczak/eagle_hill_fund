"""Job: copy the Studio's ``var/`` (results, scorecards, price cache) up to S3.

python -m ehf.research.jobs.sync_var                 # everything the deployed lab reads
python -m ehf.research.jobs.sync_var --dry-run       # count what would be uploaded
python -m ehf.research.jobs.sync_var --include results,scorecards

The bucket is the system of record for the deployed lab (ADR-0009): the site
container mirrors ``s3://<bucket>/var/`` on demand (``ehf.web.lab.mirror``).
One direction only — nothing is ever deleted from the bucket — and a file is
uploaded when the bucket lacks it, has a different size, or was last written
before the local copy. The results database is checkpointed first so the
single ``ehf.db`` file is complete on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ehf.research.jobs import (
    AlreadyRunning,
    configure_logging,
    load_dotenv_quietly,
    single_instance,
    write_summary,
)
from ehf.research.paths import VAR_DIR

JOB = "sync_var"
DEFAULT_INCLUDE = ("results", "scorecards", "cache/fmp/1day", "cache/cboe")
SKIP_SUFFIXES = ("-wal", "-shm", ".part", ".s3etag", ".lock")


@dataclass
class SyncReport:
    scanned: int = 0
    uploaded: int = 0
    skipped: int = 0
    bytes: int = 0
    failed: list[str] = field(default_factory=list)
    seconds: float = 0.0
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "uploaded": self.uploaded,
            "skipped": self.skipped,
            "bytes": self.bytes,
            "failed": self.failed,
            "seconds": round(self.seconds, 1),
            "dry_run": self.dry_run,
        }


def checkpoint_sqlite(db: Path) -> None:
    """Fold the WAL into the main file so a copy of ``ehf.db`` alone is complete."""
    if not db.exists():
        return
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def local_files(root: Path, include: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for inc in include:
        base = root / inc
        if base.is_file():
            out.append(base)
            continue
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file() and not p.name.startswith(".") and not p.name.endswith(SKIP_SUFFIXES):
                out.append(p)
    return out


def remote_objects(
    client: Any, bucket: str, prefix: str, include: tuple[str, ...]
) -> dict[str, tuple[int, float]]:
    """``key -> (size, last_modified_epoch)`` under each included prefix."""
    out: dict[str, tuple[int, float]] = {}
    paginator = client.get_paginator("list_objects_v2")
    for inc in include:
        key_prefix = f"{prefix}/{inc}".strip("/")
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                out[obj["Key"]] = (int(obj.get("Size", 0)), obj["LastModified"].timestamp())
    return out


def sync(
    client: Any,
    bucket: str,
    *,
    prefix: str = "var",
    root: Path = VAR_DIR,
    include: tuple[str, ...] = DEFAULT_INCLUDE,
    dry_run: bool = False,
    workers: int = 16,
    log: Any = None,
) -> SyncReport:
    t0 = time.time()
    prefix = prefix.strip("/")
    report = SyncReport(dry_run=dry_run)
    if "results" in include:
        checkpoint_sqlite(root / "results" / "ehf.db")
    remote = remote_objects(client, bucket, prefix, include)
    todo: list[tuple[Path, str]] = []
    for p in local_files(root, include):
        report.scanned += 1
        key = f"{prefix}/{p.relative_to(root).as_posix()}".strip("/")
        st = p.stat()
        have = remote.get(key)
        if have is not None and have[0] == st.st_size and have[1] >= st.st_mtime:
            report.skipped += 1
            continue
        todo.append((p, key))
        report.bytes += st.st_size
    if log:
        log.info(
            "%d local files, %d in s3://%s/%s, %d to upload (%.0f MB)%s",
            report.scanned,
            len(remote),
            bucket,
            prefix,
            len(todo),
            report.bytes / 1e6,
            " [dry run]" if dry_run else "",
        )
    if dry_run:
        report.uploaded = len(todo)
        report.seconds = time.time() - t0
        return report

    def one(item: tuple[Path, str]) -> str | None:
        p, key = item
        try:
            client.upload_file(str(p), bucket, key)
            return None
        except Exception as exc:
            if log:
                log.warning("upload %s failed: %s", key, exc)
            return key

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for failed in pool.map(one, todo):
            if failed is None:
                report.uploaded += 1
            else:
                report.failed.append(failed)
    report.seconds = time.time() - t0
    return report


def default_bucket() -> str:
    explicit = os.getenv("EHF_VAR_S3_BUCKET") or os.getenv("EHF_RESULTS_S3_BUCKET")
    if explicit:
        return explicit
    import boto3

    account = boto3.client("sts").get_caller_identity()["Account"]
    return f"ehf-results-{account}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EHF: copy var/ up to S3 for the deployed lab")
    p.add_argument(
        "--bucket", default=None, help="default: $EHF_VAR_S3_BUCKET or ehf-results-<acct>"
    )
    p.add_argument("--prefix", default=os.getenv("EHF_VAR_S3_PREFIX", "var"))
    p.add_argument("--include", default=",".join(DEFAULT_INCLUDE), help="comma-separated subtrees")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    load_dotenv_quietly()
    log = configure_logging(JOB, verbose=args.verbose)
    import boto3

    bucket = args.bucket or default_bucket()
    region = os.getenv("AWS_REGION", "us-east-1")
    try:
        with single_instance(JOB):
            report = sync(
                boto3.client("s3", region_name=region),
                bucket,
                prefix=args.prefix,
                include=tuple(s.strip() for s in args.include.split(",") if s.strip()),
                dry_run=args.dry_run,
                workers=args.workers,
                log=log,
            )
    except AlreadyRunning as exc:
        log.warning("%s", exc)
        return 2
    summary = {"bucket": bucket, "prefix": args.prefix, **report.as_dict()}
    path = write_summary(JOB, summary)
    log.info("done in %.0fs; summary -> %s", report.seconds, path)
    print(json.dumps(summary, indent=1))
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
