#!/usr/bin/env python3
"""Operate the site: publish write-ups to the library, watch the service.

    python infra/aws/site/ops.py publish-report --slug vix-explainer --title "..." \
        --date 2026-11-02 --summary "..." --file var/reports/VIX_SPY_Explainer.pdf \
        [--file more.md] [--tags vix,spy] [--body-md notes.md] [--author "Andrew Antczak"] \
        [--study leverage] [--run HASH=label ...] [--idea ID ...]   # the thread (ADR-0011)
    python infra/aws/site/ops.py link --slug SLUG --study vix_overlay --run HASH=label --idea ID
    python infra/aws/site/ops.py list                      # what is in the library
    python infra/aws/site/ops.py unpublish --slug SLUG --yes
    python infra/aws/site/ops.py status                    # ECS deployments + target health
    python infra/aws/site/ops.py logs [--minutes 30]       # tail the container log
    python infra/aws/site/ops.py user --email X --name Y   # invite someone (see terraform.tfvars)
    python infra/aws/site/ops.py ideas                     # the ideas box (newest first)
    python infra/aws/site/ops.py ideas --set ID --status done --note "..." --link /library/SLUG

Files are stored under library/<slug>/ in the reports bucket next to a meta.json;
the site lists whatever is there (new posts show within a minute).
"""

from __future__ import annotations

import argparse
import getpass
import json
import mimetypes
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")

REGION = os.getenv("AWS_REGION", "us-east-1")
NAME = os.getenv("EHF_SITE_NAME", "ehf-site")
PREFIX = os.getenv("EHF_LIBRARY_PREFIX", "library").strip("/")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


def bucket_name() -> str:
    explicit = os.getenv("EHF_REPORTS_BUCKET")
    if explicit:
        return explicit
    account = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    return f"ehf-market-reports-{account}"


def _content_type(path: Path) -> str:
    known = {".pdf": "application/pdf", ".md": "text/markdown; charset=utf-8"}
    return known.get(
        path.suffix.lower(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )


def _run_links(items: list[str]) -> list[dict[str, str]]:
    """``HASH`` or ``HASH=label`` -> ``{"hash", "label"}``, validated like the site does."""
    sys.path.insert(0, str(ROOT))
    from ehf.web.library import HASH_RE

    out = []
    for item in items:
        h, _, label = item.partition("=")
        h = h.strip().lower()
        if not HASH_RE.match(h):
            raise SystemExit(f"not a run hash: {h!r} (16 hex characters)")
        out.append({"hash": h, "label": label.strip()})
    return out


def _thread_fields(args: argparse.Namespace) -> dict[str, object]:
    return {
        "study": (args.study or "").strip().lower(),
        "runs": _run_links(args.run or []),
        "ideas": [i.strip() for i in (args.idea or []) if i.strip()],
    }


def publish(args: argparse.Namespace) -> int:
    if not SLUG_RE.match(args.slug):
        print(f"slug must match {SLUG_RE.pattern}", file=sys.stderr)
        return 2
    bucket = bucket_name()
    s3 = boto3.client("s3", region_name=REGION)
    files = []
    for i, f in enumerate(args.file):
        path = Path(f)
        if not path.is_file():
            print(f"missing file {path}", file=sys.stderr)
            return 2
        key = f"{PREFIX}/{args.slug}/{path.name}"
        ct = _content_type(path)
        s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": ct})
        default_label = "Read the PDF" if ct == "application/pdf" else path.name
        files.append(
            {
                "name": path.name,
                "label": args.label[i] if i < len(args.label) else default_label,
                "content_type": ct,
                "primary": i == 0,
            }
        )
        print(f"uploaded s3://{bucket}/{key} ({path.stat().st_size / 1024:.0f} KB)")
    meta = {
        "slug": args.slug,
        "title": args.title,
        "date": args.date,
        "summary": args.summary,
        "tags": [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else [],
        "body_md": Path(args.body_md).read_text() if args.body_md else "",
        "files": files,
        "author": args.author,
        "published_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **_thread_fields(args),
    }
    _put_meta(s3, bucket, args.slug, meta)
    print(f"published {args.slug}: {args.title} -> https://eaglehillfund.com/library/{args.slug}")
    return 0


def _put_meta(s3: object, bucket: str, slug: str, meta: dict[str, object]) -> None:
    s3.put_object(  # type: ignore[attr-defined]
        Bucket=bucket,
        Key=f"{PREFIX}/{slug}/meta.json",
        Body=json.dumps(meta, indent=1).encode(),
        ContentType="application/json",
    )


def link(args: argparse.Namespace) -> int:
    """Attach (or replace) the study, runs and ideas behind an existing write-up."""
    bucket = bucket_name()
    s3 = boto3.client("s3", region_name=REGION)
    key = f"{PREFIX}/{args.slug}/meta.json"
    try:
        meta = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        print(f"no write-up under slug {args.slug!r}", file=sys.stderr)
        return 1
    fields = _thread_fields(args)
    if args.clear:
        meta.update(fields)
    else:
        if fields["study"]:
            meta["study"] = fields["study"]
        have = {r["hash"]: r for r in meta.get("runs", [])}
        for r in fields["runs"]:  # type: ignore[union-attr]
            have[r["hash"]] = r
        meta["runs"] = list(have.values())
        meta["ideas"] = list(dict.fromkeys([*meta.get("ideas", []), *fields["ideas"]]))  # type: ignore[misc]
    meta["linked_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    _put_meta(s3, bucket, args.slug, meta)
    runs = ", ".join(
        f"{r['hash']}{'=' + r['label'] if r.get('label') else ''}" for r in meta["runs"]
    )
    print(
        f"{args.slug}: study={meta.get('study') or '-'} runs=[{runs}] ideas={meta.get('ideas', [])}"
    )
    return 0


def list_library() -> int:
    bucket = bucket_name()
    s3 = boto3.client("s3", region_name=REGION)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{PREFIX}/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/meta.json"):
                meta = json.loads(s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())
                names = ", ".join(f["name"] for f in meta.get("files", []))
                print(f"{meta.get('date')}  {meta.get('slug'):30s}  {meta.get('title')}  [{names}]")
    return 0


def unpublish(args: argparse.Namespace) -> int:
    if not args.yes:
        print("refusing without --yes (this deletes the post and its files)", file=sys.stderr)
        return 2
    bucket = bucket_name()
    s3 = boto3.client("s3", region_name=REGION)
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{PREFIX}/{args.slug}/"
    )
    keys = [o["Key"] for page in pages for o in page.get("Contents", [])]
    if not keys:
        print("nothing under that slug")
        return 1
    s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in keys]})
    print(
        f"deleted {len(keys)} objects under {PREFIX}/{args.slug}/ (versioning keeps them 30 days)"
    )
    return 0


def status() -> int:
    ecs = boto3.client("ecs", region_name=REGION)
    svc = ecs.describe_services(cluster=NAME, services=[NAME])["services"]
    if not svc:
        print("service not found (terraform apply first)")
        return 1
    s = svc[0]
    print(
        f"{s['serviceName']}: {s['status']} running {s['runningCount']}/{s['desiredCount']} "
        f"pending {s['pendingCount']}"
    )
    for d in s["deployments"]:
        td = d["taskDefinition"].rsplit("/", 1)[-1]
        print(
            f"  deployment {d['status']:8s} {d.get('rolloutState', ''):12s} {td}  "
            f"running {d['runningCount']} failed {d.get('failedTasks', 0)}"
        )
    for e in s["events"][:3]:
        print(f"  {e['createdAt']:%Y-%m-%d %H:%M} {e['message']}")
    elb = boto3.client("elbv2", region_name=REGION)
    for tg in elb.describe_target_groups()["TargetGroups"]:
        if tg["TargetGroupName"] != NAME:
            continue
        for h in elb.describe_target_health(TargetGroupArn=tg["TargetGroupArn"])[
            "TargetHealthDescriptions"
        ]:
            t, th = h["Target"], h["TargetHealth"]
            print(f"  target {t['Id']}:{t['Port']} {th['State']} {th.get('Reason', '')}")
    return 0


def logs(args: argparse.Namespace) -> int:
    cw = boto3.client("logs", region_name=REGION)
    start = int((time.time() - args.minutes * 60) * 1000)
    kw: dict[str, object] = {
        "logGroupName": f"/ecs/{NAME}",
        "startTime": start,
        "interleaved": True,
    }
    while True:
        resp = cw.filter_log_events(**kw)
        for e in resp.get("events", []):
            stamp = datetime.fromtimestamp(e["timestamp"] / 1000, tz=UTC).strftime("%H:%M:%S")
            print(stamp, e["message"].rstrip())
        if "nextToken" not in resp:
            return 0
        kw["nextToken"] = resp["nextToken"]


def add_user(args: argparse.Namespace) -> int:
    cog = boto3.client("cognito-idp", region_name=REGION)
    pools = [p for p in cog.list_user_pools(MaxResults=20)["UserPools"] if p["Name"] == NAME]
    if not pools:
        print("user pool not found")
        return 1
    cog.admin_create_user(
        UserPoolId=pools[0]["Id"],
        Username=args.email,
        UserAttributes=[
            {"Name": "email", "Value": args.email},
            {"Name": "email_verified", "Value": "true"},
            {"Name": "name", "Value": args.name},
        ],
        DesiredDeliveryMediums=["EMAIL"],
    )
    print(f"invited {args.email}; a temporary password is on its way (add to terraform.tfvars too)")
    return 0


def reset_password(args: argparse.Namespace) -> int:
    """Give a user a temporary password by hand (eaglehillfund.com has no mailbox,
    so Cognito's own reset email never arrives). They choose a new one on next sign-in."""
    cog = boto3.client("cognito-idp", region_name=REGION)
    pools = [p for p in cog.list_user_pools(MaxResults=20)["UserPools"] if p["Name"] == NAME]
    if not pools:
        print("user pool not found")
        return 1
    pw = getpass.getpass("temporary password (12+ chars, upper, lower, digit; not echoed): ")
    if pw != getpass.getpass("again: "):
        print("passwords differ")
        return 1
    cog.admin_set_user_password(
        UserPoolId=pools[0]["Id"], Username=args.email, Password=pw, Permanent=False
    )
    print(f"{args.email}: sign in with that temporary password; Cognito will ask for a new one")
    return 0


def ideas(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT))
    from ehf.web.ideas import STATUSES, IdeaBox

    box = IdeaBox(boto3.client("s3", region_name=REGION), bucket_name(), ttl_s=0)
    if args.set:
        if not args.status:
            print(f"--status is required: {', '.join(STATUSES)}", file=sys.stderr)
            return 2
        idea = box.set_status(args.set, args.status, args.note, args.link)
        print(f"{idea.id}: {idea.status}  {idea.title}{'  -> ' + idea.link if idea.link else ''}")
        return 0
    rows = box.list()
    if args.status:
        rows = [i for i in rows if i.status == args.status]
    if args.json:
        print(json.dumps([i.__dict__ for i in rows], indent=1))
        return 0
    for i in rows:
        print(f"{i.id}  [{i.status:8s}] {i.kind:8s} {i.author:18s} {i.title}")
        if i.details:
            for line in i.details.splitlines():
                print(f"{'':24s}  {line}")
        if i.note:
            print(f"{'':24s}  -> {i.note}")
        if i.link:
            print(f"{'':24s}  -> https://eaglehillfund.com{i.link}")
    print(f"{len(rows)} idea(s)")
    return 0


def _thread_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--study", default=None, help="study key its scorecards carry (vix_overlay, leverage)"
    )
    sp.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="HASH[=LABEL]",
        help="a run behind it; repeatable",
    )
    sp.add_argument(
        "--idea", action="append", default=[], metavar="ID", help="an ideas-box id; repeatable"
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    pub = sub.add_parser("publish-report")
    pub.add_argument("--slug", required=True)
    pub.add_argument("--title", required=True)
    pub.add_argument("--date", required=True, help="YYYY-MM-DD")
    pub.add_argument("--summary", required=True)
    pub.add_argument(
        "--file", action="append", required=True, help="first file is the primary one; repeatable"
    )
    pub.add_argument(
        "--label", action="append", default=[], help="button label per --file, in order"
    )
    pub.add_argument("--tags", default="")
    pub.add_argument(
        "--body-md", default=None, help="markdown file rendered above the embedded PDF"
    )
    pub.add_argument("--author", default="Andrew Antczak")
    for sp in (pub,):
        _thread_args(sp)
    ln = sub.add_parser("link", help="attach the study / runs / ideas behind a published write-up")
    ln.add_argument("--slug", required=True)
    _thread_args(ln)
    ln.add_argument("--clear", action="store_true", help="replace the lists instead of merging")
    sub.add_parser("list")
    un = sub.add_parser("unpublish")
    un.add_argument("--slug", required=True)
    un.add_argument("--yes", action="store_true")
    sub.add_parser("status")
    lg = sub.add_parser("logs")
    lg.add_argument("--minutes", type=int, default=30)
    us = sub.add_parser("user")
    us.add_argument("--email", required=True)
    us.add_argument("--name", required=True)
    pw = sub.add_parser("password", help="hand a user a temporary password (no reset email here)")
    pw.add_argument("--email", required=True)
    idp = sub.add_parser("ideas")
    idp.add_argument("--set", metavar="ID", help="update this idea instead of listing")
    idp.add_argument("--status", help="new | planned | building | done | declined")
    idp.add_argument("--note", help="a sentence shown under the idea on the site")
    idp.add_argument("--link", help="site path of the result, e.g. /library/<slug>")
    idp.add_argument("--json", action="store_true")
    a = p.parse_args()
    commands = {
        "publish-report": lambda: publish(a),
        "link": lambda: link(a),
        "list": list_library,
        "unpublish": lambda: unpublish(a),
        "status": status,
        "logs": lambda: logs(a),
        "user": lambda: add_user(a),
        "password": lambda: reset_password(a),
        "ideas": lambda: ideas(a),
    }
    return int(commands[a.cmd]())


if __name__ == "__main__":
    sys.exit(main())
