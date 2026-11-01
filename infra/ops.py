#!/usr/bin/env python3
"""Operational helpers for the daily report (boto3; reads .env for keys).

    python infra/ops.py set-secrets                 # push FMP_API_KEY / ANTHROPIC_API_KEY into Secrets Manager
    python infra/ops.py invoke [--dry-run] [--max-symbols N] [--no-narrative] [--date YYYY-MM-DD]
    python infra/ops.py logs [--minutes 60]         # tail the Lambda log group
    python infra/ops.py latest                      # print the latest report.md from S3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

REGION = os.getenv("AWS_REGION", "us-east-1")
FUNCTION = os.getenv("EHF_REPORT_FUNCTION", "ehf-daily-report")
SECRETS = {"ehf/fmp": "FMP_API_KEY", "ehf/anthropic": "ANTHROPIC_API_KEY"}


def set_secrets() -> int:
    sm = boto3.client("secretsmanager", region_name=REGION)
    for name, env in SECRETS.items():
        value = os.getenv(env, "").strip()
        if not value:
            print(f"skip {name}: {env} not set in .env / environment")
            continue
        sm.put_secret_value(SecretId=name, SecretString=value)
        print(f"set {name} from {env} ({len(value)} chars)")
    return 0


def invoke(args: argparse.Namespace) -> int:
    payload: dict[str, object] = {"dry_run": args.dry_run}
    if args.max_symbols:
        payload["max_symbols"] = args.max_symbols
    if args.no_narrative:
        payload["narrative"] = False
    if args.no_email:
        payload["email"] = False
    if args.date:
        payload["date"] = args.date
    lam = boto3.client("lambda", region_name=REGION)
    print(f"invoking {FUNCTION} with {payload} ...")
    t0 = time.time()
    resp = lam.invoke(FunctionName=FUNCTION, InvocationType="RequestResponse", Payload=json.dumps(payload))
    body = resp["Payload"].read().decode()
    print(f"status {resp['StatusCode']} in {time.time() - t0:.0f}s; FunctionError={resp.get('FunctionError')}")
    try:
        print(json.dumps(json.loads(body), indent=1))
    except ValueError:
        print(body)
    return 1 if resp.get("FunctionError") else 0


def logs(args: argparse.Namespace) -> int:
    cw = boto3.client("logs", region_name=REGION)
    start = int((time.time() - args.minutes * 60) * 1000)
    kwargs: dict[str, object] = {"logGroupName": f"/aws/lambda/{FUNCTION}", "startTime": start, "interleaved": True}
    while True:
        page = cw.filter_log_events(**kwargs)
        for e in page.get("events", []):
            print(e["message"].rstrip())
        if "nextToken" not in page:
            return 0
        kwargs["nextToken"] = page["nextToken"]


def latest() -> int:
    s3 = boto3.client("s3", region_name=REGION)
    bucket = os.getenv("EHF_REPORT_BUCKET") or f"ehf-market-reports-{boto3.client('sts').get_caller_identity()['Account']}"
    obj = s3.get_object(Bucket=bucket, Key="daily-report/latest/report.md")
    sys.stdout.write(obj["Body"].read().decode())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("set-secrets")
    inv = sub.add_parser("invoke")
    inv.add_argument("--dry-run", action="store_true")
    inv.add_argument("--max-symbols", type=int)
    inv.add_argument("--no-narrative", action="store_true")
    inv.add_argument("--no-email", action="store_true", help="store to S3 but skip SES")
    inv.add_argument("--date")
    lg = sub.add_parser("logs")
    lg.add_argument("--minutes", type=int, default=60)
    sub.add_parser("latest")
    args = p.parse_args()
    if args.cmd == "set-secrets":
        return set_secrets()
    if args.cmd == "invoke":
        return invoke(args)
    if args.cmd == "logs":
        return logs(args)
    return latest()


if __name__ == "__main__":
    sys.exit(main())
