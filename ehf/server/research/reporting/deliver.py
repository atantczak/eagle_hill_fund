"""Delivery: publish artifacts to S3 and send the HTML report by SES email.

boto3 is imported lazily so the pure modules never pay for it (and local runs
without AWS credentials work with ``--dry-run``).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

log = logging.getLogger(__name__)

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".json": "application/json",
    ".csv": "text/csv",
    ".gz": "application/gzip",
}


def s3_keys_for(prefix: str, run_date: date, names: list[str]) -> dict[str, tuple[str, str]]:
    """Pure: ``name -> (dated key, latest key)``."""
    day = f"{prefix.rstrip('/')}/{run_date:%Y/%m/%d}"
    latest = f"{prefix.rstrip('/')}/latest"
    return {n: (f"{day}/{n}", f"{latest}/{n}") for n in names}


def publish_s3(
    bucket: str, prefix: str, run_date: date, files: dict[str, bytes], *, region: str
) -> list[str]:
    import boto3  # type: ignore[import-untyped]

    s3 = boto3.client("s3", region_name=region)
    written: list[str] = []
    for name, (dated, latest) in s3_keys_for(prefix, run_date, list(files)).items():
        ext = name[name.rfind(".") :] if "." in name else ""
        ctype = _CONTENT_TYPES.get(ext, "application/octet-stream")
        for key in (dated, latest):
            s3.put_object(Bucket=bucket, Key=key, Body=files[name], ContentType=ctype)
            written.append(f"s3://{bucket}/{key}")
    log.info("published %d objects to s3://%s/%s", len(written), bucket, prefix)
    return written


def send_email(
    *, sender: str, recipients: list[str], subject: str, html_body: str, text_body: str, region: str
) -> str:
    import boto3

    ses: Any = boto3.client("ses", region_name=region)
    resp = ses.send_email(
        Source=sender,
        Destination={"ToAddresses": recipients},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {
                "Text": {"Data": text_body, "Charset": "UTF-8"},
                "Html": {"Data": html_body, "Charset": "UTF-8"},
            },
        },
    )
    message_id = str(resp.get("MessageId", ""))
    log.info("email sent to %s (MessageId=%s)", recipients, message_id)
    return message_id
