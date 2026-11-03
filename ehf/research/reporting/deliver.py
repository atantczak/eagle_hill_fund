"""Delivery: publish artifacts to S3, presign links, send the report by SES email.

boto3 is imported lazily so the pure modules never pay for it (and local runs
without AWS credentials work with ``--dry-run``).
"""

from __future__ import annotations

import logging
from datetime import date
from email.message import EmailMessage
from email.policy import SMTP
from typing import Any

log = logging.getLogger(__name__)

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".json": "application/json",
    ".csv": "text/csv",
    ".gz": "application/gzip",
    ".pdf": "application/pdf",
}
MAX_PRESIGN_SECONDS = 7 * 24 * 3600  # hard S3 limit for SigV4


def s3_keys_for(prefix: str, run_date: date, names: list[str]) -> dict[str, tuple[str, str]]:
    """Pure: ``name -> (dated key, latest key)``."""
    day = f"{prefix.rstrip('/')}/{run_date:%Y/%m/%d}"
    latest = f"{prefix.rstrip('/')}/latest"
    return {n: (f"{day}/{n}", f"{latest}/{n}") for n in names}


def publish_s3(
    bucket: str, prefix: str, run_date: date, files: dict[str, bytes], *, region: str
) -> list[str]:
    import boto3

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


def presign(
    bucket: str,
    keys: dict[str, str],
    *,
    region: str,
    days: int,
    key_id: str | None = None,
    secret: str | None = None,
) -> dict[str, str]:
    """``name -> https URL`` valid for ``days`` (capped at S3's 7-day maximum).

    With ``key_id``/``secret`` the URLs are signed by the long-lived signer user; without
    them the caller's credentials are used, which for a Lambda means the links die when
    the role session expires (hours). ``ContentDisposition`` is set so the browser opens
    the HTML inline and offers the PDF with a readable filename.
    """
    import boto3
    from botocore.config import Config

    expires = min(max(int(days), 1) * 24 * 3600, MAX_PRESIGN_SECONDS)
    kwargs: dict[str, Any] = {"region_name": region, "config": Config(signature_version="s3v4")}
    if key_id and secret:
        kwargs.update(aws_access_key_id=key_id, aws_secret_access_key=secret)
    else:
        log.warning("presigning with the caller's temporary credentials; links will be short-lived")
    s3 = boto3.client("s3", **kwargs)
    out: dict[str, str] = {}
    for name, key in keys.items():
        filename = key.rsplit("/", 1)[-1]
        disposition = "attachment" if filename.endswith(".pdf") else "inline"
        out[name] = s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ResponseContentDisposition": f'{disposition}; filename="{filename}"',
            },
            ExpiresIn=expires,
        )
    return out


def build_email(
    *,
    sender: str,
    recipients: list[str],
    subject: str,
    html_body: str,
    text_body: str,
    attachments: dict[str, bytes] | None = None,
) -> EmailMessage:
    """Pure: a MIME message (text + HTML alternatives, optional file attachments)."""
    msg = EmailMessage(policy=SMTP)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(text_body, charset="utf-8")
    msg.add_alternative(html_body, subtype="html", charset="utf-8")
    for filename, blob in (attachments or {}).items():
        maintype, subtype = _mime_for(filename)
        msg.add_attachment(blob, maintype=maintype, subtype=subtype, filename=filename)
    return msg


def _mime_for(filename: str) -> tuple[str, str]:
    if filename.endswith(".pdf"):
        return "application", "pdf"
    if filename.endswith(".html"):
        return "text", "html"
    if filename.endswith(".md"):
        return "text", "markdown"
    return "application", "octet-stream"


def verified_recipients(recipients: list[str], *, region: str) -> tuple[list[str], list[str]]:
    """Split recipients into (verified, unverified) SES identities.

    In the SES sandbox a single unverified address rejects the whole send, so a newly
    added recipient who has not clicked the verification link must be dropped, not
    allowed to block everyone else. Lookup failures count the address as verified so a
    permissions hiccup never silences the report.
    """
    import boto3
    from botocore.exceptions import ClientError

    ses = boto3.client("sesv2", region_name=region)
    ok: list[str] = []
    bad: list[str] = []
    for addr in recipients:
        try:
            ident = ses.get_email_identity(EmailIdentity=addr)
            (ok if ident.get("VerifiedForSendingStatus") else bad).append(addr)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "NotFoundException":
                bad.append(addr)
            else:
                log.warning("could not check SES identity %s (%s); sending anyway", addr, code)
                ok.append(addr)
    return ok, bad


def send_email(
    *,
    sender: str,
    recipients: list[str],
    subject: str,
    html_body: str,
    text_body: str,
    region: str,
    attachments: dict[str, bytes] | None = None,
) -> str:
    import boto3

    msg = build_email(
        sender=sender,
        recipients=recipients,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        attachments=attachments,
    )
    ses: Any = boto3.client("ses", region_name=region)
    resp = ses.send_raw_email(
        Source=sender, Destinations=recipients, RawMessage={"Data": msg.as_bytes()}
    )
    message_id = str(resp.get("MessageId", ""))
    log.info(
        "email sent to %s (%d attachment(s), MessageId=%s)",
        recipients,
        len(attachments or {}),
        message_id,
    )
    return message_id
