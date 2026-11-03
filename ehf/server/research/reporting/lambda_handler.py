"""AWS Lambda entry point. Secrets come from Secrets Manager; config from env.

Event (all optional)::

    {"dry_run": false, "date": "YYYY-MM-DD", "max_symbols": N, "narrative": true, "email": true}
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Any

from ehf.server.research.reporting.config import ReportConfig
from ehf.server.research.reporting.job import run

logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

_SECRET_ENV = {
    "EHF_FMP_SECRET_ARN": "FMP_API_KEY",
    "EHF_ANTHROPIC_SECRET_ARN": "ANTHROPIC_API_KEY",
}
_cache: dict[str, str] = {}


def _secret_value(arn: str) -> str | None:
    if arn in _cache:
        return _cache[arn]
    import boto3  # type: ignore[import-untyped]

    try:
        resp = boto3.client("secretsmanager").get_secret_value(SecretId=arn)
    except Exception as exc:  # noqa: BLE001 - a missing optional secret is not fatal
        log.warning("secret %s unavailable: %s", arn, exc)
        return None
    raw = resp.get("SecretString") or ""
    # Accept either a bare string or {"api_key": "..."} / {"value": "..."}.
    value = raw
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            value = str(
                obj.get("api_key")
                or obj.get("value")
                or obj.get("key")
                or next(iter(obj.values()), "")
            )
    except ValueError:
        pass
    value = value.strip()
    if value and not value.upper().startswith("REPLACE"):
        _cache[arn] = value
        return value
    return None


def _signer_keys(arn: str) -> tuple[str, str] | None:
    """The link-signer IAM user's keys: ``{"access_key_id": .., "secret_access_key": ..}``."""
    if arn in _cache:
        raw = _cache[arn]
    else:
        import boto3

        try:
            raw = boto3.client("secretsmanager").get_secret_value(SecretId=arn)["SecretString"]
        except Exception as exc:  # noqa: BLE001 - links degrade to short-lived ones
            log.warning("signer secret %s unavailable: %s", arn, exc)
            return None
        _cache[arn] = raw
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    kid, sec = str(obj.get("access_key_id", "")), str(obj.get("secret_access_key", ""))
    if kid and sec and not kid.upper().startswith("REPLACE"):
        return kid, sec
    return None


def load_secrets() -> None:
    for env_arn, env_key in _SECRET_ENV.items():
        arn = os.getenv(env_arn)
        if arn and not os.getenv(env_key):
            value = _secret_value(arn)
            if value:
                os.environ[env_key] = value
    signer_arn = os.getenv("EHF_REPORT_SIGNER_SECRET_ARN")
    if signer_arn and not os.getenv("EHF_REPORT_SIGNER_KEY_ID"):
        keys = _signer_keys(signer_arn)
        if keys:
            os.environ["EHF_REPORT_SIGNER_KEY_ID"], os.environ["EHF_REPORT_SIGNER_SECRET"] = keys


def handler(event: dict[str, Any] | None, context: Any) -> dict[str, Any]:
    event = event or {}
    load_secrets()
    cfg = ReportConfig.from_env()
    overrides: dict[str, Any] = {}
    if event.get("max_symbols"):
        overrides["max_symbols"] = int(event["max_symbols"])
    if "narrative" in event:
        overrides["narrative"] = bool(event["narrative"])
    if event.get("email") is False:  # e.g. before the SES identity is verified
        overrides["email_to"] = ()
    if overrides:
        cfg = ReportConfig(**{**cfg.__dict__, **overrides})
    if cfg.narrative and not os.getenv("ANTHROPIC_API_KEY"):
        log.warning("ANTHROPIC_API_KEY not set; running without narrative")
        cfg = ReportConfig(**{**cfg.__dict__, "narrative": False})
    run_date = date.fromisoformat(event["date"]) if event.get("date") else None
    out = run(cfg, run_date=run_date, dry_run=bool(event.get("dry_run", False)))
    summary = out.summary()
    log.info("done: %s", json.dumps(summary, default=str))
    return {"status": "ok", **summary}
