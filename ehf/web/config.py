"""Environment-driven settings for the site (set by the ECS task definition)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SiteConfig:
    site_url: str = "http://localhost:8080"
    reports_bucket: str = ""
    library_prefix: str = "library"
    daily_prefix: str = "daily-report"
    ideas_prefix: str = "ideas"  # the ideas box, in the same bucket
    cognito_domain: str = ""  # e.g. ehf-site-123.auth.us-east-1.amazoncognito.com
    cognito_client_id: str = ""
    region: str = "us-east-1"
    cache_ttl_s: float = 60.0
    org_name: str = "The Eagle Hill Fund"
    lab: str = "auto"  # "auto" (on when ehf.research imports), "on", "off"
    # Deployed lab: mirror the Studio's var/ from s3://<var_bucket>/<var_prefix>/ into
    # EHF_VAR_DIR on demand (ehf.web.lab.mirror). Empty = read var/ straight off disk.
    var_bucket: str = ""
    var_prefix: str = "var"

    @classmethod
    def from_env(cls) -> SiteConfig:
        d = cls()
        return cls(
            site_url=os.getenv("EHF_SITE_URL", d.site_url).rstrip("/"),
            reports_bucket=os.getenv("EHF_REPORTS_BUCKET", d.reports_bucket),
            library_prefix=os.getenv("EHF_LIBRARY_PREFIX", d.library_prefix).strip("/"),
            daily_prefix=os.getenv("EHF_REPORT_PREFIX", d.daily_prefix).strip("/"),
            ideas_prefix=os.getenv("EHF_IDEAS_PREFIX", d.ideas_prefix).strip("/"),
            cognito_domain=os.getenv("EHF_COGNITO_DOMAIN", d.cognito_domain),
            cognito_client_id=os.getenv("EHF_COGNITO_CLIENT_ID", d.cognito_client_id),
            region=os.getenv("AWS_REGION", d.region),
            cache_ttl_s=float(os.getenv("EHF_SITE_CACHE_TTL", d.cache_ttl_s)),
            lab=os.getenv("EHF_LAB", d.lab).strip().lower(),
            var_bucket=os.getenv("EHF_VAR_S3_BUCKET", d.var_bucket).strip(),
            var_prefix=os.getenv("EHF_VAR_S3_PREFIX", d.var_prefix).strip("/"),
        )
