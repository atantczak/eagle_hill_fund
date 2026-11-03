"""Who is looking at the page.

The ALB authenticates against Cognito and forwards the user's claims in the
``x-amzn-oidc-data`` header (a JWT signed by the ALB). Authentication already
happened by the time a request reaches us and the task's security group only
accepts traffic from the ALB, so the claims are read for display and never
treated as a credential.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlencode

from ehf.server.site.config import SiteConfig

OIDC_HEADER = "x-amzn-oidc-data"
ALB_COOKIE_PREFIX = "AWSELBAuthSessionCookie"


@dataclass(frozen=True)
class User:
    email: str
    name: str

    @property
    def display(self) -> str:
        return self.name or self.email


def _b64url_json(segment: str) -> dict[str, object]:
    padded = segment + "=" * (-len(segment) % 4)
    raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    data = json.loads(raw)
    return dict(data) if isinstance(data, dict) else {}


def user_from_headers(headers: Mapping[str, str]) -> User | None:
    """Pure: parse the ALB identity header; ``None`` when absent or unparseable."""
    token = headers.get(OIDC_HEADER) or headers.get(OIDC_HEADER.upper())
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        claims = _b64url_json(parts[1])
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    email = str(claims.get("email") or claims.get("username") or "")
    if not email:
        return None
    return User(email=email, name=str(claims.get("name") or ""))


def logout_url(cfg: SiteConfig) -> str:
    """Cognito hosted-UI logout, which then bounces back to the landing page."""
    if not cfg.cognito_domain:
        return "/"
    q = urlencode({"client_id": cfg.cognito_client_id, "logout_uri": f"{cfg.site_url}/"})
    return f"https://{cfg.cognito_domain}/logout?{q}"


def alb_cookie_names(cookie_header: str) -> list[str]:
    """Names of the ALB session cookies present in a Cookie header (to expire them)."""
    names = []
    for part in cookie_header.split(";"):
        name = part.strip().split("=", 1)[0]
        if name.startswith(ALB_COOKIE_PREFIX):
            names.append(name)
    return names
