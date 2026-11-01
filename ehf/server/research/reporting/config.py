"""Report configuration. All knobs in one frozen dataclass, overridable from env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ReportConfig:
    """Every threshold the report uses. Defaults are the v1 research choices."""

    # --- universe -----------------------------------------------------------
    min_market_cap: float = 2e9
    min_avg_volume: float = 1_000_000.0
    exchanges: tuple[str, ...] = ("NASDAQ", "NYSE")
    include_sp500: bool = True
    max_symbols: int | None = None  # debug: cap the universe for fast local runs

    # --- data windows ---------------------------------------------------------
    history_calendar_days: int = 400  # ~ 270 trading days: enough for 252d features
    fetch_workers: int = 16
    insider_lookback_days: int = 14
    grades_lookback_days: int = 5
    earnings_lookback_days: int = 4  # calendar days back to look for "just reported"
    earnings_lookahead_days: int = 7
    news_lookback_days: int = 3
    news_per_symbol: int = 4
    max_enriched_symbols: int = 40  # per-symbol enrichment calls are capped here

    # --- screen thresholds ----------------------------------------------------
    outlier_z: float = 3.0
    outlier_min_abs_ret: float = 0.05
    min_dollar_volume: float = 20e6  # 20-day avg $ volume floor for any hit
    dislocation_ret_21d: float = -0.12
    dislocation_dd_63d: float = -0.20
    quality_min_piotroski: float = 6
    quality_min_altman: float = 2.6
    quality_min_fcf_yield: float = 0.03
    quality_min_roic: float = 0.08
    quality_max_net_debt_ebitda: float = 3.0
    quality_min_market_cap: float = 5e9
    pead_min_surprise: float = 0.05
    pead_min_reaction: float = 0.03
    breakout_mom_quantile: float = 0.80
    breakout_min_vol_ratio: float = 1.3
    insider_min_insiders: int = 2
    insider_min_value: float = 500_000.0
    revision_min_net: int = 2
    max_hits_per_screen: int = 12

    # --- delivery -------------------------------------------------------------
    s3_bucket: str | None = None
    s3_prefix: str = "daily-report"
    email_from: str | None = None
    email_to: tuple[str, ...] = field(default_factory=tuple)
    aws_region: str = "us-east-1"

    # --- narrative ------------------------------------------------------------
    narrative: bool = True
    anthropic_model: str = "claude-opus-5-5"
    narrative_effort: str = "high"

    @classmethod
    def from_env(cls) -> ReportConfig:
        """Build from ``EHF_REPORT_*`` environment variables (unset -> defaults)."""
        to_raw = os.getenv("EHF_REPORT_EMAIL_TO", "")
        email_to = tuple(s.strip() for s in to_raw.split(",") if s.strip())
        max_symbols_raw = os.getenv("EHF_REPORT_MAX_SYMBOLS")
        return cls(
            min_market_cap=_env_float("EHF_REPORT_MIN_MARKET_CAP", cls.min_market_cap),
            min_avg_volume=_env_float("EHF_REPORT_MIN_AVG_VOLUME", cls.min_avg_volume),
            max_symbols=int(max_symbols_raw) if max_symbols_raw else None,
            fetch_workers=_env_int("EHF_REPORT_FETCH_WORKERS", cls.fetch_workers),
            s3_bucket=os.getenv("EHF_REPORT_BUCKET") or None,
            s3_prefix=os.getenv("EHF_REPORT_PREFIX", cls.s3_prefix),
            email_from=os.getenv("EHF_REPORT_EMAIL_FROM") or None,
            email_to=email_to,
            aws_region=os.getenv("AWS_REGION", cls.aws_region),
            narrative=_env_bool("EHF_REPORT_NARRATIVE", cls.narrative),
            anthropic_model=os.getenv("EHF_REPORT_MODEL", cls.anthropic_model),
            narrative_effort=os.getenv("EHF_REPORT_EFFORT", cls.narrative_effort),
        )
