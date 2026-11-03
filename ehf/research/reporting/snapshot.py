"""Orchestration: fetch everything once, build the ``Snapshot`` the report renders from.

A feed that fails (insiders, grades, calendars...) logs and degrades to empty —
the report still ships with whatever did arrive. Only the universe and the
price bars are hard requirements.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, TypeVar

import pandas as pd

from ehf.research.reporting import market as mk
from ehf.research.reporting.config import ReportConfig
from ehf.research.reporting.features import price_features
from ehf.research.reporting.fmp import FMPReportClient
from ehf.research.reporting.fundamentals import (
    KEY_METRICS_COLS,
    RATIO_COLS,
    SCORE_COLS,
    merge_fundamentals,
)
from ehf.research.reporting.screens import (
    SCREEN_ORDER,
    Hit,
    apply_enrichment,
    run_all_screens,
    symbols_to_enrich,
)
from ehf.research.reporting.universe import build_universe, fill_from_quotes

log = logging.getLogger(__name__)
Json = dict[str, Any]
T = TypeVar("T")

INDEX_SYMBOL = "SPY"
VIX_SYMBOL = "^VIX"


@dataclass
class Snapshot:
    run_date: date
    as_of: date  # last bar date in the data
    universe: pd.DataFrame
    features: pd.DataFrame
    fundamentals: pd.DataFrame
    hits: dict[str, list[Hit]]
    market: Json
    api_calls: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_hits(self) -> int:
        return sum(len(v) for v in self.hits.values())

    def hits_json(self) -> list[Json]:
        return [h.to_json() for screen in SCREEN_ORDER for h in self.hits.get(screen, [])]

    def table(self) -> pd.DataFrame:
        """Universe-wide features + fundamentals, one row per symbol (for the stored snapshot)."""
        base = self.universe.set_index("symbol")
        return base.join(self.features, how="left").join(self.fundamentals, how="left")

    def narrative_payload(self, *, per_screen: int = 8) -> Json:
        """Compact JSON for the narrative model: market context + top hits with context."""
        return {
            "run_date": self.run_date.isoformat(),
            "data_as_of": self.as_of.isoformat(),
            "universe_size": int(len(self.universe)),
            "market": self.market,
            "screens": {
                screen: [h.to_json() for h in self.hits.get(screen, [])[:per_screen]]
                for screen in SCREEN_ORDER
            },
            "warnings": self.warnings,
        }


def _timed(timings: dict[str, float], name: str, fn: Callable[[], T]) -> T:
    t0 = time.perf_counter()
    try:
        return fn()
    finally:
        timings[name] = round(time.perf_counter() - t0, 2)


def _rows() -> list[Json]:
    return []


def _no_enrichment() -> dict[str, Json]:
    return {}


def _soft(warnings: list[str], name: str, fn: Callable[[], T], default: T) -> T:
    """Run a non-critical fetch; on failure record a warning and return ``default``."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - any feed failure must not sink the report
        log.exception("feed %s failed", name)
        warnings.append(f"{name} unavailable: {type(exc).__name__}: {exc}")
        return default


def build_snapshot(
    client: FMPReportClient, cfg: ReportConfig, *, run_date: date | None = None
) -> Snapshot:
    run_date = run_date or date.today()
    timings: dict[str, float] = {}
    warnings: list[str] = []

    # 1. universe + quotes (names / caps for S&P names missing from the screener)
    universe = _timed(timings, "universe", lambda: build_universe(client, cfg))
    symbols = universe["symbol"].tolist()
    quotes = _soft(warnings, "quotes", lambda: client.batch_quotes(symbols), pd.DataFrame())
    universe = fill_from_quotes(universe, quotes)
    log.info("universe: %d symbols", len(symbols))

    # 2. bars (the one expensive fetch: ~1 call per symbol)
    start = run_date - timedelta(days=cfg.history_calendar_days)
    bars = _timed(
        timings, "bars", lambda: client.daily_bars([*symbols, INDEX_SYMBOL], start, run_date)
    )
    if bars.empty:
        raise RuntimeError("no price bars returned; cannot build a report")
    spy_bars = bars[bars["symbol"] == INDEX_SYMBOL]
    bars = bars[bars["symbol"] != INDEX_SYMBOL]
    as_of = bars["date"].max().date()
    features = _timed(timings, "features", lambda: price_features(bars))
    missing = sorted(set(symbols) - set(features.index))
    if missing:
        warnings.append(f"{len(missing)} symbols had no bars (e.g. {', '.join(missing[:8])})")

    # 3. fundamentals: four whole-market CSVs, filtered to the universe
    def _fund() -> pd.DataFrame:
        km = _soft(
            warnings,
            "key-metrics-ttm-bulk",
            lambda: client.key_metrics_ttm_bulk(KEY_METRICS_COLS),
            pd.DataFrame(),
        )
        ra = _soft(
            warnings, "ratios-ttm-bulk", lambda: client.ratios_ttm_bulk(RATIO_COLS), pd.DataFrame()
        )
        sc = _soft(warnings, "scores-bulk", lambda: client.scores_bulk(SCORE_COLS), pd.DataFrame())
        co = _soft(warnings, "consensus-bulk", lambda: client.consensus_bulk(), pd.DataFrame())
        return merge_fundamentals(symbols, km, ra, sc, co)

    fundamentals = _timed(timings, "fundamentals", _fund)

    # 4. market-wide feeds
    def _feeds() -> dict[str, Any]:
        return {
            "earnings": _soft(
                warnings,
                "earnings-calendar",
                lambda: client.earnings_calendar(
                    run_date - timedelta(days=cfg.earnings_lookback_days),
                    run_date + timedelta(days=cfg.earnings_lookahead_days),
                ),
                _rows(),
            ),
            "insiders": _soft(
                warnings,
                "insider-trading",
                lambda: client.insider_trades(
                    since=run_date - timedelta(days=cfg.insider_lookback_days)
                ),
                _rows(),
            ),
            "grades": _soft(
                warnings,
                "grades-news",
                lambda: client.grades_news(
                    since=run_date - timedelta(days=cfg.grades_lookback_days)
                ),
                _rows(),
            ),
            "pt": _soft(
                warnings,
                "price-target-news",
                lambda: client.price_target_news(
                    since=run_date - timedelta(days=cfg.grades_lookback_days)
                ),
                _rows(),
            ),
            "econ": _soft(
                warnings,
                "economic-calendar",
                lambda: client.economic_calendar(run_date, run_date + timedelta(days=7)),
                _rows(),
            ),
            "treasury": _soft(
                warnings,
                "treasury-rates",
                lambda: client.treasury_rates(run_date - timedelta(days=10), run_date),
                _rows(),
            ),
            "sectors": _soft(
                warnings, "sector-snapshot", lambda: client.sector_snapshot(as_of), _rows()
            ),
            "vix": _soft(warnings, "vix-quote", lambda: client.quote(VIX_SYMBOL), dict[str, Any]()),
            "vix_bars": _soft(
                warnings,
                "vix-bars",
                lambda: client.daily_bars(
                    [VIX_SYMBOL, mk.VIX_LONG_SYMBOL],
                    run_date - timedelta(days=cfg.dial_history_calendar_days),
                    run_date,
                ),
                pd.DataFrame(),
            ),
        }

    feeds = _timed(timings, "feeds", _feeds)

    # 5. screens
    universe_syms = set(symbols)
    upcoming = {
        str(r["symbol"]): str(r["date"])[:10]
        for r in feeds["earnings"]
        if r.get("symbol") in universe_syms and str(r.get("date", ""))[:10] > as_of.isoformat()
    }
    hits = _timed(
        timings,
        "screens",
        lambda: run_all_screens(
            features=features,
            fundamentals=fundamentals,
            universe=universe,
            bars=bars,
            earnings_rows=feeds["earnings"],
            insider_rows=feeds["insiders"],
            grades_rows=feeds["grades"],
            pt_rows=feeds["pt"],
            cfg=cfg,
            as_of=as_of,
            upcoming_earnings=upcoming,
        ),
    )

    # 6. per-symbol enrichment for the names that matter
    to_enrich = symbols_to_enrich(hits, cfg.max_enriched_symbols)
    enrichment: dict[str, Json] = _timed(
        timings,
        "enrichment",
        lambda: _soft(
            warnings,
            "enrichment",
            lambda: client.enrich(
                to_enrich,
                news_since=run_date - timedelta(days=cfg.news_lookback_days),
                news_limit=cfg.news_per_symbol,
            ),
            _no_enrichment(),
        ),
    )
    apply_enrichment(hits, enrichment)

    # 7. market context
    market: Json = {
        "index": mk.index_context(spy_bars),
        "vix": mk.vix_context(feeds["vix"]),
        "dial": _soft(
            warnings,
            "market-dial",
            lambda: mk.dial_context(spy_bars, feeds["vix_bars"], cfg),
            dict[str, Any](),
        ),
        "breadth": mk.breadth(features),
        "treasury": mk.treasury_context(feeds["treasury"]),
        "sectors": mk.sector_context(feeds["sectors"]),
        "econ_ahead": mk.econ_context(feeds["econ"]),
        "earnings_ahead": mk.earnings_ahead(feeds["earnings"], universe, after=as_of),
    }

    return Snapshot(
        run_date=run_date,
        as_of=as_of,
        universe=universe,
        features=features,
        fundamentals=fundamentals,
        hits=hits,
        market=market,
        api_calls=client.calls,
        timings=timings,
        warnings=warnings,
    )
