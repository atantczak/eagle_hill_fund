"""A small, dependency-light FMP client for the daily report.

Deliberately separate from the legacy ``integrations/financial`` wrapper: that
one imports ``requests_mock`` and ``psutil`` at module load, which we do not
want inside a Lambda. This client knows only the endpoints the report uses,
retries 429/5xx with backoff, and fans per-symbol calls out through a bounded
thread pool (FMP is rate-limited I/O, not CPU).

Every method returns plain ``pandas`` objects or JSON-shaped lists; no caching.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://financialmodelingprep.com/stable"
_RETRYABLE = {429, 500, 502, 503, 504}
JsonList = list[dict[str, Any]]


class FMPReportClient:
    """Thin FMP client. ``api_key`` is required; nothing is read from env here."""

    def __init__(
        self,
        api_key: str,
        *,
        workers: int = 16,
        timeout: float = 60.0,
        max_retries: int = 4,
        base_url: str = BASE_URL,
    ) -> None:
        if not api_key:
            raise ValueError("FMP api_key is required")
        self.api_key = api_key
        self.workers = workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
        self._session.mount("https://", adapter)
        self.calls = 0  # simple counter so a run can report its API budget

    # --- transport -----------------------------------------------------------

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> requests.Response:
        q: dict[str, Any] = dict(params or {})
        q["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self.calls += 1
                resp = self._session.get(url, params=q, timeout=self.timeout)
                if resp.status_code in _RETRYABLE and attempt < self.max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else 1.5 * (2**attempt)
                    log.warning(
                        "FMP %s -> %s; backing off %.1fs", endpoint, resp.status_code, delay
                    )
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:  # network / timeouts
                last = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (2**attempt))
                    continue
                raise
        raise RuntimeError(f"FMP request failed: {endpoint}") from last

    def get_json(self, endpoint: str, **params: Any) -> Any:
        resp = self._get(endpoint, params)
        try:
            return resp.json()
        except ValueError:
            return []

    def get_csv(
        self, endpoint: str, *, usecols: Sequence[str] | None = None, **params: Any
    ) -> pd.DataFrame:
        """Bulk endpoints return CSV; parse only ``usecols`` to keep memory sane."""
        resp = self._get(endpoint, params)
        return pd.read_csv(io.StringIO(resp.text), usecols=list(usecols) if usecols else None)

    def _parallel(self, fn: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]:
        items = list(items)
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=min(self.workers, len(items))) as pool:
            return list(pool.map(fn, items))

    # --- universe --------------------------------------------------------------

    def sp500_constituents(self) -> JsonList:
        out = self.get_json("sp500-constituent")
        return out if isinstance(out, list) else []

    def screener(
        self,
        *,
        market_cap_more_than: float,
        volume_more_than: float,
        exchanges: Sequence[str],
        limit: int = 10_000,
    ) -> JsonList:
        out = self.get_json(
            "company-screener",
            marketCapMoreThan=market_cap_more_than,
            volumeMoreThan=volume_more_than,
            exchange=",".join(exchanges),
            isEtf="false",
            isFund="false",
            isActivelyTrading="true",
            limit=limit,
        )
        return out if isinstance(out, list) else []

    # --- prices ------------------------------------------------------------------

    def batch_quotes(self, symbols: Sequence[str], *, chunk: int = 500) -> pd.DataFrame:
        rows: JsonList = []
        for i in range(0, len(symbols), chunk):
            part = self.get_json("batch-quote", symbols=",".join(symbols[i : i + chunk]))
            if isinstance(part, list):
                rows.extend(r for r in part if isinstance(r, dict))
        return pd.DataFrame(rows)

    def quote(self, symbol: str) -> dict[str, Any]:
        out = self.get_json("quote", symbol=symbol)
        return out[0] if isinstance(out, list) and out else {}

    def daily_bars(self, symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """One year-ish of daily OHLCV per symbol, fetched in a bounded pool.

        Returns a long frame ``symbol, date, open, high, low, close, volume``
        (dates ascending). Symbols that fail or return nothing are dropped and
        logged — a missing name should never sink the whole report.
        """

        def one(sym: str) -> JsonList:
            try:
                rows = self.get_json(
                    "historical-price-eod/full",
                    symbol=sym,
                    **{"from": start.isoformat(), "to": end.isoformat()},
                )
            except requests.RequestException as exc:
                log.warning("bars failed for %s: %s", sym, exc)
                return []
            if not isinstance(rows, list):
                return []
            return [{**r, "symbol": sym} for r in rows if isinstance(r, dict)]

        chunks = self._parallel(one, symbols)
        flat = [r for ch in chunks for r in ch]
        if not flat:
            return pd.DataFrame(
                columns=["symbol", "date", "open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(flat)[["symbol", "date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # --- fundamentals (bulk) ----------------------------------------------------------

    def key_metrics_ttm_bulk(self, usecols: Sequence[str]) -> pd.DataFrame:
        return self.get_csv("key-metrics-ttm-bulk", usecols=usecols)

    def ratios_ttm_bulk(self, usecols: Sequence[str]) -> pd.DataFrame:
        return self.get_csv("ratios-ttm-bulk", usecols=usecols)

    def scores_bulk(self, usecols: Sequence[str]) -> pd.DataFrame:
        return self.get_csv("scores-bulk", usecols=usecols)

    def consensus_bulk(self) -> pd.DataFrame:
        return self.get_csv("upgrades-downgrades-consensus-bulk")

    # --- per-symbol enrichment -----------------------------------------------------------

    def income_growth(self, symbol: str) -> dict[str, Any]:
        out = self.get_json("income-statement-growth", symbol=symbol, period="quarter", limit=1)
        return out[0] if isinstance(out, list) and out else {}

    def price_target_consensus(self, symbol: str) -> dict[str, Any]:
        out = self.get_json("price-target-consensus", symbol=symbol)
        return out[0] if isinstance(out, list) and out else {}

    def earnings(self, symbol: str, limit: int = 4) -> JsonList:
        out = self.get_json("earnings", symbol=symbol, limit=limit)
        return out if isinstance(out, list) else []

    def stock_news(self, symbol: str, *, since: date, limit: int = 5) -> JsonList:
        out = self.get_json(
            "news/stock", symbols=symbol, limit=limit, **{"from": since.isoformat()}
        )
        return out if isinstance(out, list) else []

    def enrich(
        self, symbols: Sequence[str], *, news_since: date, news_limit: int
    ) -> dict[str, dict[str, Any]]:
        """Per-symbol context for the top candidates only (bounded by the caller)."""

        def one(sym: str) -> tuple[str, dict[str, Any]]:
            out: dict[str, Any] = {}
            for key, fn in (
                ("growth", lambda: self.income_growth(sym)),
                ("price_target", lambda: self.price_target_consensus(sym)),
                ("earnings", lambda: self.earnings(sym)),
                ("news", lambda: self.stock_news(sym, since=news_since, limit=news_limit)),
            ):
                try:
                    out[key] = fn()
                except requests.RequestException as exc:
                    log.warning("enrich %s/%s failed: %s", sym, key, exc)
                    out[key] = None
            return sym, out

        return dict(self._parallel(one, symbols))

    # --- market-wide feeds (paged) ---------------------------------------------------------

    def _paged_until(
        self, endpoint: str, *, date_key: str, cutoff: date, max_pages: int, limit: int = 100
    ) -> JsonList:
        """Walk ``page=0..`` until a page's oldest row is older than ``cutoff``."""
        rows: JsonList = []
        for page in range(max_pages):
            batch = self.get_json(endpoint, page=page, limit=limit)
            if not isinstance(batch, list) or not batch:
                break
            rows.extend(r for r in batch if isinstance(r, dict))
            oldest = min((str(r.get(date_key, ""))[:10] for r in batch), default="")
            if oldest and oldest < cutoff.isoformat():
                break
        return [r for r in rows if str(r.get(date_key, ""))[:10] >= cutoff.isoformat()]

    def insider_trades(self, *, since: date, max_pages: int = 120) -> JsonList:
        return self._paged_until(
            "insider-trading/latest", date_key="filingDate", cutoff=since, max_pages=max_pages
        )

    def grades_news(self, *, since: date, max_pages: int = 30) -> JsonList:
        return self._paged_until(
            "grades-latest-news", date_key="publishedDate", cutoff=since, max_pages=max_pages
        )

    def price_target_news(self, *, since: date, max_pages: int = 30) -> JsonList:
        return self._paged_until(
            "price-target-latest-news", date_key="publishedDate", cutoff=since, max_pages=max_pages
        )

    # --- calendars / macro ---------------------------------------------------------------------

    def earnings_calendar(self, start: date, end: date) -> JsonList:
        """Per-day calls: FMP caps a window at 4000 rows, a single day stays under it."""
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

        def one(d: date) -> JsonList:
            out = self.get_json("earnings-calendar", **{"from": d.isoformat(), "to": d.isoformat()})
            return out if isinstance(out, list) else []

        return [r for ch in self._parallel(one, days) for r in ch if isinstance(r, dict)]

    def economic_calendar(self, start: date, end: date) -> JsonList:
        out = self.get_json(
            "economic-calendar", **{"from": start.isoformat(), "to": end.isoformat()}
        )
        return out if isinstance(out, list) else []

    def treasury_rates(self, start: date, end: date) -> JsonList:
        out = self.get_json("treasury-rates", **{"from": start.isoformat(), "to": end.isoformat()})
        return out if isinstance(out, list) else []

    def sector_snapshot(self, d: date) -> JsonList:
        out = self.get_json("sector-performance-snapshot", date=d.isoformat())
        return out if isinstance(out, list) else []
