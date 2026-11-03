"""FMP adapter: implements ``DataProvider`` by wrapping the existing FMP tooling.

This is the ONLY module in the platform that knows the data vendor is Financial
Modeling Prep. It delegates network/rate-limit/parallelism to the legacy
``FinancialDataCompiler`` (unchanged), then funnels the result through
``normalize_bars`` so callers receive canonical bars.

Normalization is split out as a module-level pure function
(:func:`normalize_fmp_records`) so it can be unit-tested without any network.
"""

from __future__ import annotations

import os
import random
import warnings
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from ehf.research.data.cache import BarCache
from ehf.research.data.provider import DataProvider, Universe
from ehf.research.data.quality import DataQualityWarning, warn_survivorship
from ehf.research.data.schema import Timeframe, normalize_bars, validate_bars

# FMP timeframe label -> canonical Timeframe.
_FMP_INTERVAL = {
    Timeframe.MINUTE: "1min",
    Timeframe.HOUR: "1hour",
    Timeframe.DAY: "1day",
}

# FMP EOD/intraday field -> canonical column.
_FMP_COLUMN_MAP = {
    "date": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


def normalize_fmp_records(
    records: pd.DataFrame,
    *,
    timeframe: Timeframe,
    adjusted: bool,
) -> pd.DataFrame:
    """Map a raw FMP price frame (long, with a 'symbol' column) into canonical bars.

    Pure function — no network, no key. Unit-tested directly.
    """
    if records.empty:
        empty = pd.DataFrame(columns=["symbol", *(_FMP_COLUMN_MAP.keys())])
        return normalize_bars(
            empty, column_map=_FMP_COLUMN_MAP, timeframe=timeframe, adjusted=adjusted
        )
    return normalize_bars(
        records,
        column_map=_FMP_COLUMN_MAP,
        timeframe=timeframe,
        adjusted=adjusted,
        assume_tz="America/New_York",  # FMP timestamps are US/Eastern wall time
    )


class FMPProvider(DataProvider):
    """Canonical data access backed by Financial Modeling Prep."""

    def __init__(self, cache: BarCache | None = None, *, fetch_workers: int = 16):
        self.cache = cache if cache is not None else BarCache(vendor="fmp")
        self._compiler = None  # lazy: importing legacy tooling pulls psutil etc.
        self._delisted: list[dict[str, Any]] | None = None  # memoized delisted list
        # Bounded concurrency for intraday chunk fetches. FMP is rate-limited
        # network I/O, so this is tuned to SATURATE the rate limit, not the CPU —
        # firing hundreds of threads just earns HTTP 429s. ~16 in-flight requests
        # comfortably saturates FMP's per-minute cap; retries absorb transient 429s.
        self.fetch_workers = fetch_workers

    @property
    def compiler(self):  # type: ignore[no-untyped-def]
        """Lazily construct the wrapped ``FinancialDataCompiler``."""
        if self._compiler is None:
            from ehf.tools.financial.data.tool import (
                FinancialDataCompiler,
            )

            self._compiler = FinancialDataCompiler()
        return self._compiler

    def get_bars(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date,
        timeframe: Timeframe | str = Timeframe.DAY,
        *,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        tf = Timeframe(timeframe) if not isinstance(timeframe, Timeframe) else timeframe
        start_s, end_s = str(start), str(end)
        symbols = [s.upper() for s in symbols]

        # Tolerance so a requested start/end on a weekend or holiday still counts
        # as covered (the first/last real bar can be a few days inside the range).
        tol = pd.Timedelta(days=4)
        req_lo = pd.Timestamp(start_s, tz="UTC")
        req_hi = pd.Timestamp(end_s, tz="UTC")

        frames: list[pd.DataFrame] = []
        to_fetch: list[str] = []
        for sym in symbols:
            cached = self.cache.read(sym, tf, adjusted)
            covered = self.cache.covered_range(sym, tf, adjusted)
            if cached is not None and covered is not None:
                lo, hi = covered
                if lo <= req_lo + tol and hi >= req_hi - tol:
                    frames.append(cached)
                    continue
            to_fetch.append(sym)

        if to_fetch:
            fetched = self._fetch(to_fetch, start_s, end_s, tf, adjusted)
            if not fetched.empty:
                self.cache.write(fetched, tf, adjusted)
            frames.append(fetched)

        if not frames:
            return normalize_fmp_records(pd.DataFrame(), timeframe=tf, adjusted=adjusted)

        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
            .sort_values(["symbol", "timestamp"])
            .reset_index(drop=True)
        )
        # Slice to the requested window.
        lo = pd.Timestamp(start_s, tz="UTC")
        hi = pd.Timestamp(end_s, tz="UTC") + pd.Timedelta(days=1)
        combined = combined[(combined["timestamp"] >= lo) & (combined["timestamp"] < hi)]
        combined = combined.reset_index(drop=True)
        combined.attrs["adjusted"] = adjusted
        combined.attrs["timeframe"] = tf.value
        _warn_if_truncated(combined, lo, hi, tf)
        return validate_bars(combined)

    def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Live quotes keyed by symbol: ``price``, ``change_pct``, ``time`` (ISO, UTC).

        FMP's quote is a few seconds behind the tape (measured: 1-10 s), far
        fresher than the end-of-day history the bars come from. One batch
        request; a vendor failure or an unknown symbol just means no entry.
        """
        import requests

        key = os.getenv("FMP_API_KEY")
        if not key or not symbols:
            return {}
        r = requests.get(
            "https://financialmodelingprep.com/stable/batch-quote",
            params={"symbols": ",".join(s.upper() for s in symbols), "apikey": key},
            timeout=10,
        )
        r.raise_for_status()
        out: dict[str, dict[str, Any]] = {}
        for row in r.json() if isinstance(r.json(), list) else []:
            sym, price, ts = row.get("symbol"), row.get("price"), row.get("timestamp")
            if not sym or price is None:
                continue
            pct = row.get("changePercentage")
            out[str(sym).upper()] = {
                "price": float(price),
                "change_pct": float(pct) / 100.0 if pct is not None else None,
                "time": datetime.fromtimestamp(int(ts), UTC).isoformat() if ts else None,
            }
        return out

    def intraday(self, symbol: str, *, interval: str = "1min", days: int = 7) -> pd.DataFrame:
        """Recent intraday bars for one symbol, straight from FMP, never cached.

        Regular session only (FMP returns 09:30-15:59 ET), the last couple of
        sessions. Canonical columns, UTC timestamps; empty when unknown. The
        latest bar is about a minute behind the tape.
        """
        import requests

        key = os.getenv("FMP_API_KEY")
        if not key:
            return pd.DataFrame(
                columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
            )
        end_d = date.today()
        r = requests.get(
            f"https://financialmodelingprep.com/stable/historical-chart/{interval}",
            params={
                "symbol": symbol.upper(),
                "from": (end_d - timedelta(days=days)).isoformat(),
                "to": end_d.isoformat(),
                "apikey": key,
            },
            timeout=15,
        )
        r.raise_for_status()
        rows = r.json() if isinstance(r.json(), list) else []
        if not rows:
            return pd.DataFrame(
                columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(rows)
        ts = pd.to_datetime(df["date"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
        out = pd.DataFrame(
            {
                "symbol": symbol.upper(),
                "timestamp": ts,
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": df["close"].astype(float),
                "volume": df["volume"].astype(float),
            }
        )
        return out.sort_values("timestamp").reset_index(drop=True)

    def refresh_tail(
        self,
        symbols: list[str],
        *,
        days: int = 10,
        timeframe: Timeframe | str = Timeframe.DAY,
        adjusted: bool = True,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """Re-fetch the last ``days`` calendar days for ``symbols`` and merge them in.

        ``get_bars`` treats a range as covered when the cache reaches within a
        few days of its end, which is right for research and wrong for "what
        does the dial say today": this pulls the tail regardless, so a page read
        after the close sees that close (and, during the session, the partial
        bar). Returns the fetched rows; a vendor failure propagates.

        The pull always starts at or before the last cached bar. Coverage is
        judged by the cache's min and max, so a ten-day tail stitched onto a cache
        that stopped months ago would hide the hole for good; pulling from the
        cache's end forward closes it instead.
        """
        tf = Timeframe(timeframe) if not isinstance(timeframe, Timeframe) else timeframe
        end_d = date.fromisoformat(str(end)) if end else date.today()
        tail_start = end_d - timedelta(days=days)
        by_start: dict[date, list[str]] = {}
        for sym in (s.upper() for s in symbols):
            covered = self.cache.covered_range(sym, tf, adjusted)
            start_d = tail_start
            if covered is not None:
                start_d = min(start_d, covered[1].date() - timedelta(days=1))
            by_start.setdefault(start_d, []).append(sym)
        frames = []
        for start_d, syms in by_start.items():
            fetched = self._fetch(syms, start_d.isoformat(), end_d.isoformat(), tf, adjusted)
            if not fetched.empty:
                self.cache.write(fetched, tf, adjusted)
                frames.append(fetched)
        if not frames:
            return normalize_fmp_records(pd.DataFrame(), timeframe=tf, adjusted=adjusted)
        return pd.concat(frames, ignore_index=True)

    def _fetch(
        self, symbols: list[str], start: str, end: str, tf: Timeframe, adjusted: bool
    ) -> pd.DataFrame:
        """Hit the wrapped FMP tooling and normalize. Isolated for mocking."""
        if tf is Timeframe.DAY:
            # FMP caps a daily request at 5,000 rows (~19.8 years) and silently
            # returns only the most recent ones, so long ranges are fetched in
            # date chunks and unioned. Within a chunk the compiler fans symbols
            # out in parallel.
            parts = []
            for c_start, c_end in daily_chunks(start, end):
                part = self.compiler.compile_price_data(
                    symbols=symbols, from_date=c_start, to_date=c_end, interval=_FMP_INTERVAL[tf]
                )
                if part is not None and len(part):
                    parts.append(part)
            raw = pd.concat(parts, ignore_index=True) if parts else None
        else:
            # Intraday: fetch every (symbol, chunk) concurrently in one bounded pool.
            raw = self._fetch_intraday_parallel(symbols, start, end, _FMP_INTERVAL[tf])

        if raw is None or len(raw) == 0:
            return normalize_fmp_records(pd.DataFrame(), timeframe=tf, adjusted=adjusted)
        if "symbol" not in raw.columns and len(symbols) == 1:
            raw = raw.assign(symbol=symbols[0])
        return normalize_fmp_records(raw, timeframe=tf, adjusted=adjusted)

    def _fetch_intraday_parallel(
        self, symbols: list[str], start: str, end: str, interval: str
    ) -> pd.DataFrame:
        """Flatten (symbol x chunk) into one bounded, retrying parallel fetch.

        The legacy compiler chunks intraday per symbol but fetches those chunks
        *sequentially*; for a long history that serializes hundreds of requests.
        Here every chunk across every symbol is one task in a single pool capped
        at ``fetch_workers`` (saturates FMP's rate limit without 429-storming),
        with retries on transient failures. Each record is stamped with its symbol.
        """
        chunker = self.compiler._chunk_intraday_date_ranges
        price_tool = self.compiler.price_data_tool
        tasks = [(sym, cs, ce) for sym in symbols for (cs, ce) in chunker(start, end, interval)]
        if not tasks:
            return pd.DataFrame()

        def fetch_chunk(task: tuple[str, str, str]) -> list[dict[str, Any]]:
            sym, chunk_start, chunk_end = task
            rows = price_tool.get_intraday_prices(
                symbol=sym, interval=interval, from_date=chunk_start, to_date=chunk_end
            )
            if not isinstance(rows, list):
                return []
            for record in rows:
                record["symbol"] = sym
            return rows

        workers = min(self.fetch_workers, len(tasks))
        results = self.compiler.parallel_tool.parallel_map_with_retry(
            fetch_chunk, tasks, max_retries=2, max_workers=workers
        )
        all_rows = [record for chunk in results if chunk for record in chunk]
        return pd.DataFrame(all_rows)

    def get_universe(self, name: str, as_of: str | date) -> Universe:
        """Resolve a named universe through FMP.

        * ``sp500``: true point-in-time membership, rebuilt from today's list and
          the constituent-change log (``survivorship_safe=True``).
        * ``us_liquid``: the current screener above the shared liquidity floors
          (``ehf.research.data.universes.US_LIQUID``). FMP only exposes
          CURRENT membership here, so a historical ``as_of`` is survivorship-biased:
          we warn and flag it.
        * anything else: the legacy screener / stock list (also current-only).
        """
        as_of_d = as_of if isinstance(as_of, date) else date.fromisoformat(str(as_of))
        key = name.lower()
        if key == "sp500":
            from ehf.research.data.universes import sp500_members_as_of

            members = sp500_members_as_of(self._sp500_current(), self._sp500_changes(), as_of_d)
            return Universe(
                name=key, as_of=as_of_d, symbols=sorted(members), survivorship_safe=True
            )
        if key == "us_liquid":
            from ehf.research.data.universes import US_LIQUID

            rows = self.compiler.screen_companies(
                market_cap_more_than=US_LIQUID.min_market_cap,
                price_more_than=US_LIQUID.min_price,
                volume_more_than=US_LIQUID.min_volume,
                is_etf=False,
                is_fund=False,
                is_actively_trading=True,
            )
            symbols = sorted(_clean_symbols(rows, exchanges=US_LIQUID.exchanges))
        else:
            symbols = self._screen_symbols(name)
        warn_survivorship(name, as_of)
        return Universe(name=key, as_of=as_of_d, symbols=symbols, survivorship_safe=False)

    def _sp500_current(self) -> list[str]:
        rows = self._json("/sp500-constituent")
        return [str(r["symbol"]) for r in rows if isinstance(r, dict) and r.get("symbol")]

    def _sp500_changes(self) -> list[dict[str, Any]]:
        return [r for r in self._json("/historical-sp500-constituent") if isinstance(r, dict)]

    def _json(self, endpoint: str, **params: Any) -> list[Any]:
        """GET a JSON-list endpoint through the wrapped client (adds the API key)."""
        resp = self.compiler.get(endpoint, params={"apikey": self.compiler.api_key, **params})
        body = resp.json() if hasattr(resp, "json") else resp
        return body if isinstance(body, list) else []

    def fetch_and_cache(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date,
        timeframe: Timeframe | str = Timeframe.DAY,
        *,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Force-fetch ``symbols`` from the vendor and merge into the cache.

        Bypasses the coverage check (``get_bars`` only fetches what the cache
        lacks); the nightly refresh uses this for symbols whose cached tail is
        stale. Returns the fetched canonical bars (possibly empty).
        """
        tf = Timeframe(timeframe) if not isinstance(timeframe, Timeframe) else timeframe
        fetched = self._fetch([s.upper() for s in symbols], str(start), str(end), tf, adjusted)
        if not fetched.empty:
            self.cache.write(fetched, tf, adjusted)
        return fetched

    def _screen_symbols(self, name: str) -> list[str]:
        """Resolve a universe name to symbols via the wrapped FMP screener."""
        if name.lower() in {"all", "us_equities"}:
            rows = self.compiler.get_stock_list(actively_trading=True)
        else:
            rows = self.compiler.screen_companies(is_actively_trading=True)
        return sorted({r["symbol"] for r in rows if isinstance(r, dict) and r.get("symbol")})

    def random_symbols(
        self,
        n: int,
        *,
        as_of: str | date | None = None,
        include_delisted: bool = True,
        seed: int | None = None,
        min_market_cap: float = 2e9,
        min_price: float = 5.0,
        min_volume: float = 1_000_000,
        exchanges: Sequence[str] = ("NASDAQ", "NYSE"),
    ) -> list[str]:
        """Sample ``n`` random liquid US common stocks. Two modes:

        * ``as_of=None`` (default): sample from stocks trading *today* above the
          liquidity floors. Fast and convenient, but **survivorship-biased** for a
          historical backtest — names that died before today can never be drawn.

        * ``as_of="YYYY-MM-DD"`` (or a ``date``): sample from stocks that were
          **alive on that date** — the current liquid names PLUS every company
          that was listed on ``as_of`` and has since delisted (``include_delisted``;
          matched by ``ipoDate <= as_of < delistedDate``). FMP serves price history
          for those dead tickers (e.g. Lehman, SVB → ~$0), so a backtest started
          near ``as_of`` now sees the failures, not only the survivors. This
          removes the dominant survivorship bias.

        Bake the returned list into ``UniverseSpec(symbols=...)`` so the run stays
        reproducible; pass ``seed`` to make the *draw* reproducible too.

        Honest residual limits of the FMP screener: the *survivor* set is filtered
        on **today's** cap/price/volume (a mild look-ahead in the floors) and is
        **not** IPO-date filtered (the screener carries no IPO date), so a name
        that listed after ``as_of`` can still be drawn — though its pre-listing
        bars are absent, so it can't trade before it existed. The delisted set is
        date-correct. For a fully point-in-time universe, use a vendor PIT
        constituent feed via ``get_universe``.
        """
        active = self.compiler.screen_companies(
            market_cap_more_than=min_market_cap,
            price_more_than=min_price,
            volume_more_than=min_volume,
            is_etf=False,
            is_fund=False,
            is_actively_trading=True,
        )
        pool = _clean_symbols(active, exchanges=exchanges)
        if as_of is not None and include_delisted:
            pool |= delisted_alive_on(self._delisted_companies(), as_of, exchanges=exchanges)
        return _sample(pool, n, seed=seed)

    def _delisted_companies(self, *, max_pages: int = 300) -> list[dict[str, Any]]:
        """All FMP delisted companies (paged, memoized per instance).

        FMP caps ``/delisted-companies`` at 100 rows/page regardless of ``limit``,
        so we page until an empty response (``max_pages`` is a runaway guard).
        Rows carry ``symbol, exchange, ipoDate, delistedDate`` — enough to know who
        was tradeable on any historical date.
        """
        if self._delisted is not None:
            return self._delisted
        rows: list[dict[str, Any]] = []
        for page in range(max_pages):
            resp = self.compiler.get(
                "/delisted-companies",
                params={"apikey": self.compiler.api_key, "page": page, "limit": 100},
            )
            batch = resp.json() if hasattr(resp, "json") else resp
            if not isinstance(batch, list) or not batch:
                break
            rows.extend(batch)
        else:
            warnings.warn(
                f"delisted-companies paging hit the {max_pages}-page cap; "
                "the delisted set may be truncated.",
                DataQualityWarning,
                stacklevel=2,
            )
        self._delisted = rows
        return rows


DAILY_ROW_CAP = 5_000  # FMP: max rows per historical-price-eod request
DAILY_CHUNK_YEARS = 15  # comfortably under the cap (~252 sessions/year)


def daily_chunks(start: str, end: str, years: int = DAILY_CHUNK_YEARS) -> list[tuple[str, str]]:
    """Split ``[start, end]`` into consecutive inclusive date ranges of ``years`` (pure).

    FMP returns at most ``DAILY_ROW_CAP`` daily rows per request — the most
    recent ones — so a 25-year request silently loses its first years. Chunking
    keeps every request under the cap; the cache merge unions the pieces.
    """
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    if hi < lo:
        return []
    out: list[tuple[str, str]] = []
    cur = lo
    while cur <= hi:
        nxt = min(cur + pd.DateOffset(years=years) - pd.Timedelta(days=1), hi)
        out.append((cur.date().isoformat(), nxt.date().isoformat()))
        cur = nxt + pd.Timedelta(days=1)
    return out


def _clean_symbols(
    rows: object,
    *,
    exchanges: Sequence[str],
    exchange_key: str = "exchangeShortName",
) -> set[str]:
    """Keep clean US common-stock tickers on the allowed exchanges (pure).

    Drops share-class/odd tickers (containing '.' or '-') and anything off the
    allowed exchanges. ``exchange_key`` differs by endpoint: the screener uses
    ``exchangeShortName``; the delisted endpoint uses ``exchange``.
    """
    if not isinstance(rows, list):
        raise TypeError(f"expected a list of screener rows, got {type(rows).__name__}")
    allowed = {e.upper() for e in exchanges}
    out: set[str] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        sym = r.get("symbol")
        if not sym or "." in sym or "-" in sym:
            continue
        if allowed and str(r.get(exchange_key, "")).upper() not in allowed:
            continue
        out.add(sym)
    return out


def _sample(symbols: set[str] | list[str], n: int, *, seed: int | None = None) -> list[str]:
    """Reproducible sample of ``n`` symbols without replacement (sorted output)."""
    pool = sorted(symbols)
    if n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n))


def _is_iso_date(s: object) -> bool:
    """True for a plausible 'YYYY-MM-DD' string (lexical date compares work then)."""
    return isinstance(s, str) and len(s) >= 10 and s[4] == "-" and s[7] == "-"


def delisted_alive_on(
    rows: object,
    as_of: str | date,
    *,
    exchanges: Sequence[str] = ("NASDAQ", "NYSE"),
) -> set[str]:
    """Symbols listed & trading on ``as_of`` that have *since* delisted (pure).

    Given FMP ``/delisted-companies`` rows (``symbol, exchange, ipoDate,
    delistedDate``), keep clean common-stock tickers on the allowed exchanges whose
    listing window brackets the date — ``ipoDate <= as_of < delistedDate``. These
    are exactly the names a *current* screen drops, so including them is the core
    of the survivorship fix. ISO date strings compare lexically — no parsing.
    """
    if not isinstance(rows, list):
        raise TypeError(f"expected a list of delisted rows, got {type(rows).__name__}")
    as_of_s = as_of.isoformat() if isinstance(as_of, date) else str(as_of)[:10]
    allowed = {e.upper() for e in exchanges}
    out: set[str] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        sym, ipo, delisted = r.get("symbol"), r.get("ipoDate"), r.get("delistedDate")
        if not sym or "." in sym or "-" in sym:
            continue
        if not (_is_iso_date(ipo) and _is_iso_date(delisted)):
            continue
        if not (str(ipo) <= as_of_s < str(delisted)):
            continue
        if allowed and str(r.get("exchange", "")).upper() not in allowed:
            continue
        out.add(sym)
    return out


def sample_screen_rows(
    rows: object,
    n: int,
    *,
    exchanges: Sequence[str] = ("NASDAQ", "NYSE"),
    seed: int | None = None,
) -> list[str]:
    """Filter screener rows to clean common-stock tickers and sample ``n`` (pure)."""
    return _sample(_clean_symbols(rows, exchanges=exchanges), n, seed=seed)


def _warn_if_truncated(
    bars: pd.DataFrame, requested_lo: pd.Timestamp, requested_hi: pd.Timestamp, tf: Timeframe
) -> None:
    """Warn when the vendor returned far less history than requested.

    FMP caps intraday history (often ~a few months) regardless of the ``from``
    date, so a multi-year intraday request silently returns a short recent
    window. Surfacing this prevents a backtest from running on a much shorter
    span than the user asked for.
    """
    if bars.empty:
        return
    actual_lo = bars["timestamp"].min()
    requested_days = max((requested_hi - requested_lo).days, 1)
    if requested_days <= 60:
        return
    coverage = (requested_hi - actual_lo).days / requested_days
    gap_days = (actual_lo - requested_lo).days
    if coverage < 0.8 and gap_days > 30:
        warnings.warn(
            f"Requested {tf.value} bars from {requested_lo.date()} but the vendor "
            f"only returned data from {actual_lo.date()} "
            f"(~{coverage:.0%} of the requested span). FMP caps intraday history; "
            f"your backtest will run on the SHORTER window. Use daily bars for long "
            f"histories, or a longer-history intraday source.",
            DataQualityWarning,
            stacklevel=3,
        )
