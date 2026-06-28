"""FMP adapter: implements ``DataProvider`` by wrapping the existing FMP tooling.

This is the ONLY module in the platform that knows the data vendor is Financial
Modeling Prep. It delegates network/rate-limit/parallelism to the legacy
``FinancialDataCompiler`` (unchanged), then funnels the result through
``normalize_bars`` so callers receive canonical bars.

Normalization is split out as a module-level pure function
(:func:`normalize_fmp_records`) so it can be unit-tested without any network.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd

from ehf.server.research.data.cache import BarCache
from ehf.server.research.data.provider import DataProvider, Universe
from ehf.server.research.data.quality import DataQualityWarning, warn_survivorship
from ehf.server.research.data.schema import Timeframe, normalize_bars, validate_bars

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
            from ehf.server.tools.financial.data.tool import (
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

    def _fetch(
        self, symbols: list[str], start: str, end: str, tf: Timeframe, adjusted: bool
    ) -> pd.DataFrame:
        """Hit the wrapped FMP tooling and normalize. Isolated for mocking."""
        if tf is Timeframe.DAY:
            # Daily isn't chunked; the compiler already fans symbols out in parallel.
            raw = self.compiler.compile_price_data(
                symbols=symbols, from_date=start, to_date=end, interval=_FMP_INTERVAL[tf]
            )
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
        """Build a universe from FMP's screener / stock list.

        FMP only exposes CURRENT membership, so any historical ``as_of`` is
        survivorship-biased — we warn and flag it.
        """
        symbols = self._screen_symbols(name)
        warn_survivorship(name, as_of)
        as_of_d = as_of if isinstance(as_of, date) else date.fromisoformat(str(as_of))
        return Universe(name=name, as_of=as_of_d, symbols=symbols, survivorship_safe=False)

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
        if not (_is_iso_date(ipo) and _is_iso_date(delisted) and ipo <= as_of_s < delisted):
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
