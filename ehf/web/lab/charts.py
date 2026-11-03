"""Any ticker, charted: candles, volume, moving averages, and the fear gauge.

The symbol's own sessions anchor the chart. If the VIX family loads, the page
also gets the VIX itself and the exposure the ``vix_vote`` dial would have held
in that symbol — so you can see what "in" and "out" would have looked like on
any stock, not just SPY.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from ehf.research import signals
from ehf.research.backtest.panel import Panel
from ehf.research.data.provider import DataProvider
from ehf.research.data.schema import Timeframe
from ehf.research.data.vix import VIX, load_vix_bars
from ehf.research.strategies.base import build_strategy
from ehf.web.lab.data import session_dates

log = logging.getLogger(__name__)

SMA_WINDOWS = (50, 200)
WARMUP_DAYS = 400
MAX_LINES = 4  # per kind (SMA, EMA)
MAX_WINDOW = 500


def parse_windows(raw: str | None, *, default: tuple[int, ...] = ()) -> tuple[int, ...]:
    """``"20, 50"`` -> (20, 50); empty means none, ``None`` means the default."""
    if raw is None:
        return default
    out: list[int] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except ValueError:
            raise ValueError(f"not a whole number of days: {part!r}") from None
        if not 2 <= n <= MAX_WINDOW:
            raise ValueError(f"a window must be between 2 and {MAX_WINDOW} days, not {n}")
        if n not in out:
            out.append(n)
    if len(out) > MAX_LINES:
        raise ValueError(f"at most {MAX_LINES} lines per indicator")
    return tuple(out)


def parse_indicators(
    sma: str | None = None, ema: str | None = None, bb: str | None = None, rsi: str | None = None
) -> dict[str, Any]:
    """Validate the chart's indicator settings (query-string strings) into one dict.

    ``sma``/``ema``: comma lists of windows; ``bb``: ``"window,n_std"``; ``rsi``: a
    window. Anything absent falls back to the page's defaults (SMA 50 and 200,
    nothing else).
    """
    out: dict[str, Any] = {
        "sma": parse_windows(sma, default=SMA_WINDOWS),
        "ema": parse_windows(ema),
        "bb": None,
        "rsi": None,
    }
    if bb:
        parts = [x.strip() for x in bb.split(",")]
        try:
            window = int(parts[0])
            n_std = float(parts[1]) if len(parts) > 1 and parts[1] else 2.0
        except ValueError:
            raise ValueError("Bollinger wants 'window,standard deviations', e.g. 20,2") from None
        if not 2 <= window <= MAX_WINDOW or not 0.5 <= n_std <= 5.0:
            raise ValueError("Bollinger window 2-500 days, width 0.5-5 standard deviations")
        out["bb"] = {"window": window, "n_std": n_std}
    if rsi:
        (window,) = parse_windows(rsi) or (0,)
        if window:
            out["rsi"] = {"window": window}
    return out


def indicator_key(ind: dict[str, Any]) -> str:
    bb, r = ind.get("bb"), ind.get("rsi")
    return "|".join(
        [
            ",".join(map(str, ind.get("sma", ()))),
            ",".join(map(str, ind.get("ema", ()))),
            f"{bb['window']},{bb['n_std']:g}" if bb else "",
            str(r["window"]) if r else "",
        ]
    )


def max_window(ind: dict[str, Any]) -> int:
    wins = [*ind.get("sma", ()), *ind.get("ema", ())]
    if ind.get("bb"):
        wins.append(int(ind["bb"]["window"]))
    if ind.get("rsi"):
        wins.append(int(ind["rsi"]["window"]))
    return max(wins, default=0)


TAIL_TTL_S = 900.0  # re-pull the last days of bars at most this often per symbol
REFRESH_MIN_S = 60.0  # ...and no more often than this even when someone hits Refresh
TAIL_DAYS = 10
INTRADAY_TTL_S = 60.0  # minute bars are ~1 min behind the tape; no point asking more often
INTRADAY_MIN_S = 15.0  # ...but let Refresh pull a little sooner
ET = ZoneInfo("America/New_York")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Charts:
    """Any-ticker chart payloads, cache-first, with a freshness guarantee.

    The provider's bar cache counts a range as covered when it ends within a few
    days of today, so a symbol looked at on Monday would still show Monday's close
    on Thursday. ``symbol`` therefore re-pulls the tail (the last ``TAIL_DAYS``) at
    most every ``TAIL_TTL_S`` per symbol, or on demand through ``refresh=True``
    (throttled to ``REFRESH_MIN_S``), and attaches a live quote on every call.
    """

    def __init__(
        self,
        provider: DataProvider,
        *,
        tail_ttl_s: float = TAIL_TTL_S,
        refresh_min_s: float = REFRESH_MIN_S,
    ) -> None:
        self.provider = provider
        self.tail_ttl_s = tail_ttl_s
        self.refresh_min_s = refresh_min_s
        self._cache: OrderedDict[tuple[str, str, str, str], dict[str, Any]] = OrderedDict()
        self._tail: dict[str, float] = {}  # symbol -> when its tail was last pulled
        self._intra: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    # -- freshness -------------------------------------------------------------------------

    def freshen(self, symbol: str, *, force: bool = False) -> bool:
        """Pull the last days of ``symbol`` (and the VIX) if due; True when it did."""
        refresh = getattr(self.provider, "refresh_tail", None)
        if not callable(refresh):
            return False
        now = time.time()
        with self._lock:
            last = self._tail.get(symbol)
            due = last is None or now - last >= (self.refresh_min_s if force else self.tail_ttl_s)
            if not due:
                return False
            self._tail[symbol] = now  # claim it before the network call
        try:
            refresh([symbol, VIX], days=TAIL_DAYS)
        except Exception as exc:  # stale beats nothing; the next view retries
            log.warning("charts: tail refresh of %s failed: %s", symbol, exc)
            with self._lock:
                self._tail.pop(symbol, None)
            return False
        with self._lock:
            for k in [k for k in self._cache if k[0] == symbol]:
                del self._cache[k]
        return True

    def live(self, symbol: str) -> dict[str, Any]:
        """What the browser needs to say how fresh the page is (never cached)."""
        quote: dict[str, Any] | None = None
        q = getattr(self.provider, "quote", None)
        if callable(q):
            try:
                quote = q([symbol]).get(symbol)
            except Exception as exc:
                log.info("charts: quote for %s unavailable: %s", symbol, exc)
        with self._lock:
            pulled = self._tail.get(symbol)
        return {
            "quote": quote,
            "bars_pulled_at": datetime.fromtimestamp(pulled, UTC).isoformat(timespec="seconds")
            if pulled
            else None,
            "generated_at": now_iso(),
        }

    # -- the payload -----------------------------------------------------------------------

    def symbol(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        dial: str = "vix_vote",
        refresh: bool = False,
        indicators: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        if not symbol or len(symbol) > 12 or not all(c.isalnum() or c in ".-^" for c in symbol):
            raise ValueError("that does not look like a ticker")
        date.fromisoformat(start)
        date.fromisoformat(end)
        ind = indicators if indicators is not None else parse_indicators()
        self.freshen(symbol, force=refresh)
        return {**self._payload(symbol, start, end, dial, ind), "live": self.live(symbol)}

    # -- today, minute by minute ---------------------------------------------------------

    def intraday(self, symbol: str, *, refresh: bool = False) -> dict[str, Any]:
        """The latest session in 1-minute candles, with the VIX alongside.

        Pulled straight from the vendor (never the bar cache, which is daily and
        would count a partial day as covered), kept for ``INTRADAY_TTL_S``.
        """
        symbol = symbol.strip().upper()
        if not symbol or len(symbol) > 12 or not all(c.isalnum() or c in ".-^" for c in symbol):
            raise ValueError("that does not look like a ticker")
        fetch = getattr(self.provider, "intraday", None)
        if not callable(fetch):
            raise LookupError("intraday bars are not available from this data source")
        now = time.time()
        with self._lock:
            hit = self._intra.get(symbol)
        if hit and now - hit[0] < (INTRADAY_MIN_S if refresh else INTRADAY_TTL_S):
            return {**hit[1], "live": self.live(symbol)}

        bars = fetch(symbol)
        if bars is None or bars.empty:
            raise LookupError(f"no intraday bars for {symbol}")
        bars = bars.sort_values("timestamp")
        et = bars["timestamp"].dt.tz_convert(ET)
        session = et.iloc[-1].date()
        day = bars[et.dt.date == session]
        vix_series: list[dict[str, Any]] = []
        if symbol != VIX:
            try:
                v = fetch(VIX)
                if v is not None and not v.empty:
                    v = v.sort_values("timestamp")
                    v = v[v["timestamp"].dt.tz_convert(ET).dt.date == session]
                    vix_series = [
                        {"time": _et_epoch(ts), "value": float(c)}
                        for ts, c in zip(v["timestamp"], v["close"], strict=True)
                    ]
            except Exception as exc:
                log.info("intraday VIX unavailable: %s", exc)

        prev_close = self._prev_close(symbol, session)
        candles = [
            {
                "time": _et_epoch(ts),
                "open": float(o),
                "high": float(h),
                "low": float(lw),
                "close": float(c),
                "volume": float(vol) if math.isfinite(float(vol)) else 0.0,
            }
            for ts, o, h, lw, c, vol in zip(
                day["timestamp"],
                day["open"],
                day["high"],
                day["low"],
                day["close"],
                day["volume"],
                strict=True,
            )
        ]
        last_close = candles[-1]["close"]
        payload = {
            "symbol": symbol,
            "interval": "1min",
            "session": session.isoformat(),
            "candles": candles,
            "vix": vix_series,
            "prev_close": prev_close,
            "stats": {
                "open": candles[0]["open"],
                "high": max(c["high"] for c in candles),
                "low": min(c["low"] for c in candles),
                "last": last_close,
                "last_time": et.iloc[-1].isoformat(timespec="minutes"),
                "change": (last_close / prev_close - 1.0) if prev_close else None,
                "volume": sum(c["volume"] for c in candles),
                "bars": len(candles),
            },
        }
        with self._lock:
            self._intra[symbol] = (now, payload)
            while len(self._intra) > 32:
                self._intra.pop(next(iter(self._intra)))
        return {**payload, "live": self.live(symbol)}

    def _prev_close(self, symbol: str, session: date) -> float | None:
        """The daily close before ``session`` (what the day's change is measured from)."""
        try:
            raw = self.provider.get_bars(
                [symbol],
                (session - timedelta(days=12)).isoformat(),
                session.isoformat(),
                Timeframe.DAY,
            )
        except Exception as exc:
            log.info("prev close for %s unavailable: %s", symbol, exc)
            return None
        own = raw.loc[raw["symbol"] == symbol].sort_values("timestamp")
        before = own[own["timestamp"].dt.tz_convert(ET).dt.date < session]
        return float(before["close"].iloc[-1]) if not before.empty else None

    def _payload(
        self, symbol: str, start: str, end: str, dial: str, ind: dict[str, Any]
    ) -> dict[str, Any]:
        key = (symbol, start, end, indicator_key(ind))
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit

        # enough history before ``start`` that the longest window is warm on day one
        warm = max(WARMUP_DAYS, int(max_window(ind) * 1.6) + 30)
        data_start = (date.fromisoformat(start) - timedelta(days=warm)).isoformat()
        vix_bars: pd.DataFrame | None = None
        try:
            vix_bars = load_vix_bars(
                self.provider, data_start, end, asset=symbol, offline_skew=True
            )
            bars = vix_bars.loc[vix_bars["symbol"] == symbol].reset_index(drop=True)
        except Exception as exc:  # no VIX family (or the symbol is itself an index)
            log.info("vix dataset unavailable for %s (%s); plain bars", symbol, exc)
            raw = self.provider.get_bars([symbol], data_start, end, Timeframe.DAY)
            bars = raw.loc[raw["symbol"] == symbol].reset_index(drop=True)
        if bars.empty:
            raise LookupError(f"no bars for {symbol}")

        bars = bars.sort_values("timestamp")
        close = bars.set_index("timestamp")["close"]
        lo = pd.Timestamp(start, tz="UTC")
        overlays: dict[str, Any] = {}
        for w in ind["sma"]:
            s = cast(pd.Series, signals.sma(close, w))
            overlays[f"sma{w}"] = _series(s[s.index >= lo])
        for w in ind["ema"]:
            s = cast(pd.Series, signals.ema(close, w))
            overlays[f"ema{w}"] = _series(s[s.index >= lo])
        if ind["bb"]:
            mid, upper, lower = signals.bollinger(close, ind["bb"]["window"], ind["bb"]["n_std"])
            overlays["bb"] = {
                **ind["bb"],
                "mid": _series(cast(pd.Series, mid)[mid.index >= lo]),
                "upper": _series(cast(pd.Series, upper)[upper.index >= lo]),
                "lower": _series(cast(pd.Series, lower)[lower.index >= lo]),
            }
        if ind["rsi"]:
            r = cast(pd.Series, signals.rsi(close, ind["rsi"]["window"]))
            overlays["rsi"] = {**ind["rsi"], "values": _series(r[r.index >= lo])}

        vix_series: list[dict[str, Any]] = []
        exposure: list[dict[str, Any]] = []
        if vix_bars is not None:
            panel = Panel.from_bars(vix_bars)
            if VIX in panel.close.columns:
                v = panel.close[VIX]
                vix_series = _series(v[v.index >= lo])
            try:
                strat = build_strategy(dial, {"asset": symbol})
                e = strat.exposure(panel).reindex(panel.index).clip(0.0, 1.0).fillna(0.0)  # type: ignore[attr-defined]
                exposure = _series(e[e.index >= lo])
            except Exception as exc:
                log.info("dial %s failed on %s: %s", dial, symbol, exc)

        shown = bars[bars["timestamp"] >= lo]
        candles = [
            {
                "time": d,
                "open": float(o),
                "high": float(h),
                "low": float(lw),
                "close": float(c),
                "volume": float(v) if math.isfinite(float(v)) else 0.0,
            }
            for d, o, h, lw, c, v in zip(
                session_dates(shown["timestamp"]),
                shown["open"],
                shown["high"],
                shown["low"],
                shown["close"],
                shown["volume"],
                strict=True,
            )
        ]
        last = shown.iloc[-1] if not shown.empty else None
        prev = shown.iloc[-2] if len(shown) > 1 else None
        payload = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "candles": candles,
            "overlays": overlays,
            "indicators": {
                "sma": list(ind["sma"]),
                "ema": list(ind["ema"]),
                "bb": ind["bb"],
                "rsi": ind["rsi"],
            },
            "vix": vix_series,
            "exposure": exposure,
            "dial": dial,
            "last": {
                "date": session_dates([last["timestamp"]])[0] if last is not None else None,
                "close": float(last["close"]) if last is not None else None,
                "change": (float(last["close"]) / float(prev["close"]) - 1.0)
                if last is not None and prev is not None and float(prev["close"])
                else None,
                "high_52w": float(shown["high"].tail(252).max()) if last is not None else None,
                "low_52w": float(shown["low"].tail(252).min()) if last is not None else None,
            },
        }
        with self._lock:
            self._cache[key] = payload
            while len(self._cache) > 16:
                self._cache.popitem(last=False)
        return payload


def _et_epoch(ts: pd.Timestamp) -> int:
    """Seconds since the epoch of the *Eastern wall-clock* time, read as if UTC.

    Lightweight Charts labels unix times in UTC; shifting them by the Eastern
    offset makes the axis read 9:30 to 4:00 for every viewer, which is what a
    session chart should say.
    """
    local = ts.tz_convert(ET).tz_localize(None)
    return int((local - pd.Timestamp("1970-01-01")).total_seconds())


def _series(s: pd.Series) -> list[dict[str, Any]]:
    out = []
    for d, v in zip(session_dates(s.index), s.to_numpy(dtype=float), strict=True):
        if math.isfinite(v):
            out.append({"time": d, "value": float(v)})
    return out
