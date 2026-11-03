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
from collections import OrderedDict
from datetime import date, timedelta
from typing import Any, cast

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


class Charts:
    def __init__(self, provider: DataProvider) -> None:
        self.provider = provider
        self._cache: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def symbol(
        self, symbol: str, start: str, end: str, *, dial: str = "vix_vote"
    ) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        if not symbol or len(symbol) > 12 or not all(c.isalnum() or c in ".-^" for c in symbol):
            raise ValueError("that does not look like a ticker")
        date.fromisoformat(start)
        date.fromisoformat(end)
        key = (symbol, start, end)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit

        data_start = (date.fromisoformat(start) - timedelta(days=WARMUP_DAYS)).isoformat()
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
        overlays: dict[str, list[dict[str, Any]]] = {}
        for w in SMA_WINDOWS:
            s = cast(pd.Series, signals.sma(close, w))
            overlays[f"sma{w}"] = _series(s[s.index >= lo])

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


def _series(s: pd.Series) -> list[dict[str, Any]]:
    out = []
    for d, v in zip(session_dates(s.index), s.to_numpy(dtype=float), strict=True):
        if math.isfinite(v):
            out.append({"time": d, "value": float(v)})
    return out
