"""The playbook: the rules that earned a place, read live, with what they say today.

The VIX study (``docs``: *The Fear Gauge, Put to the Test*) blessed two of the
nine rules: the **surprise count** (rule 6) as the dial and the **fear curve
flip** (rule 1) as the sanity check, with one combination rule: when both say
out, that is the strongest signal; when they disagree, half in. This page makes
that actionable. Every number comes from the same strategy classes and signal
code the backtests ran (``ehf.research.strategies.vix_overlay``), on the same
cached bars, with the last few sessions re-fetched from the vendor so the
reading is as of the latest close.

    GET /playbook             the page (server-rendered state + charts)
    GET /playbook/api/today   the payload behind it
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ehf.research import signals
from ehf.research.backtest.panel import Panel
from ehf.research.data.provider import DataProvider
from ehf.research.data.schema import Timeframe
from ehf.research.data.vix import VIX, VIX3M, align_to_sessions
from ehf.research.strategies.base import build_strategy
from ehf.web.lab.data import LabData, session_dates
from ehf.web.lab.routes import Page

log = logging.getLogger(__name__)

ASSET = "SPY"
HISTORY_YEARS = 3  # what the charts show
DATA_YEARS = 4  # what is loaded (warm-up for the rolling windows)


@dataclass(frozen=True)
class Rule:
    key: str  # registered strategy key
    number: int  # numbering from the explainer
    title: str
    role: str  # "the dial" / "the sanity check"
    reading: str  # what the number is
    reading_fmt: str  # "pct" | "ratio"
    params: dict[str, Any]  # the settings the study used (the defaults)
    in_below: float
    out_above: float
    plain: str  # one paragraph for someone who does not do this for a living


RULES: tuple[Rule, ...] = (
    Rule(
        key="vix_surprise",
        number=6,
        title="Surprise count",
        role="the dial",
        reading="share of the last 21 sessions that beat the VIX-implied move",
        reading_fmt="pct",
        params={"window": 21, "out_above": 0.35, "band": 0.10},
        in_below=0.25,
        out_above=0.35,
        plain=(
            "Each day the VIX implies how big a move the market expects, roughly VIX divided "
            "by 16, in percent. The surprise count is the share of the last 21 trading days "
            "whose actual move was bigger than that. Under 25%: the market is behaving, stay "
            "fully in. Between 25% and 35%: it is getting surprised, go half in. Over 35%: it "
            "is being surprised again and again, get out. In the test this kept almost all of "
            "the market's return while cutting the worst loss by about half, and it gets back "
            "in early because surprises stop long before prices recover."
        ),
    ),
    Rule(
        key="vix_term",
        number=1,
        title="Fear curve flip",
        role="the sanity check",
        reading="VIX divided by VIX3M",
        reading_fmt="ratio",
        params={"out_above": 1.0, "band": 0.05, "confirm_bars": 1},
        in_below=0.95,
        out_above=1.0,
        plain=(
            "The VIX prices fear over the next 30 days; the VIX3M prices it over three months. "
            "Normally the longer one is higher, because more can go wrong in three months than "
            "in one. When 30-day fear climbs above 3-month fear (the ratio goes over 1.00) the "
            "curve has flipped: something is happening right now, get out. Back in once the "
            "ratio falls under 0.95. It is the simplest switch in the study, it barely cares "
            "about its settings, and it was the best of all in the fast crashes."
        ),
    ),
)


def dial_label(exposure: float | None) -> str:
    """The same three words the daily report uses."""
    if exposure is None or not math.isfinite(exposure):
        return "unknown"
    if exposure >= 0.75:
        return "IN"
    if exposure <= 0.25:
        return "OUT"
    return "HALF"


def combine(dial: str, check: str) -> tuple[str, str]:
    """The playbook's answer from the two rules: (state, plain sentence)."""
    if dial == "unknown" or check == "unknown":
        return "unknown", "One of the rules has no reading today, so the playbook is silent."
    if dial == "OUT" and check == "OUT":
        return "OUT", (
            "Both rules say OUT. That is the strongest signal this playbook gives: "
            "be out of the market, in cash."
        )
    if dial == "IN" and check == "IN":
        return "IN", "Both rules say IN. Be fully invested."
    return "HALF", (
        f"The rules disagree: the surprise count says {dial}, the fear curve says {check}. "
        "When they disagree the playbook's answer is half in."
    )


def switches(times: list[str], labels: list[str], limit: int = 12) -> list[dict[str, str]]:
    """The last ``limit`` state changes, newest first."""
    out: list[dict[str, str]] = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            out.append({"date": times[i], "from": labels[i - 1], "to": labels[i]})
    return list(reversed(out[-limit:]))


def streak(labels: list[str]) -> int:
    n = 0
    for lab in reversed(labels):
        if lab != labels[-1]:
            break
        n += 1
    return n


def _f(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _series(s: pd.Series) -> list[float | None]:
    return [_f(v) for v in s.to_numpy()]


class Playbook:
    """Builds (and caches for ``ttl_s``) the page payload from the provider's bars."""

    def __init__(
        self,
        provider: DataProvider,
        data: LabData | None = None,
        *,
        ttl_s: float = 900.0,
        rules: tuple[Rule, ...] = RULES,
        asset: str = ASSET,
    ) -> None:
        self.provider = provider
        self.data = data
        self.ttl_s = ttl_s
        self.rules = rules
        self.asset = asset
        self._cache: tuple[float, dict[str, Any]] | None = None
        self._lock = threading.Lock()

    # -- data ----------------------------------------------------------------------------

    def bars(self, end: date) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        symbols = [self.asset, VIX, VIX3M]
        refresh = getattr(self.provider, "refresh_tail", None)
        if callable(refresh):
            try:
                refresh(symbols, days=10, end=end)
            except Exception as exc:  # stale cache beats no page
                log.warning("playbook: tail refresh failed, using the cache: %s", exc)
        start = date(end.year - DATA_YEARS, end.month, min(end.day, 28))
        raw = self.provider.get_bars(symbols, start, end, Timeframe.DAY)
        asset_bars = raw[raw["symbol"] == self.asset].sort_values("timestamp")
        asset_bars = asset_bars.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        if asset_bars.empty:
            raise LookupError(f"no bars for {self.asset}")
        sessions = pd.DatetimeIndex(asset_bars["timestamp"])
        index_bars = raw[raw["symbol"] != self.asset]
        aligned = align_to_sessions(index_bars, sessions) if not index_bars.empty else index_bars
        bars = pd.concat([asset_bars, aligned], ignore_index=True)
        bars.attrs["timeframe"] = Timeframe.DAY.value
        return bars, sessions

    # -- the payload -----------------------------------------------------------------------

    def today(self, end: date | None = None) -> dict[str, Any]:
        end = end or date.today()
        with self._lock:
            if self._cache and time.time() - self._cache[0] < self.ttl_s:
                return self._cache[1]
        payload = self.build(end)
        with self._lock:
            self._cache = (time.time(), payload)
        return payload

    def build(self, end: date) -> dict[str, Any]:
        bars, sessions = self.bars(end)
        panel = Panel.from_bars(bars)
        close = panel.close
        for col in (self.asset, VIX, VIX3M):
            if col not in close.columns:
                raise LookupError(f"{col} is missing from the data")
        spy, vix, vix3m = close[self.asset], close[VIX], close[VIX3M]
        times = session_dates(panel.index)
        cut = max(0, len(times) - int(HISTORY_YEARS * 252))

        rules_out: list[dict[str, Any]] = []
        states: list[str] = []
        for rule in self.rules:
            strat = build_strategy(rule.key, dict(rule.params))
            exposure = strat.generate_signals(panel)[self.asset].astype(float)
            reading = self._reading(rule, spy, vix, vix3m)
            labels = [dial_label(_f(e)) for e in exposure.to_numpy()]
            state = labels[-1]
            states.append(state)
            days = streak(labels)
            since = times[len(times) - days] if days and days <= len(times) else times[0]
            rules_out.append(
                {
                    "key": rule.key,
                    "number": rule.number,
                    "title": rule.title,
                    "role": rule.role,
                    "reading_label": rule.reading,
                    "reading_fmt": rule.reading_fmt,
                    "reading": _f(reading.iloc[-1]),
                    "prev_reading": _f(reading.iloc[-2]) if len(reading) > 1 else None,
                    "in_below": rule.in_below,
                    "out_above": rule.out_above,
                    "exposure": _f(exposure.iloc[-1]),
                    "state": state,
                    "days_in_state": days,
                    "since": since,
                    "switches": switches(times, labels),
                    "params": dict(rule.params),
                    "plain": rule.plain,
                    "study": self._study(rule.key),
                    "series": {
                        "time": times[cut:],
                        "reading": _series(reading.iloc[cut:]),
                        "exposure": _series(exposure.iloc[cut:]),
                    },
                }
            )

        combined_state, combined_text = combine(states[0], states[1] if len(states) > 1 else "IN")
        sma200 = cast(pd.Series, signals.sma(spy, 200))
        last_spy, prev_spy = _f(spy.iloc[-1]), _f(spy.iloc[-2]) if len(spy) > 1 else None
        last_vix = _f(vix.iloc[-1])
        last_sma = _f(sma200.iloc[-1])
        return {
            "as_of": times[-1],
            "generated": date.today().isoformat(),
            "asset": self.asset,
            "combined": {"state": combined_state, "text": combined_text},
            "market": {
                "close": last_spy,
                "change": (last_spy / prev_spy - 1.0) if last_spy and prev_spy else None,
                "vix": last_vix,
                "vix3m": _f(vix3m.iloc[-1]),
                "implied_move": (last_vix / 16.0 / 100.0) if last_vix else None,
                "sma200": last_sma,
                "vs_sma200": (last_spy / last_sma - 1.0) if last_spy and last_sma else None,
            },
            "rules": rules_out,
            "series": {"time": times[cut:], "close": _series(spy.iloc[cut:])},
        }

    def _reading(self, rule: Rule, spy: pd.Series, vix: pd.Series, vix3m: pd.Series) -> pd.Series:
        if rule.key == "vix_surprise":
            return signals.surprise_rate(spy, vix, int(rule.params["window"]))
        if rule.key == "vix_term":
            return signals.term_structure_ratio(vix, vix3m)
        raise KeyError(rule.key)

    def _study(self, key: str) -> dict[str, Any] | None:
        """The latest walk-forward scorecard for the rule, trimmed to what the page shows."""
        if self.data is None:
            return None
        try:
            cards = [s for s in self.data.scorecards() if s.strategy == key]
        except Exception as exc:
            log.warning("playbook: scorecards unavailable: %s", exc)
            return None
        if not cards:
            return None
        card = cards[0]
        m = card.payload.get("oos_metrics", {}) or {}
        extra = card.payload.get("extra", {}) or {}
        wf = card.payload.get("walk_forward", {}) or {}
        return {
            "wf_id": card.wf_id,
            "run": card.full_hash or card.stitched_hash,
            "n_windows": int(m.get("n_windows") or 0),
            "cagr": _f(m.get("cagr")),
            "max_drawdown": _f(m.get("max_drawdown")),
            "sharpe": _f(m.get("sharpe")),
            "switches_per_year": _f(extra.get("switches_per_year")),
            "time_in_market": _f(extra.get("time_in_market")),
            "dd_2020": _f(extra.get("dd_2020")),
            "dd_2022": _f(extra.get("dd_2022")),
            "dd_2025": _f(extra.get("dd_2025")),
            "train_months": wf.get("train_months"),
            "test_months": wf.get("test_months"),
        }


def mount_playbook(app: FastAPI, page: Page, playbook: Playbook) -> None:
    @app.get("/playbook", response_class=HTMLResponse)
    def playbook_page(request: Request) -> HTMLResponse:
        try:
            today = playbook.today()
            error = None
        except Exception as exc:
            log.exception("playbook failed")
            today, error = None, f"{type(exc).__name__}: {exc}"
        return page(request, "playbook.html", today=today, error=error)

    @app.get("/playbook/api/today")
    def playbook_api() -> JSONResponse:
        return JSONResponse(playbook.today())


__all__ = ["Playbook", "Rule", "RULES", "combine", "dial_label", "mount_playbook", "switches"]
