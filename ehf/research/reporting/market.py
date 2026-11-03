"""Pure market-level context: regime, breadth, macro, calendars.

Everything here is a plain ``dict`` so it serialises straight into the snapshot
JSON and the narrative prompt.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ehf.research.reporting.config import ReportConfig
from ehf.research.signals import core as sig
from ehf.research.signals import vix as vixsig

Json = dict[str, Any]


def _f(x: object) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def index_context(spy_bars: pd.DataFrame) -> Json:
    """SPY trend + realized-vol regime from its own daily bars (pure)."""
    if spy_bars.empty:
        return {}
    s = spy_bars.sort_values("date").set_index("date")["close"].astype(float)
    rets = s.pct_change()
    vol20 = rets.rolling(20).std() * np.sqrt(252.0)
    sma50 = sig.sma(s, 50)
    sma200 = sig.sma(s, 200)
    vol_hist = vol20.dropna()
    years = pd.DatetimeIndex(s.index).year
    vol_pct = float((vol_hist <= vol_hist.iloc[-1]).mean()) if len(vol_hist) > 20 else None
    out: Json = {
        "as_of": str(s.index[-1].date()),
        "close": _f(s.iloc[-1]),
        "ret_1d": _f(rets.iloc[-1]),
        "ret_5d": _f(s.iloc[-1] / s.iloc[-6] - 1) if len(s) > 6 else None,
        "ret_21d": _f(s.iloc[-1] / s.iloc[-22] - 1) if len(s) > 22 else None,
        "ret_63d": _f(s.iloc[-1] / s.iloc[-64] - 1) if len(s) > 64 else None,
        "ret_ytd": _f(s.iloc[-1] / s.loc[years == years[-1]].iloc[0] - 1),
        "pct_from_252d_high": _f(s.iloc[-1] / s.tail(252).max() - 1),
        "above_sma50": bool(s.iloc[-1] > sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None,
        "above_sma200": bool(s.iloc[-1] > sma200.iloc[-1])
        if not np.isnan(sma200.iloc[-1])
        else None,
        "sma50_above_sma200": (
            bool(sma50.iloc[-1] > sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None
        ),
        "realized_vol_20d": _f(vol20.iloc[-1]),
        "realized_vol_20d_pctile_1y": vol_pct,
        "drawdown_from_252d_high": _f(s.iloc[-1] / s.tail(252).max() - 1),
    }
    out["regime"] = classify_regime(out)
    return out


def classify_regime(ctx: Json) -> str:
    """Coarse, honest label: trend from the 200dma, stress from vol percentile."""
    above200 = ctx.get("above_sma200")
    vol_pct = ctx.get("realized_vol_20d_pctile_1y")
    if above200 is None:
        return "unknown"
    stressed = vol_pct is not None and vol_pct >= 0.85
    if above200 and not stressed:
        return "risk-on (uptrend, calm vol)"
    if above200 and stressed:
        return "uptrend under stress (vol spike inside an uptrend)"
    if not above200 and stressed:
        return "risk-off (downtrend, high vol)"
    return "downtrend, calm vol (grind lower)"


def breadth(features: pd.DataFrame) -> Json:
    """Universe breadth from the feature table (pure)."""
    if features.empty:
        return {}
    n = int(len(features))
    med = features.median(numeric_only=True)
    return {
        "n_symbols": n,
        "pct_above_sma50": _f(features["above_50"].mean()),
        "pct_above_sma200": _f(features["above_200"].mean()),
        "new_52w_highs": int(features["new_high_252"].sum()),
        "new_52w_lows": int(features["new_low_252"].sum()),
        "pct_up_1d": _f((features["ret_1d"] > 0).mean()),
        "median_ret_1d": _f(med.get("ret_1d")),
        "median_ret_21d": _f(med.get("ret_21d")),
        "pct_down_20_from_52w_high": _f((features["pct_from_high_252"] <= -0.20).mean()),
        "median_rsi_14": _f(med.get("rsi_14")),
    }


def treasury_context(rows: list[Json]) -> Json:
    """Latest curve points + slopes (pure). FMP rows: month3, year2, year10, ..."""
    if not rows:
        return {}
    latest = sorted(rows, key=lambda r: str(r.get("date", "")))[-1]
    y10, y2, m3 = _f(latest.get("year10")), _f(latest.get("year2")), _f(latest.get("month3"))
    return {
        "date": latest.get("date"),
        "3m": m3,
        "2y": y2,
        "10y": y10,
        "30y": _f(latest.get("year30")),
        "slope_10y_2y": (y10 - y2) if (y10 is not None and y2 is not None) else None,
        "slope_10y_3m": (y10 - m3) if (y10 is not None and m3 is not None) else None,
    }


def sector_context(rows: list[Json], *, exchange: str = "NASDAQ") -> list[Json]:
    """Sector snapshot for one exchange, sorted best to worst (pure)."""
    picked = [r for r in rows if str(r.get("exchange", "")).upper() == exchange.upper()] or rows
    out = [
        {"sector": r.get("sector"), "avg_change_pct": _f(r.get("averageChange"))} for r in picked
    ]
    return sorted(out, key=lambda r: -(r["avg_change_pct"] or 0.0))


def econ_context(
    rows: list[Json], *, country: str = "US", impacts: tuple[str, ...] = ("High",)
) -> list[Json]:
    """Upcoming high-impact releases (pure)."""
    keep = [
        r
        for r in rows
        if str(r.get("country", "")).upper() == country and str(r.get("impact", "")) in impacts
    ]
    keep.sort(key=lambda r: str(r.get("date", "")))
    return [
        {
            "date": r.get("date"),
            "event": r.get("event"),
            "previous": r.get("previous"),
            "estimate": r.get("estimate"),
            "actual": r.get("actual"),
        }
        for r in keep
    ]


def earnings_ahead(
    calendar_rows: list[Json], universe: pd.DataFrame, *, after: date, min_market_cap: float = 25e9
) -> list[Json]:
    """Large-cap universe names reporting in the look-ahead window (pure)."""
    caps = (
        universe.set_index("symbol")["market_cap"] if not universe.empty else pd.Series(dtype=float)
    )
    names = universe.set_index("symbol")["name"] if not universe.empty else pd.Series(dtype=object)
    out: list[Json] = []
    for r in calendar_rows:
        sym = r.get("symbol")
        d = str(r.get("date", ""))[:10]
        if sym not in caps.index or d <= after.isoformat():
            continue
        cap = _f(caps.get(sym))
        if cap is None or cap < min_market_cap:
            continue
        out.append(
            {
                "symbol": sym,
                "name": names.get(sym),
                "date": d,
                "eps_estimate": _f(r.get("epsEstimated")),
                "revenue_estimate": _f(r.get("revenueEstimated")),
                "market_cap": cap,
            }
        )
    out.sort(key=lambda r: (r["date"], -(r["market_cap"] or 0)))
    return out


def vix_context(quote: Json) -> Json:
    if not quote:
        return {}
    return {
        "level": _f(quote.get("price")),
        "change_pct": _f(quote.get("changePercentage")),
        "avg_50d": _f(quote.get("priceAvg50")),
        "year_high": _f(quote.get("yearHigh")),
        "year_low": _f(quote.get("yearLow")),
    }


# --- the market dial --------------------------------------------------------------

VIX_LONG_SYMBOL = "^VIX3M"


def dial_label(exposure: float | None) -> str:
    if exposure is None:
        return "unknown"
    if exposure >= 0.75:
        return "IN"
    if exposure <= 0.25:
        return "OUT"
    return "HALF"


def dial_context(spy_bars: pd.DataFrame, vix_bars: pd.DataFrame, cfg: ReportConfig) -> Json:
    """The in/out dial from the three core VIX rules (term, residual, VRP), read live.

    Same code path as the backtests (``signals.vix.exposure_vote``): SPY closes
    and the VIX / VIX3M closes aligned on SPY's sessions, forward-filled. Returns
    the vote, each rule's reading and exposure, and whether the dial just moved.
    """
    if spy_bars.empty or vix_bars.empty:
        return {}
    spy = spy_bars.sort_values("date").set_index("date")["close"].astype(float)
    spy = spy[~spy.index.duplicated(keep="last")]

    def _idx(symbol: str) -> pd.Series | None:
        sub = vix_bars[vix_bars["symbol"] == symbol].sort_values("date")
        if sub.empty:
            return None
        s_ = sub.set_index("date")["close"].astype(float)
        s_ = s_[~s_.index.duplicated(keep="last")]
        return s_.reindex(spy.index).ffill(limit=10)

    vix = _idx("^VIX")
    vix3m = _idx(VIX_LONG_SYMBOL)
    if vix is None or vix3m is None or vix.dropna().empty or vix3m.dropna().empty:
        return {}

    term = {"out_above": cfg.dial_term_out_above, "band": cfg.dial_term_band}
    resid = {
        "window": cfg.dial_resid_window,
        "warn_z": cfg.dial_resid_warn_z,
        "cooldown": cfg.dial_resid_cooldown,
    }
    vrp = {
        "rv_window": cfg.dial_vrp_rv_window,
        "out_below": cfg.dial_vrp_out_below,
        "full_above": cfg.dial_vrp_full_above,
    }
    votes = vixsig.exposure_vote(spy, vix, vix3m, term=term, resid=resid, vrp=vrp)
    last = votes.iloc[-1]
    prev = votes.iloc[-2] if len(votes) > 1 else last
    ratio = vixsig.term_structure_ratio(vix, vix3m)
    resid_z = vixsig.residual_zscore(spy, vix, cfg.dial_resid_window)
    vrp_pts = vixsig.variance_risk_premium(vix, spy, cfg.dial_vrp_rv_window)

    labels = votes["vote"].map(dial_label)
    streak = int((labels[::-1] == labels.iloc[-1]).cummin().sum())
    exposure = _f(last["vote"])
    return {
        "as_of": str(pd.Timestamp(spy.index[-1]).date()),
        "exposure": exposure,
        "label": dial_label(exposure),
        "prev_exposure": _f(prev["vote"]),
        "changed": bool(abs(float(last["vote"]) - float(prev["vote"])) > 1e-9),
        "days_in_state": streak,
        "term": {
            "ratio": _f(ratio.iloc[-1]),
            "exposure": _f(last["term"]),
            "out_above": cfg.dial_term_out_above,
        },
        "resid": {
            "z": _f(resid_z.iloc[-1]),
            "exposure": _f(last["resid"]),
            "warn_z": cfg.dial_resid_warn_z,
        },
        "vrp": {
            "points": _f(vrp_pts.iloc[-1]),
            "exposure": _f(last["vrp"]),
            "out_below": cfg.dial_vrp_out_below,
            "full_above": cfg.dial_vrp_full_above,
        },
        "vix": _f(vix.iloc[-1]),
        "vix3m": _f(vix3m.iloc[-1]),
    }
