"""The screens: pure rules that turn features + feeds into ranked ``Hit`` records.

Design notes (why these six, and why daily movers are the *least* important):

* ``outlier_move``        — today's statistically abnormal moves. Mostly noise on
                             their own; included so the narrative can *explain* them
                             and so we can later measure whether they mean anything.
* ``quality_dislocation`` — "fell hard, fundamentals intact". The research prompt
                             Andrew asked for. Quality is scored from Piotroski /
                             Altman / FCF yield / ROIC / leverage, all point-in-time
                             from FMP's TTM bulk tables.
* ``pead``                — post-earnings-announcement drift: beat + positive
                             reaction tends to keep drifting for weeks (and the
                             mirror image for misses). The best-documented
                             short-horizon anomaly we can compute from this data.
* ``breakout_52w``        — new 52-week high with strong 12-1 momentum and a volume
                             confirmation (George & Hwang style). Trend following.
* ``insider_cluster``     — several insiders buying in the open market within two
                             weeks. Cluster purchases are informative; single
                             purchases and all sales are not, so they're ignored.
* ``analyst_revision``    — net upgrades / downgrades over the last few days.
                             Revision momentum, long and short.

Each ``Hit`` carries the numbers that fired it so the stored snapshot can be
scored later (``evaluate.py``) — that is the whole point of writing them down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ehf.server.research.reporting.config import ReportConfig
from ehf.server.research.reporting.features import reaction_since

Json = dict[str, Any]
SCREEN_ORDER = (
    "quality_dislocation",
    "pead",
    "insider_cluster",
    "analyst_revision",
    "breakout_52w",
    "outlier_move",
)
HORIZON = {
    "quality_dislocation": "1-3 months",
    "pead": "2-8 weeks",
    "insider_cluster": "1-6 months",
    "analyst_revision": "1-3 months",
    "breakout_52w": "1-6 months",
    "outlier_move": "explain, don't trade",
}


@dataclass
class Hit:
    symbol: str
    screen: str
    direction: str  # long | short | watch
    score: float
    headline: str
    metrics: Json = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    context: Json = field(default_factory=dict)  # filled by enrichment (news, growth, PT)

    @property
    def horizon(self) -> str:
        return HORIZON.get(self.screen, "")

    def to_json(self) -> Json:
        return {
            "symbol": self.symbol,
            "screen": self.screen,
            "direction": self.direction,
            "score": round(float(self.score), 4),
            "horizon": self.horizon,
            "headline": self.headline,
            "metrics": {k: _jsonable(v) for k, v in self.metrics.items()},
            "reasons": list(self.reasons),
            "context": self.context,
        }


def _jsonable(v: object) -> object:
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else round(f, 6)
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    return v


def _num(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _ok(x: float) -> bool:
    return not math.isnan(x)


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _liquid(features: pd.DataFrame, cfg: ReportConfig) -> pd.Series:
    return features["dollar_vol_20"].fillna(0.0) >= cfg.min_dollar_volume


def _name(universe: pd.DataFrame, sym: str) -> str:
    row = universe.loc[universe["symbol"] == sym]
    return str(row["name"].iloc[0]) if not row.empty and pd.notna(row["name"].iloc[0]) else sym


def _sector(universe: pd.DataFrame, sym: str) -> str:
    row = universe.loc[universe["symbol"] == sym]
    return str(row["sector"].iloc[0]) if not row.empty and pd.notna(row["sector"].iloc[0]) else ""


# --- 1. abnormal daily moves ----------


def screen_outlier_moves(
    features: pd.DataFrame, universe: pd.DataFrame, cfg: ReportConfig
) -> list[Hit]:
    if features.empty:
        return []
    z = features["z_1d"]
    mask = (z.abs() >= cfg.outlier_z) & (features["ret_1d"].abs() >= cfg.outlier_min_abs_ret)
    mask &= _liquid(features, cfg)
    hits: list[Hit] = []
    for sym, row in features.loc[mask].iterrows():
        s = str(sym)
        r1, zz, vr = _num(row["ret_1d"]), _num(row["z_1d"]), _num(row["vol_ratio_20"])
        hits.append(
            Hit(
                symbol=s,
                screen="outlier_move",
                direction="watch",
                score=abs(zz),
                headline=f"{_name(universe, s)} {_pct(r1)} on the day ({zz:+.1f} sigma vs 60d vol)",
                metrics={
                    "ret_1d": r1,
                    "z_1d": zz,
                    "vol_ratio_20": vr,
                    "ret_21d": _num(row["ret_21d"]),
                    "pct_from_high_252": _num(row["pct_from_high_252"]),
                    "rsi_14": _num(row["rsi_14"]),
                    "close": _num(row["close"]),
                },
                reasons=[
                    f"1-day move {_pct(r1)} is {abs(zz):.1f}x its 60-day daily vol",
                    f"volume {vr:.1f}x the 20-day average"
                    if _ok(vr)
                    else "volume ratio unavailable",
                ],
            )
        )
    return _rank(hits, cfg)


# --- 2. quality on sale ----------


def screen_quality_dislocation(
    features: pd.DataFrame,
    fundamentals: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: ReportConfig,
    *,
    upcoming_earnings: dict[str, str] | None = None,
) -> list[Hit]:
    if features.empty or fundamentals.empty:
        return []
    j = features.join(fundamentals, how="inner")
    if j.empty:
        return []
    sectors = universe.set_index("symbol")["sector"].reindex(j.index)
    caps = universe.set_index("symbol")["market_cap"].reindex(j.index)
    financial = sectors.fillna("").str.contains("Financial", case=False)

    dislocated = (j["ret_21d"] <= cfg.dislocation_ret_21d) | (
        j["dd_from_high_63"] <= cfg.dislocation_dd_63d
    )
    quality = (
        (j["piotroski"] >= cfg.quality_min_piotroski)
        & ((j["altman_z"] >= cfg.quality_min_altman) | (financial & j["altman_z"].isna()))
        & (j["fcf_yield"] >= cfg.quality_min_fcf_yield)
        & (j["roic"] >= cfg.quality_min_roic)
        & ((j["net_debt_ebitda"] <= cfg.quality_max_net_debt_ebitda) | j["net_debt_ebitda"].isna())
    )
    big = caps.fillna(0.0) >= cfg.quality_min_market_cap
    mask = dislocated & quality & big & _liquid(j, cfg)

    hits: list[Hit] = []
    upcoming = upcoming_earnings or {}
    for sym, row in j.loc[mask].iterrows():
        s = str(sym)
        r21, dd63 = _num(row["ret_21d"]), _num(row["dd_from_high_63"])
        pio, alt = _num(row["piotroski"]), _num(row["altman_z"])
        fcf, roic, nde = _num(row["fcf_yield"]), _num(row["roic"]), _num(row["net_debt_ebitda"])
        drop = min(r21 if _ok(r21) else 0.0, dd63 if _ok(dd63) else 0.0)
        q = (min(pio, 9) / 9.0 + min(fcf, 0.15) / 0.15 + min(roic, 0.40) / 0.40) / 3.0
        reasons = [
            f"down {_pct(r21)} over 21 days; {_pct(dd63)} from its 63-day high",
            f"Piotroski {pio:.0f}/9, FCF yield {fcf * 100:.1f}%, ROIC {roic * 100:.1f}%",
            (f"Altman Z {alt:.1f}" if _ok(alt) else "Altman n/a (financial)")
            + (f", net debt/EBITDA {nde:.1f}x" if _ok(nde) else ", net cash / n/a"),
        ]
        ctx: Json = {}
        if s in upcoming:
            reasons.append(f"CAUTION: reports earnings on {upcoming[s]}")
            ctx["earnings_date_ahead"] = upcoming[s]
        hits.append(
            Hit(
                symbol=s,
                screen="quality_dislocation",
                direction="long",
                score=abs(drop) * q * 100.0,
                headline=(
                    f"{_name(universe, s)} {_pct(drop)} with fundamentals intact (quality {q:.2f})"
                ),
                metrics={
                    "ret_21d": r21,
                    "dd_from_high_63": dd63,
                    "pct_from_high_252": _num(row["pct_from_high_252"]),
                    "piotroski": pio,
                    "altman_z": alt,
                    "fcf_yield": fcf,
                    "roic": roic,
                    "net_debt_ebitda": nde,
                    "pe": _num(row["pe"]),
                    "ev_ebitda": _num(row["ev_ebitda"]),
                    "rsi_14": _num(row["rsi_14"]),
                    "quality_score": q,
                    "market_cap": _num(caps.get(s)),
                    "close": _num(row["close"]),
                    "sector": _sector(universe, s),
                },
                reasons=reasons,
                context=ctx,
            )
        )
    return _rank(hits, cfg)


# --- 3. post-earnings drift ----------------------------------------------------------------------


def screen_pead(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    earnings_rows: list[Json],
    universe: pd.DataFrame,
    cfg: ReportConfig,
    *,
    as_of: date,
) -> list[Hit]:
    if features.empty or not earnings_rows:
        return []
    since = (as_of - timedelta(days=cfg.earnings_lookback_days)).isoformat()
    liquid = _liquid(features, cfg)
    hits: list[Hit] = []
    seen: set[str] = set()
    for r in earnings_rows:
        sym = str(r.get("symbol", ""))
        d = str(r.get("date", ""))[:10]
        if sym in seen or sym not in features.index or not (since <= d <= as_of.isoformat()):
            continue
        if not bool(liquid.get(sym, False)):
            continue
        a, e = _num(r.get("epsActual")), _num(r.get("epsEstimated"))
        if not (_ok(a) and _ok(e)) or e == 0:
            continue
        surprise = (a - e) / abs(e)
        ra, re_ = _num(r.get("revenueActual")), _num(r.get("revenueEstimated"))
        rev_surprise = (ra - re_) / abs(re_) if _ok(ra) and _ok(re_) and re_ != 0 else float("nan")
        reaction = reaction_since(bars, sym, pd.Timestamp(d))
        if reaction is None:
            continue
        long_ok = surprise >= cfg.pead_min_surprise and reaction >= cfg.pead_min_reaction
        short_ok = surprise <= -cfg.pead_min_surprise and reaction <= -cfg.pead_min_reaction
        if not (long_ok or short_ok):
            continue
        seen.add(sym)
        direction = "long" if long_ok else "short"
        hits.append(
            Hit(
                symbol=sym,
                screen="pead",
                direction=direction,
                score=min(abs(surprise), 0.5) * 100.0 + abs(reaction) * 100.0,
                headline=(
                    f"{_name(universe, sym)} {'beat' if surprise > 0 else 'missed'} EPS by "
                    f"{abs(surprise) * 100:.0f}% on {d}; stock {_pct(reaction)} since"
                ),
                metrics={
                    "report_date": d,
                    "eps_actual": a,
                    "eps_estimate": e,
                    "eps_surprise": surprise,
                    "revenue_surprise": rev_surprise,
                    "reaction": reaction,
                    "ret_21d": _num(features.at[sym, "ret_21d"]),
                    "vol_ratio_20": _num(features.at[sym, "vol_ratio_20"]),
                    "close": _num(features.at[sym, "close"]),
                },
                reasons=[
                    f"EPS {a:.2f} vs {e:.2f} est ({surprise * 100:+.0f}%)"
                    + (f", revenue {rev_surprise * 100:+.1f}% vs est" if _ok(rev_surprise) else ""),
                    f"price reaction {_pct(reaction)} from the pre-report close",
                    "drift tends to continue in the direction of the surprise for weeks",
                ],
            )
        )
        if _ok(rev_surprise) and abs(rev_surprise) < 0.01 and abs(surprise) > 0.15:
            hits[-1].reasons.append(
                "revenue was in line: a large EPS surprise with flat revenue is often a GAAP vs "
                "adjusted mismatch or a one-off; verify before acting"
            )
    return _rank(hits, cfg)


# --- 4. momentum breakouts ----------


def screen_breakouts(
    features: pd.DataFrame, universe: pd.DataFrame, cfg: ReportConfig
) -> list[Hit]:
    if features.empty:
        return []
    mom = features["mom_12_1"]
    cut = mom.quantile(cfg.breakout_mom_quantile) if mom.notna().any() else float("nan")
    mask = (
        features["new_high_252"]
        & features["above_200"]
        & (mom >= cut)
        & (features["vol_ratio_20"] >= cfg.breakout_min_vol_ratio)
        & _liquid(features, cfg)
    )
    hits: list[Hit] = []
    for sym, row in features.loc[mask].iterrows():
        s = str(sym)
        m, vr, r1 = _num(row["mom_12_1"]), _num(row["vol_ratio_20"]), _num(row["ret_1d"])
        hits.append(
            Hit(
                symbol=s,
                screen="breakout_52w",
                direction="long",
                score=max(m, 0.0) * 100.0 * min(vr, 3.0),
                headline=(
                    f"{_name(universe, s)} new 52-week high on {vr:.1f}x volume, "
                    f"12-1 momentum {_pct(m)}"
                ),
                metrics={
                    "mom_12_1": m,
                    "vol_ratio_20": vr,
                    "ret_1d": r1,
                    "ret_21d": _num(row["ret_21d"]),
                    "rsi_14": _num(row["rsi_14"]),
                    "close": _num(row["close"]),
                    "mom_cutoff": _num(cut),
                },
                reasons=[
                    f"closed at a new 52-week high; 12-1 momentum {_pct(m)} is top "
                    f"{int((1 - cfg.breakout_mom_quantile) * 100)}%",
                    f"volume {vr:.1f}x its 20-day average confirms the move",
                ],
            )
        )
    return _rank(hits, cfg)


# --- 5. insider clusters ----------


def screen_insider_clusters(
    insider_rows: list[Json], universe: pd.DataFrame, features: pd.DataFrame, cfg: ReportConfig
) -> list[Hit]:
    if not insider_rows:
        return []
    symbols = set(universe["symbol"])
    buys = [
        r
        for r in insider_rows
        if str(r.get("transactionType", "")).upper().startswith("P-PURCHASE")
        and str(r.get("acquisitionOrDisposition", "A")).upper() == "A"
        and r.get("symbol") in symbols
    ]
    if not buys:
        return []
    df = pd.DataFrame(buys)
    if "typeOfOwner" not in df.columns:
        df["typeOfOwner"] = ""
    owner = df["typeOfOwner"].astype(str).str.lower()
    df["is_officer"] = owner.str.contains("officer|director|chief|president|ceo|cfo", regex=True)
    df["value"] = pd.to_numeric(df["securitiesTransacted"], errors="coerce").fillna(
        0
    ) * pd.to_numeric(df["price"], errors="coerce").fillna(0)
    g = df.groupby("symbol").agg(
        n_insiders=("reportingName", "nunique"),
        n_officers=("reportingName", lambda s: s[df.loc[s.index, "is_officer"]].nunique()),
        n_filings=("symbol", "size"),
        total_value=("value", "sum"),
        last_filing=("filingDate", "max"),
        names=("reportingName", lambda s: ", ".join(sorted(set(map(str, s)))[:4])),
    )
    hits: list[Hit] = []
    for sym, row in g.iterrows():
        s = str(sym)
        n, val = int(row["n_insiders"]), float(row["total_value"])
        n_off = int(row["n_officers"])
        cluster = n_off >= cfg.insider_min_insiders and val >= cfg.insider_min_value
        whale = val >= 10 * cfg.insider_min_value
        if not (cluster or whale):
            continue
        kind = "cluster" if cluster else "large holder"
        feat = features.loc[s] if s in features.index else None
        hits.append(
            Hit(
                symbol=s,
                screen="insider_cluster",
                direction="long",
                score=n + math.log10(max(val, 1.0) / 1e5),
                headline=(
                    f"{_name(universe, s)}: {n} insider(s) bought ${val / 1e6:.1f}M "
                    f"in the open market ({kind})"
                ),
                metrics={
                    "n_insiders": n,
                    "n_officers": n_off,
                    "kind": kind,
                    "n_filings": int(row["n_filings"]),
                    "total_value": val,
                    "last_filing": str(row["last_filing"]),
                    "ret_21d": _num(feat["ret_21d"]) if feat is not None else float("nan"),
                    "pct_from_high_252": _num(feat["pct_from_high_252"])
                    if feat is not None
                    else float("nan"),
                    "close": _num(feat["close"]) if feat is not None else float("nan"),
                },
                reasons=[
                    f"{n} distinct insiders ({n_off} officers/directors), "
                    f"{int(row['n_filings'])} open-market purchase filings "
                    f"in {cfg.insider_lookback_days} days"
                    + (
                        ""
                        if cluster
                        else "; a single large buyer, often a 10% holder, not a cluster"
                    ),
                    f"buyers: {row['names']}",
                    "clustered open-market buying is informative; sales and exercises are ignored",
                ],
            )
        )
    return _rank(hits, cfg)


# --- 6. analyst revisions ----------


def screen_analyst_revisions(
    grades_rows: list[Json],
    pt_rows: list[Json],
    universe: pd.DataFrame,
    features: pd.DataFrame,
    cfg: ReportConfig,
) -> list[Hit]:
    symbols = set(universe["symbol"])
    tally: dict[str, dict[str, Any]] = {}
    for r in grades_rows:
        sym = str(r.get("symbol", ""))
        if sym not in symbols:
            continue
        action = str(r.get("action", "")).lower()
        t = tally.setdefault(sym, {"up": 0, "down": 0, "firms_up": [], "firms_down": [], "pt": []})
        if action == "upgrade":
            t["up"] += 1
            t["firms_up"].append(str(r.get("gradingCompany", "")))
        elif action == "downgrade":
            t["down"] += 1
            t["firms_down"].append(str(r.get("gradingCompany", "")))
    for r in pt_rows:
        sym = str(r.get("symbol", ""))
        if sym not in symbols:
            continue
        pt, px = _num(r.get("priceTarget")), _num(r.get("priceWhenPosted"))
        if _ok(pt) and _ok(px) and px > 0:
            tally.setdefault(sym, {"up": 0, "down": 0, "firms_up": [], "firms_down": [], "pt": []})[
                "pt"
            ].append(pt / px - 1.0)
    hits: list[Hit] = []
    for sym, t in tally.items():
        net = int(t["up"] - t["down"])
        if abs(net) < cfg.revision_min_net:
            continue
        direction = "long" if net > 0 else "short"
        pts = [p for p in t["pt"] if _ok(p)]
        avg_upside = float(np.mean(pts)) if pts else float("nan")
        feat = features.loc[sym] if sym in features.index else None
        firms = t["firms_up"] if net > 0 else t["firms_down"]
        hits.append(
            Hit(
                symbol=sym,
                screen="analyst_revision",
                direction=direction,
                score=abs(net) + 0.5 * len(pts),
                headline=(
                    f"{_name(universe, sym)}: {t['up']} upgrade(s) / {t['down']} downgrade(s) "
                    f"in {cfg.grades_lookback_days} days"
                ),
                metrics={
                    "upgrades": int(t["up"]),
                    "downgrades": int(t["down"]),
                    "net": net,
                    "n_price_targets": len(pts),
                    "avg_pt_upside": avg_upside,
                    "ret_5d": _num(feat["ret_5d"]) if feat is not None else float("nan"),
                    "ret_21d": _num(feat["ret_21d"]) if feat is not None else float("nan"),
                    "close": _num(feat["close"]) if feat is not None else float("nan"),
                },
                reasons=[
                    f"net {net:+d} rating changes ({', '.join(f for f in firms if f)[:120]})",
                    (
                        f"{len(pts)} price-target notes, avg implied upside {_pct(avg_upside)}"
                        if pts
                        else "no price-target notes"
                    ),
                ],
            )
        )
    return _rank(hits, cfg)


# --- ranking + enrichment plumbing ----------


def _rank(hits: list[Hit], cfg: ReportConfig) -> list[Hit]:
    hits.sort(key=lambda h: -h.score)
    return hits[: cfg.max_hits_per_screen]


def run_all_screens(
    *,
    features: pd.DataFrame,
    fundamentals: pd.DataFrame,
    universe: pd.DataFrame,
    bars: pd.DataFrame,
    earnings_rows: list[Json],
    insider_rows: list[Json],
    grades_rows: list[Json],
    pt_rows: list[Json],
    cfg: ReportConfig,
    as_of: date,
    upcoming_earnings: dict[str, str] | None = None,
) -> dict[str, list[Hit]]:
    """Run every screen; keyed by screen name in presentation order.

    Also stamps each hit with ``context["also_in"]``: the other screens the same
    symbol fired on. Confluence is the strongest single signal in this report.
    """
    hits = {
        "quality_dislocation": screen_quality_dislocation(
            features, fundamentals, universe, cfg, upcoming_earnings=upcoming_earnings
        ),
        "pead": screen_pead(features, bars, earnings_rows, universe, cfg, as_of=as_of),
        "insider_cluster": screen_insider_clusters(insider_rows, universe, features, cfg),
        "analyst_revision": screen_analyst_revisions(grades_rows, pt_rows, universe, features, cfg),
        "breakout_52w": screen_breakouts(features, universe, cfg),
        "outlier_move": screen_outlier_moves(features, universe, cfg),
    }
    industry = universe.set_index("symbol")["industry"] if "industry" in universe.columns else None
    for screen in ("quality_dislocation", "breakout_52w"):
        hits[screen] = diversify(hits[screen], industry, max_per_group=3)
    where: dict[str, list[str]] = {}
    for screen, group in hits.items():
        for h in group:
            where.setdefault(h.symbol, []).append(screen)
    for screen, group in hits.items():
        for h in group:
            h.context["also_in"] = [s for s in where[h.symbol] if s != screen]
    return hits


def diversify(hits: list[Hit], industry: pd.Series | None, *, max_per_group: int) -> list[Hit]:
    """Keep at most ``max_per_group`` hits per industry; note the crowd on the survivors.

    Seven gold miners on the same day is one theme, not seven ideas. The dropped
    names are listed on the kept ones so the information isn't lost.
    """
    if industry is None or not hits:
        return hits
    kept: list[Hit] = []
    count: dict[str, int] = {}
    first: dict[str, Hit] = {}
    dropped: dict[str, list[str]] = {}
    for h in hits:
        grp = str(industry.get(h.symbol, "") or "")
        if count.get(grp, 0) < max_per_group:
            count[grp] = count.get(grp, 0) + 1
            kept.append(h)
            first.setdefault(grp, h)
        else:
            dropped.setdefault(grp, []).append(h.symbol)
    for grp, syms in dropped.items():
        first[grp].context["same_industry_also"] = syms
        first[grp].reasons.append(
            f"theme, not a single name: {len(syms) + max_per_group} {grp or 'peer'} names fired "
            f"(also {', '.join(syms)})"
        )
    return kept


def symbols_to_enrich(hits: dict[str, list[Hit]], limit: int) -> list[str]:
    """Unique symbols in priority order (research screens first), capped."""
    out: list[str] = []
    for screen in SCREEN_ORDER:
        for h in hits.get(screen, []):
            if h.symbol not in out:
                out.append(h.symbol)
    return out[:limit]


def apply_enrichment(hits: dict[str, list[Hit]], enrichment: dict[str, Json]) -> None:
    """Attach per-symbol context (growth, price target, next earnings, news) in place."""
    for group in hits.values():
        for h in group:
            e = enrichment.get(h.symbol)
            if not e:
                continue
            ctx = h.context
            growth = e.get("growth") or {}
            if growth:
                ctx["revenue_growth_qoq_yoy"] = _jsonable(_num(growth.get("growthRevenue")))
                ctx["net_income_growth"] = _jsonable(_num(growth.get("growthNetIncome")))
                ctx["eps_growth"] = _jsonable(_num(growth.get("growthEPS")))
                ctx["growth_period"] = f"{growth.get('period')} {growth.get('fiscalYear')}"
            pt = e.get("price_target") or {}
            close = _num(h.metrics.get("close"))
            if pt and _ok(_num(pt.get("targetConsensus"))) and _ok(close) and close > 0:
                ctx["pt_consensus"] = _jsonable(_num(pt.get("targetConsensus")))
                ctx["pt_upside"] = _jsonable(_num(pt.get("targetConsensus")) / close - 1.0)
            earn = e.get("earnings") or []
            upcoming = [r for r in earn if r.get("epsActual") is None and r.get("date")]
            if upcoming:
                ctx["next_earnings"] = str(min(str(r["date"]) for r in upcoming))[:10]
            news = e.get("news") or []
            ctx["news"] = [
                {
                    "date": str(n.get("publishedDate", ""))[:10],
                    "title": n.get("title"),
                    "site": n.get("site"),
                }
                for n in news[:5]
            ]
