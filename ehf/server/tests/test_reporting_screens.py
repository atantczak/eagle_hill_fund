"""Screen rules fire on crafted rows and stay quiet on controls (no network)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ehf.server.research.reporting.config import ReportConfig
from ehf.server.research.reporting.features import FEATURE_COLUMNS
from ehf.server.research.reporting.fundamentals import FUNDAMENTAL_COLUMNS
from ehf.server.research.reporting.screens import (
    Hit,
    apply_enrichment,
    run_all_screens,
    screen_analyst_revisions,
    screen_breakouts,
    screen_insider_clusters,
    screen_outlier_moves,
    screen_pead,
    screen_quality_dislocation,
    symbols_to_enrich,
)
from ehf.server.tests.test_reporting_features import make_bars

CFG = ReportConfig()


def feat_row(**over: object) -> dict[str, object]:
    base: dict[str, object] = {c: np.nan for c in FEATURE_COLUMNS}
    base.update(
        {
            "close": 100.0,
            "ret_1d": 0.0,
            "ret_5d": 0.0,
            "ret_21d": 0.0,
            "dd_from_high_63": 0.0,
            "pct_from_high_252": -0.05,
            "z_1d": 0.0,
            "rsi_14": 50.0,
            "mom_12_1": 0.1,
            "vol_ratio_20": 1.0,
            "dollar_vol_20": 50e6,
            "new_high_252": False,
            "new_low_252": False,
            "above_50": True,
            "above_200": True,
            "golden_cross_5d": False,
            "death_cross_5d": False,
        }
    )
    base.update(over)
    return base


def features(rows: dict[str, dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "symbol"
    return df


def universe(symbols: list[str], sector: str = "Technology") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbols,
            "name": [f"{s} Inc" for s in symbols],
            "sector": sector,
            "industry": "x",
            "market_cap": 20e9,
            "avg_volume": 5e6,
            "in_sp500": True,
        }
    )


def fund_row(**over: object) -> dict[str, object]:
    base: dict[str, object] = {c: np.nan for c in FUNDAMENTAL_COLUMNS}
    base.update(
        {"piotroski": 7, "altman_z": 4.0, "fcf_yield": 0.05, "roic": 0.15, "net_debt_ebitda": 1.0}
    )
    base.update(over)
    return base


def test_outlier_moves_requires_sigma_and_size_and_liquidity() -> None:
    f = features(
        {
            "BIG": feat_row(ret_1d=-0.12, z_1d=-6.0, vol_ratio_20=4.0),
            "SMALLMOVE": feat_row(ret_1d=-0.02, z_1d=-3.5),  # sigma ok, size too small
            "ILLIQ": feat_row(ret_1d=-0.12, z_1d=-6.0, dollar_vol_20=1e6),
        }
    )
    hits = screen_outlier_moves(f, universe(list(f.index)), CFG)
    assert [h.symbol for h in hits] == ["BIG"]
    assert hits[0].direction == "watch" and "sigma" in hits[0].headline


def test_quality_dislocation_needs_both_drop_and_quality() -> None:
    f = features(
        {
            "GOOD": feat_row(ret_21d=-0.15, dd_from_high_63=-0.2),
            "JUNK": feat_row(ret_21d=-0.15, dd_from_high_63=-0.2),
            "FINE": feat_row(ret_21d=0.02),
            "BANK": feat_row(ret_21d=-0.15),
            "SOON": feat_row(ret_21d=-0.15),
        }
    )
    fund = pd.DataFrame.from_dict(
        {
            "GOOD": fund_row(),
            "JUNK": fund_row(piotroski=3),
            "FINE": fund_row(),
            "BANK": fund_row(altman_z=np.nan),
            "SOON": fund_row(),
        },
        orient="index",
    )
    uni = universe(list(f.index))
    uni.loc[uni["symbol"] == "BANK", "sector"] = "Financial Services"
    hits = screen_quality_dislocation(f, fund, uni, CFG, upcoming_earnings={"SOON": "2026-11-04"})
    syms = {h.symbol for h in hits}
    assert syms == {"GOOD", "BANK", "SOON"}
    soon = next(h for h in hits if h.symbol == "SOON")
    assert (
        any("CAUTION" in r for r in soon.reasons)
        and soon.context["earnings_date_ahead"] == "2026-11-04"
    )
    assert all(h.direction == "long" and h.horizon for h in hits)


def test_quality_dislocation_respects_market_cap_floor() -> None:
    f = features({"TINY": feat_row(ret_21d=-0.2)})
    fund = pd.DataFrame.from_dict({"TINY": fund_row()}, orient="index")
    uni = universe(["TINY"])
    uni["market_cap"] = 1e9
    assert screen_quality_dislocation(f, fund, uni, CFG) == []


def test_pead_long_and_short() -> None:
    n = 80
    up = np.full(n, 100.0)
    up[-2:] = 108.0  # jumped after the report
    dn = np.full(n, 100.0)
    dn[-2:] = 92.0
    bars = make_bars({"BEAT": up, "MISS": dn, "MEH": np.full(n, 100.0)})
    last = bars["date"].max().date()
    report_day = bars["date"].sort_values().unique()[-2]
    f = features({s: feat_row() for s in ("BEAT", "MISS", "MEH")})
    rows = [
        {
            "symbol": "BEAT",
            "date": str(report_day)[:10],
            "epsActual": 1.2,
            "epsEstimated": 1.0,
            "revenueActual": 10,
            "revenueEstimated": 9,
        },
        {"symbol": "MISS", "date": str(report_day)[:10], "epsActual": 0.8, "epsEstimated": 1.0},
        {
            "symbol": "MEH",
            "date": str(report_day)[:10],
            "epsActual": 1.2,
            "epsEstimated": 1.0,
        },  # beat, no reaction
        {"symbol": "OLD", "date": "2020-01-01", "epsActual": 9, "epsEstimated": 1},
    ]
    hits = screen_pead(f, bars, rows, universe(["BEAT", "MISS", "MEH"]), CFG, as_of=last)
    by = {h.symbol: h for h in hits}
    assert set(by) == {"BEAT", "MISS"}
    assert by["BEAT"].direction == "long" and by["BEAT"].metrics["eps_surprise"] == pytest.approx(
        0.2
    )
    assert by["MISS"].direction == "short"
    assert by["BEAT"].metrics["revenue_surprise"] > 0


def test_breakouts_need_new_high_momentum_and_volume() -> None:
    rows = {f"S{i}": feat_row(mom_12_1=i / 10) for i in range(10)}  # momentum spread 0..0.9
    rows["HIT"] = feat_row(new_high_252=True, mom_12_1=0.95, vol_ratio_20=2.0)
    rows["WEAKMOM"] = feat_row(new_high_252=True, mom_12_1=0.05, vol_ratio_20=2.0)
    rows["NOVOL"] = feat_row(new_high_252=True, mom_12_1=0.95, vol_ratio_20=0.9)
    hits = screen_breakouts(features(rows), universe(list(rows)), CFG)
    assert [h.symbol for h in hits] == ["HIT"]


def test_insider_cluster_counts_distinct_open_market_buyers() -> None:
    rows = [
        {
            "symbol": "HOLDER",
            "filingDate": "2026-10-30",
            "transactionType": "P-Purchase",
            "acquisitionOrDisposition": "A",
            "reportingName": "Parent Co",
            "typeOfOwner": "10 percent owner",
            "securitiesTransacted": 100_000,
            "price": 50,
        },
        {
            "symbol": "HOLDER",
            "filingDate": "2026-10-30",
            "transactionType": "P-Purchase",
            "acquisitionOrDisposition": "A",
            "reportingName": "Parent Co 2",
            "typeOfOwner": "10 percent owner",
            "securitiesTransacted": 100_000,
            "price": 50,
        },
        {
            "symbol": "CLU",
            "filingDate": "2026-10-30",
            "transactionType": "P-Purchase",
            "acquisitionOrDisposition": "A",
            "reportingName": "A",
            "securitiesTransacted": 10_000,
            "price": 50,
        },
        {
            "symbol": "CLU",
            "filingDate": "2026-10-29",
            "transactionType": "P-Purchase",
            "acquisitionOrDisposition": "A",
            "reportingName": "B",
            "securitiesTransacted": 5_000,
            "price": 50,
        },
        {
            "symbol": "ONE",
            "filingDate": "2026-10-29",
            "transactionType": "P-Purchase",
            "acquisitionOrDisposition": "A",
            "reportingName": "A",
            "securitiesTransacted": 5_000,
            "price": 50,
        },
        {
            "symbol": "WHALE",
            "filingDate": "2026-10-29",
            "transactionType": "P-Purchase",
            "acquisitionOrDisposition": "A",
            "reportingName": "A",
            "securitiesTransacted": 200_000,
            "price": 50,
        },
        {
            "symbol": "SELL",
            "filingDate": "2026-10-29",
            "transactionType": "S-Sale",
            "acquisitionOrDisposition": "D",
            "reportingName": "A",
            "securitiesTransacted": 900_000,
            "price": 50,
        },
        {
            "symbol": "SELL",
            "filingDate": "2026-10-29",
            "transactionType": "S-Sale",
            "acquisitionOrDisposition": "D",
            "reportingName": "B",
            "securitiesTransacted": 900_000,
            "price": 50,
        },
        {
            "symbol": "OUTSIDE",
            "filingDate": "2026-10-29",
            "transactionType": "P-Purchase",
            "acquisitionOrDisposition": "A",
            "reportingName": "A",
            "securitiesTransacted": 900_000,
            "price": 50,
        },
    ]
    for r in rows:
        r.setdefault("typeOfOwner", "officer: CEO")
    uni = universe(["CLU", "ONE", "WHALE", "SELL", "HOLDER"])
    f = features({s: feat_row() for s in uni["symbol"]})
    hits = screen_insider_clusters(rows, uni, f, CFG)
    by = {h.symbol: h for h in hits}
    assert set(by) == {"CLU", "WHALE", "HOLDER"}
    assert by["HOLDER"].metrics["kind"] == "large holder" and by["CLU"].metrics["kind"] == "cluster"
    assert by["CLU"].metrics["n_insiders"] == 2 and by["CLU"].metrics["total_value"] == 750_000
    assert screen_insider_clusters([], uni, f, CFG) == []


def test_analyst_revisions_net_threshold_and_direction() -> None:
    grades = [
        {"symbol": "UP", "action": "upgrade", "gradingCompany": "X"},
        {"symbol": "UP", "action": "upgrade", "gradingCompany": "Y"},
        {"symbol": "UP", "action": "downgrade", "gradingCompany": "Z"},
        {"symbol": "UP", "action": "upgrade", "gradingCompany": "W"},
        {"symbol": "DN", "action": "downgrade", "gradingCompany": "X"},
        {"symbol": "DN", "action": "downgrade", "gradingCompany": "Y"},
        {"symbol": "MIX", "action": "upgrade", "gradingCompany": "X"},
        {"symbol": "MIX", "action": "downgrade", "gradingCompany": "Y"},
        {"symbol": "NOTINUNI", "action": "upgrade", "gradingCompany": "X"},
        {"symbol": "NOTINUNI", "action": "upgrade", "gradingCompany": "X"},
    ]
    pts = [{"symbol": "UP", "priceTarget": 120, "priceWhenPosted": 100}]
    uni = universe(["UP", "DN", "MIX"])
    f = features({s: feat_row() for s in uni["symbol"]})
    hits = screen_analyst_revisions(grades, pts, uni, f, CFG)
    by = {h.symbol: h for h in hits}
    assert set(by) == {"UP", "DN"}
    assert by["UP"].direction == "long" and by["UP"].metrics["net"] == 2
    assert by["UP"].metrics["avg_pt_upside"] == pytest.approx(0.2)
    assert by["DN"].direction == "short"


def test_run_all_screens_orders_and_caps_and_enrichment_plumbing() -> None:
    cfg = ReportConfig(max_hits_per_screen=1, max_enriched_symbols=3)
    rows = {f"O{i}": feat_row(ret_1d=-0.2, z_1d=-9.0 - i) for i in range(3)}
    f = features(rows)
    uni = universe(list(rows))
    bars = make_bars({s: np.full(30, 100.0) for s in rows})
    hits = run_all_screens(
        features=f,
        fundamentals=pd.DataFrame(),
        universe=uni,
        bars=bars,
        earnings_rows=[],
        insider_rows=[],
        grades_rows=[],
        pt_rows=[],
        cfg=cfg,
        as_of=bars["date"].max().date(),
    )
    assert list(hits) == [
        "quality_dislocation",
        "pead",
        "insider_cluster",
        "analyst_revision",
        "breakout_52w",
        "outlier_move",
    ]
    assert [h.symbol for h in hits["outlier_move"]] == ["O2"]  # best score, capped to 1
    assert symbols_to_enrich(hits, 10) == ["O2"]
    assert hits["outlier_move"][0].context["also_in"] == []
    apply_enrichment(
        hits,
        {
            "O2": {
                "growth": {
                    "growthRevenue": 0.12,
                    "growthNetIncome": 0.3,
                    "growthEPS": 0.25,
                    "period": "Q3",
                    "fiscalYear": 2026,
                },
                "price_target": {"targetConsensus": 120.0},
                "earnings": [
                    {"date": "2027-01-20", "epsActual": None},
                    {"date": "2026-10-20", "epsActual": 1.0},
                ],
                "news": [
                    {
                        "publishedDate": "2026-10-31 09:00:00",
                        "title": "Something happened",
                        "site": "x",
                    }
                ],
            }
        },
    )
    ctx = hits["outlier_move"][0].context
    assert ctx["pt_upside"] == 0.2 and ctx["next_earnings"] == "2027-01-20"
    assert ctx["revenue_growth_qoq_yoy"] == 0.12 and ctx["news"][0]["date"] == "2026-10-31"


def test_diversify_caps_per_industry_and_notes_the_theme() -> None:
    from ehf.server.research.reporting.screens import diversify

    hits = [Hit(f"G{i}", "quality_dislocation", "long", 10 - i, "h") for i in range(5)]
    hits.append(Hit("T1", "quality_dislocation", "long", 1, "h"))
    industry = pd.Series({**{f"G{i}": "Gold" for i in range(5)}, "T1": "Tech"})
    out = diversify(hits, industry, max_per_group=3)
    assert [h.symbol for h in out] == ["G0", "G1", "G2", "T1"]
    assert out[0].context["same_industry_also"] == ["G3", "G4"] and "theme" in out[0].reasons[-1]
    assert diversify(hits, None, max_per_group=3) is hits


def test_hit_to_json_drops_nan_and_numpy_types() -> None:
    h = Hit(
        "X",
        "outlier_move",
        "watch",
        np.float64(2.5),
        "hl",
        metrics={"a": float("nan"), "b": np.int64(3), "c": np.bool_(True)},
    )
    j = h.to_json()
    assert j["metrics"] == {"a": None, "b": 3, "c": True} and j["score"] == 2.5 and j["horizon"]
    assert not math.isnan(j["score"])
