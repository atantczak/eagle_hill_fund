"""Universe / fundamentals mapping, rendering, delivery keys, evaluation (no network)."""

from __future__ import annotations

import gzip
import json
from datetime import date

import numpy as np
import pandas as pd

from ehf.server.research.reporting.config import ReportConfig
from ehf.server.research.reporting.deliver import s3_keys_for
from ehf.server.research.reporting.evaluate import forward_returns, hits_frame, scorecard
from ehf.server.research.reporting.fundamentals import FUNDAMENTAL_COLUMNS, merge_fundamentals
from ehf.server.research.reporting.job import build_artifacts
from ehf.server.research.reporting.render import md_to_html, render_html, render_markdown
from ehf.server.research.reporting.screens import Hit
from ehf.server.research.reporting.snapshot import Snapshot
from ehf.server.research.reporting.universe import fill_from_quotes, universe_from_rows
from ehf.server.tests.test_reporting_features import make_bars


def test_universe_merges_screener_and_sp500_and_cleans_symbols() -> None:
    screener = [
        {
            "symbol": "AAPL",
            "companyName": "Apple",
            "sector": "Tech",
            "industry": "HW",
            "marketCap": 3e12,
            "avgVolume": 5e7,
            "exchangeShortName": "NASDAQ",
        },
        {"symbol": "BRK.B", "companyName": "Berkshire", "exchangeShortName": "NYSE"},
        {"symbol": "TSX1", "companyName": "Canadian", "exchangeShortName": "TSX"},
    ]
    sp = [
        {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Tech"},
        {"symbol": "XOM", "name": "Exxon", "sector": "Energy", "subSector": "Oil"},
    ]
    u = universe_from_rows(screener, sp, exchanges=("NASDAQ", "NYSE"))
    assert u["symbol"].tolist() == ["AAPL", "XOM"]
    assert (
        u.set_index("symbol").loc["AAPL", "in_sp500"]
        and u.set_index("symbol").loc["XOM", "industry"] == "Oil"
    )
    assert np.isnan(u.set_index("symbol").loc["XOM", "market_cap"])
    filled = fill_from_quotes(
        u,
        pd.DataFrame(
            [{"symbol": "XOM", "name": "Exxon Mobil", "marketCap": 5e11, "avgVolume": 1e7}]
        ),
    )
    assert filled.set_index("symbol").loc["XOM", "market_cap"] == 5e11
    assert universe_from_rows(screener, sp, exchanges=("NASDAQ", "NYSE"), include_sp500=False)[
        "symbol"
    ].tolist() == ["AAPL"]


def test_merge_fundamentals_maps_columns_and_restricts_to_universe() -> None:
    km = pd.DataFrame(
        {
            "symbol": ["A", "B", "Z"],
            "freeCashFlowYieldTTM": [0.05, 0.01, 9],
            "returnOnInvestedCapitalTTM": [0.2, 0.1, 9],
        }
    )
    ra = pd.DataFrame({"symbol": ["A"], "priceToEarningsRatioTTM": [15.0]})
    sc = pd.DataFrame({"symbol": ["A", "B"], "altmanZScore": [3.1, 1.0], "piotroskiScore": [8, 2]})
    co = pd.DataFrame({"symbol": ["A"], "consensus": ["Buy"], "buy": [10]})
    f = merge_fundamentals(["A", "B", "C"], km, ra, sc, co)
    assert list(f.columns) == FUNDAMENTAL_COLUMNS and f.index.tolist() == ["A", "B", "C"]
    assert (
        f.loc["A", "fcf_yield"] == 0.05
        and f.loc["A", "pe"] == 15.0
        and f.loc["A", "rating_consensus"] == "Buy"
    )
    assert np.isnan(f.loc["C", "piotroski"]) and np.isnan(f.loc["B", "pe"])
    assert merge_fundamentals(["A"], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()).empty
    bad = merge_fundamentals(
        ["X"],
        km.iloc[0:0],
        ra.iloc[0:0],
        pd.DataFrame({"symbol": ["X"], "altmanZScore": [4733.9], "piotroskiScore": [8]}),
    )
    assert np.isnan(bad.loc["X", "altman_z"]) and bad.loc["X", "piotroski"] == 8


def _snapshot() -> Snapshot:
    uni = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "name": ["Aaa Corp"],
            "sector": ["Tech"],
            "industry": ["x"],
            "market_cap": [1e10],
            "avg_volume": [1e6],
            "in_sp500": [True],
        }
    )
    feats = pd.DataFrame(
        {"close": [100.0], "ret_21d": [-0.15]}, index=pd.Index(["AAA"], name="symbol")
    )
    fund = pd.DataFrame({"piotroski": [7.0]}, index=pd.Index(["AAA"], name="symbol"))
    hit = Hit(
        "AAA",
        "quality_dislocation",
        "long",
        12.0,
        "Aaa Corp -15.0% with fundamentals intact",
        metrics={"close": 100.0, "ret_21d": -0.15, "piotroski": 7, "altman_z": float("nan")},
        reasons=["r1", "r2"],
        context={
            "pt_consensus": 120.0,
            "pt_upside": 0.2,
            "news": [{"date": "2026-10-31", "title": "T <b>", "site": "s"}],
        },
    )
    market = {
        "index": {
            "regime": "risk-on (uptrend, calm vol)",
            "ret_1d": -0.01,
            "ret_5d": 0.0,
            "ret_21d": 0.02,
            "ret_ytd": 0.1,
            "pct_from_252d_high": -0.03,
            "realized_vol_20d": 0.15,
            "realized_vol_20d_pctile_1y": 0.4,
        },
        "vix": {
            "level": 19.4,
            "change_pct": 12.9,
            "avg_50d": 16.0,
            "year_high": 35.0,
            "year_low": 13.0,
        },
        "breadth": {
            "n_symbols": 1,
            "pct_above_sma200": 0.6,
            "pct_above_sma50": 0.5,
            "new_52w_highs": 1,
            "new_52w_lows": 0,
            "pct_up_1d": 0.4,
            "pct_down_20_from_52w_high": 0.2,
        },
        "treasury": {
            "date": "2026-10-30",
            "3m": 3.9,
            "2y": 3.5,
            "10y": 4.2,
            "slope_10y_2y": 0.7,
            "slope_10y_3m": 0.3,
        },
        "sectors": [{"sector": "Tech", "avg_change_pct": 1.5}],
        "econ_ahead": [
            {"date": "2026-11-05", "event": "CPI", "previous": 0.2, "estimate": 0.3, "actual": None}
        ],
        "earnings_ahead": [
            {
                "date": "2026-11-05",
                "symbol": "AAA",
                "name": "Aaa Corp",
                "eps_estimate": 1.0,
                "revenue_estimate": 1e9,
                "market_cap": 1e10,
            }
        ],
    }
    return Snapshot(
        run_date=date(2026, 11, 2),
        as_of=date(2026, 10, 30),
        universe=uni,
        features=feats,
        fundamentals=fund,
        hits={"quality_dislocation": [hit]},
        market=market,
        api_calls=42,
        timings={"bars": 1.5},
        warnings=["insider-trading unavailable: x"],
    )


def test_render_markdown_and_html_cover_every_section() -> None:
    snap = _snapshot()
    md = render_markdown(snap, "## Market read\nAll good.\n- bullet **bold**\n")
    assert "# Eagle Hill daily market report — 2026-11-02" in md
    assert "## Research assistant read" in md and "All good." in md
    assert "Regime: risk-on" in md and "| AAA |" in md and "  - r1" in md
    assert "Nothing fired today." in md and "consensus PT 120.00 (+20.0%)" in md
    assert "Warning: insider-trading unavailable" in md
    assert "10y–2y +70bp" in md and "60.0% above 200dma" in md and "$10.0B" in md
    html = render_html(snap, "## Market read\nAll good.")
    assert "<h3>Market read</h3>" in html and "<b>AAA</b>" in html and "T &lt;b&gt;" in html
    assert "AltZ" in html and "–" in html  # NaN rendered as a dash
    assert render_html(snap, None).count("Research assistant read") == 0


def test_md_to_html_subset() -> None:
    out = md_to_html("# T\npara one\nstill one\n\n- a\n- **b**\n\n1. x\n2. y\n\n## S")
    assert out.startswith("<h2>T</h2>") and "<p>para one still one</p>" in out
    assert "<ul><li>a</li><li><b>b</b></li></ul>".replace("><", ">\n<") in out.replace("><", ">\n<")
    assert "<ol>" in out and "</ol>" in out and out.endswith("<h3>S</h3>")


def test_build_artifacts_round_trip() -> None:
    snap = _snapshot()
    arts = build_artifacts(snap, "narr", "md", "<html>", ReportConfig())
    assert set(arts) == {
        "report.md",
        "report.html",
        "hits.json",
        "market.json",
        "universe_table.csv.gz",
        "meta.json",
        "narrative.md",
    }
    hits = json.loads(arts["hits.json"])
    assert (
        hits["data_as_of"] == "2026-10-30"
        and hits["hits"][0]["symbol"] == "AAA"
        and hits["hits"][0]["metrics"]["altman_z"] is None
    )
    table = pd.read_csv(
        gzip.decompress(arts["universe_table.csv.gz"]).decode().splitlines().__iter__().__class__
        and __import__("io").StringIO(gzip.decompress(arts["universe_table.csv.gz"]).decode())
    )
    assert table.loc[0, "symbol"] == "AAA" and table.loc[0, "piotroski"] == 7.0
    meta = json.loads(arts["meta.json"])
    assert (
        meta["config"]["exchanges"] == ["NASDAQ", "NYSE"]
        and meta["narrative_model"] == "claude-opus-5-5"
    )


def test_s3_keys_and_config_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    keys = s3_keys_for("daily-report/", date(2026, 11, 2), ["report.md"])
    assert keys["report.md"] == (
        "daily-report/2026/11/02/report.md",
        "daily-report/latest/report.md",
    )
    monkeypatch.setenv("EHF_REPORT_EMAIL_TO", "a@x.com, b@x.com")
    monkeypatch.setenv("EHF_REPORT_NARRATIVE", "false")
    monkeypatch.setenv("EHF_REPORT_MAX_SYMBOLS", "50")
    cfg = ReportConfig.from_env()
    assert (
        cfg.email_to == ("a@x.com", "b@x.com") and cfg.narrative is False and cfg.max_symbols == 50
    )


def test_forward_returns_and_scorecard() -> None:
    n = 100
    up = 100.0 * np.exp(np.linspace(0, 0.5, n))  # rises steadily
    flat = np.full(n, 100.0)
    bars = make_bars({"UP": up, "DN": 200.0 - up, "SPY": flat})
    dates = sorted(bars["date"].unique())
    as_of = dates[10]
    hits = hits_frame(
        [
            {
                "run_date": "x",
                "data_as_of": str(as_of)[:10],
                "hits": [
                    {"symbol": "UP", "screen": "s", "direction": "long", "score": 1},
                    {"symbol": "DN", "screen": "s", "direction": "short", "score": 1},
                ],
            },
            {
                "run_date": "y",
                "data_as_of": str(dates[-2])[:10],
                "hits": [{"symbol": "UP", "screen": "s", "direction": "long", "score": 1}],
            },
        ]
    )
    scored = forward_returns(hits, bars, horizons=(5, 21))
    assert (
        scored.loc[0, "ret_5"] > 0 and scored.loc[0, "excess_5"] == scored.loc[0, "ret_5"]
    )  # SPY flat
    assert scored.loc[1, "ret_5"] < 0
    assert np.isnan(scored.loc[2, "ret_5"])  # future not in yet
    card = scorecard(scored, horizons=(5, 21))
    long_row = card[(card["screen"] == "s") & (card["direction"] == "long")].iloc[0]
    short_row = card[(card["screen"] == "s") & (card["direction"] == "short")].iloc[0]
    assert long_row["n"] == 2 and long_row["n_5"] == 1 and long_row["hit_rate_5"] == 1.0
    assert short_row["hit_rate_5"] == 1.0  # short of a faller counts as a win
    assert scorecard(pd.DataFrame()).empty
