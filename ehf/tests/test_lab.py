"""The lab (``/lab``): stored runs, the playground, symbol charts. Synthetic data only."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ehf.research.backtest.engine import run_backtest
from ehf.research.config.spec import RunSpec, UniverseSpec
from ehf.research.data.provider import DataProvider, Universe
from ehf.research.data.schema import Timeframe, validate_bars
from ehf.research.results.registry import ExperimentRegistry
from ehf.research.results.store import LocalResultStore
from ehf.tests.test_site import FakeS3
from ehf.web.app import create_app
from ehf.web.config import SiteConfig
from ehf.web.lab.charts import Charts
from ehf.web.lab.data import LabData, benchmark_symbol, held_weights, primary_asset, session_date
from ehf.web.lab.playground import Playground, catalog, param_spec

SYMBOLS = ("AAA", "BBB", "CCC")


def synthetic_bars(symbols: tuple[str, ...], start: str, end: str, seed: int = 7) -> pd.DataFrame:
    idx = pd.bdate_range(start, end, tz="UTC") + pd.Timedelta(hours=5)
    rng = np.random.default_rng(seed)
    frames = []
    for i, sym in enumerate(symbols):
        r = rng.normal(0.0004, 0.012, len(idx))
        close = 100.0 * (i + 1) * np.cumprod(1 + r)
        open_ = close * (1 + rng.normal(0, 0.002, len(idx)))
        high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.003, len(idx))))
        low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.003, len(idx))))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "timestamp": idx,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": rng.integers(1e5, 1e6, len(idx)).astype(float),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out.attrs["timeframe"] = Timeframe.DAY.value
    out.attrs["adjusted"] = True
    return validate_bars(out)


class FakeProvider(DataProvider):
    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = bars
        self.calls = 0

    def get_bars(
        self,
        symbols: list[str],
        start: Any,
        end: Any,
        timeframe: Any = Timeframe.DAY,
        *,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        self.calls += 1
        syms = {s.upper() for s in symbols}
        lo, hi = pd.Timestamp(str(start), tz="UTC"), pd.Timestamp(str(end), tz="UTC")
        b = self.bars[self.bars["symbol"].isin(syms)]
        b = b[(b["timestamp"] >= lo) & (b["timestamp"] <= hi + pd.Timedelta(days=1))]
        out = b.reset_index(drop=True)
        out.attrs.update(self.bars.attrs)
        return out

    def get_universe(self, name: str, as_of: Any) -> Universe:
        raise NotImplementedError


def _window(i: int, year: str, params: dict[str, Any], sharpe: float) -> dict[str, Any]:
    return {
        "window": i,
        "test_start": f"{year}-01-01",
        "test_end": f"{year}-12-31",
        "params": params,
        "oos_sharpe": sharpe,
    }


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return synthetic_bars((*SYMBOLS, "SPY"), "2017-01-02", "2021-12-31")


@pytest.fixture
def world(tmp_path: Path, bars: pd.DataFrame) -> dict[str, Any]:
    store = LocalResultStore(tmp_path / "results")
    registry = ExperimentRegistry(store)
    spec = RunSpec(
        strategy="tsmom",
        params={"lookback": 60, "vol_window": 20},
        universe=UniverseSpec(symbols=SYMBOLS),
        start="2018-01-01",
        end="2021-12-31",
    )
    res = run_backtest(spec, bars)
    h = registry.log_run(spec, res)
    cards = tmp_path / "scorecards"
    cards.mkdir()
    (cards / "wf_tsmom_abcd1234.json").write_text(
        json.dumps(
            {
                "wf_id": "wf_tsmom_abcd1234",
                "strategy": "tsmom",
                "base_spec": json.loads(spec.canonical_json()),
                "grid": {"lookback": [40, 60]},
                "walk_forward": {"train_months": 12, "test_months": 12, "objective": "sharpe"},
                "stitched_hash": h,
                "oos_metrics": {"sharpe": 0.5, "cagr": 0.04, "max_drawdown": -0.2},
                "windows": [
                    {
                        "window": 0,
                        "test_start": "2019-01-01",
                        "test_end": "2019-12-31",
                        "params": {"lookback": 40},
                        "oos_sharpe": 0.3,
                    },
                    {
                        "window": 1,
                        "test_start": "2020-01-01",
                        "test_end": "2020-12-31",
                        "params": {"lookback": 60},
                        "oos_sharpe": 0.7,
                    },
                ],
                "verdict": {"status": "pass", "failed": []},
                "extra": {"time_in_market": 0.8, "switches_per_year": 3.0},
                "notes": "Full-sample default-param run `0000000000000000` (in-sample).",
            }
        )
    )
    provider = FakeProvider(bars)
    data = LabData(store=store, provider=provider, scorecard_dir=cards)
    play = Playground(provider, registry=registry)
    return {
        "store": store,
        "hash": h,
        "data": data,
        "play": play,
        "charts": Charts(provider),
        "provider": provider,
    }


@pytest.fixture
def client(world: dict[str, Any]) -> TestClient:
    from ehf.web.library import Library

    cfg = SiteConfig(lab="on", reports_bucket="b")
    lib = Library(FakeS3({}), "b")
    app = create_app(cfg, library=lib, lab=(world["data"], world["play"], world["charts"]))
    return TestClient(app)


# -- helpers -----------------------------------------------------------------------------


def test_session_date_is_new_york_calendar_day() -> None:
    assert session_date(pd.Timestamp("2024-03-05 05:00:00", tz="UTC")) == "2024-03-05"
    assert session_date(pd.Timestamp("2024-07-05 04:00:00", tz="UTC")) == "2024-07-05"


def test_held_weights_restart_at_resets() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=6, tz="UTC"))
    trades = pd.DataFrame(
        {
            "timestamp": [idx[0], idx[3]],
            "symbol": ["SPY", "SPY"],
            "delta_weight": [1.0, 1.0],
            "side": ["buy", "buy"],
            "price": [1.0, 1.0],
        }
    )
    plain = held_weights(trades, idx)["SPY"].tolist()
    assert plain == [1, 1, 1, 2, 2, 2]
    reset = held_weights(trades, idx, resets=[idx[3]])["SPY"].tolist()
    assert reset == [1, 1, 1, 1, 1, 1]


def test_primary_asset_prefers_dial_asset_then_first_real_symbol() -> None:
    assert primary_asset({"params": {"asset": "qqq"}, "universe": {"symbols": ["SPY"]}}) == "QQQ"
    assert primary_asset({"params": {}, "universe": {"symbols": ["^VIX", "CASH", "IWM"]}}) == "IWM"


def test_benchmark_is_the_timed_asset_for_dials_and_spy_otherwise() -> None:
    assert benchmark_symbol({"params": {"asset": "qqq"}}) == "QQQ"
    assert benchmark_symbol({"params": {}, "universe": {"symbols": ["HRL", "GRMN"]}}) == "SPY"


def test_param_spec_ranges() -> None:
    s = param_spec("window", 200, [100, 150, 200, 250])
    assert s["type"] == "int" and s["min"] >= 0 and s["max"] > 250 and s["step"] >= 1
    f = param_spec("band", 0.05, None)
    assert f["type"] == "float" and f["min"] < 0.05 < f["max"]
    assert param_spec("use_ratio", False, None)["type"] == "bool"


def test_catalog_lists_registered_strategies() -> None:
    keys = {s["key"] for s in catalog()}
    assert {"tsmom", "vix_vote", "dial_sma"} <= keys
    vote = next(s for s in catalog() if s["key"] == "vix_vote")
    assert vote["kind"] == "dial"
    assert "asset" not in {p["name"] for p in vote["params"]}  # the asset is the universe input
    assert "vix" not in {p["name"] for p in vote["params"]}
    boll = next(s for s in catalog() if s["key"] == "bollinger_reversion")
    assert "Base class" not in boll["doc"]  # no inherited docstring


# -- browser + run ------------------------------------------------------------------------


def test_lab_pages_render(client: TestClient, world: dict[str, Any]) -> None:
    r = client.get("/lab")
    assert r.status_code == 200 and "wf_tsmom_abcd1234" in r.text and "Playground" in r.text
    assert client.get(f"/lab/run/{world['hash']}").status_code == 200
    assert client.get("/lab/run/nope").status_code == 404
    assert client.get("/lab/play").status_code == 200
    assert client.get("/lab/chart?symbol=aaa").status_code == 200
    assert 'href="/lab"' in client.get("/").text  # nav shows the lab


def test_studies_and_runs_api(client: TestClient, world: dict[str, Any]) -> None:
    s = client.get("/lab/api/studies").json()["studies"]
    assert s[0]["strategy"] == "tsmom" and s[0]["n_windows"] == 2 and s[0]["verdict"] == "pass"
    assert s[0]["full_hash"] == "0000000000000000"
    runs = client.get("/lab/api/runs").json()["runs"]
    assert [r["hash"] for r in runs] == [world["hash"]]
    assert runs[0]["metrics"]["sharpe"] is not None
    assert client.get("/lab/api/runs?strategy=rise_fall").json()["runs"] == []


def test_run_payload(client: TestClient, world: dict[str, Any]) -> None:
    p = client.get(f"/lab/api/run/{world['hash']}").json()
    n = len(p["series"]["time"])
    assert n > 900
    for k in ("equity", "drawdown", "exposure", "gross", "benchmark", "benchmark_drawdown"):
        assert len(p["series"][k]) == n, k
    assert (
        p["asset"] == "AAA" and p["candles"] and p["candles"][0]["time"] == p["series"]["time"][0]
    )
    assert p["benchmark"] == "SPY"  # not the first stock in the basket
    assert p["benchmark_metrics"]["cagr"] is not None
    assert p["windows"] and p["windows"][1]["params"] == {"lookback": 60}
    assert p["study"]["wf_id"] == "wf_tsmom_abcd1234"
    assert {y["year"] for y in p["yearly"]} == {2018, 2019, 2020, 2021}
    assert max(p["series"]["drawdown"]) <= 0.0
    assert client.get("/lab/api/run/ffffffffffffffff").status_code == 404


# -- playground ---------------------------------------------------------------------------


def test_playground_runs_and_caches(client: TestClient, world: dict[str, Any]) -> None:
    body = {
        "strategy": "tsmom",
        "params": {"lookback": "90", "vol_window": 20},
        "symbols": ["aaa", "bbb"],
        "start": "2019-01-01",
        "end": "2021-06-30",
        "slippage_bps": 2,
    }
    r = client.post("/lab/api/play", json=body)
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["params"] == {"lookback": 90, "vol_window": 20}
    assert p["strategy"] == "tsmom" and p["logged"] is False and p["source"] == "playground"
    assert p["benchmark"] == "SPY" and p["asset"] == "AAA"
    assert len(p["series"]["benchmark"]) == len(p["series"]["time"])
    assert p["series"]["time"][0] >= "2019-01-01" and p["series"]["time"][-1] <= "2021-06-30"
    assert p["costs"]["slippage_bps"] == 2.0
    assert not world["store"].has_run(p["hash"])
    calls = world["provider"].calls
    again = client.post("/lab/api/play", json={**body, "log": True}).json()
    assert again["hash"] == p["hash"] and again["logged"] is True
    assert world["provider"].calls == calls  # bars came from the cache
    assert world["store"].has_run(p["hash"])


def test_playground_validation(client: TestClient) -> None:
    assert client.post("/lab/api/play", json={"strategy": "nope"}).status_code == 400
    bad = {
        "strategy": "tsmom",
        "params": {"bogus": 1},
        "symbols": ["AAA"],
        "start": "2019-01-01",
        "end": "2020-01-01",
    }
    assert client.post("/lab/api/play", json=bad).status_code == 400
    r = client.post(
        "/lab/api/play", json={**bad, "params": {}, "start": "2020-01-01", "end": "2019-01-01"}
    )
    assert r.status_code == 400
    assert client.get("/lab/api/catalog").json()["strategies"]


# -- charts -------------------------------------------------------------------------------


def test_symbol_chart_falls_back_without_vix(client: TestClient) -> None:
    d = client.get("/lab/api/bars/BBB?start=2020-01-01&end=2021-12-31").json()
    assert d["symbol"] == "BBB" and d["candles"][0]["time"] >= "2020-01-01"
    assert d["overlays"]["sma50"] and d["overlays"]["sma200"]
    assert d["vix"] == [] and d["exposure"] == []
    assert d["last"]["close"] and d["last"]["high_52w"] >= d["last"]["low_52w"]
    assert client.get("/lab/api/bars/ZZZ?start=2020-01-01&end=2021-12-31").status_code == 404
    assert client.get("/lab/api/bars/bad%20name?start=2020-01-01&end=2021-01-01").status_code == 400


class LiveProvider(FakeProvider):
    """A provider that can also re-pull a tail and quote, like FMP."""

    def __init__(self, bars: pd.DataFrame) -> None:
        super().__init__(bars)
        self.tails: list[list[str]] = []
        self.quotes = 0

    def refresh_tail(self, symbols: list[str], *, days: int = 10, **_: Any) -> pd.DataFrame:
        self.tails.append(list(symbols))
        return self.bars.iloc[0:0]

    def intraday(self, symbol: str, **_: Any) -> pd.DataFrame:
        """Two sessions of flat one-minute bars (the second one half done)."""
        frames = []
        for day, n in (("2021-12-30", 390), ("2021-12-31", 200)):
            idx = pd.date_range(f"{day} 09:30", periods=n, freq="min", tz="America/New_York")
            px = 100.0 if symbol != "^VIX" else 20.0
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "timestamp": idx.tz_convert("UTC"),
                        "open": px,
                        "high": px + 1,
                        "low": px - 1,
                        "close": px + 0.5,
                        "volume": 10.0,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        self.quotes += 1
        q = {"price": 101.5, "change_pct": 0.004, "time": "2021-12-31T20:59:50+00:00"}
        return {s: dict(q) for s in symbols}


def test_charts_freshen_tail_on_a_timer_and_on_demand(bars: pd.DataFrame, monkeypatch: Any) -> None:
    prov = LiveProvider(bars)
    ch = Charts(prov, tail_ttl_s=900, refresh_min_s=60)
    clock = [1_000_000.0]
    monkeypatch.setattr("ehf.web.lab.charts.time.time", lambda: clock[0])
    d = ch.symbol("BBB", "2020-01-01", "2021-12-31")
    assert prov.tails == [["BBB", "^VIX"]]  # first view pulls the tail
    assert d["live"]["quote"]["price"] == 101.5 and d["live"]["bars_pulled_at"]
    ch.symbol("BBB", "2020-01-01", "2021-12-31")
    ch.symbol("BBB", "2020-01-01", "2021-12-31", refresh=True)  # too soon even when forced
    assert len(prov.tails) == 1
    clock[0] += 61
    ch.symbol("BBB", "2020-01-01", "2021-12-31", refresh=True)  # forced, past the minimum
    assert len(prov.tails) == 2
    clock[0] += 300
    ch.symbol("BBB", "2020-01-01", "2021-12-31")  # not due on the timer yet
    assert len(prov.tails) == 2
    clock[0] += 900
    ch.symbol("BBB", "2020-01-01", "2021-12-31")  # due
    assert len(prov.tails) == 3
    assert prov.quotes == 6  # a quote on every view, never cached
    assert Charts(FakeProvider(bars)).symbol("BBB", "2020-01-01", "2021-12-31")["live"] == {
        "quote": None,
        "bars_pulled_at": None,
        "generated_at": ANY_ISO,
    }


class _AnyIso:
    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and other.endswith("+00:00")


ANY_ISO = _AnyIso()


def test_intraday_payload_is_the_latest_session(bars: pd.DataFrame, monkeypatch: Any) -> None:
    prov = LiveProvider(bars)
    ch = Charts(prov)
    clock = [2_000_000.0]
    monkeypatch.setattr("ehf.web.lab.charts.time.time", lambda: clock[0])
    d = ch.intraday("BBB")
    assert d["session"] == "2021-12-31" and d["interval"] == "1min"
    assert len(d["candles"]) == 200 and len(d["vix"]) == 200
    assert d["stats"]["bars"] == 200 and d["stats"]["last"] == 100.5
    assert d["stats"]["last_time"].startswith("2021-12-31T12:49")
    # chart times are Eastern wall-clock read as UTC: 09:30 ET -> 09:30 "UTC"
    assert d["candles"][0]["time"] % 86400 == 9 * 3600 + 30 * 60
    assert d["prev_close"] is not None and d["stats"]["change"] is not None
    # BBB's daily close on 2021-12-30 is the prior close (synthetic bars have it)
    prior = bars[(bars["symbol"] == "BBB") & (bars["timestamp"] < "2021-12-31")]["close"].iloc[-1]
    assert d["prev_close"] == pytest.approx(float(prior))
    assert d["live"]["quote"]["price"] == 101.5
    calls = prov.calls
    ch.intraday("BBB")  # served from the minute cache
    ch.intraday("BBB", refresh=True)  # a forced refresh still waits 15 s
    assert prov.calls == calls
    clock[0] += 16
    ch.intraday("BBB", refresh=True)
    assert prov.calls > calls


def test_parse_indicators() -> None:
    from ehf.web.lab.charts import indicator_key, max_window, parse_indicators

    d = parse_indicators()
    assert d["sma"] == (50, 200) and d["ema"] == () and d["bb"] is None and d["rsi"] is None
    e = parse_indicators(sma="20, 50;200", ema="21", bb="20,2.5", rsi="14")
    assert e["sma"] == (20, 50, 200) and e["ema"] == (21,)
    assert e["bb"] == {"window": 20, "n_std": 2.5} and e["rsi"] == {"window": 14}
    assert indicator_key(e) == "20,50,200|21|20,2.5|14" and max_window(e) == 200
    assert parse_indicators(sma="")["sma"] == ()  # blank turns the default lines off
    assert parse_indicators(bb="20")["bb"]["n_std"] == 2.0
    for bad in ({"sma": "x"}, {"sma": "1"}, {"sma": "600"}, {"sma": "2,3,4,5,6"}, {"bb": "20,9"}):
        with pytest.raises(ValueError):
            parse_indicators(**bad)


def test_bars_api_indicators(client: TestClient) -> None:
    base = "/lab/api/bars/BBB?start=2020-01-01&end=2021-12-31"
    d = client.get(base).json()
    assert set(d["overlays"]) == {"sma50", "sma200"} and d["indicators"]["sma"] == [50, 200]
    assert d["overlays"]["sma200"][0]["time"] == d["candles"][0]["time"]  # warm on day one
    e = client.get(f"{base}&sma=20&ema=21&bb=20,2&rsi=14").json()
    assert set(e["overlays"]) == {"sma20", "ema21", "bb", "rsi"}
    assert e["overlays"]["bb"]["upper"][0]["value"] > e["overlays"]["bb"]["lower"][0]["value"]
    assert all(0 <= x["value"] <= 100 for x in e["overlays"]["rsi"]["values"])
    assert e["overlays"]["rsi"]["values"][0]["time"] == e["candles"][0]["time"]
    assert client.get(f"{base}&sma=nope").status_code == 400
    assert client.get(f"{base}&sma=").json()["overlays"] == {}


def test_intraday_api(client: TestClient) -> None:
    # the default fixture provider has no minute bars -> a clear 404, not a crash
    r = client.get("/lab/api/intraday/BBB")
    assert r.status_code == 404 and "not available" in r.json()["detail"]
    assert client.get("/lab/api/intraday/bad%20name").status_code == 400


def test_bars_api_accepts_refresh(client: TestClient) -> None:
    r = client.get("/lab/api/bars/BBB?start=2020-01-01&end=2021-12-31&refresh=1")
    assert r.status_code == 200 and "live" in r.json()


def test_mirror_mode_hides_the_log_box(world: dict[str, Any]) -> None:
    from ehf.web.library import Library

    lib = Library(FakeS3({}), "b")
    cfg = SiteConfig(lab="on", reports_bucket="b", var_bucket="ehf-results-x")
    app = create_app(cfg, library=lib, lab=(world["data"], world["play"], world["charts"]))
    html = TestClient(app).get("/lab/play").text
    assert "f-log" not in html and "Nothing you do here is saved" in html


def test_lab_off_hides_everything() -> None:
    from ehf.web.library import Library

    app = create_app(SiteConfig(lab="off"), library=Library(FakeS3({}), "b"))
    c = TestClient(app)
    assert c.get("/lab").status_code == 404
    assert 'href="/lab"' not in c.get("/").text


def test_today_is_a_valid_default_end() -> None:
    assert date.today().isoformat() >= "2026-01-01"


# -- the short version (ehf.web.lab.explain) ----------------------------------------------


def _payload(**over: Any) -> dict[str, Any]:
    """A run payload the way ``run_payload`` shapes it, small enough to write by hand."""
    base: dict[str, Any] = {
        "hash": "abcdef0123456789",
        "strategy": "vix_term",
        "params": {"asset": "SPY", "out_above": 1.05, "band": 0.1, "confirm_bars": 3},
        "universe": {"symbols": ["SPY", "CASH", "^VIX", "^VIX3M"]},
        "start": "2006-08-01",
        "end": "2026-10-30",
        "initial_capital": 1_000_000.0,
        "costs": {"slippage_bps": 1.0},
        "fill_model": "next_open",
        "asset": "SPY",
        "benchmark": "SPY",
        "metrics": {
            "n_bars": 5040.0,
            "cagr": 0.075,
            "total_return": 3.3,
            "max_drawdown": -0.33,
            "sharpe": 0.64,
            "ann_vol": 0.13,
        },
        "benchmark_metrics": {
            "cagr": 0.092,
            "total_return": 4.9,
            "max_drawdown": -0.565,
            "sharpe": 0.55,
            "ann_vol": 0.19,
        },
        "series": {"exposure": [1.0, 1.0, 0.0, 1.0], "gross": [1.0, 1.0, 0.0, 1.0]},
        "n_trades": 355,
        "yearly": [
            {"year": 2020, "strategy": 0.10, "benchmark": 0.18},
            {"year": 2022, "strategy": -0.265, "benchmark": -0.18},
        ],
        "windows": None,
        "study": None,
    }
    base.update(over)
    return base


def test_explain_puts_the_numbers_in_the_sentences() -> None:
    from ehf.web.lab.explain import explain

    e = explain(_payload())
    assert e["name"] == "Fear curve flip" and e["tag"] == "VIX rule 1" and e["kind"] == "dial"
    assert e["headline"].startswith("7.5% a year against 9.2% for holding SPY")
    assert "VIX3M" in e["what"] and "cash" in e["what"]
    out, back = e["rules"]
    assert out["when"] == "Step out of SPY when" and "above 1.05" in out["html"]
    assert "3 days in a row" in out["html"]  # confirm_bars=3
    assert "below 0.95" in back["html"]  # out_above - band
    assert "VIX<sub>t</sub>" in e["math"]
    assert "$1,000,000 became $4,300,000" in e["results"][0]
    assert "worse than the market" in e["results"][0]
    assert "-33.0%" in e["results"][1] and "better than holding the market" in e["results"][1]
    assert "In the market <b>75%</b>" in " ".join(e["results"])
    assert "355 trades, about 18 a year" in " ".join(e["results"])
    assert "Beat SPY in <b>0 of 2</b>" in e["results"][-1] and "-26.5% in 2022" in e["results"][-1]
    assert "2006-08-01" in e["setup"][0] and "20.0 years" in e["setup"][0]
    assert "Treasury bill" in e["setup"][1] and "1 basis point of slippage" in e["setup"][2]
    assert e["verdict"] is None and not any("Walk-forward" in s for s in e["setup"])


def test_explain_walk_forward_names_the_grid_and_the_verdict() -> None:
    from ehf.web.lab.explain import explain

    p = _payload(
        params={
            "base_params": {},
            "grid": {"out_above": [0.95, 1.0, 1.05], "band": [0.02, 0.05], "confirm_bars": [1]},
            "walk_forward": {"train_months": 24, "test_months": 12},
        },
        windows=[{"window": 0, "test_start": "2008-08-01", "test_end": "2009-07-31"}] * 19,
        study={
            "verdict": "killed",
            "failed": ["dsr", "trades", "pbo"],
            "objective": "calmar",
            "train_months": 24,
            "test_months": 12,
            "dd_2008": -0.21,
            "dd_2020": -0.15,
            "checks": [
                {"name": "dsr", "value": 0.12, "threshold": 0.95, "passed": False},
                {"name": "trades", "value": 19.0, "threshold": 100.0, "passed": False},
                {"name": "pbo", "value": None, "threshold": 0.3, "passed": False},
            ],
        },
    )
    e = explain(p)
    wf = next(s for s in e["setup"] if "Walk-forward" in s)
    assert "19 out-of-sample windows" in wf and "previous 24 months" in wf and "Calmar" in wf
    assert "<b>out_above</b>: 0.95, 1 or 1.05" in wf and "<b>band</b>: 0.02 or 0.05" in wf
    assert "confirm_bars" not in wf  # a one-cell grid is not a choice
    assert "above 1" in e["rules"][0]["html"]  # defaults, since base_params is empty
    assert any("-21% in 2008, -15% in 2020" in r for r in e["results"])
    v = e["verdict"]
    assert v["status"] == "killed"
    assert "how many settings were tried (0.12 against 0.95 needed)" in v["html"]
    assert "too few trades to trust the numbers (19 against 100 needed)" in v["html"]
    assert "overfit" in v["html"] and "could not even be computed" in v["html"]

    p["study"]["verdict"], p["study"]["failed"] = "pass", []
    assert explain(p)["verdict"]["status"] == "pass"


def test_explain_leverage_extended_pair() -> None:
    from ehf.web.lab.explain import explain

    p = _payload(
        strategy="lev_swing",
        params={
            "asset": "TQQQ.X",
            "base": "QQQ",
            "benchmark": "QQQ",
            "up_pct": 0.15,
            "down_pct": 0.2,
            "lookback": 126,
            "min_dwell": 5,
        },
        universe={"symbols": ["QQQ", "TQQQ.X", "^VIX"]},
        asset="TQQQ.X",
        benchmark="QQQ",
    )
    e = explain(p)
    assert e["kind"] == "leverage" and e["name"] == "Bull and bear swings"
    assert "3× cousin TQQQ" in e["what"] and "never from TQQQ's" in e["what"]
    assert "up 15% or more from its lowest close of the last 126 days" in e["rules"][0]["html"]
    assert e["rules"][1]["when"] == "Back to QQQ when" and "down 20%" in e["rules"][1]["html"]
    assert "at least 5 trading days (about a week)" in e["what"]  # min_dwell
    assert any("extended series" in s and "3× QQQ" in s for s in e["setup"])
    assert "In the leveraged fund <b>75%</b>" in " ".join(e["results"])


def test_explain_portfolio_uses_gross_exposure_and_escapes_symbols() -> None:
    from ehf.web.lab.explain import explain

    p = _payload(
        strategy="tsmom",
        params={"lookback": 60, "vol_window": 20},
        universe={"symbols": ["AAA", "<b>EVIL</b>", "CCC"]},
        asset="<b>EVIL</b>",
        benchmark="SPY",
        series={"exposure": [0.0, 0.0, 0.0, 0.0], "gross": [0.5, 1.0, 1.0, 0.5]},
    )
    e = explain(p)
    assert e["kind"] == "portfolio" and e["name"] == "Trend following"
    assert "each of the 3 symbols" in e["what"] and "last 60 trading days" in e["what"]
    assert "0.10 / σ" not in e["math"] and "0.1 / σ<sub>i,20</sub>" in e["math"]  # defaults fill in
    assert "In the market <b>75%</b>" in " ".join(e["results"])  # gross, not the first symbol
    joined = json.dumps(e)
    assert "<b>EVIL</b>" not in joined and "&lt;b&gt;EVIL&lt;/b&gt;" in joined


def test_explain_never_breaks_the_page() -> None:
    from ehf.web.lab.explain import explain

    e = explain(
        _payload(strategy="no_such_strategy", params={"k": 1}, metrics={}, series={}, yearly=[])
    )
    assert e["name"] == "no such strategy" and "k = 1" in e["what"] and e["results"] == []
    # a known strategy with a parameter missing falls back to its docstring, not a 500
    e = explain(_payload(strategy="vix_sized", params={"asset": "SPY", "target_vol": "bad"}))
    assert e["name"] == "Size by fear" and e["what"]


def test_run_payload_carries_the_short_version(client: TestClient, world: dict[str, Any]) -> None:
    p = client.get(f"/lab/api/run/{world['hash']}").json()
    e = p["explain"]
    assert e["name"] == "Trend following" and e["kind"] == "portfolio"
    assert "last 60 trading days" in e["what"]
    assert e["verdict"]["status"] == "pass"  # the fixture's study passed
    wf = next(s for s in e["setup"] if "Walk-forward" in s)
    assert "2 out-of-sample windows" in wf and "<b>lookback</b>: 40 or 60" in wf
    assert p["study"]["checks"] == [] and p["study"]["grid"] == {"lookback": [40, 60]}
    play = client.post(
        "/lab/api/play",
        json={
            "strategy": "tsmom",
            "params": {},
            "symbols": ["aaa"],
            "start": "2019-01-01",
            "end": "2021-06-30",
        },
    ).json()
    assert play["explain"]["name"] == "Trend following" and play["explain"]["verdict"] is None


def test_blurb_is_one_or_two_sentences_with_the_numbers() -> None:
    from ehf.web.lab.explain import blurb

    b = blurb(
        "lev_sma",
        {"asset": "TQQQ.X", "base": "QQQ", "benchmark": "QQQ", "window": 150, "band": 0.03},
        universe="QQQ,TQQQ.X,CASH,^VIX",
        asset="TQQQ.X",
        metrics={"cagr": 0.254, "max_drawdown": -0.755, "sharpe": 0.71},
        market={"cagr": 0.124, "max_drawdown": -0.536},
        out_of_sample=True,
    )
    assert b["name"] == "Above the average" and b["tag"] == "Leverage rule"
    assert b["text"].startswith(
        "Into TQQQ when QQQ closes more than 3% above its 150-day average; "
        "back to QQQ when QQQ closes more than 3% below its 150-day average. "
    )
    assert "Made <b>25.4%</b> a year against 12.4% for the market" in b["text"]
    assert "worst fall of <b>-76%</b> versus -54%; Sharpe 0.71 out of sample." in b["text"]
    # no rule boxes: the first sentence of the description; no numbers: no second sentence
    b = blurb("dial_hold", {"asset": "SPY"}, universe=["SPY", "CASH"])
    assert b["text"] == "Buy SPY on day one and never sell."
    # a hostile symbol is escaped
    b = blurb("dial_sma", {"asset": "<i>x</i>"}, universe=["<i>x</i>"])
    assert "<i>" not in b["text"] and "&lt;i&gt;x&lt;/i&gt;" in b["text"]


def test_rows_carry_blurbs(client: TestClient, world: dict[str, Any]) -> None:
    s = client.get("/lab/api/studies").json()["studies"][0]
    assert s["blurb"]["name"] == "Trend following"
    assert "last 60 trading days is positive" in s["blurb"]["text"]
    assert "Made <b>4.0%</b> a year" in s["blurb"]["text"] and "out of sample" in s["blurb"]["text"]
    r = client.get("/lab/api/runs").json()["runs"][0]
    assert r["blurb"]["name"] == "Trend following" and "Made <b>" in r["blurb"]["text"]
    page = client.get("/lab").text
    assert 'data-tip-name="Trend following"' in page and "data-tip=" in page
