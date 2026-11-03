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
