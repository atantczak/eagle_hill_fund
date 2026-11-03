"""Run a backtest on demand: a strategy, a few sliders, a date range, go.

The catalog turns the strategy registry into something a form can render: every
parameter with its default, type, and a sensible slider range (taken from the
study grids where one exists, otherwise widened around the default). Dial
strategies (one asset vs. cash, steered by the VIX family) get the VIX dataset
assembled for *any* asset; portfolio strategies take a symbol list.

Results are cached by config hash, so dragging a slider back to a previous
value is instant, and may optionally be logged to the registry like any other
run — the playground is a front door to the same engine, not a toy.
"""

from __future__ import annotations

import inspect
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from ehf.research import strategies  # noqa: F401  (import registers strategies)
from ehf.research.backtest.engine import run_backtest
from ehf.research.config.spec import CostConfig, FillModel, RunSpec, UniverseSpec
from ehf.research.data.provider import DataProvider
from ehf.research.data.schema import Timeframe
from ehf.research.data.universes import STATIC_UNIVERSES
from ehf.research.data.vix import CASH, INDEX_SYMBOLS, SKEW, load_vix_bars
from ehf.research.results.registry import ExperimentRegistry
from ehf.research.strategies.base import _REGISTRY, Strategy
from ehf.research.strategies.dial import DialStrategy
from ehf.research.strategies.vix_overlay import GRIDS
from ehf.web.lab.data import run_payload, spec_dict

HIDDEN_PARAMS = {"vix", "vix3m", "vvix", "skew", "cash"}
WARMUP_DAYS = 600  # calendar days of history before ``start`` for signal warm-up
MAX_SYMBOLS = 40


def _kind(cls: type[Strategy]) -> str:
    return "dial" if issubclass(cls, DialStrategy) else "portfolio"


def _doc(cls: type[Strategy]) -> str:
    doc = inspect.getdoc(cls) or ""
    return doc.strip().splitlines()[0] if doc.strip() else cls.__name__


def _nice_step(span: float) -> float:
    """A slider step that gives ~40 stops across ``span``."""
    if span <= 0:
        return 1.0
    raw = span / 40.0
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    for m in (1, 2, 5, 10):
        if m * mag >= raw:
            return float(m * mag)
    return float(mag * 10)


def param_spec(name: str, default: Any, grid: Sequence[Any] | None) -> dict[str, Any]:
    spec: dict[str, Any] = {"name": name, "default": default}
    if isinstance(default, bool):
        spec["type"] = "bool"
        return spec
    if isinstance(default, int | float) and not isinstance(default, bool):
        is_int = isinstance(default, int)
        vals = [float(v) for v in (grid or []) if isinstance(v, int | float)]
        if vals:
            lo, hi = min(vals), max(vals)
            span = hi - lo
            lo, hi = lo - 0.5 * span, hi + 0.5 * span
            if min(vals) >= 0:
                lo = max(lo, 0.0)
        else:
            d = float(default)
            if d == 0:
                lo, hi = -1.0, 1.0
            elif d > 0:
                lo, hi = d / 4.0, d * 4.0
            else:
                lo, hi = d * 4.0, d / 4.0
        if is_int:
            lo, hi = float(int(lo)), float(max(int(hi), int(lo) + 1))
            step = max(1.0, float(round(_nice_step(hi - lo))))
            spec.update(type="int", min=int(lo), max=int(hi), step=int(step))
        else:
            step = _nice_step(hi - lo)
            spec.update(type="float", min=round(lo, 6), max=round(hi, 6), step=step)
        spec["grid"] = list(grid) if grid else None
        return spec
    spec["type"] = "str"
    spec["grid"] = list(grid) if grid else None
    return spec


def catalog() -> list[dict[str, Any]]:
    """Every registered strategy with its editable parameters."""
    out = []
    for key, cls in sorted(_REGISTRY.items()):
        grid = GRIDS.get(key, {})
        kind = _kind(cls)
        params = []
        for name, default in cls.defaults.items():
            if name in HIDDEN_PARAMS:
                continue
            if kind == "dial" and name == "asset":
                continue  # the asset is the universe input, not a slider
            params.append(param_spec(name, default, grid.get(name)))
        out.append(
            {
                "key": key,
                "name": cls.__name__,
                "doc": _doc(cls),
                "kind": kind,
                "params": params,
            }
        )
    return out


def universe_presets() -> list[dict[str, Any]]:
    return [
        {"name": k, "symbols": list(v)}
        for k, v in STATIC_UNIVERSES.items()
        if isinstance(v, tuple | list) and 0 < len(v) <= MAX_SYMBOLS
    ]


@dataclass(frozen=True)
class PlayRequest:
    strategy: str
    params: dict[str, Any]
    start: str
    end: str
    asset: str = "SPY"
    symbols: tuple[str, ...] = ()
    slippage_bps: float = 1.0
    fill_model: str = "next_open"
    initial_capital: float = 1_000_000.0
    log: bool = False

    @classmethod
    def parse(cls, body: dict[str, Any]) -> PlayRequest:
        strategy = str(body.get("strategy") or "")
        if strategy not in _REGISTRY:
            raise ValueError(f"unknown strategy {strategy!r}")
        params = coerce_params(strategy, body.get("params") or {})
        start, end = str(body.get("start") or ""), str(body.get("end") or "")
        date.fromisoformat(start)
        date.fromisoformat(end)
        if start >= end:
            raise ValueError("start must be before end")
        symbols = tuple(
            s.strip().upper()
            for s in (body.get("symbols") or [])
            if isinstance(s, str) and s.strip()
        )
        if len(symbols) > MAX_SYMBOLS:
            raise ValueError(f"at most {MAX_SYMBOLS} symbols")
        fill = str(body.get("fill_model") or "next_open")
        FillModel(fill)
        return cls(
            strategy=strategy,
            params=params,
            start=start,
            end=end,
            asset=str(body.get("asset") or "SPY").strip().upper() or "SPY",
            symbols=symbols,
            slippage_bps=float(body.get("slippage_bps", 1.0)),
            fill_model=fill,
            initial_capital=float(body.get("initial_capital", 1_000_000.0)),
            log=bool(body.get("log", False)),
        )


def coerce_params(strategy: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Cast form values to the types the strategy's defaults declare."""
    cls = _REGISTRY[strategy]
    out: dict[str, Any] = {}
    for name, value in raw.items():
        if name not in cls.defaults:
            raise ValueError(f"{strategy} has no parameter {name!r}")
        default = cls.defaults[name]
        if isinstance(default, bool):
            out[name] = value in (True, "true", "True", 1, "1", "on")
        elif isinstance(default, int):
            out[name] = int(round(float(value)))
        elif isinstance(default, float):
            out[name] = float(value)
        else:
            out[name] = value
    return out


class Playground:
    """On-demand runs with two small caches: assembled bars, and finished payloads."""

    def __init__(
        self,
        provider: DataProvider,
        registry: ExperimentRegistry | None = None,
        *,
        max_results: int = 64,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self._bars: OrderedDict[tuple[Any, ...], pd.DataFrame] = OrderedDict()
        self._results: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_results = max_results
        self._lock = threading.Lock()

    # -- data ------------------------------------------------------------------------

    def _data_start(self, start: str) -> str:
        return (date.fromisoformat(start) - timedelta(days=WARMUP_DAYS)).isoformat()

    def bars_for(self, req: PlayRequest) -> tuple[pd.DataFrame, tuple[str, ...]]:
        kind = _kind(_REGISTRY[req.strategy])
        data_start = self._data_start(req.start)
        if kind == "dial":
            key: tuple[Any, ...] = ("vix", req.asset, data_start, req.end)
        else:
            if not req.symbols:
                raise ValueError("this strategy needs a symbol list")
            key = ("symbols", req.symbols, data_start, req.end)
        with self._lock:
            hit = self._bars.get(key)
        if hit is None:
            if kind == "dial":
                hit = load_vix_bars(
                    self.provider, data_start, req.end, asset=req.asset, offline_skew=True
                )
            else:
                hit = self.provider.get_bars(list(req.symbols), data_start, req.end, Timeframe.DAY)
            with self._lock:
                self._bars[key] = hit
                while len(self._bars) > 8:
                    self._bars.popitem(last=False)
        symbols = tuple(sorted(hit["symbol"].unique()))
        return hit, symbols

    # -- runs --------------------------------------------------------------------------

    def spec_for(self, req: PlayRequest, symbols: tuple[str, ...]) -> RunSpec:
        kind = _kind(_REGISTRY[req.strategy])
        params = dict(req.params)
        if kind == "dial":
            params["asset"] = req.asset
        return RunSpec(
            strategy=req.strategy,
            params=params,
            universe=UniverseSpec(symbols=symbols),
            start=req.start,
            end=req.end,
            initial_capital=req.initial_capital,
            costs=CostConfig(slippage_bps=req.slippage_bps),
            fill_model=FillModel(req.fill_model),
        )

    def run(self, req: PlayRequest) -> dict[str, Any]:
        bars, symbols = self.bars_for(req)
        spec = self.spec_for(req, symbols)
        h = spec.config_hash
        with self._lock:
            cached = self._results.get(h)
            if cached is not None:
                self._results.move_to_end(h)
        if cached is not None:
            if req.log and self.registry is not None and not cached.get("logged"):
                self._log(spec, bars, cached)
            return cached

        res = run_backtest(spec, bars)
        asset = req.asset if _kind(_REGISTRY[req.strategy]) == "dial" else symbols[0]
        asset_bars = bars.loc[bars["symbol"] == asset].reset_index(drop=True)
        payload = run_payload(
            spec=spec_dict(spec),
            metrics=res.metrics,
            equity=res.equity,
            trades=res.trades,
            asset_bars=asset_bars,
            meta={"hash": h, "logged": False, "source": "playground"},
        )
        if req.log and self.registry is not None:
            self.registry.log_run(spec, res)
            payload["logged"] = True
        with self._lock:
            self._results[h] = payload
            while len(self._results) > self._max_results:
                self._results.popitem(last=False)
        return payload

    def _log(self, spec: RunSpec, bars: pd.DataFrame, payload: dict[str, Any]) -> None:
        assert self.registry is not None
        res = run_backtest(spec, bars)
        self.registry.log_run(spec, res)
        payload["logged"] = True


def dial_dataset_symbols(asset: str) -> tuple[str, ...]:
    return (asset.upper(), CASH, SKEW, *INDEX_SYMBOLS)
