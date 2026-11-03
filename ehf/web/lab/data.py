"""Read-only access to stored results, and the one payload shape every chart uses.

``LabData`` wraps the local results store (SQLite + parquet) and the scorecard
directory. ``run_payload`` turns a run — stored or just computed — into the JSON
the browser draws: equity vs. buy-and-hold, drawdowns, exposure rebuilt from the
trade ledger, candles of the primary asset with trade markers, yearly returns,
and the walk-forward windows when the run is a stitched out-of-sample curve.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ehf.research.backtest.metrics import core_metrics
from ehf.research.config.spec import RunSpec
from ehf.research.data.provider import DataProvider
from ehf.research.data.schema import Timeframe
from ehf.research.paths import SCORECARD_DIR
from ehf.research.results.store import LocalResultStore

SESSION_TZ = "America/New_York"
MAX_MARKERS = 4000
_FULL_RUN_RE = re.compile(r"run `([0-9a-f]{16})`")

PRIMARY_METRICS = ("cagr", "sharpe", "max_drawdown", "calmar", "ann_vol", "total_return")


def _clean(v: Any) -> Any:
    """JSON-safe scalar: NaN/inf -> None, numpy -> python."""
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return session_date(v)
    return v


def session_date(ts: pd.Timestamp) -> str:
    """UTC bar stamp -> the New York calendar date it belongs to (YYYY-MM-DD)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(SESSION_TZ).strftime("%Y-%m-%d")


def session_dates(index: Iterable[pd.Timestamp]) -> list[str]:
    idx = pd.DatetimeIndex(list(index))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return list(idx.tz_convert(SESSION_TZ).strftime("%Y-%m-%d"))


def _floats(s: pd.Series) -> list[float | None]:
    arr = s.to_numpy(dtype=float)
    return [float(x) if math.isfinite(x) else None for x in arr]


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def held_weights(
    trades: pd.DataFrame, index: pd.DatetimeIndex, resets: Iterable[pd.Timestamp] = ()
) -> pd.DataFrame:
    """Rebuild the weight held in every symbol on every bar from the trade ledger.

    ``resets`` mark bars where the book is known to start flat (the first bar of
    each walk-forward test window), so the running sum restarts there.
    """
    if trades.empty:
        return pd.DataFrame(index=index)
    deltas = (
        trades.pivot_table(
            index="timestamp", columns="symbol", values="delta_weight", aggfunc="sum"
        )
        .reindex(index)
        .fillna(0.0)
    )
    reset_pos = sorted({int(index.searchsorted(pd.Timestamp(r))) for r in resets})
    segment = np.zeros(len(index), dtype=int)
    for p in reset_pos:
        if 0 < p < len(index):
            segment[p:] += 1
    return deltas.groupby(segment).cumsum()


def primary_asset(spec: dict[str, Any]) -> str:
    """The symbol whose price the run is charted against (SPY for the dials)."""
    params = spec.get("params") or {}
    if isinstance(params.get("asset"), str):
        return str(params["asset"]).upper()
    uni = spec.get("universe") or {}
    symbols = [s for s in (uni.get("symbols") or ()) if not str(s).startswith("^") and s != "CASH"]
    return str(symbols[0]).upper() if symbols else "SPY"


def yearly_returns(net: pd.Series) -> pd.Series:
    idx = pd.DatetimeIndex(net.index).tz_convert(SESSION_TZ)
    return (1.0 + net).groupby(idx.year).prod() - 1.0


@dataclass
class Scorecard:
    wf_id: str
    strategy: str
    path: Path
    payload: dict[str, Any]

    @property
    def stitched_hash(self) -> str:
        return str(self.payload.get("stitched_hash", ""))

    @property
    def full_hash(self) -> str | None:
        m = _FULL_RUN_RE.search(str(self.payload.get("notes", "")))
        return m.group(1) if m else None

    def summary(self) -> dict[str, Any]:
        m = self.payload.get("oos_metrics") or {}
        v = self.payload.get("verdict") or {}
        wf = self.payload.get("walk_forward") or {}
        extra = self.payload.get("extra") or {}
        windows = self.payload.get("windows") or []
        base = self.payload.get("base_spec") or {}
        return {
            "wf_id": self.wf_id,
            "strategy": self.strategy,
            "stitched_hash": self.stitched_hash,
            "full_hash": self.full_hash,
            "verdict": v.get("status"),
            "failed": v.get("failed") or [],
            "objective": wf.get("objective"),
            "train_months": wf.get("train_months"),
            "test_months": wf.get("test_months"),
            "n_windows": len(windows),
            "start": windows[0]["test_start"] if windows else base.get("start"),
            "end": windows[-1]["test_end"] if windows else base.get("end"),
            "asset": primary_asset(base),
            "metrics": {k: _clean(val) for k, val in m.items()},
            "time_in_market": _clean(extra.get("time_in_market")),
            "switches_per_year": _clean(extra.get("switches_per_year")),
            "dd_2008": _clean(extra.get("dd_2008")),
            "dd_2020": _clean(extra.get("dd_2020")),
            "dd_2022": _clean(extra.get("dd_2022")),
            "modified": self.path.stat().st_mtime,
        }


@dataclass
class LabData:
    """Everything the browser screens read. One instance per app, thread-safe."""

    store: LocalResultStore
    provider: DataProvider | None = None
    scorecard_dir: Path = SCORECARD_DIR
    _bars_cache: dict[tuple[str, str, str], pd.DataFrame] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- scorecards (walk-forward studies) ------------------------------------------

    def scorecards(self) -> list[Scorecard]:
        out: list[Scorecard] = []
        if not self.scorecard_dir.is_dir():
            return out
        for p in sorted(self.scorecard_dir.glob("wf_*.json")):
            try:
                payload = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            wf_id = str(payload.get("wf_id") or p.stem)
            out.append(Scorecard(wf_id, str(payload.get("strategy", "")), p, payload))
        out.sort(key=lambda s: s.path.stat().st_mtime, reverse=True)
        return out

    def scorecard(self, wf_id: str) -> Scorecard | None:
        for s in self.scorecards():
            if s.wf_id == wf_id:
                return s
        return None

    def scorecard_for_hash(self, config_hash: str) -> Scorecard | None:
        for s in self.scorecards():
            if s.stitched_hash == config_hash:
                return s
        return None

    def studies(self) -> list[dict[str, Any]]:
        return [s.summary() for s in self.scorecards()]

    # -- runs ------------------------------------------------------------------------

    def strategies(self) -> list[dict[str, Any]]:
        df = self.store.query("")
        if df.empty:
            return []
        g = df.groupby("strategy").agg(n=("config_hash", "size"), last=("created_at", "max"))
        return [
            {"strategy": k, "n": int(r["n"]), "last": r["last"]}
            for k, r in g.sort_values("n", ascending=False).iterrows()
        ]

    def runs(
        self,
        *,
        strategy: str | None = None,
        standalone: bool = True,
        sweep_id: str | None = None,
        sort: str = "created_at",
        descending: bool = True,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if strategy:
            where.append("strategy = ?")
            params.append(strategy)
        if sweep_id:
            where.append("sweep_id = ?")
            params.append(sweep_id)
        elif standalone:
            where.append("sweep_id IS NULL")
        df = self.store.query("WHERE " + " AND ".join(where) if where else "", tuple(params))
        if df.empty:
            return []
        if sort not in df.columns:
            sort = "created_at"
        df = df.sort_values(sort, ascending=not descending, na_position="last").head(limit)
        rows = []
        for _, r in df.iterrows():
            rows.append(
                {
                    "hash": r["config_hash"],
                    "strategy": r["strategy"],
                    "params": json.loads(r["params_json"]),
                    "universe": r["universe"],
                    "start": r["data_start"],
                    "end": r["data_end"],
                    "sweep_id": r["sweep_id"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "metrics": {c[2:]: _clean(r[c]) for c in df.columns if c.startswith("m_")},
                }
            )
        return rows

    def run(self, config_hash: str) -> dict[str, Any]:
        """Full payload for one stored run (raises KeyError if unknown)."""
        rec = self.store.read_run(config_hash)
        spec = rec["spec"]
        card = self.scorecard_for_hash(config_hash)
        windows = card.payload.get("windows") if card else None
        return run_payload(
            spec=spec,
            metrics=rec["metrics"],
            equity=rec["equity"],
            trades=rec["trades"],
            asset_bars=self.asset_bars(primary_asset(spec), spec["start"], spec["end"]),
            windows=windows,
            meta={
                "hash": config_hash,
                "created_at": rec.get("created_at"),
                "git_sha": rec.get("git_sha"),
                "dirty": bool(rec.get("dirty")),
                "status": rec.get("status"),
                "sweep_id": rec.get("sweep_id"),
                "study": card.summary() if card else None,
            },
        )

    # -- prices ------------------------------------------------------------------------

    def asset_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        """Daily bars for one symbol (cache-first via the provider); None if unavailable."""
        if self.provider is None:
            return None
        key = (symbol.upper(), str(start), str(end))
        with self._lock:
            hit = self._bars_cache.get(key)
        if hit is not None:
            return hit
        try:
            bars = self.provider.get_bars([symbol], start, end, Timeframe.DAY)
        except Exception:
            return None
        bars = bars.loc[bars["symbol"] == symbol.upper()].reset_index(drop=True)
        with self._lock:
            if len(self._bars_cache) > 32:
                self._bars_cache.clear()
            self._bars_cache[key] = bars
        return bars


def benchmark_from_bars(
    bars: pd.DataFrame | None, index: pd.DatetimeIndex, initial_capital: float
) -> pd.Series | None:
    """Buy-and-hold equity of the asset on the run's bars (close-to-close)."""
    if bars is None or bars.empty:
        return None
    close = bars.set_index("timestamp")["close"].sort_index()
    close = close.reindex(index, method="ffill")
    if close.isna().all():
        return None
    first = float(close.dropna().iloc[0])
    out: pd.Series = close / first * initial_capital
    return out


def run_payload(
    *,
    spec: dict[str, Any],
    metrics: dict[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    asset_bars: pd.DataFrame | None,
    windows: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eq = equity.set_index("timestamp").sort_index()
    index = pd.DatetimeIndex(eq.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
        eq.index = index
    asset = primary_asset(spec)
    capital = float(spec.get("initial_capital") or 1_000_000.0)

    resets: list[pd.Timestamp] = []
    if windows:
        for w in windows:
            ts = pd.Timestamp(w["test_start"], tz="UTC")
            pos = int(index.searchsorted(ts))
            if pos < len(index):
                resets.append(index[pos])
    held = held_weights(trades, index, resets)
    gross = held.abs().sum(axis=1) if not held.empty else pd.Series(0.0, index=index)
    asset_w = held[asset] if asset in held.columns else pd.Series(0.0, index=index)

    bench = benchmark_from_bars(asset_bars, index, capital)
    bench_net = bench.pct_change().fillna(0.0) if bench is not None else None
    bench_metrics: dict[str, Any] | None = None
    if bench is not None and bench_net is not None:
        ppy = Timeframe(spec.get("timeframe", "1day")).periods_per_year
        try:
            bm = core_metrics(bench_net, bench, trades.iloc[0:0], ppy)
            bench_metrics = {k: _clean(v) for k, v in bm.items()}
        except Exception:
            bench_metrics = None

    yr = yearly_returns(eq["net_return"])
    yr_b = yearly_returns(bench_net) if bench_net is not None else None
    yearly = [
        {
            "year": int(str(y)),
            "strategy": _clean(v),
            "benchmark": _clean(yr_b.get(y)) if yr_b is not None else None,
        }
        for y, v in yr.items()
    ]

    candles: list[dict[str, Any]] = []
    if asset_bars is not None and not asset_bars.empty:
        b = asset_bars.sort_values("timestamp")
        lo, hi = index[0], index[-1]
        b = b[(b["timestamp"] >= lo) & (b["timestamp"] <= hi)]
        dates = session_dates(b["timestamp"])
        for d, o, h, lw, c, v in zip(
            dates, b["open"], b["high"], b["low"], b["close"], b["volume"], strict=True
        ):
            candles.append(
                {
                    "time": d,
                    "open": float(o),
                    "high": float(h),
                    "low": float(lw),
                    "close": float(c),
                    "volume": float(v) if math.isfinite(float(v)) else 0.0,
                }
            )

    markers: list[dict[str, Any]] = []
    if not trades.empty:
        t = trades.sort_values("timestamp")
        own = t[t["symbol"] == asset] if asset in set(t["symbol"]) else t
        if len(own) > MAX_MARKERS:
            own = own.iloc[:: int(math.ceil(len(own) / MAX_MARKERS))]
        for day, sym, dw, side, px in zip(
            session_dates(own["timestamp"]),
            own["symbol"],
            own["delta_weight"],
            own["side"],
            own["price"],
            strict=True,
        ):
            markers.append(
                {
                    "time": day,
                    "symbol": str(sym),
                    "side": str(side),
                    "delta_weight": _clean(dw),
                    "price": _clean(px),
                }
            )

    win_out = None
    if windows:
        win_out = [
            {
                "window": int(w["window"]) if w.get("window") is not None else i,
                "test_start": w["test_start"],
                "test_end": w["test_end"],
                "params": w.get("params") or {},
                "oos_sharpe": _clean(w.get("oos_sharpe")),
                "oos_cagr": _clean(w.get("oos_cagr")),
                "oos_max_drawdown": _clean(w.get("oos_max_drawdown")),
            }
            for i, w in enumerate(windows)
        ]

    return {
        **(meta or {}),
        "strategy": spec.get("strategy"),
        "params": spec.get("params") or {},
        "universe": spec.get("universe") or {},
        "start": spec.get("start"),
        "end": spec.get("end"),
        "timeframe": spec.get("timeframe", "1day"),
        "initial_capital": capital,
        "costs": spec.get("costs") or {},
        "fill_model": spec.get("fill_model"),
        "asset": asset,
        "metrics": {k: _clean(v) for k, v in metrics.items()},
        "benchmark_metrics": bench_metrics,
        "series": {
            "time": session_dates(index),
            "equity": _floats(eq["equity"]),
            "benchmark": _floats(bench) if bench is not None else None,
            "drawdown": _floats(drawdown(eq["equity"])),
            "benchmark_drawdown": _floats(drawdown(bench)) if bench is not None else None,
            "exposure": _floats(asset_w if asset in held.columns else gross),
            "gross": _floats(gross),
        },
        "candles": candles,
        "markers": markers,
        "n_trades": int(len(trades)),
        "yearly": yearly,
        "windows": win_out,
    }


def spec_dict(spec: RunSpec) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(spec.canonical_json())
    return out
