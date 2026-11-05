"""The grand sweep: every strategy family x every target x a matrix of test settings.

An unattended, resumable queue for a day the Studio would otherwise sit idle.
Each queue item is one full walk-forward study (optimize on train months, trade
the winner on the next test months, stitch) followed by the gauntlet:

    item = (strategy, target, grid, train_months, objective, slippage_bps)

Targets are either a **basket** (a named ETF universe; the strategy picks
weights across it) or a single-asset **dial** (asset vs. a T-bill CASH leg,
driven by its own price or by the VIX family).

Robustness comes from the matrix, not from any one number: the same
(strategy, target) is re-run with 24/36/60-month training, a Sharpe or Calmar
objective, and 5 bp slippage. A rule that only looks good under one setting is
a fitted accident; the leaderboard ranks by the *worst* setting as well as the
headline one, and against the passive benchmark for the same target
(``ew_hold`` for baskets, ``dial_hold`` for dials).

Items run in tiers so a partial run is still useful: tier 0 is every
(strategy, target) at the canonical setting, later tiers are the variations.
Every finished item appends a row to ``results.jsonl`` and the leaderboard is
rewritten, so a crash or a stop loses at most the item in flight; re-running
with the same ``--tag`` skips finished items (keyed by the deterministic wf_id).

Results go to their own registry under ``var/sweeps/<tag>/results`` — NOT the
main ``var/results`` — so the nightly S3 sync and the deployed lab are not
flooded with hundreds of thousands of train-window cells. Promote a winner by
re-running it with the normal study tools.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ehf.research.analysis.gauntlet import GauntletConfig, market_proxy_returns, run_gauntlet
from ehf.research.analysis.scorecard import write_scorecard
from ehf.research.backtest.walkforward import (
    Grid,
    WalkForwardSpec,
    walk_forward,
)
from ehf.research.config.spec import CostConfig, RunSpec, UniverseSpec
from ehf.research.data.provider import DataProvider
from ehf.research.data.schema import Timeframe
from ehf.research.data.universes import STATIC_UNIVERSES
from ehf.research.data.vix import load_vix_bars
from ehf.research.paths import VAR_DIR
from ehf.research.results.registry import ExperimentRegistry
from ehf.research.results.store import LocalResultStore
from ehf.research.strategies.vix_overlay import GRIDS as DIAL_GRIDS

log = logging.getLogger(__name__)

STUDY = "grand_sweep"
SWEEP_DIR = VAR_DIR / "sweeps"

# --- the plan -------------------------------------------------------------------------

BASKETS: tuple[str, ...] = ("etf_asset_classes", "etf_sectors", "etf_core")

BASKET_GRIDS: dict[str, Grid] = {
    "ew_hold": {"rebalance_every": [21]},
    "tsmom": {
        "lookback": [21, 63, 126, 252],
        "target_vol": [0.10, 0.20],
        "vol_window": [21, 63],
        "max_leverage": [1.0, 2.0],
        "rebalance_every": [1, 5, 21],
    },
    "xsmom": {
        "lookback": [63, 126, 252],
        "skip": [0, 21],
        "quantile": [0.1, 0.2, 0.3],
        "long_only": [False, True],
        "rebalance_every": [5, 21],
    },
    "bollinger_reversion": {
        "window": [20, 50, 100, 200],
        "n_std": [1.5, 2.0, 2.5],
        "target_pct": [0.03, 0.06, 0.10],
        "stop_pct": [0.05, 0.10, 0.20],
        "max_hold": [10, 30, 60],
    },
    "rise_fall": {
        "drop_pct": [0.005, 0.01, 0.02],
        "rise_pct": [0.02, 0.04],
        "stop_pct": [0.05, 0.10],
        "atr_mult": [0.5, 1.0],
        "trend_window": [100, 200],
        "max_hold": [10, 20],
    },
    "tsmom_regime": {
        "lookback": [63, 252],
        "rebalance_every": [5, 21],
        "pe_max": [0.95, 0.99],
        "hurst_min": [0.35, 0.45],
        "hurst_window": [100, 150],
    },
}

DIAL_ASSETS: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "TLT", "GLD", "VNQ", "HYG", "XLK", "XLF", "XLE",
)  # fmt: skip

DIAL_RULES: tuple[str, ...] = (
    "dial_hold", "dial_sma", "vix_term", "vix_vote", "vix_sized", "vix_spike", "vix_resid",
)  # fmt: skip

BENCHMARK = {"basket": "ew_hold", "dial": "dial_hold"}

# --- the alternative-data family (FMP fundamentals, analysts, insiders, Congress, macro) ----

ALT_PAIRS: tuple[tuple[str, str], ...] = (
    ("pead", "sp500_pead"),
    ("grades_momo", "sp500_grades"),
    ("fscore_value", "sp500_fund"),
    ("shareholder_yield", "sp500_fund"),
    ("insider_cluster", "sp500_insider"),
    ("congress_follow", "sp500_congress"),
    ("curve_dial", "macro_SPY"),
    ("curve_dial", "macro_QQQ"),
    ("curve_dial", "macro_IWM"),
    ("claims_dial", "macro_SPY"),
    ("claims_dial", "macro_QQQ"),
    ("claims_dial", "macro_IWM"),
    ("cugo_rotation", "cugo_sectors"),
    ("dollar_em", "dollar_em"),
)

ALT_GRIDS: dict[str, Grid] = {
    "member_ew": {"rebalance_every": [21]},
    "pead": {
        "min_surprise": [0.02, 0.05, 0.10, 0.20],
        "hold": [10, 20, 40, 60],
        "rebalance_every": [1, 5],
    },
    "grades_momo": {"window": [21, 63, 126], "min_net": [1, 2, 3], "rebalance_every": [5, 21]},
    "fscore_value": {
        "min_fscore": [5, 6, 7, 8],
        "quantile": [0.1, 0.2, 0.3, 0.5],
        "rebalance_every": [21, 63],
    },
    "shareholder_yield": {
        "quantile": [0.1, 0.2, 0.3],
        "min_fscore": [0, 5],
        "rebalance_every": [21, 63],
    },
    "insider_cluster": {
        "window": [10, 21, 42],
        "min_buyers": [1, 2, 3],
        "hold": [21, 63, 126],
        "rebalance_every": [5],
    },
    "congress_follow": {
        "window": [21, 63, 126, 252],
        "min_net": [1, 2, 3],
        "rebalance_every": [5, 21],
    },
    "curve_dial": {
        "threshold": [-0.5, 0.0, 0.5],
        "mode": ["inverted", "after", "both"],
        "memory": [126, 252, 504],
        "sma_confirm": [False, True],
    },
    "claims_dial": {
        "lookback": [126, 252],
        "threshold": [0.10, 0.15, 0.20, 0.30],
        "exit_frac": [0.5, 1.0],
        "sma_confirm": [False, True],
    },
    "cugo_rotation": {
        "lookback": [21, 63, 126, 252],
        "mode": ["mom", "sma"],
        "rebalance_every": [5, 21],
    },
    "dollar_em": {
        "window": [50, 100, 200],
        "band": [0.0, 0.01, 0.02],
        "foreign": ["EEM", "EFA", "both"],
        "rebalance_every": [5, 21],
    },
    "ew_hold": {"rebalance_every": [21]},
    "dial_hold": {"min_dwell": [1]},
}


def benchmark_of(family: str, target: str) -> str:
    if family != "alt":
        return BENCHMARK[family]
    if target.startswith("sp500_"):
        return "member_ew"
    return "dial_hold" if target.startswith("macro_") else "ew_hold"


def alt_start(target: str) -> str:
    if target == "sp500_congress":
        return "2015-01-01"
    return "2012-01-01" if target.startswith("sp500_") else "2006-01-01"


CANONICAL = {"train_months": 36, "objective": "sharpe", "slippage_bps": 1.0}
VARIANTS: tuple[dict[str, Any], ...] = (
    {"train_months": 24, "objective": "sharpe", "slippage_bps": 1.0},
    {"train_months": 60, "objective": "sharpe", "slippage_bps": 1.0},
    {"train_months": 36, "objective": "calmar", "slippage_bps": 1.0},
    {"train_months": 36, "objective": "sharpe", "slippage_bps": 5.0},
    {"train_months": 60, "objective": "calmar", "slippage_bps": 1.0},
)

SPAN = {
    "basket": {"data_start": "2003-01-01", "start": "2006-01-01"},
    "dial": {"data_start": "2004-01-01", "start": "2006-08-01"},
    "alt": {"data_start": "2003-01-01", "start": "2006-01-01"},  # start: see alt_start()
}


@dataclass(frozen=True)
class Item:
    family: str  # "basket" | "dial"
    strategy: str
    target: str  # universe name (basket) or asset symbol (dial)
    train_months: int
    objective: str
    slippage_bps: float
    tier: int

    @property
    def key(self) -> str:
        return (
            f"{self.strategy}|{self.target}|tr{self.train_months}|{self.objective}"
            f"|{self.slippage_bps:g}bp"
        )

    @property
    def is_canonical(self) -> bool:
        return (
            self.train_months == CANONICAL["train_months"]
            and self.objective == CANONICAL["objective"]
            and self.slippage_bps == CANONICAL["slippage_bps"]
        )

    def grid(self) -> Grid:
        if self.family == "basket":
            return BASKET_GRIDS[self.strategy]
        if self.family == "alt":
            return ALT_GRIDS[self.strategy]
        return DIAL_GRIDS[self.strategy]


def build_plan(
    *,
    strategies: Iterable[str] | None = None,
    targets: Iterable[str] | None = None,
    families: Iterable[str] = ("basket", "dial"),
    max_tier: int = 99,
) -> list[Item]:
    """Every item, tier-ordered (canonical first, then variants; benchmarks lead each tier)."""
    want_s = set(strategies) if strategies else None
    want_t = set(targets) if targets else None
    settings = [CANONICAL, *VARIANTS]
    items: list[Item] = []
    pairs: list[tuple[str, str, str]] = []
    if "basket" in families:
        pairs += [("basket", s, u) for s in BASKET_GRIDS for u in BASKETS]
    if "dial" in families:
        pairs += [("dial", s, a) for s in DIAL_RULES for a in DIAL_ASSETS]
    if "alt" in families:
        targets_alt = list(dict.fromkeys(t for _, t in ALT_PAIRS))
        pairs += [("alt", benchmark_of("alt", t), t) for t in targets_alt]
        pairs += [("alt", s, t) for s, t in ALT_PAIRS]
    for tier, setting in enumerate(settings):
        for fam, strat, tgt in pairs:
            bench = benchmark_of(fam, tgt)
            if want_s and strat not in want_s and strat != bench:
                continue
            if want_t and tgt not in want_t:
                continue
            # benchmarks have a one-cell grid: the objective/train variants are identical
            # runs, so they only need the canonical setting and the cost stress
            if strat == bench and setting["objective"] != "sharpe":
                continue
            items.append(Item(fam, strat, tgt, tier=tier, **setting))
    items = [i for i in items if i.tier <= max_tier]
    # within a tier: benchmarks first, then the cheapest families, so early results are broad
    order = {"dial": 0, "basket": 1, "alt": 2}
    items.sort(
        key=lambda i: (i.tier, i.strategy != benchmark_of(i.family, i.target), order[i.family])
    )
    return items


# --- data -----------------------------------------------------------------------------


class DataBook:
    """Loads each target's bars once (from the price cache) and keeps them."""

    def __init__(self, provider: DataProvider, end: str):
        self.provider = provider
        self.end = end
        self._bars: dict[tuple[str, str], pd.DataFrame] = {}
        self._market: pd.Series | None = None

    def bars(self, item: Item) -> pd.DataFrame:
        k = (item.family, item.target)
        if k not in self._bars:
            span = SPAN[item.family]
            if item.family == "alt":
                from ehf.research.data.altdata import load_dataset

                b = load_dataset(item.target, self.provider, self.end)
            elif item.family == "basket":
                syms = list(STATIC_UNIVERSES[item.target])
                b = self.provider.get_bars(syms, span["data_start"], self.end, Timeframe.DAY)
            else:
                b = load_vix_bars(
                    self.provider,
                    span["data_start"],
                    self.end,
                    asset=item.target,
                    include_skew=False,
                )
            log.info(
                "loaded %s %s: %d symbols x %d sessions",
                item.family,
                item.target,
                b["symbol"].nunique(),
                b["timestamp"].nunique(),
            )
            self._bars[k] = b
        return self._bars[k]

    def market(self) -> pd.Series | None:
        if self._market is None:
            spy = self.provider.get_bars(["SPY"], "2003-01-01", self.end, Timeframe.DAY)
            self._market = market_proxy_returns(spy, "SPY")
        return self._market


# --- one item -------------------------------------------------------------------------


def item_spec(item: Item, bars: pd.DataFrame, end: str) -> tuple[RunSpec, WalkForwardSpec]:
    params: dict[str, Any] = {}
    start = SPAN[item.family]["start"]
    if item.family == "dial":
        params = {"asset": item.target, "cash": "CASH"}
        symbols = tuple(sorted(bars["symbol"].unique()))
    elif item.family == "alt":
        symbols = tuple(sorted(bars["symbol"].astype(str).unique()))
        start = alt_start(item.target)
        if item.target.startswith("macro_"):
            params = {"asset": item.target.split("_", 1)[1], "cash": "CASH"}
    else:
        symbols = tuple(STATIC_UNIVERSES[item.target])
    spec = RunSpec(
        strategy=item.strategy,
        params=params,
        universe=UniverseSpec(symbols=symbols),
        start=start,
        end=end,
        costs=CostConfig(slippage_bps=item.slippage_bps),
    )
    wf = WalkForwardSpec(
        train_months=item.train_months,
        test_months=12,
        objective=item.objective,
        warmup_bars=252,
        min_trades=2 if (item.family == "dial" or item.target.startswith("macro_")) else 10,
    )
    return spec, wf


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def run_item(
    item: Item,
    book: DataBook,
    *,
    end: str,
    registry: ExperimentRegistry,
    out_dir: Path,
    n_workers: int | None,
) -> dict[str, Any]:
    t0 = time.time()
    bars = book.bars(item)
    spec, wf_spec = item_spec(item, bars, end)
    grid = item.grid()
    wf = walk_forward(spec, grid, bars, wf_spec, registry=registry, n_workers=n_workers)
    verdict = run_gauntlet(
        wf, registry=registry, market_returns=book.market(), cfg=GauntletConfig()
    )
    notes = f"Study `{STUDY}` item `{item.key}` (tier {item.tier})."
    write_scorecard(wf, verdict, notes=notes, out_dir=out_dir / "scorecards", study=STUDY)
    m = wf.oos_metrics
    checks = {c.name: c.value for c in verdict.checks}
    years = max(_f(m.get("n_bars")) or 0.0, 1.0) / 252.0
    params = wf.windows["params"].tolist()
    return {
        **asdict(item),
        "key": item.key,
        "wf_id": wf.wf_id,
        "stitched_hash": wf.stitched_hash,
        "verdict": verdict.status,
        "failed": verdict.failed,
        "n_cells": math.prod(len(v) for v in grid.values()),
        "n_windows": wf.n_windows,
        "oos_start": str(wf.windows["test_start"].iloc[0]),
        "oos_end": str(wf.windows["test_end"].iloc[-1]),
        "sharpe": _f(m.get("sharpe")),
        "cagr": _f(m.get("cagr")),
        "max_dd": _f(m.get("max_drawdown")),
        "calmar": _f(m.get("calmar")),
        "ann_vol": _f(m.get("ann_vol")),
        "sharpe_x2": _f(m.get("sharpe_cost_x2")),
        "wf_eff": _f(m.get("wf_efficiency")),
        "win_pos": _f(m.get("share_windows_positive")),
        "trades_per_year": (_f(m.get("n_trades")) or 0.0) / years,
        "dsr": _f(checks.get("dsr")),
        "pbo": _f(checks.get("pbo")),
        "last_params": params[-1] if params else None,
        "params_stable": len({json.dumps(p, sort_keys=True) for p in params}) if params else 0,
        "seconds": round(time.time() - t0, 1),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# --- the queue ------------------------------------------------------------------------


@dataclass
class SweepState:
    tag: str
    root: Path
    shard: int = 0
    done: dict[str, dict[str, Any]] = field(default_factory=dict)  # key -> row
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def rows_path(self) -> Path:
        return self.root / f"results_{self.shard}.jsonl"

    @property
    def failures_path(self) -> Path:
        return self.root / f"failures_{self.shard}.json"

    @property
    def registry_root(self) -> Path:
        return self.root / f"registry_{self.shard}"

    @classmethod
    def load(cls, tag: str, shard: int = 0) -> SweepState:
        """Every shard's rows (so each shard skips work any shard already did)."""
        root = SWEEP_DIR / tag
        root.mkdir(parents=True, exist_ok=True)
        st = cls(tag=tag, root=root, shard=shard)
        for path in sorted(root.glob("results*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    row.setdefault("registry_root", str(root / "results"))
                    st.done[row["key"]] = row
        for path in sorted(root.glob("failures*.json")):
            st.failures.update(json.loads(path.read_text()))
        return st

    def record(self, row: dict[str, Any]) -> None:
        row["registry_root"] = str(self.registry_root)
        self.done[row["key"]] = row
        self.failures.pop(row["key"], None)
        with self.rows_path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        self.failures_path.write_text(json.dumps(self.failures, indent=1))

    def fail(self, key: str, err: str) -> None:
        self.failures[key] = err
        self.failures_path.write_text(json.dumps(self.failures, indent=1))


def run_sweep_queue(
    provider: DataProvider,
    *,
    tag: str,
    end: str,
    plan: list[Item],
    hours: float | None = None,
    n_workers: int | None = None,
    retry_failures: bool = False,
    shard: int = 0,
    n_shards: int = 1,
    shard_base: int = 0,
    on_progress: Callable[[SweepState, int, int], None] | None = None,
) -> SweepState:
    state = SweepState.load(tag, shard_base + shard)
    registry = ExperimentRegistry(store=LocalResultStore(root=state.registry_root))
    book = DataBook(provider, end)
    deadline = datetime.now(UTC) + timedelta(hours=hours) if hours else None
    todo = [
        i
        for i in plan
        if i.key not in state.done and (retry_failures or i.key not in state.failures)
    ][shard::n_shards]
    (state.root / f"plan_{shard}.json").write_text(
        json.dumps([{**asdict(i), "key": i.key} for i in plan], indent=1)
    )
    log.info(
        "sweep %s shard %d/%d: %d items planned, %d done, %d to run",
        tag,
        shard,
        n_shards,
        len(plan),
        len(state.done),
        len(todo),
    )
    t_start = time.time()
    for n, item in enumerate(todo, 1):
        if deadline and datetime.now(UTC) >= deadline:
            log.warning("time budget reached; stopping with %d items left", len(todo) - n + 1)
            break
        try:
            row = run_item(
                item, book, end=end, registry=registry, out_dir=state.root, n_workers=n_workers
            )
            state.record(row)
            el = time.time() - t_start
            log.info(
                "[%d/%d] %-48s %-8s sharpe=%s cagr=%s dd=%s (%.0fs; eta %.1fh)",
                n,
                len(todo),
                item.key,
                row["verdict"],
                _fmt(row["sharpe"]),
                _pct(row["cagr"]),
                _pct(row["max_dd"]),
                row["seconds"],
                el / n * (len(todo) - n) / 3600,
            )
        except Exception as exc:  # one broken item must not sink the day
            log.exception("%s failed", item.key)
            state.fail(item.key, f"{type(exc).__name__}: {exc}")
        write_leaderboard(state, plan)
        if on_progress:
            on_progress(state, n, len(todo))
    write_leaderboard(state, plan)
    return state


# --- the leaderboard ------------------------------------------------------------------


def _fmt(v: Any, nd: int = 2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def _pct(v: Any) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.1%}"


def robustness_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """One line per (strategy, target): headline, worst and median OOS Sharpe across settings,
    and how often it beat its target's passive benchmark under the same setting."""
    bench: dict[tuple[str, str, int, float], dict[str, Any]] = {}
    for r in rows:
        if r["strategy"] == benchmark_of(r["family"], r["target"]):
            # benchmark has one cell: train length only moves the OOS start; key on it
            bench[(r["family"], r["target"], r["train_months"], r["slippage_bps"])] = r
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["strategy"], r["target"])].append(r)
    out = []
    for (fam, strat, tgt), rs in groups.items():
        sharpes = [r["sharpe"] for r in rs if r["sharpe"] is not None]
        if not sharpes:
            continue
        canon = next(
            (
                r
                for r in rs
                if Item(
                    **{
                        k: r[k]
                        for k in (
                            "family",
                            "strategy",
                            "target",
                            "train_months",
                            "objective",
                            "slippage_bps",
                            "tier",
                        )
                    }
                ).is_canonical
            ),
            None,
        )
        beats, deltas = [], []
        for r in rs:
            b = bench.get((fam, tgt, r["train_months"], r["slippage_bps"]))
            if b and b["sharpe"] is not None and r["sharpe"] is not None:
                beats.append(r["sharpe"] > b["sharpe"])
                deltas.append(r["sharpe"] - b["sharpe"])
        canon_b = (
            bench.get((fam, tgt, canon["train_months"], canon["slippage_bps"])) if canon else None
        )
        out.append(
            {
                "family": fam,
                "strategy": strat,
                "target": tgt,
                "n_settings": len(rs),
                "sharpe_canon": canon["sharpe"] if canon else None,
                "cagr_canon": canon["cagr"] if canon else None,
                "max_dd_canon": canon["max_dd"] if canon else None,
                "bench_sharpe": canon_b["sharpe"] if canon_b else None,
                "bench_cagr": canon_b["cagr"] if canon_b else None,
                "bench_max_dd": canon_b["max_dd"] if canon_b else None,
                "sharpe_min": min(sharpes),
                "sharpe_med": statistics.median(sharpes),
                "beat_bench": (sum(beats) / len(beats)) if beats else None,
                "sharpe_vs_bench_min": min(deltas) if deltas else None,
                "promoted": sum(r["verdict"] == "promoted" for r in rs),
                "trades_per_year": canon["trades_per_year"] if canon else None,
            }
        )
    return pd.DataFrame(out)


def write_leaderboard(state: SweepState, plan: list[Item]) -> Path:
    rows = list(state.done.values())
    md = state.root / "leaderboard.md"
    lines = [
        f"# Grand sweep `{state.tag}`",
        "",
        f"Updated {datetime.now(UTC).isoformat(timespec='seconds')}. "
        f"{len(rows)} of {len(plan)} walk-forward studies finished, {len(state.failures)} failed. "
        f"Every number is out-of-sample (stitched test windows). Canonical setting: "
        f"{CANONICAL['train_months']}-month train / 12-month test, Sharpe objective, "
        f"{CANONICAL['slippage_bps']:g} bp slippage. Variants: "
        + "; ".join(
            f"{v['train_months']}m/{v['objective']}/{v['slippage_bps']:g}bp" for v in VARIANTS
        )
        + ".",
        "",
    ]
    if not rows:
        md.write_text("\n".join(lines + ["Nothing finished yet."]) + "\n")
        return md
    rt = robustness_table(rows)
    rt.to_csv(state.root / "robustness.csv", index=False)
    pd.DataFrame(rows).to_csv(state.root / "results.csv", index=False)
    head = (
        "| strategy | target | settings | Sharpe | CAGR | max DD | bench Sharpe | bench CAGR "
        "| bench DD | worst Sharpe | median Sharpe | beat bench | worst vs bench | trades/yr "
        "| promoted |"
    )
    sep = "|" + "---|" * 15

    def table(df: pd.DataFrame) -> list[str]:
        out = [head, sep]
        for _, r in df.iterrows():
            out.append(
                "| "
                + " | ".join(
                    [
                        str(r["strategy"]),
                        str(r["target"]),
                        str(r["n_settings"]),
                        _fmt(r["sharpe_canon"]),
                        _pct(r["cagr_canon"]),
                        _pct(r["max_dd_canon"]),
                        _fmt(r["bench_sharpe"]),
                        _pct(r["bench_cagr"]),
                        _pct(r["bench_max_dd"]),
                        _fmt(r["sharpe_min"]),
                        _fmt(r["sharpe_med"]),
                        _pct(r["beat_bench"]),
                        _fmt(r["sharpe_vs_bench_min"]),
                        _fmt(r["trades_per_year"], 0),
                        str(r["promoted"]),
                    ]
                )
                + " |"
            )
        return out

    benches = {"ew_hold", "dial_hold", "member_ew"}
    nonbench = rt[~rt["strategy"].isin(benches)]
    robust = nonbench[nonbench["n_settings"] >= 3].sort_values("sharpe_min", ascending=False)
    lines += [
        "## Most robust (ranked by the worst Sharpe across settings; needs 3+ settings)",
        "",
        *table(robust.head(40)),
        "",
        "## Best headline (canonical setting, ranked by Sharpe)",
        "",
        *table(nonbench.sort_values("sharpe_canon", ascending=False).head(40)),
        "",
        "## Benchmarks (passive: equal-weight basket or index members, or always-in for a dial)",
        "",
        *table(rt[rt["strategy"].isin(benches)].sort_values("target")),
        "",
    ]
    promoted = [r for r in rows if r["verdict"] == "promoted"]
    lines += ["## Gauntlet survivors", ""]
    if promoted:
        lines += [
            f"- `{r['key']}` sharpe {_fmt(r['sharpe'])}, CAGR {_pct(r['cagr'])}, "
            f"DD {_pct(r['max_dd'])}, wf `{r['wf_id']}`"
            for r in promoted
        ]
    else:
        lines += ["None yet. Single-asset dials fail `dsr`/`pbo` structurally (see VIX study)."]
    kills: dict[str, int] = defaultdict(int)
    for r in rows:
        for f in r["failed"]:
            kills[f] += 1
    lines += ["", "## Why things were killed (count of failed checks)", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(kills.items(), key=lambda kv: -kv[1])]
    if state.failures:
        lines += ["", "## Errors", ""] + [f"- `{k}`: {v}" for k, v in state.failures.items()]
    _atomic_write(md, "\n".join(lines) + "\n")
    return md


def _atomic_write(path: Path, text: str) -> None:
    import os

    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


# --- export to the main registry (what the website's lab reads) ----------------------


def export_to_main(tag: str, *, scorecard_dir: Path | None = None) -> dict[str, int]:
    """Copy each finished study's stitched OOS run into the main registry and its
    scorecard into ``var/scorecards``: one run per study, never the train cells.

    Idempotent (``log_run`` skips hashes it already has; cards are overwritten)."""
    import shutil

    from ehf.research.paths import SCORECARD_DIR
    from ehf.research.results.registry import RunResult

    out_dir = scorecard_dir or SCORECARD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    state = SweepState.load(tag)
    main = ExperimentRegistry(store=LocalResultStore())
    sources: dict[str, ExperimentRegistry] = {}
    copied = cards = missing = 0
    for row in state.done.values():
        root = str(row["registry_root"])
        src = sources.get(root) or ExperimentRegistry(store=LocalResultStore(root=root))
        sources[root] = src
        h = str(row["stitched_hash"])
        if not main.store.has_run(h):
            run = src.get_run(h)
            spec = RunSpec.model_validate(run["spec"])
            if spec.config_hash != h:
                raise RuntimeError(f"hash drift on {row['key']}")
            main.log_run(
                spec,
                RunResult(
                    metrics=run["metrics"],
                    equity=run["equity"],
                    trades=run["trades"],
                    status="walk_forward",
                ),
            )
            copied += 1
        card = state.root / "scorecards" / f"{row['wf_id']}.json"
        if card.exists():
            shutil.copy2(card, out_dir / card.name)
            shutil.copy2(card.with_suffix(".md"), out_dir / card.with_suffix(".md").name)
            cards += 1
        else:
            missing += 1
    return {"runs_copied": copied, "scorecards": cards, "missing_cards": missing}
