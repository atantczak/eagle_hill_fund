"""The leverage study: hold the 3x fund when the market turns bullish, the plain fund otherwise?

Dad's question (ideas box, 2026-11-03): when the market turns bullish, what does
better, TQQQ or QQQ, and how long should you hold the 3x fund? Two answers are
produced for every pair in ``data.leverage.PAIRS``:

  1. **The event study** (model-free). Each rule's "turned bullish" bars, on its
     default settings, and what the leveraged and plain funds did over the next
     week to two years, against the base rate of any random day. This is the
     "how long to hold it" table.
  2. **The switch as a strategy.** Every ``lev_*`` rule walk-forward tested
     (optimise on train months, trade the next test months, stitched), through
     the gauntlet, with dial metrics (share of time leveraged, switches, crisis
     drawdowns), plus a full-sample default-param run. ``lev_hold`` is the "just
     hold TQQQ" record; a ``dial_hold`` run on the plain fund is "just hold QQQ".

Two variants per pair: **extended** (``TQQQ.X``: the real fund spliced onto a
synthetic rebuild, so the record starts in 1999/2000 and includes the dot-com
bust and 2008) and **real** (the fund's own history, 2009/2010 onward). The
headline numbers use the extended series; the real one is the check.

Everything is logged to the registry under deterministic ids and every scorecard
carries ``study = "leverage"`` so the site can tie each run to the write-up.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ehf.research.analysis.dial import dial_metrics, positions_from_trades
from ehf.research.analysis.events import event_study, excess_path
from ehf.research.analysis.gauntlet import (
    GauntletConfig,
    Verdict,
    market_proxy_returns,
    run_gauntlet,
)
from ehf.research.analysis.scorecard import write_scorecard
from ehf.research.backtest.engine import run_backtest
from ehf.research.backtest.walkforward import Grid, WalkForwardResult, WalkForwardSpec, walk_forward
from ehf.research.config.spec import CostConfig, RunSpec, UniverseSpec
from ehf.research.data.leverage import PAIRS, Pair, pair_universe, select_symbols
from ehf.research.data.vix import CASH
from ehf.research.paths import SCORECARD_DIR
from ehf.research.results.registry import ExperimentRegistry
from ehf.research.strategies.leverage import GRIDS

log = logging.getLogger(__name__)

STUDY = "leverage"
Loader = Callable[[Pair], tuple[pd.DataFrame, dict[str, Any]]]


@dataclass
class Outcome:
    pair: str
    extended: bool
    strategy: str
    wf: WalkForwardResult
    verdict: Verdict
    oos_dial: dict[str, Any]
    full_hash: str
    full_metrics: dict[str, float]
    full_dial: dict[str, Any]
    events: dict[str, Any]
    seconds: float
    scorecard: Path
    excess: pd.DataFrame | None = None

    @property
    def lev_symbol(self) -> str:
        return str(self.wf.base_spec.params["asset"])


@dataclass
class StudyResult:
    objective: str
    pairs: dict[str, dict[str, Any]] = field(default_factory=dict)  # pair key -> facts
    holds: dict[str, dict[str, Any]] = field(default_factory=dict)  # pair key -> hold-base run
    outcomes: list[Outcome] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    summary_md: Path | None = None
    summary_json: Path | None = None

    def table(self) -> pd.DataFrame:
        rows = []
        for o in self.outcomes:
            m, d, f, fd = o.wf.oos_metrics, o.oos_dial, o.full_metrics, o.full_dial
            best = o.events.get("best_h")
            best_row = next((r for r in o.events.get("rows", []) if r["h"] == best), None)
            rows.append(
                {
                    "pair": o.pair,
                    "variant": "extended" if o.extended else "real",
                    "strategy": o.strategy,
                    "verdict": o.verdict.status,
                    "failed": ",".join(o.verdict.failed),
                    "oos_sharpe": m.get("sharpe"),
                    "oos_cagr": m.get("cagr"),
                    "oos_max_dd": m.get("max_drawdown"),
                    "oos_sharpe_x2": m.get("sharpe_cost_x2"),
                    "win_pos": m.get("share_windows_positive"),
                    "time_lev": d.get("time_in_market"),
                    "switch_py": d.get("switches_per_year"),
                    "mkt_cagr": d.get("mkt_cagr"),
                    "mkt_max_dd": d.get("mkt_max_drawdown"),
                    "oos_dd_2008": d.get("dd_2008"),
                    "oos_dd_2020": d.get("dd_2020"),
                    "oos_dd_2022": d.get("dd_2022"),
                    "full_cagr": f.get("cagr"),
                    "full_max_dd": f.get("max_drawdown"),
                    "full_dd_2008": fd.get("dd_2008"),
                    "n_events": o.events.get("n_events"),
                    "best_h": best,
                    "best_excess": best_row["excess_median"] if best_row else None,
                    "best_hit": best_row["hit_rate"] if best_row else None,
                }
            )
        return pd.DataFrame(rows)


# --- specs ------------------------------------------------------------------------------


def base_spec(
    strategy: str, pair: Pair, *, extended: bool, start: str, end: str, slippage_bps: float = 1.0
) -> RunSpec:
    lev = pair.extended if extended else pair.lev
    params: dict[str, Any] = {"asset": lev, "base": pair.base, "benchmark": pair.base}
    if strategy == "lev_voltarget":
        params["leverage"] = pair.leverage
    return RunSpec(
        strategy=strategy,
        params=params,
        universe=UniverseSpec(symbols=pair_universe(pair, extended=extended)),
        start=start,
        end=end,
        costs=CostConfig(slippage_bps=slippage_bps),
    )


def hold_base_spec(pair: Pair, *, extended: bool, start: str, end: str) -> RunSpec:
    """Buy and hold the plain fund, logged like any other run so the lab can show it."""
    return RunSpec(
        strategy="dial_hold",
        params={"asset": pair.base, "cash": CASH},
        universe=UniverseSpec(symbols=pair_universe(pair, extended=extended)),
        start=start,
        end=end,
        costs=CostConfig(slippage_bps=1.0),
    )


def quick_grid(grid: Grid) -> Grid:
    return {k: ([v[0], v[-1]] if len(v) > 1 else list(v)) for k, v in grid.items()}


def _add_months(day: str, months: int) -> str:
    return (pd.Timestamp(day) + pd.DateOffset(months=months)).date().isoformat()


def variant_start(facts: dict[str, Any], *, extended: bool, start: str) -> str:
    """Where the evaluation window opens: a year after the plain fund's first bar for the
    extended series (signal warm-up), a month after the leveraged fund's for the real one."""
    if extended:
        return max(start, _add_months(str(facts["base_start"]), 12))
    return max(start, _add_months(str(facts["lev_start"]), 1))


# --- one rule on one pair ----------------------------------------------------------------


def _opens(bars: pd.DataFrame, symbol: str) -> pd.Series:
    s = bars.loc[bars["symbol"] == symbol].set_index("timestamp")["open"].astype(float)
    return s.sort_index()


def run_rule(
    strategy: str,
    pair: Pair,
    bars: pd.DataFrame,
    *,
    extended: bool,
    start: str,
    end: str,
    wf_spec: WalkForwardSpec,
    registry: ExperimentRegistry,
    grid: Grid | None = None,
    gauntlet: GauntletConfig | None = None,
    n_workers: int | None = None,
    out_dir: Path = SCORECARD_DIR,
    with_path: bool = False,
) -> Outcome:
    t0 = time.time()
    grid = grid if grid is not None else GRIDS[strategy]
    spec = base_spec(strategy, pair, extended=extended, start=start, end=end)
    lev = str(spec.params["asset"])
    market = market_proxy_returns(bars, pair.base)

    wf = walk_forward(spec, grid, bars, wf_spec, registry=registry, n_workers=n_workers)
    verdict = run_gauntlet(wf, registry=registry, market_returns=market, cfg=gauntlet)

    stitched = registry.get_run(wf.stitched_hash)
    eq = stitched["equity"].set_index("timestamp")
    idx = pd.DatetimeIndex(eq.index)
    resets = [pd.Timestamp(d, tz="UTC") for d in wf.windows["test_start"]]
    resets = [idx[int(idx.searchsorted(r))] for r in resets if int(idx.searchsorted(r)) < len(idx)]
    oos_dial = dial_metrics(
        eq["net_return"],
        eq["equity"],
        stitched["trades"],
        asset=lev,
        market_net=market,
        resets=resets,
    )

    full_res = run_backtest(spec, bars)
    full_hash = registry.log_run(spec, full_res)
    feq = full_res.equity.set_index("timestamp")
    fidx = pd.DatetimeIndex(feq.index)
    full_dial = dial_metrics(
        feq["net_return"], feq["equity"], full_res.trades, asset=lev, market_net=market
    )
    exposure = positions_from_trades(full_res.trades, fidx, lev)
    lev_open = _opens(bars, lev).reindex(fidx)
    base_open = _opens(bars, pair.base).reindex(fidx)
    events = event_study(exposure, lev_open, base_open)
    path = excess_path(exposure, lev_open, base_open, max_h=252, step=1) if with_path else None

    best = events.get("best_h")
    best_row = next((r for r in events["rows"] if r["h"] == best), None)
    history = "extended history" if extended else "the fund's own history"
    notes = (
        f"Study `{STUDY}`: {pair.name} — {lev} when the rule says the market turned bullish, "
        f"{pair.base} otherwise ({history}). "
        f"Full-sample default-param run `{full_hash}` (in-sample, descriptive): "
        f"CAGR {full_res.metrics['cagr']:.1%}, max DD {full_res.metrics['max_drawdown']:.1%}; "
        f"{pair.base} held: CAGR {_pct(full_dial.get('mkt_cagr'))}, "
        f"max DD {_pct(full_dial.get('mkt_max_drawdown'))}. "
        f"{events['n_events']} bull calls; "
        + (
            f"best hold {best_row['label']}: median {lev} {_pct(best_row['lev_median'])} vs "
            f"{pair.base} {_pct(best_row['base_median'])}, ahead "
            f"{_pct(best_row['hit_rate'], 0)} of the time."
            if best_row
            else "no horizon where the leveraged fund was ahead more often than not."
        )
    )
    md, _ = write_scorecard(
        wf,
        verdict,
        extra=oos_dial,
        notes=notes,
        out_dir=out_dir,
        study=STUDY,
        facts={"pair": pair.key, "extended": extended, "events": events, "full_hash": full_hash},
    )
    secs = time.time() - t0
    log.info(
        "%-9s %-8s %-13s %-8s oos_sharpe=%.2f cagr=%.1f%% maxdd=%.1f%% lev_time=%.0f%% "
        "sw/yr=%.1f events=%d best=%s (%.0fs)",
        pair.key,
        "extended" if extended else "real",
        strategy,
        verdict.status,
        wf.oos_sharpe,
        wf.oos_metrics.get("cagr", float("nan")) * 100,
        wf.oos_metrics.get("max_drawdown", float("nan")) * 100,
        oos_dial["time_in_market"] * 100,
        oos_dial["switches_per_year"],
        events["n_events"],
        best,
        secs,
    )
    return Outcome(
        pair=pair.key,
        extended=extended,
        strategy=strategy,
        wf=wf,
        verdict=verdict,
        oos_dial=oos_dial,
        full_hash=full_hash,
        full_metrics=full_res.metrics,
        full_dial=full_dial,
        events=events,
        seconds=secs,
        scorecard=md,
        excess=path,
    )


# --- the study ---------------------------------------------------------------------------


def run_study(
    load: Loader,
    *,
    pairs: tuple[str, ...] = tuple(PAIRS),
    start: str = "2000-03-01",
    end: str,
    variants: tuple[bool, ...] = (True, False),  # extended, real
    strategies: tuple[str, ...] | None = None,
    train_months: int = 24,
    test_months: int = 12,
    objective: str = "sharpe",
    quick: bool = False,
    n_workers: int | None = None,
    registry: ExperimentRegistry | None = None,
    out_dir: Path = SCORECARD_DIR,
    tag: str | None = None,
    path_pairs: tuple[str, ...] = ("QQQ/TQQQ",),
) -> StudyResult:
    """``load(pair) -> (bars, facts)`` supplies each pair's dataset
    (``data.leverage.load_pair_bars`` in the job; a synthetic frame in tests)."""
    registry = registry or ExperimentRegistry()
    wf_spec = WalkForwardSpec(
        train_months=train_months,
        test_months=test_months,
        objective=objective,
        warmup_bars=252,
        min_trades=2,
    )
    keys = strategies or tuple(GRIDS)
    result = StudyResult(objective=objective)
    for pkey in pairs:
        pair = PAIRS[pkey]
        try:
            all_bars, facts = load(pair)
        except Exception as exc:
            log.exception("%s: dataset failed", pkey)
            result.failures[pkey] = f"{type(exc).__name__}: {exc}"
            continue
        result.pairs[pkey] = facts
        for extended in variants:
            if extended and "extended" not in facts:
                continue
            uni = pair_universe(pair, extended=extended)
            bars = select_symbols(all_bars, uni)
            v_start = variant_start(facts, extended=extended, start=start)
            label = f"{pkey}:{'extended' if extended else 'real'}"
            try:
                hold = hold_base_spec(pair, extended=extended, start=v_start, end=end)
                hold_res = run_backtest(hold, bars)
                hold_hash = registry.log_run(hold, hold_res)
                result.holds[label] = {
                    "hash": hold_hash,
                    "metrics": hold_res.metrics,
                    "start": v_start,
                }
            except Exception as exc:
                log.exception("%s: hold run failed", label)
                result.failures[label] = f"{type(exc).__name__}: {exc}"
                continue
            for key in keys:
                grid = quick_grid(GRIDS[key]) if quick else GRIDS[key]
                try:
                    result.outcomes.append(
                        run_rule(
                            key,
                            pair,
                            bars,
                            extended=extended,
                            start=v_start,
                            end=end,
                            wf_spec=wf_spec,
                            registry=registry,
                            grid=grid,
                            n_workers=n_workers,
                            out_dir=out_dir,
                            with_path=extended and pkey in path_pairs,
                        )
                    )
                except Exception as exc:  # one broken rule must not sink the study
                    log.exception("%s %s failed", label, key)
                    result.failures[f"{label}:{key}"] = f"{type(exc).__name__}: {exc}"
    stamp = tag or datetime.now(UTC).strftime("%Y-%m-%d")
    result.summary_md, result.summary_json = write_summary(
        result, out_dir, stamp, wf_spec, start, end
    )
    return result


# --- summary -----------------------------------------------------------------------------


def _pct(v: Any, nd: int = 1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}%}"


def _num(v: Any, nd: int = 2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def write_summary(
    result: StudyResult, out_dir: Path, stamp: str, wf_spec: WalkForwardSpec, start: str, end: str
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = result.table()
    lines = [
        f"# Leverage study — objective `{result.objective}` — {stamp}",
        "",
        "The leveraged fund when a rule says the market turned bullish, the plain fund otherwise. "
        f"Span {start}..{end}; walk-forward train {wf_spec.train_months}m / test "
        f"{wf_spec.test_months}m rolling; signals at the close, fills at the next open; 1 bp "
        "slippage. `extended` runs use the fund's own history spliced onto a synthetic rebuild "
        "(``<LEV>.X``); `real` runs use the fund only. Event columns come from the full-sample "
        "default-param run (unoptimised).",
        "",
        "## Out-of-sample (stitched walk-forward)",
        "",
        "| pair | variant | strategy | verdict | Sharpe | CAGR | max DD | plain CAGR "
        "| plain max DD | time lev | sw/yr | DD 2008 | DD 2020 | DD 2022 | calls | best hold "
        "| median excess | ahead |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["pair"]),
                    str(r["variant"]),
                    str(r["strategy"]),
                    str(r["verdict"]),
                    _num(r["oos_sharpe"]),
                    _pct(r["oos_cagr"]),
                    _pct(r["oos_max_dd"]),
                    _pct(r["mkt_cagr"]),
                    _pct(r["mkt_max_dd"]),
                    _pct(r["time_lev"], 0),
                    _num(r["switch_py"], 1),
                    _pct(r["oos_dd_2008"], 0),
                    _pct(r["oos_dd_2020"], 0),
                    _pct(r["oos_dd_2022"], 0),
                    str(int(r["n_events"]) if r["n_events"] is not None else "n/a"),
                    str(r["best_h"] if r["best_h"] is not None else "none"),
                    _pct(r["best_excess"]),
                    _pct(r["best_hit"], 0),
                ]
            )
            + " |"
        )
    if result.failures:
        lines += ["", "## Failures", ""] + [f"- {k}: {v}" for k, v in result.failures.items()]
    lines += ["", "## Scorecards", ""] + [
        f"- {o.pair} {'extended' if o.extended else 'real'} {o.strategy}: `{o.scorecard.name}` "
        f"(wf `{o.wf.wf_id}`, full `{o.full_hash}`)"
        for o in result.outcomes
    ]
    md_path = out_dir / f"{STUDY}_{result.objective}_{stamp}.md"
    json_path = out_dir / f"{STUDY}_{result.objective}_{stamp}.json"
    md_path.write_text("\n".join(lines) + "\n")
    payload = {
        "study": STUDY,
        "objective": result.objective,
        "stamp": stamp,
        "start": start,
        "end": end,
        "walk_forward": wf_spec.model_dump(mode="json"),
        "pairs": result.pairs,
        "holds": result.holds,
        "rows": t.to_dict(orient="records"),
        "outcomes": [
            {
                "pair": o.pair,
                "extended": o.extended,
                "strategy": o.strategy,
                "lev": o.lev_symbol,
                "wf_id": o.wf.wf_id,
                "stitched_hash": o.wf.stitched_hash,
                "full_hash": o.full_hash,
                "verdict": o.verdict.as_dict(),
                "oos_metrics": o.wf.oos_metrics,
                "oos_dial": o.oos_dial,
                "full_metrics": o.full_metrics,
                "full_dial": o.full_dial,
                "events": o.events,
                "excess_path": o.excess.reset_index().to_dict(orient="list")
                if o.excess is not None
                else None,
                "chosen_params": o.wf.windows["params"].tolist(),
                "windows": o.wf.windows[["test_start", "test_end"]]
                .astype(str)
                .to_dict(orient="records"),
                "seconds": o.seconds,
            }
            for o in result.outcomes
        ],
        "failures": result.failures,
    }
    json_path.write_text(json.dumps(payload, indent=1, default=str))
    return md_path, json_path
