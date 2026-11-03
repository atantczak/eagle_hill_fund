"""The VIX overlay study: can the fear gauge tell you when to be in or out of SPY?

For every strategy in ``strategies.vix_overlay.GRIDS`` (the two benchmarks, the
nine rules, the vote):

  1. walk-forward (optimize on train months, validate on test months, stitched)
  2. the gauntlet's fixed kill criteria
  3. dial metrics on the stitched out-of-sample curve (time in market, switches,
     drawdown in 2008 / 2020 / 2022 / 2025, vs. holding)
  4. a full-sample run on *default* params (descriptive, in-sample) so the
     crisis table covers all of 2008, which the OOS windows only reach in August
  5. a scorecard per strategy, and one summary for the study

Everything is logged to the registry under deterministic ids, so the study is
re-runnable and every number traces to a stored run.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ehf.research.analysis.dial import dial_metrics
from ehf.research.analysis.gauntlet import (
    GauntletConfig,
    Verdict,
    market_proxy_returns,
    run_gauntlet,
)
from ehf.research.analysis.scorecard import write_scorecard
from ehf.research.backtest.engine import run_backtest
from ehf.research.backtest.walkforward import (
    Grid,
    WalkForwardResult,
    WalkForwardSpec,
    walk_forward,
)
from ehf.research.config.spec import CostConfig, RunSpec, UniverseSpec
from ehf.research.data.vix import ASSET, VIX_UNIVERSE
from ehf.research.paths import SCORECARD_DIR
from ehf.research.results.registry import ExperimentRegistry
from ehf.research.strategies.vix_overlay import GRIDS

log = logging.getLogger(__name__)

STUDY = "vix_overlay"


@dataclass
class StrategyOutcome:
    strategy: str
    wf: WalkForwardResult
    verdict: Verdict
    oos_dial: dict[str, Any]
    full_hash: str
    full_metrics: dict[str, float]
    full_dial: dict[str, Any]
    seconds: float
    scorecard: Path


@dataclass
class StudyResult:
    objective: str
    outcomes: list[StrategyOutcome] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    summary_md: Path | None = None
    summary_json: Path | None = None

    def table(self) -> pd.DataFrame:
        rows = []
        for o in self.outcomes:
            m, d, f, fd = o.wf.oos_metrics, o.oos_dial, o.full_metrics, o.full_dial
            rows.append(
                {
                    "strategy": o.strategy,
                    "verdict": o.verdict.status,
                    "failed": ",".join(o.verdict.failed),
                    "oos_sharpe": m.get("sharpe"),
                    "oos_cagr": m.get("cagr"),
                    "oos_max_dd": m.get("max_drawdown"),
                    "oos_sharpe_x2": m.get("sharpe_cost_x2"),
                    "wf_eff": m.get("wf_efficiency"),
                    "win_pos": m.get("share_windows_positive"),
                    "time_in": d.get("time_in_market"),
                    "switch_py": d.get("switches_per_year"),
                    "oos_dd_2020": d.get("dd_2020"),
                    "oos_dd_2022": d.get("dd_2022"),
                    "oos_dd_2025": d.get("dd_2025"),
                    "full_cagr": f.get("cagr"),
                    "full_max_dd": f.get("max_drawdown"),
                    "full_dd_2008": fd.get("dd_2008"),
                    "full_dd_2020": fd.get("dd_2020"),
                    "full_dd_2022": fd.get("dd_2022"),
                }
            )
        return pd.DataFrame(rows)


def base_spec(strategy: str, start: str, end: str, *, slippage_bps: float = 1.0) -> RunSpec:
    return RunSpec(
        strategy=strategy,
        universe=UniverseSpec(symbols=VIX_UNIVERSE),
        start=start,
        end=end,
        costs=CostConfig(slippage_bps=slippage_bps),
    )


def quick_grid(grid: Grid) -> Grid:
    """First and last value of every knob — a smoke-sized sweep."""
    return {k: ([v[0], v[-1]] if len(v) > 1 else list(v)) for k, v in grid.items()}


def run_strategy(
    strategy: str,
    bars: pd.DataFrame,
    *,
    start: str,
    end: str,
    wf_spec: WalkForwardSpec,
    registry: ExperimentRegistry,
    grid: Grid | None = None,
    gauntlet: GauntletConfig | None = None,
    n_workers: int | None = None,
    out_dir: Path = SCORECARD_DIR,
) -> StrategyOutcome:
    t0 = time.time()
    grid = grid if grid is not None else GRIDS[strategy]
    spec = base_spec(strategy, start, end)
    market = market_proxy_returns(bars, ASSET)

    wf = walk_forward(spec, grid, bars, wf_spec, registry=registry, n_workers=n_workers)
    verdict = run_gauntlet(wf, registry=registry, market_returns=market, cfg=gauntlet)

    stitched = registry.get_run(wf.stitched_hash)
    eq = stitched["equity"].set_index("timestamp")
    resets = [pd.Timestamp(d, tz="UTC") for d in wf.windows["test_start"]]
    # the first OOS bar of each window may sit a day or two after test_start; snap to bars
    idx = pd.DatetimeIndex(eq.index)
    resets = [idx[int(idx.searchsorted(r))] for r in resets if int(idx.searchsorted(r)) < len(idx)]
    oos_dial = dial_metrics(
        eq["net_return"],
        eq["equity"],
        stitched["trades"],
        asset=ASSET,
        market_net=market,
        resets=resets,
    )

    # full-sample defaults (descriptive): one logged run, complete crisis coverage
    full_res = run_backtest(spec, bars)
    full_hash = registry.log_run(spec, full_res)
    feq = full_res.equity.set_index("timestamp")
    full_dial = dial_metrics(
        feq["net_return"], feq["equity"], full_res.trades, asset=ASSET, market_net=market
    )

    notes = (
        f"Study `{STUDY}`: dial on {ASSET} vs CASH (T-bill accrual), signals from the VIX family. "
        f"Full-sample default-param run `{full_hash}` (in-sample, descriptive): "
        f"CAGR {full_res.metrics['cagr']:.1%}, max DD {full_res.metrics['max_drawdown']:.1%}, "
        f"2008 DD {_pct(full_dial.get('dd_2008'))} vs market {_pct(full_dial.get('mkt_dd_2008'))}."
    )
    md, _ = write_scorecard(wf, verdict, extra=oos_dial, notes=notes, out_dir=out_dir)
    secs = time.time() - t0
    log.info(
        "%-13s %-8s oos_sharpe=%.2f cagr=%.1f%% maxdd=%.1f%% time_in=%.0f%% sw/yr=%.1f (%.0fs)",
        strategy,
        verdict.status,
        wf.oos_sharpe,
        wf.oos_metrics.get("cagr", float("nan")) * 100,
        wf.oos_metrics.get("max_drawdown", float("nan")) * 100,
        oos_dial["time_in_market"] * 100,
        oos_dial["switches_per_year"],
        secs,
    )
    return StrategyOutcome(
        strategy=strategy,
        wf=wf,
        verdict=verdict,
        oos_dial=oos_dial,
        full_hash=full_hash,
        full_metrics=full_res.metrics,
        full_dial=full_dial,
        seconds=secs,
        scorecard=md,
    )


def run_study(
    bars: pd.DataFrame,
    *,
    start: str,
    end: str,
    strategies: tuple[str, ...] | None = None,
    train_months: int = 24,
    test_months: int = 12,
    objective: str = "sharpe",
    quick: bool = False,
    n_workers: int | None = None,
    registry: ExperimentRegistry | None = None,
    out_dir: Path = SCORECARD_DIR,
    tag: str | None = None,
) -> StudyResult:
    registry = registry or ExperimentRegistry()
    wf_spec = WalkForwardSpec(
        train_months=train_months,
        test_months=test_months,
        objective=objective,
        warmup_bars=252,
        min_trades=2,  # a dial that switched once in two years is still a dial
    )
    keys = strategies or tuple(GRIDS)
    result = StudyResult(objective=objective)
    for key in keys:
        grid = quick_grid(GRIDS[key]) if quick else GRIDS[key]
        try:
            result.outcomes.append(
                run_strategy(
                    key,
                    bars,
                    start=start,
                    end=end,
                    wf_spec=wf_spec,
                    registry=registry,
                    grid=grid,
                    n_workers=n_workers,
                    out_dir=out_dir,
                )
            )
        except Exception as exc:  # one broken rule must not sink the study
            log.exception("%s failed", key)
            result.failures[key] = f"{type(exc).__name__}: {exc}"
    stamp = tag or datetime.now(UTC).strftime("%Y-%m-%d")
    result.summary_md, result.summary_json = write_summary(
        result, out_dir, stamp, wf_spec, start, end
    )
    return result


def _pct(v: Any) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.1%}"


def _num(v: Any, nd: int = 2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def write_summary(
    result: StudyResult, out_dir: Path, stamp: str, wf_spec: WalkForwardSpec, start: str, end: str
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = result.table()
    lines = [
        f"# VIX overlay study — objective `{result.objective}` — {stamp}",
        "",
        f"Dial on {ASSET} vs CASH (13-week bill accrual). Span {start}..{end}; walk-forward train "
        f"{wf_spec.train_months}m / test {wf_spec.test_months}m rolling; signals at the close, "
        f"fills at the next open; 1 bp slippage (2x stress in the gauntlet). Full-sample columns "
        f"use each rule's *default* params (in-sample, descriptive); every other column is "
        f"out-of-sample.",
        "",
        "## Out-of-sample (stitched walk-forward)",
        "",
        "| strategy | verdict | failed | Sharpe | CAGR | max DD | Sharpe@2x | WF eff | win+ "
        "| time in | sw/yr | DD 2020 | DD 2022 | DD 2025 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        cells = [
            str(r["strategy"]),
            str(r["verdict"]),
            str(r["failed"] or "-"),
            _num(r["oos_sharpe"]),
            _pct(r["oos_cagr"]),
            _pct(r["oos_max_dd"]),
            _num(r["oos_sharpe_x2"]),
            _num(r["wf_eff"]),
            _pct(r["win_pos"]),
            _pct(r["time_in"]),
            _num(r["switch_py"], 1),
            _pct(r["oos_dd_2020"]),
            _pct(r["oos_dd_2022"]),
            _pct(r["oos_dd_2025"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Full sample, default params (in-sample, descriptive)",
        "",
        "| strategy | CAGR | max DD | DD 2008 | DD 2020 | DD 2022 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['strategy']} | {_pct(r['full_cagr'])} | {_pct(r['full_max_dd'])} | "
            f"{_pct(r['full_dd_2008'])} | {_pct(r['full_dd_2020'])} | {_pct(r['full_dd_2022'])} |"
        )
    if result.outcomes:
        o = result.outcomes[0]
        fd = o.full_dial
        lines += [
            "",
            f"Market ({ASSET}) over the same full span: CAGR {_pct(fd.get('mkt_cagr'))}, "
            f"max DD {_pct(fd.get('mkt_max_drawdown'))}, 2008 {_pct(fd.get('mkt_dd_2008'))}, "
            f"2020 {_pct(fd.get('mkt_dd_2020'))}, 2022 {_pct(fd.get('mkt_dd_2022'))}.",
        ]
    if result.failures:
        lines += ["", "## Failures", ""] + [f"- {k}: {v}" for k, v in result.failures.items()]
    lines += ["", "## Scorecards", ""] + [
        f"- {o.strategy}: `{o.scorecard.name}` (wf `{o.wf.wf_id}`, full `{o.full_hash}`)"
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
        "rows": t.to_dict(orient="records"),
        "outcomes": {
            o.strategy: {
                "wf_id": o.wf.wf_id,
                "stitched_hash": o.wf.stitched_hash,
                "full_hash": o.full_hash,
                "verdict": o.verdict.as_dict(),
                "oos_metrics": o.wf.oos_metrics,
                "oos_dial": o.oos_dial,
                "full_metrics": o.full_metrics,
                "full_dial": o.full_dial,
                "chosen_params": o.wf.windows["params"].tolist(),
                "seconds": o.seconds,
            }
            for o in result.outcomes
        },
        "failures": result.failures,
    }
    import json

    json_path.write_text(json.dumps(payload, indent=1, default=str))
    return md_path, json_path
