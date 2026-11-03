"""Scorecards — the human-readable record of one walk-forward study.

``write_scorecard`` turns a ``WalkForwardResult`` + gauntlet ``Verdict`` (+ any
extra metrics the study computed) into ``var/scorecards/<wf_id>.md`` and
``.json``. The markdown is what the weekly review reads; the JSON is what a
digest or a dashboard loads. Same shape as the hand-written scorecards from the
first TSMOM studies, now produced by code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ehf.research.analysis.gauntlet import Verdict
from ehf.research.backtest.walkforward import WalkForwardResult
from ehf.research.paths import SCORECARD_DIR


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "n/a" if not math.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


def _short_params(params: dict[str, Any], base: dict[str, Any]) -> str:
    """Only the knobs that differ from the base spec, compactly."""
    diff = {k: v for k, v in params.items() if base.get(k) != v}
    return " ".join(f"{k}={v}" for k, v in sorted(diff.items())) or "(defaults)"


def windows_table(wf: WalkForwardResult) -> str:
    base = wf.base_spec.params
    lines = [
        "| test_start | test_end | params | is_sharpe | oos_sharpe | oos_cagr | oos_max_dd "
        "| oos_trades |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in wf.windows.iterrows():
        lines.append(
            f"| {r['test_start']} | {r['test_end']} | {_short_params(r['params'], base)} | "
            f"{_fmt(float(r['is_sharpe']), 2)} | {_fmt(float(r['oos_sharpe']), 2)} | "
            f"{_fmt(float(r['oos_cagr']), 3)} | {_fmt(float(r['oos_max_drawdown']), 3)} | "
            f"{int(r['oos_n_trades'])} |"
        )
    return "\n".join(lines)


def render_markdown(
    wf: WalkForwardResult,
    verdict: Verdict,
    *,
    title: str | None = None,
    extra: dict[str, Any] | None = None,
    notes: str = "",
) -> str:
    spec, w = wf.base_spec, wf.wf_spec
    where = ",".join(spec.universe.symbols) or spec.universe.name
    head = title or f"{wf.wf_id} — {spec.strategy} on {where}"
    mode = "anchored" if w.anchored else "rolling"
    parts = [
        f"# {head}",
        "",
        f"Windows: {wf.n_windows} (train {w.train_months}m / test {w.test_months}m, {mode}), "
        f"objective {w.objective}, grid {json.dumps({k: list(v) for k, v in wf.grid.items()})}, "
        f"costs {spec.costs.slippage_bps:g} bps, span {spec.start}..{spec.end}.",
        "",
    ]
    if notes:
        parts += [notes, ""]
    parts += ["## OOS metrics", ""]
    parts += [f"- {k}: {_fmt(v)}" for k, v in wf.oos_metrics.items()]
    if extra:
        parts += ["", "## Dial metrics (OOS)", ""]
        parts += [f"- {k}: {_fmt(v)}" for k, v in extra.items()]
    parts += [
        "",
        "## Windows",
        "",
        windows_table(wf),
        "",
        "## Gauntlet",
        "",
        verdict.to_markdown(),
        "",
    ]
    return "\n".join(parts)


def write_scorecard(
    wf: WalkForwardResult,
    verdict: Verdict,
    *,
    title: str | None = None,
    extra: dict[str, Any] | None = None,
    notes: str = "",
    out_dir: Path = SCORECARD_DIR,
) -> tuple[Path, Path]:
    """Write ``<wf_id>.md`` + ``<wf_id>.json`` and return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(wf, verdict, title=title, extra=extra, notes=notes)
    payload = {
        "wf_id": wf.wf_id,
        "strategy": wf.base_spec.strategy,
        "base_spec": json.loads(wf.base_spec.canonical_json()),
        "grid": {k: list(v) for k, v in wf.grid.items()},
        "walk_forward": wf.wf_spec.model_dump(mode="json"),
        "stitched_hash": wf.stitched_hash,
        "oos_metrics": wf.oos_metrics,
        "stress": {str(k): v for k, v in wf.stress.items()},
        "windows": wf.windows.to_dict(orient="records"),
        "verdict": verdict.as_dict(),
        "extra": extra or {},
        "notes": notes,
    }
    md_path = out_dir / f"{wf.wf_id}.md"
    json_path = out_dir / f"{wf.wf_id}.json"
    md_path.write_text(md)
    json_path.write_text(json.dumps(payload, indent=1, default=_json_default))
    return md_path, json_path


def _json_default(o: Any) -> Any:
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if hasattr(o, "item"):
        return o.item()
    return str(o)
