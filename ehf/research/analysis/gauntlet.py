"""The acceptance gauntlet — fixed-in-advance kill criteria for any strategy.

A walk-forward result walks in; a ``Verdict`` walks out. Every threshold lives in
``GauntletConfig`` and is decided *before* looking at results, so an autonomous
research loop can pass/kill without a human in the loop — and a human reviewing
a scorecard sees exactly which bar was missed and by how much.

Checks (all must pass):
  windows        enough out-of-sample windows to mean anything
  oos_sharpe     the stitched OOS Sharpe clears a floor
  psr            Probabilistic Sharpe of the OOS curve (length/skew/kurtosis aware)
  dsr            Deflated Sharpe vs. the number of darts thrown (every train cell)
  efficiency     OOS Sharpe / mean in-sample Sharpe — how much of the backtest survived
  pbo            Probability of Backtest Overfitting on the last train sweep (CSCV)
  cost_stress    still positive when slippage is multiplied
  trades         enough OOS trades for the statistics to be about trading, not luck
  windows_pos    a majority of test windows positive (not one lucky year)
  regimes        positive in most market trend regimes, or the edge is regime-specific
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from ehf.research.analysis import metrics
from ehf.research.analysis.regime import regime_breakdown
from ehf.research.analysis.runs import sweep_return_matrix
from ehf.research.backtest.walkforward import WalkForwardResult
from ehf.research.results.registry import ExperimentRegistry


class GauntletConfig(BaseModel):
    """Kill criteria. Change them deliberately, in one place, with a reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_windows: int = 3
    min_oos_sharpe: float = 0.3
    min_psr: float = 0.90
    min_deflated_sharpe: float = 0.95
    min_wf_efficiency: float = 0.5
    max_pbo: float = 0.30
    stress_multiplier: float = 2.0  # must match a multiplier the walk-forward ran
    min_stress_sharpe: float = 0.0
    min_oos_trades: int = 100
    min_share_windows_positive: float = 0.5
    min_regime_positive_share: float = 0.6  # of trend regimes (bull/bear/chop) with Sharpe > 0
    periods_per_year: float = 252.0


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    threshold: float
    op: str  # ">=" or "<="
    note: str = ""

    @property
    def passed(self) -> bool:
        if not math.isfinite(self.value):
            return False
        return self.value >= self.threshold if self.op == ">=" else self.value <= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": None if not math.isfinite(self.value) else round(self.value, 4),
            "threshold": self.threshold,
            "op": self.op,
            "passed": self.passed,
            "note": self.note,
        }


@dataclass
class Verdict:
    strategy: str
    wf_id: str
    stitched_hash: str
    checks: list[Check]
    config: GauntletConfig

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    @property
    def status(self) -> str:
        return "promoted" if self.passed else "killed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "wf_id": self.wf_id,
            "stitched_hash": self.stitched_hash,
            "status": self.status,
            "failed": self.failed,
            "checks": [c.as_dict() for c in self.checks],
            "config": self.config.model_dump(mode="json"),
        }

    def to_markdown(self) -> str:
        head = f"**Verdict: {self.status.upper()}**"
        if self.failed:
            head += f" — failed: {', '.join(self.failed)}"
        lines = [head, "", "| check | value | bar | pass |", "|---|---|---|---|"]
        for c in self.checks:
            val = "n/a" if not math.isfinite(c.value) else f"{c.value:.3f}"
            mark = "✅" if c.passed else "❌"
            note = f" <br><sub>{c.note}</sub>" if c.note else ""
            lines.append(f"| {c.name} | {val} | {c.op} {c.threshold:g} | {mark}{note} |")
        return "\n".join(lines)


def run_gauntlet(
    wf: WalkForwardResult,
    *,
    registry: ExperimentRegistry,
    market_returns: pd.Series | None = None,
    cfg: GauntletConfig | None = None,
) -> Verdict:
    """Score a walk-forward study against the fixed criteria."""
    cfg = cfg or GauntletConfig()
    r = wf.oos_returns.dropna()
    m = wf.oos_metrics
    checks: list[Check] = []

    checks.append(Check("windows", float(wf.n_windows), float(cfg.min_windows), ">="))
    checks.append(Check("oos_sharpe", wf.oos_sharpe, cfg.min_oos_sharpe, ">="))
    checks.append(Check("psr", metrics.probabilistic_sharpe_ratio(r, 0.0), cfg.min_psr, ">="))

    n_trials = int(len(wf.trial_sharpes))
    dsr = metrics.deflated_sharpe_ratio(r, wf.trial_sharpes) if n_trials >= 2 else float("nan")
    checks.append(
        Check("dsr", dsr, cfg.min_deflated_sharpe, ">=", note=f"{n_trials} trials across windows")
    )
    checks.append(
        Check(
            "efficiency",
            wf.efficiency,
            cfg.min_wf_efficiency,
            ">=",
            note=f"OOS {wf.oos_sharpe:.2f} / IS {wf.mean_is_sharpe:.2f}",
        )
    )

    pbo = float("nan")
    note = ""
    if wf.train_sweep_ids:
        sid = wf.train_sweep_ids[-1]
        matrix, _ = sweep_return_matrix(registry, sid)
        if matrix.size:
            pbo = metrics.probability_of_backtest_overfitting(matrix)
            note = f"CSCV on {sid} ({matrix.shape[1]} cells x {matrix.shape[0]} bars)"
    checks.append(Check("pbo", pbo, cfg.max_pbo, "<=", note=note))

    key = f"sharpe_cost_x{cfg.stress_multiplier:g}"
    checks.append(
        Check(
            "cost_stress",
            float(m.get(key, float("nan"))),
            cfg.min_stress_sharpe,
            ">=",
            note=f"OOS Sharpe at {cfg.stress_multiplier:g}x slippage",
        )
    )
    checks.append(Check("trades", float(m.get("n_trades", 0.0)), float(cfg.min_oos_trades), ">="))
    checks.append(
        Check(
            "windows_pos",
            float(m.get("share_windows_positive", float("nan"))),
            cfg.min_share_windows_positive,
            ">=",
        )
    )

    share, rnote = regime_positive_share(r, market_returns, periods_per_year=cfg.periods_per_year)
    checks.append(Check("regimes", share, cfg.min_regime_positive_share, ">=", note=rnote))

    return Verdict(
        strategy=wf.base_spec.strategy,
        wf_id=wf.wf_id,
        stitched_hash=wf.stitched_hash,
        checks=checks,
        config=cfg,
    )


def regime_positive_share(
    returns: pd.Series, market_returns: pd.Series | None, *, periods_per_year: float
) -> tuple[float, str]:
    """Share of trend regimes (bull / bear / chop) in which the OOS Sharpe is > 0."""
    if len(returns) < 100:
        return float("nan"), "too short for a regime split"
    proxy = market_returns.reindex(returns.index) if market_returns is not None else None
    table = regime_breakdown(returns, proxy, periods_per_year=periods_per_year)
    trend = table[[str(i).startswith("trend:") for i in table.index]]
    if trend.empty:
        return float("nan"), "no regime rows"
    pos = (trend["sharpe"] > 0).sum()
    parts = ", ".join(f"{str(i).split(':')[1]} {v:+.2f}" for i, v in trend["sharpe"].items())
    source = "vs market proxy" if market_returns is not None else "self-conditioned"
    return float(pos / len(trend)), f"{parts} ({source})"


def market_proxy_returns(bars: pd.DataFrame, symbol: str = "SPY") -> pd.Series | None:
    """Close-to-close returns of ``symbol`` from canonical bars, or None if absent."""
    px = bars.loc[bars["symbol"] == symbol.upper(), ["timestamp", "close"]]
    if px.empty:
        return None
    s = px.set_index("timestamp")["close"].sort_index().pct_change()
    return s.replace([np.inf, -np.inf], np.nan)
