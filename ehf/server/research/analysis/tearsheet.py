"""One-call tearsheets rendered from stored runs (by hash, no rerun).

``tearsheet(registry, config_hash)`` returns a matplotlib Figure: equity,
drawdown, rolling Sharpe, return distribution, the core metrics, and the
robustness scorecard. ``comparison_tearsheet`` overlays several runs.
"""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ehf.server.research.analysis import runs
from ehf.server.research.results.registry import ExperimentRegistry


def _rolling_sharpe(returns: pd.Series, window: int, ppy: float) -> pd.Series:
    mean = returns.rolling(window).mean()
    sd = returns.rolling(window).std()
    return cast(pd.Series, (mean / sd * np.sqrt(ppy)).where(sd > 0))


def tearsheet(
    registry: ExperimentRegistry,
    config_hash: str,
    *,
    sweep_id: str | None = None,
    periods_per_year: float = 252.0,
    rolling_window: int = 63,
    save_path: str | None = None,
) -> Figure:
    """Render a full tearsheet for one stored run."""
    run = registry.get_run(config_hash)
    eq = run["equity"].set_index("timestamp")
    equity = eq["equity"]
    ret = eq["net_return"]
    metrics = run["metrics"]
    report = runs.robustness_report(
        registry, config_hash, sweep_id=sweep_id, periods_per_year=periods_per_year
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle(
        f"{run['strategy']}  ·  {config_hash}" + (f"  ·  sweep {sweep_id}" if sweep_id else ""),
        fontsize=13,
        weight="bold",
    )

    # 1. Equity curve
    ax = axes[0, 0]
    ax.plot(equity.index, equity, lw=1.3)
    ax.set_title("Equity")
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # 2. Drawdown
    ax = axes[0, 1]
    dd = equity / equity.cummax() - 1.0
    ax.fill_between(dd.index, dd, 0, color="crimson", alpha=0.4)
    ax.set_title("Drawdown")
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    # 3. Rolling Sharpe
    ax = axes[1, 0]
    rs = _rolling_sharpe(ret, rolling_window, periods_per_year)
    ax.plot(rs.index, rs, lw=1, color="darkgreen")
    ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.set_title(f"Rolling Sharpe ({rolling_window} bars)")
    ax.grid(alpha=0.3)

    # 4. Return distribution
    ax = axes[1, 1]
    ax.hist(ret.dropna(), bins=60, color="steelblue", alpha=0.8)
    ax.axvline(0, ls="--", color="gray", lw=0.8)
    ax.set_title("Per-bar return distribution")
    ax.grid(alpha=0.3)

    # 5. Core metrics
    ax = axes[2, 0]
    ax.axis("off")
    core = [
        ("CAGR", f"{metrics.get('cagr', 0):.1%}"),
        ("Total return", f"{metrics.get('total_return', 0):.1%}"),
        ("Ann. vol", f"{metrics.get('ann_vol', 0):.1%}"),
        ("Sharpe", f"{metrics.get('sharpe', 0):.2f}"),
        ("Sortino", f"{metrics.get('sortino', 0):.2f}"),
        ("Max drawdown", f"{metrics.get('max_drawdown', 0):.1%}"),
        ("Calmar", f"{metrics.get('calmar', 0):.2f}"),
        ("Hit rate", f"{metrics.get('hit_rate', 0):.1%}"),
        ("# trades", f"{int(metrics.get('n_trades', 0))}"),
    ]
    ax.set_title("Performance", loc="left", weight="bold")
    ax.table(cellText=core, colWidths=[0.6, 0.4], loc="center", cellLoc="left", edges="open")

    # 6. Robustness scorecard
    ax = axes[2, 1]
    ax.axis("off")
    mc = report.get("monte_carlo", {})
    rob = [("Prob. Sharpe (>0)", f"{report.get('psr_vs_zero', float('nan')):.2f}")]
    if "deflated_sharpe" in report:
        rob.append(
            (
                "Deflated Sharpe",
                f"{report['deflated_sharpe']:.2f}  (N={report.get('n_trials', '?')})",
            )
        )
    if "pbo" in report:
        rob.append(("PBO (overfit prob.)", f"{report['pbo']:.2f}"))
    if mc:
        rob.append(("MC Sharpe [p5, p95]", f"[{mc['sharpe_p05']:.2f}, {mc['sharpe_p95']:.2f}]"))
        rob.append(("MC worst drawdown", f"{mc['maxdd_worst']:.1%}"))
        rob.append(("MC P(Sharpe>0)", f"{mc['prob_sharpe_positive']:.0%}"))
    ax.set_title("Robustness", loc="left", weight="bold")
    ax.table(cellText=rob, colWidths=[0.55, 0.45], loc="center", cellLoc="left", edges="open")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def comparison_tearsheet(
    registry: ExperimentRegistry, config_hashes: list[str], *, save_path: str | None = None
) -> Figure:
    """Overlay equity + drawdown for several runs and tabulate their metrics."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[3, 2])
    rows = []
    for h in config_hashes:
        run = registry.get_run(h)
        eq = run["equity"].set_index("timestamp")["equity"]
        axes[0].plot(eq.index, eq / eq.iloc[0], lw=1.2, label=f"{run['strategy']}·{h[:8]}")
        m = run["metrics"]
        rows.append(
            [
                h[:8],
                run["strategy"],
                f"{m.get('sharpe', 0):.2f}",
                f"{m.get('cagr', 0):.1%}",
                f"{m.get('max_drawdown', 0):.1%}",
            ]
        )
    axes[0].set_title("Equity (normalized to 1.0)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].axis("off")
    axes[1].table(
        cellText=rows,
        colLabels=["hash", "strategy", "Sharpe", "CAGR", "MaxDD"],
        loc="center",
        cellLoc="center",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig
