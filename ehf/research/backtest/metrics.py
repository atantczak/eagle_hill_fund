"""Core return/risk metrics shared by every run-producing path.

One implementation, used by the engine (single backtest), the walk-forward
harness (stitched out-of-sample curve) and anything else that turns a per-bar
net return series into the standard metric set. The robustness suite
(Deflated Sharpe, PBO, Monte Carlo) lives in ``ehf.research.analysis``.
"""

from __future__ import annotations

import pandas as pd


def core_metrics(
    net: pd.Series, equity: pd.Series, trades: pd.DataFrame, periods_per_year: float
) -> dict[str, float]:
    """Standard metric set from per-bar net returns + equity curve + trade ledger."""
    net = net.dropna()
    n = len(net)
    if n == 0 or equity.empty:
        return {"n_bars": 0.0}

    growth = float(equity.iloc[-1] / equity.iloc[0]) if equity.iloc[0] else 0.0
    total_return = growth - 1.0
    years = n / periods_per_year
    if years > 0 and growth > 0:
        cagr = float(growth ** (1.0 / years) - 1.0)
    else:
        # growth <= 0 means the account was wiped out -> -100% CAGR (and avoids
        # raising a negative base to a fractional power, which returns complex).
        cagr = -1.0 if growth <= 0 else 0.0
    sd = float(net.std())
    ann_vol = sd * (periods_per_year**0.5)
    sharpe = float(net.mean() / sd * (periods_per_year**0.5)) if sd > 0 else 0.0
    downside = net[net < 0].std()
    sortino = (
        float(net.mean() / downside * (periods_per_year**0.5)) if downside and downside > 0 else 0.0
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "n_bars": float(n),
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate": float((net > 0).mean()),
        "n_trades": float(len(trades)),
        "avg_turnover": (
            float(trades["delta_weight"].abs().sum()) / n if not trades.empty else 0.0
        ),
    }


def equity_from_returns(net: pd.Series, initial_capital: float) -> pd.DataFrame:
    """Equity frame ``[timestamp, equity, net_return]`` from per-bar net returns."""
    equity_vals = (1.0 + net).cumprod() * initial_capital
    return pd.DataFrame(
        {"timestamp": net.index, "equity": equity_vals.to_numpy(), "net_return": net.to_numpy()}
    ).reset_index(drop=True)
