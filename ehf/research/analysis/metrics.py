"""Robustness & honesty metrics — the part most retail skips.

A great in-sample Sharpe is cheap; these measures ask whether it's *real*. All
operate on return series / matrices and are pure (no I/O), so they're easy to
test against known properties.

  * Probabilistic Sharpe Ratio (PSR) — confidence that the true Sharpe > a
    benchmark, given track-record length, skew and kurtosis (Bailey-LdP 2012).
  * Deflated Sharpe Ratio (DSR) — PSR with the benchmark raised to the expected
    *maximum* Sharpe under N trials, i.e. corrected for selection/multiple
    testing. The number every strategy must clear before it goes near "paper".
  * Probability of Backtest Overfitting (PBO) — CSCV: how often the in-sample
    best config underperforms out-of-sample (Bailey-Borwein-LdP-Zhu 2017).
  * Monte Carlo — bootstrap the return stream into a distribution of outcomes,
    so you see a cloud of equity paths, not one lucky curve.
"""

from __future__ import annotations

from itertools import combinations
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

_EULER_MASCHERONI = 0.5772156649015329


def _as_returns(x: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    r = np.asarray(x, dtype=float)
    return cast(np.ndarray, r[~np.isnan(r)])


def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualized Sharpe of a per-period return series (risk-free = 0)."""
    r = _as_returns(returns)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray, benchmark_sharpe: float = 0.0
) -> float:
    """P(true Sharpe > benchmark), accounting for length, skew, and kurtosis.

    Returns a probability in [0, 1]. ``benchmark_sharpe`` is an *annualized*
    Sharpe (default 0). PSR rises with track length and falls with negative
    skew / fat tails.
    """
    r = _as_returns(returns)
    n = len(r)
    sd = r.std(ddof=1)
    if n < 3 or sd == 0:
        return float("nan")
    sr = r.mean() / sd  # per-period Sharpe
    sr_star = benchmark_sharpe / np.sqrt(252.0)  # to per-period (consistent units)
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=False, bias=False))  # Pearson (normal = 3)
    denom = np.sqrt(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2)
    if denom <= 0:
        return float("nan")
    z = (sr - sr_star) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(trial_sharpes: pd.Series | np.ndarray) -> float:
    """Expected *maximum* annualized Sharpe from N independent trials of skill 0.

    The selection-bias yardstick for the Deflated Sharpe: with enough tries, a
    high Sharpe is expected by chance alone. Uses the variance of the trial
    Sharpes and the count N.
    """
    s = _as_returns(trial_sharpes)
    n = len(s)
    if n < 2:
        return 0.0
    v = s.std(ddof=1)
    return float(
        v
        * (
            (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n)
            + _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n * np.e))
        )
    )


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray, trial_sharpes: pd.Series | np.ndarray
) -> float:
    """PSR against the expected-max-Sharpe benchmark from the trial set.

    ``trial_sharpes`` are the annualized Sharpes of every config tried (e.g. a
    whole parameter sweep) — that's what makes this honest about how many darts
    were thrown. DSR < PSR(0) whenever more than one trial was run.
    """
    sr_star = expected_max_sharpe(trial_sharpes)
    return probabilistic_sharpe_ratio(returns, benchmark_sharpe=sr_star * np.sqrt(252.0))


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray, n_splits: int = 16, max_combos: int = 1000
) -> float:
    """PBO via Combinatorially Symmetric Cross-Validation (CSCV).

    ``returns_matrix`` is (T observations x N configs) of per-bar returns. The
    rows are cut into ``n_splits`` blocks; for every way to choose half as
    in-sample, the IS-best config's out-of-sample rank is measured. PBO is the
    fraction of splits where the IS winner lands below the OOS median — i.e. how
    often "best in backtest" fails to carry over.

    Returns a probability in [0, 1]; ~0.5 means the selection is no better than
    chance (overfit), near 0 means the edge is real.
    """
    m = np.asarray(returns_matrix, dtype=float)
    t, n = m.shape
    if n < 2 or t < n_splits * 2:
        return float("nan")
    if n_splits % 2 != 0:
        n_splits += 1

    block = t // n_splits
    blocks = [m[i * block : (i + 1) * block] for i in range(n_splits)]

    def sr(mat: np.ndarray) -> np.ndarray:
        sd = mat.std(axis=0, ddof=1)
        mean = mat.mean(axis=0)
        return np.where(sd > 0, mean / sd, 0.0)

    all_combos = list(combinations(range(n_splits), n_splits // 2))
    if len(all_combos) > max_combos:
        idx = np.linspace(0, len(all_combos) - 1, max_combos).astype(int)
        all_combos = [all_combos[i] for i in idx]

    logits = []
    for is_idx in all_combos:
        oos_idx = [i for i in range(n_splits) if i not in is_idx]
        is_mat = np.vstack([blocks[i] for i in is_idx])
        oos_mat = np.vstack([blocks[i] for i in oos_idx])
        best = int(np.argmax(sr(is_mat)))
        oos_perf = sr(oos_mat)
        # relative rank of the IS-best config among OOS performance, in (0,1)
        rank = (oos_perf < oos_perf[best]).sum() + 1
        omega = rank / (n + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(np.log(omega / (1.0 - omega)))

    return float(np.mean(np.array(logits) <= 0.0))


def monte_carlo_paths(
    returns: pd.Series | np.ndarray,
    n_sims: int = 1000,
    periods_per_year: float = 252.0,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the return stream into a distribution of outcomes.

    Resamples per-bar returns with replacement ``n_sims`` times and reports the
    spread of annualized Sharpe, CAGR, and max drawdown — so you judge a single
    backtest against the cloud of paths it could have produced.
    """
    r = _as_returns(returns)
    if len(r) < 2:
        return {}
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_sims)
    cagrs = np.empty(n_sims)
    maxdds = np.empty(n_sims)
    n = len(r)
    years = n / periods_per_year
    for i in range(n_sims):
        sample = r[rng.integers(0, n, n)]
        sd = sample.std(ddof=1)
        sharpes[i] = sample.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0
        eq = np.cumprod(1.0 + np.clip(sample, -1.0, None))
        cagrs[i] = eq[-1] ** (1.0 / years) - 1.0 if years > 0 and eq[-1] > 0 else -1.0
        maxdds[i] = float((eq / np.maximum.accumulate(eq) - 1.0).min())

    def pct(a: np.ndarray, p: float) -> float:
        return float(np.percentile(a, p))

    return {
        "sharpe_p05": pct(sharpes, 5),
        "sharpe_median": pct(sharpes, 50),
        "sharpe_p95": pct(sharpes, 95),
        "cagr_p05": pct(cagrs, 5),
        "cagr_median": pct(cagrs, 50),
        "cagr_p95": pct(cagrs, 95),
        "maxdd_p05": pct(maxdds, 5),
        "maxdd_median": pct(maxdds, 50),
        "maxdd_worst": float(maxdds.min()),
        "prob_sharpe_positive": float((sharpes > 0).mean()),
    }
