"""Information-theoretic measures: transfer entropy and mutual information.

Transfer entropy (Schreiber 2000) measures *directed*, model-free information
flow X -> Y: how much knowing X's past reduces uncertainty about Y's future
beyond Y's own past. This is the highest-value measure in the library — the
basis of the lead-lag application.

Estimator: histogram (binning) with quantile bins, which is robust on
heavy-tailed financial data. Finite-sample TE is positively biased; use
:func:`effective_transfer_entropy` (shuffle-corrected) when an absolute,
bias-corrected value matters. For *directionality* (TE(X->Y) vs TE(Y->X)) the
raw estimator is sufficient since the bias is symmetric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ehf.server.research.complexity.measures.entropy import ArrayLike, _as_array


def _symbolize(a: np.ndarray, bins: int) -> np.ndarray:
    """Map values to integer symbols 0..bins-1 using quantile edges."""
    edges = np.quantile(a, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.zeros(len(a), dtype=int)
    sym = np.digitize(a, edges[1:-1])
    return np.asarray(np.clip(sym, 0, len(edges) - 2), dtype=int)


def transfer_entropy(source: ArrayLike, target: ArrayLike, *, k: int = 1, bins: int = 6) -> float:
    """Transfer entropy TE(source -> target) in bits, history length ``k``.

    TE = sum p(y+, y-, x-) * log2[ p(y+ | y-, x-) / p(y+ | y-) ], where y+ is the
    target's next value, y- its past ``k`` (collapsed to one symbol here for
    k=1), and x- the source's past.
    """
    s = _as_array(source)
    t = _as_array(target)
    if len(s) != len(t):
        raise ValueError("source and target must be the same length")
    if k != 1:
        raise NotImplementedError("only k=1 history is implemented")
    if len(t) < bins * 3:
        return float("nan")

    y_future = _symbolize(t[1:], bins)
    y_past = _symbolize(t[:-1], bins)
    x_past = _symbolize(s[:-1], bins)

    h3 = np.zeros((bins, bins, bins))
    np.add.at(h3, (y_future, y_past, x_past), 1.0)
    total = h3.sum()
    if total == 0:
        return float("nan")
    p3 = h3 / total

    p_bc = p3.sum(axis=0)  # p(y-, x-)
    p_yfyp = p3.sum(axis=2)  # p(y+, y-)
    p_yp = p3.sum(axis=(0, 2))  # p(y-)

    te = 0.0
    for a in range(bins):
        for b in range(bins):
            for c in range(bins):
                pj = p3[a, b, c]
                if pj <= 0 or p_bc[b, c] <= 0 or p_yfyp[a, b] <= 0 or p_yp[b] <= 0:
                    continue
                cond_full = pj / p_bc[b, c]
                cond_marg = p_yfyp[a, b] / p_yp[b]
                te += pj * np.log2(cond_full / cond_marg)
    return float(max(te, 0.0))


def effective_transfer_entropy(
    source: ArrayLike,
    target: ArrayLike,
    *,
    k: int = 1,
    bins: int = 6,
    n_shuffles: int = 20,
    seed: int = 0,
) -> float:
    """Bias-corrected TE: raw TE minus the mean TE of shuffled-source surrogates.

    Shuffling the source destroys X->Y coupling, so the surrogate TE estimates
    the finite-sample bias floor. Effective TE = raw - mean(surrogate), clipped
    at 0.
    """
    raw = transfer_entropy(source, target, k=k, bins=bins)
    if np.isnan(raw):
        return raw
    s = _as_array(source)
    t = _as_array(target)
    rng = np.random.default_rng(seed)
    surrogates = [
        transfer_entropy(rng.permutation(s), t, k=k, bins=bins) for _ in range(n_shuffles)
    ]
    return float(max(raw - float(np.nanmean(surrogates)), 0.0))


def mutual_information(x: ArrayLike, y: ArrayLike, *, bins: int = 8) -> float:
    """Mutual information I(X; Y) in bits (undirected dependence)."""
    a = _symbolize(_as_array(x), bins)
    b = _symbolize(_as_array(y), bins)
    h = np.zeros((bins, bins))
    np.add.at(h, (a, b), 1.0)
    total = h.sum()
    if total == 0:
        return float("nan")
    pxy = h / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px * py)
        terms = pxy * np.log2(ratio)
    return float(np.nansum(terms[pxy > 0]))


def transfer_entropy_matrix(
    returns: pd.DataFrame, *, k: int = 1, bins: int = 6, effective: bool = False
) -> pd.DataFrame:
    """N x N directed TE matrix over a returns panel.

    Entry [i, j] is TE(i -> j): information flowing from the row symbol's past
    into the column symbol's future. The diagonal is left at 0.
    """
    cols = list(returns.columns)
    out = pd.DataFrame(0.0, index=cols, columns=cols)
    est = effective_transfer_entropy if effective else transfer_entropy
    for src in cols:
        for tgt in cols:
            if src == tgt:
                continue
            pair = returns[[src, tgt]].dropna()
            if len(pair) < bins * 3:
                out.loc[src, tgt] = np.nan
                continue
            out.loc[src, tgt] = est(pair[src].to_numpy(), pair[tgt].to_numpy(), k=k, bins=bins)
    return out
