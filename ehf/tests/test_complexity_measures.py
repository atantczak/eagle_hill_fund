"""Acceptance gauntlet: every measure must recover ground truth on known systems.

White noise / AR(1) / random walk / Lorenz / Hénon / unidirectional coupling.
A measure that can't pass these is not allowed near price data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ehf.research.complexity import reference as ref
from ehf.research.complexity.measures import (
    dfa,
    effective_transfer_entropy,
    hurst,
    mutual_information,
    permutation_entropy,
    transfer_entropy,
    transfer_entropy_matrix,
)

N = 5000


# ---- Permutation entropy -------------------------------------------------


def test_permutation_entropy_white_is_maximal() -> None:
    assert permutation_entropy(ref.white_noise(N, seed=1), order=4) > 0.97


def test_permutation_entropy_chaos_is_low() -> None:
    pe_white = permutation_entropy(ref.white_noise(N, seed=1), order=4)
    pe_lorenz = permutation_entropy(ref.lorenz(N, component=0), order=4)
    pe_henon = permutation_entropy(ref.henon(N), order=4)
    assert pe_lorenz < 0.5  # deterministic chaos -> low ordinal complexity
    assert pe_lorenz < pe_henon < pe_white


def test_permutation_entropy_ar1_below_white() -> None:
    pe_white = permutation_entropy(ref.white_noise(N, seed=1), order=4)
    pe_ar1 = permutation_entropy(ref.ar1(N, phi=0.9, seed=2), order=4)
    assert pe_ar1 < pe_white


# ---- Hurst (R/S) ---------------------------------------------------------


def test_hurst_white_near_half() -> None:
    assert 0.45 < hurst(ref.white_noise(N, seed=1)) < 0.65


def test_hurst_detects_persistence_and_antipersistence() -> None:
    h_white = hurst(ref.white_noise(N, seed=1))
    h_persistent = hurst(ref.ar1(N, phi=0.9, seed=2))
    h_anti = hurst(ref.ar1(N, phi=-0.6, seed=3))
    assert h_persistent > h_white > h_anti
    assert h_persistent > 0.6
    assert h_anti < 0.5


# ---- DFA -----------------------------------------------------------------


def test_dfa_white_and_random_walk() -> None:
    assert 0.4 < dfa(ref.white_noise(N, seed=1)) < 0.6
    assert 1.3 < dfa(ref.random_walk(N, seed=4)) < 1.6  # integrated noise ~ 1.5


def test_dfa_ar1_more_correlated_than_white() -> None:
    assert dfa(ref.ar1(N, phi=0.9, seed=2)) > dfa(ref.white_noise(N, seed=1))


# ---- Transfer entropy ----------------------------------------------------


def test_transfer_entropy_is_directional() -> None:
    x, y = ref.coupled_linear(N, c=0.7, seed=5)
    te_fwd = transfer_entropy(x, y)
    te_bwd = transfer_entropy(y, x)
    assert te_fwd > 0.1
    assert te_fwd > te_bwd + 0.1  # X drives Y, not the reverse


def test_effective_transfer_entropy_kills_bias() -> None:
    x, y = ref.coupled_linear(N, c=0.7, seed=5)
    assert effective_transfer_entropy(x, y) > 0.1  # real flow survives
    assert effective_transfer_entropy(y, x) < 0.05  # wrong direction ~ 0
    xi, yi = ref.white_noise(N, seed=6), ref.white_noise(N, seed=7)
    assert effective_transfer_entropy(xi, yi) < 0.05  # independent ~ 0


def test_transfer_entropy_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        transfer_entropy(np.zeros(10), np.zeros(11))


def test_mutual_information_self_vs_independent() -> None:
    wn = ref.white_noise(N, seed=1)
    assert mutual_information(wn, wn, bins=8) > 2.5  # ~ log2(8) = 3
    assert mutual_information(ref.white_noise(N, seed=6), ref.white_noise(N, seed=7)) < 0.1


def test_transfer_entropy_matrix_shape_and_direction() -> None:
    x, y = ref.coupled_linear(N, c=0.7, seed=5)
    z = ref.white_noise(N, seed=8)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})
    m = transfer_entropy_matrix(df)
    assert list(m.index) == ["X", "Y", "Z"] and list(m.columns) == ["X", "Y", "Z"]
    assert (np.diag(m.to_numpy()) == 0).all()  # diagonal left at 0
    assert m.loc["X", "Y"] > m.loc["Y", "X"]  # flow X -> Y dominates
