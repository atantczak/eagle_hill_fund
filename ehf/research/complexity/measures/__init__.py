"""Pure complexity measures — the bottom layer (analog of ``ehf.research.signals``).

Scalar, vectorized, fully unit-tested estimators validated against known systems
(white noise, AR(1), Lorenz/Hénon) before they are trusted on price data. The
rolling, signal-shaped wrappers live in ``ehf.research.signals`` so strategies consume
regime measures like any other signal.
"""

from ehf.research.complexity.measures.entropy import permutation_entropy, shannon_entropy
from ehf.research.complexity.measures.information import (
    effective_transfer_entropy,
    mutual_information,
    transfer_entropy,
    transfer_entropy_matrix,
)
from ehf.research.complexity.measures.memory import dfa, hurst

__all__ = [
    "permutation_entropy",
    "shannon_entropy",
    "hurst",
    "dfa",
    "transfer_entropy",
    "effective_transfer_entropy",
    "mutual_information",
    "transfer_entropy_matrix",
]
