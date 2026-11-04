"""Rebalance throttle — trade on a calendar or past a drift band, not every bar.

The engine charges costs on every change in target weight. A vol-targeted book
(TSMOM) nudges every weight every bar, so it trades ~daily on every name — the
first walk-forward study (``wf_tsmom_d07fbfca``) logged ~5,800 trades a year on
23 ETFs. This throttle is the generic fix, applied by ``Strategy.target_weights``
whenever a strategy's params carry either knob:

  * ``rebalance_every`` (bars, >1): only re-read the target every N bars and hold
    the last traded book in between.
  * ``rebalance_band`` (weight, >0): on a rebalance bar, leave a name alone unless
    its target moved more than the band from the held weight. A sign flip
    (long <-> flat <-> short) always trades.

Pure and look-ahead-safe: the held book at bar t is built only from targets at
bars <= t. Absent both knobs, weights pass through untouched (and run hashes of
existing strategies are unchanged, since params only hash what a spec sets).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def throttle(weights: pd.DataFrame, every: int = 1, band: float = 0.0) -> pd.DataFrame:
    """Hold weights between scheduled rebalances and skip changes inside ``band``."""
    every = max(int(every or 1), 1)
    band = float(band or 0.0)
    if every == 1 and band <= 0.0:
        return weights
    target = weights.fillna(0.0).to_numpy(dtype=float)
    held = np.zeros_like(target)
    cur = np.zeros(target.shape[1])
    for t in range(target.shape[0]):
        if t % every == 0:
            want = target[t]
            if band > 0.0:
                flip = np.sign(want) != np.sign(cur)
                move = (np.abs(want - cur) > band) | flip
                cur = np.where(move, want, cur)
            else:
                cur = want.copy()
        held[t] = cur
    return pd.DataFrame(held, index=weights.index, columns=weights.columns)
