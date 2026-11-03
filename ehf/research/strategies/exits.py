"""Pluggable exit components. Shared across strategies, never copy-pasted.

These resolve discrete entry triggers into sustained holdings with target/stop/
time exits. The path-dependent ones scan per symbol (O(T) per column) — exact and
clear for daily data; vectorized math stays in the engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hold_until_target(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float | None = None,
    max_hold: int | None = None,
    gate: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn entry triggers into a 0/1 long-holding matrix.

    On an entry bar (``entries`` truthy and currently flat), enter long at that
    bar's close. Exit when cumulative return from entry reaches ``+target_pct``,
    falls to ``-stop_pct`` (if set), ``max_hold`` bars elapse, or ``gate`` (if
    given) turns False. Returns a frame aligned to ``close``.

    Causal by construction: bar *t*'s holding uses only returns through *t*.
    """
    holding = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for sym in close.columns:
        px = close[sym].to_numpy()
        ent = entries[sym].fillna(0).to_numpy().astype(bool)
        gate_col = None if gate is None else gate[sym].fillna(False).to_numpy().astype(bool)
        holding[sym] = _scan_symbol(px, ent, target_pct, stop_pct, max_hold, gate_col)
    return holding


def _scan_symbol(
    px: np.ndarray,
    ent: np.ndarray,
    target_pct: float,
    stop_pct: float | None,
    max_hold: int | None,
    gate: np.ndarray | None,
) -> np.ndarray:
    out = np.zeros(len(px), dtype=float)
    in_pos = False
    entry_px = np.nan
    bars_held = 0
    for i in range(len(px)):
        if in_pos:
            ret = px[i] / entry_px - 1.0
            bars_held += 1
            hit_target = ret >= target_pct
            hit_stop = stop_pct is not None and ret <= -stop_pct
            timed_out = max_hold is not None and bars_held >= max_hold
            gated_out = gate is not None and not gate[i]
            if hit_target or hit_stop or timed_out or gated_out:
                in_pos = False
                entry_px = np.nan
                bars_held = 0
            else:
                out[i] = 1.0
        if not in_pos and ent[i] and not np.isnan(px[i]):
            in_pos = True
            entry_px = px[i]
            bars_held = 0
            out[i] = 1.0
    return out
