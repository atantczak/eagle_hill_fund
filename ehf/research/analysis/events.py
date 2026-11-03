"""Event studies: what happened in the days after a rule said "the market turned".

A dial's exposure series flips from 0 to 1 on the bar the rule turned bullish. For
every such bar this measures what the leveraged fund and the plain fund did over
the next ``h`` sessions, from the next open (the first price a decision made at
that close could actually trade at), and summarises per horizon: how often the
leveraged fund came out ahead, by how much in the typical case, and how bad the
worst case was. The same statistics over *every* bar are the base rate, so a
signal can be judged against "any day at random" and not only against zero.

Pure pandas; no engine, no costs. The answer to "how long should you hold it?" is
read off the horizon table, not optimised.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

#: Trading-day horizons and the labels the write-up uses.
HORIZONS: dict[int, str] = {
    5: "1 week",
    10: "2 weeks",
    21: "1 month",
    42: "2 months",
    63: "3 months",
    126: "6 months",
    252: "1 year",
    504: "2 years",
}


def turn_events(exposure: pd.Series, *, threshold: float = 0.5) -> pd.DatetimeIndex:
    """Bars where the dial crossed up through ``threshold`` (the first bar of a bull call)."""
    on = (exposure.astype(float) >= threshold).astype(int)
    flips = (on.diff() > 0) & on.astype(bool)
    return pd.DatetimeIndex(exposure.index[flips.to_numpy()])


def forward_returns(open_: pd.Series, events: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """Open-to-open return from the session *after* each event, ``horizon`` sessions on.
    Events too close to the end of the series are dropped (no peeking, no padding)."""
    idx = open_.index
    pos = idx.get_indexer(events)
    pos = pos[pos >= 0]
    entry = pos + 1
    exit_ = entry + horizon
    keep = exit_ < len(idx)
    entry, exit_ = entry[keep], exit_[keep]
    vals = open_.to_numpy(dtype=float)
    out = vals[exit_] / vals[entry] - 1.0
    return pd.Series(out, index=idx[pos[keep]])


def horizon_stats(lev: pd.Series, base: pd.Series) -> dict[str, float]:
    """Summary of paired forward returns (same events) at one horizon."""
    both = pd.concat({"lev": lev, "base": base}, axis=1).dropna()
    if both.empty:
        return {"n": 0.0}
    ex = both["lev"] - both["base"]
    return {
        "n": float(len(both)),
        "lev_median": float(both["lev"].median()),
        "lev_mean": float(both["lev"].mean()),
        "base_median": float(both["base"].median()),
        "base_mean": float(both["base"].mean()),
        "excess_median": float(ex.median()),
        "excess_mean": float(ex.mean()),
        "hit_rate": float((ex > 0).mean()),
        "lev_positive": float((both["lev"] > 0).mean()),
        "lev_worst": float(both["lev"].min()),
        "lev_p10": float(both["lev"].quantile(0.10)),
        "lev_p90": float(both["lev"].quantile(0.90)),
        "base_worst": float(both["base"].min()),
    }


def event_study(
    exposure: pd.Series,
    lev_open: pd.Series,
    base_open: pd.Series,
    *,
    horizons: Sequence[int] = tuple(HORIZONS),
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Per-horizon statistics after each bull call, and the same for every bar (base rate).

    ``exposure``, ``lev_open`` and ``base_open`` share one index. Returns::

        {"n_events": int, "events": [iso dates], "rows": [{"h", "label", <stats>,
         "base_rate_excess_median", "base_rate_hit_rate"}], "best_h": int | None}

    ``best_h`` is the horizon with the highest median excess among those where the
    leveraged fund beat the plain one more often than not; None if there is none.
    """
    exposure = exposure.reindex(lev_open.index)
    events = turn_events(exposure, threshold=threshold)
    all_bars = pd.DatetimeIndex(lev_open.index)
    rows: list[dict[str, Any]] = []
    best: tuple[float, int] | None = None
    for h in horizons:
        st = horizon_stats(
            forward_returns(lev_open, events, h), forward_returns(base_open, events, h)
        )
        bs = horizon_stats(
            forward_returns(lev_open, all_bars, h), forward_returns(base_open, all_bars, h)
        )
        row: dict[str, Any] = {"h": int(h), "label": HORIZONS.get(h, f"{h} sessions"), **st}
        row["base_rate_excess_median"] = bs.get("excess_median", float("nan"))
        row["base_rate_hit_rate"] = bs.get("hit_rate", float("nan"))
        row["base_rate_lev_median"] = bs.get("lev_median", float("nan"))
        rows.append(row)
        if st.get("n", 0) >= 5 and st.get("hit_rate", 0.0) > 0.5:
            cand = (float(st["excess_median"]), int(h))
            if best is None or cand[0] > best[0]:
                best = cand
    return {
        "n_events": int(len(events)),
        "events": [str(e.date()) for e in events],
        "rows": rows,
        "best_h": best[1] if best else None,
    }


def excess_path(
    exposure: pd.Series,
    lev_open: pd.Series,
    base_open: pd.Series,
    *,
    max_h: int = 252,
    step: int = 1,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Median and mean excess (leveraged minus plain) at every horizon up to ``max_h``:
    the curve the write-up draws. Columns: ``median``, ``mean``, ``hit_rate``, ``n``."""
    events = turn_events(exposure.reindex(lev_open.index), threshold=threshold)
    out = []
    for h in range(step, max_h + 1, step):
        st = horizon_stats(
            forward_returns(lev_open, events, h), forward_returns(base_open, events, h)
        )
        out.append(
            {
                "h": h,
                "median": st.get("excess_median", np.nan),
                "mean": st.get("excess_mean", np.nan),
                "hit_rate": st.get("hit_rate", np.nan),
                "n": st.get("n", 0.0),
            }
        )
    return pd.DataFrame(out).set_index("h")
