"""Score past hits by their forward returns — the reason the snapshots are stored.

    python -m ehf.research.reporting.evaluate --source s3://bucket/daily-report
    python -m ehf.research.reporting.evaluate --source ./var/reports

For every stored ``hits.json`` we look up each symbol's close on the hit date
and at +5 / +21 / +63 trading days, compute the return and the excess over SPY,
and summarise by screen and direction. Pure parts (``forward_returns``,
``scorecard``) are unit-tested; data loading talks to S3 / FMP.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ehf.research.reporting.fmp import FMPReportClient

log = logging.getLogger(__name__)
HORIZONS = (5, 21, 63)
INDEX = "SPY"


def hits_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Pure: flatten stored ``hits.json`` payloads (one per day) into a frame."""
    rows: list[dict[str, Any]] = []
    for payload in records:
        as_of = str(payload.get("data_as_of") or payload.get("run_date"))
        for h in payload.get("hits", []):
            rows.append(
                {
                    "as_of": pd.Timestamp(as_of),
                    "run_date": payload.get("run_date"),
                    "symbol": h["symbol"],
                    "screen": h["screen"],
                    "direction": h["direction"],
                    "score": h.get("score"),
                }
            )
    return pd.DataFrame(
        rows, columns=["as_of", "run_date", "symbol", "screen", "direction", "score"]
    )


def forward_returns(
    hits: pd.DataFrame, bars: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Pure: attach ``ret_{h}`` / ``excess_{h}`` (vs SPY) per hit; NaN until the future is in."""
    if hits.empty or bars.empty:
        out = hits.copy()
        for h in horizons:
            out[f"ret_{h}"] = np.nan
            out[f"excess_{h}"] = np.nan
        return out
    close = bars.pivot(index="date", columns="symbol", values="close").sort_index()
    pos = {d: i for i, d in enumerate(close.index)}

    def fwd(sym: str, d: pd.Timestamp, h: int) -> float:
        if sym not in close.columns:
            return float("nan")
        # anchor on the first bar on/after as_of (the hit is known after that close)
        idx = int(close.index.searchsorted(d))
        if idx >= len(close.index) or close.index[idx] != d:
            idx = min(idx, len(close.index) - 1)
        j = idx + h
        if j >= len(close.index):
            return float("nan")
        a, b = close[sym].iloc[idx], close[sym].iloc[j]
        return float(b / a - 1.0) if a and not np.isnan(a) and not np.isnan(b) else float("nan")

    out = hits.copy()
    for h in horizons:
        out[f"ret_{h}"] = [fwd(s, d, h) for s, d in zip(out["symbol"], out["as_of"], strict=True)]
        spy = [fwd(INDEX, d, h) for d in out["as_of"]]
        out[f"excess_{h}"] = out[f"ret_{h}"] - np.array(spy)
    _ = pos  # (kept for clarity; searchsorted does the lookup)
    return out


def scorecard(scored: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Pure: by screen x direction: n, mean/median excess, hit rate (sign-adjusted for shorts)."""
    if scored.empty:
        return pd.DataFrame()
    df = scored.copy()
    sign = np.where(df["direction"] == "short", -1.0, 1.0)
    rows = []
    for (screen, direction), g in df.groupby(["screen", "direction"]):
        row: dict[str, Any] = {"screen": screen, "direction": direction, "n": int(len(g))}
        s = sign[g.index] if isinstance(sign, np.ndarray) else 1.0
        for h in horizons:
            ex = g[f"excess_{h}"].to_numpy(dtype=float) * s
            ex = ex[~np.isnan(ex)]
            row[f"n_{h}"] = int(len(ex))
            row[f"mean_excess_{h}"] = float(ex.mean()) if len(ex) else np.nan
            row[f"median_excess_{h}"] = float(np.median(ex)) if len(ex) else np.nan
            row[f"hit_rate_{h}"] = float((ex > 0).mean()) if len(ex) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["screen", "direction"]).reset_index(drop=True)


# --- loading ------------------------------------------------------------------------------------


def load_hits_local(root: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text())
        for p in sorted(Path(root).rglob("hits.json"))
        if "latest" not in p.parts
    ]


def load_hits_s3(uri: str) -> list[dict[str, Any]]:
    import boto3

    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")
    out: list[dict[str, Any]] = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix.rstrip("/") + "/"
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/hits.json") and "/latest/" not in key:
                out.append(json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read()))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score stored report hits by forward returns")
    p.add_argument(
        "--source", required=True, help="s3://bucket/prefix or a local directory of dated folders"
    )
    p.add_argument("--csv", help="write the per-hit scored table here")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    records = (
        load_hits_s3(args.source)
        if args.source.startswith("s3://")
        else load_hits_local(args.source)
    )
    hits = hits_frame(records)
    if hits.empty:
        print("no hits found")
        return 0
    client = FMPReportClient(os.getenv("FMP_API_KEY", ""))
    start = hits["as_of"].min().date() - timedelta(days=5)
    symbols = sorted({*hits["symbol"].unique(), INDEX})
    bars = client.daily_bars(symbols, start, date.today())
    scored = forward_returns(hits, bars)
    if args.csv:
        scored.to_csv(args.csv, index=False)
    with pd.option_context(
        "display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format
    ):
        print(scorecard(scored))
    return 0


if __name__ == "__main__":
    sys.exit(main())
