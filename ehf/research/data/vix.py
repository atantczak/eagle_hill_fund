"""The VIX-overlay dataset: SPY + the VIX family + SKEW + a synthetic cash leg.

One loader builds the canonical bars every ``vix_*`` dial strategy consumes, so
the alignment rules live in exactly one place:

  * **Sessions are SPY's.** Index series (``^VIX``, ``^VIX3M``, ``^VIX9D``,
    ``^VVIX``, ``^IRX``) are reindexed onto SPY's trading days and forward-filled
    (a stale fear reading is a better default than a NaN that flattens the book).
    Days where only an index prints are dropped — nothing is tradable there.
  * **SKEW** comes from Cboe (``data.cboe``) and is lifted into a bar row whose
    OHLC all equal the index level, volume 0.
  * **``CASH``** is a synthetic T-bill whose price accrues the 13-week yield
    (``^IRX``, a percent): ``price_t = price_{t-1} * (1 + irx_{t-1}/100/252)``.
    Open == close, so a NEXT_OPEN fill earns exactly the accrual. This is what
    lets "out of the market" earn the bill rate instead of zero.

Everything downstream is schema-only: the engine sees a bars frame with a few
extra symbols, and the dial strategies put weight on ``SPY`` and ``CASH`` and
zero on the index columns (they are signals, not positions).
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from ehf.research.data.cboe import skew_history
from ehf.research.data.provider import DataProvider
from ehf.research.data.schema import BARS_COLUMNS, Timeframe, validate_bars

log = logging.getLogger(__name__)

ASSET = "SPY"
CASH = "CASH"
VIX = "^VIX"
VIX3M = "^VIX3M"
VIX9D = "^VIX9D"
VVIX = "^VVIX"
IRX = "^IRX"
SKEW = "^SKEW"
INDEX_SYMBOLS: tuple[str, ...] = (VIX, VIX3M, VIX9D, VVIX, IRX)
#: Every symbol the loader emits — the universe a ``vix_*`` RunSpec should name.
VIX_UNIVERSE: tuple[str, ...] = (ASSET, CASH, SKEW, *INDEX_SYMBOLS)

_FFILL_LIMIT = 10  # sessions an index value may be carried forward before going NaN


def _wide(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    w = bars.pivot(index="timestamp", columns="symbol", values=field).sort_index()
    w.columns.name = None
    return w


def cash_bars_from_yield(irx: pd.Series, *, periods_per_year: float = 252.0) -> pd.DataFrame:
    """Synthetic ``CASH`` bars accruing yesterday's 13-week yield (percent) each session."""
    y = irx.clip(lower=0.0, upper=25.0).ffill().fillna(0.0)
    daily = (y.shift(1).fillna(0.0) / 100.0) / periods_per_year
    price = 100.0 * (1.0 + daily).cumprod()
    return pd.DataFrame(
        {
            "symbol": CASH,
            "timestamp": price.index,
            "open": price.to_numpy(),
            "high": price.to_numpy(),
            "low": price.to_numpy(),
            "close": price.to_numpy(),
            "volume": 0.0,
        }
    )


def index_bars_from_series(symbol: str, values: pd.Series) -> pd.DataFrame:
    """Lift a level-only series (e.g. SKEW) into flat OHLC bars."""
    v = values.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": values.index,
            "open": v,
            "high": v,
            "low": v,
            "close": v,
            "volume": 0.0,
        }
    )


def align_to_sessions(bars: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex every symbol in ``bars`` onto ``sessions`` with a bounded forward-fill."""
    out = []
    for sym, grp in bars.groupby("symbol", sort=False):
        g = grp.set_index("timestamp").reindex(sessions)
        g[["open", "high", "low", "close", "volume"]] = g[
            ["open", "high", "low", "close", "volume"]
        ].ffill(limit=_FFILL_LIMIT)
        g["symbol"] = sym
        g.index.name = "timestamp"
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def session_dates(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC bar stamps -> naive America/New_York calendar dates (what Cboe keys on)."""
    return pd.DatetimeIndex(sessions.tz_convert("America/New_York").normalize().tz_localize(None))


def assemble_vix_bars(
    asset_bars: pd.DataFrame,
    index_bars: pd.DataFrame,
    skew: pd.Series | None,
    *,
    asset: str = ASSET,
) -> pd.DataFrame:
    """Pure assembly (no I/O): align, add SKEW + CASH, validate. Unit-testable."""
    sessions = pd.DatetimeIndex(
        asset_bars.loc[asset_bars["symbol"] == asset, "timestamp"].sort_values().unique()
    )
    if sessions.empty:
        raise ValueError(f"no {asset} bars to anchor sessions on")
    pieces = [asset_bars.loc[asset_bars["symbol"] == asset]]
    if not index_bars.empty:
        pieces.append(align_to_sessions(index_bars, sessions))

    if skew is not None and not skew.empty:
        dates = session_dates(sessions)
        lifted = pd.Series(skew.reindex(dates).ffill(limit=_FFILL_LIMIT).to_numpy(), index=sessions)
        pieces.append(index_bars_from_series(SKEW, lifted))

    irx = _wide(pieces[1], "close")[IRX] if IRX in set(index_bars["symbol"]) else None
    if irx is not None:
        pieces.append(cash_bars_from_yield(irx.reindex(sessions)))
    else:
        log.warning("no %s in index bars; CASH will accrue 0%%", IRX)
        pieces.append(cash_bars_from_yield(pd.Series(0.0, index=sessions)))

    out = pd.concat(pieces, ignore_index=True)[list(BARS_COLUMNS)]
    out["symbol"] = out["symbol"].astype("string").str.upper()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = out[col].astype("float64")
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out.attrs["adjusted"] = bool(asset_bars.attrs.get("adjusted", True))
    out.attrs["timeframe"] = str(asset_bars.attrs.get("timeframe", Timeframe.DAY.value))
    return validate_bars(out)


def load_vix_bars(
    provider: DataProvider,
    start: str | date,
    end: str | date,
    *,
    asset: str = ASSET,
    include_skew: bool = True,
    offline_skew: bool = False,
) -> pd.DataFrame:
    """Fetch (cached) SPY + VIX family, pull SKEW from Cboe, and assemble the dataset."""
    raw = provider.get_bars([asset, *INDEX_SYMBOLS], start, end, Timeframe.DAY)
    asset_bars = raw.loc[raw["symbol"] == asset].reset_index(drop=True)
    asset_bars.attrs.update(raw.attrs)
    index_bars = raw.loc[raw["symbol"] != asset].reset_index(drop=True)
    skew = None
    if include_skew:
        try:
            skew = skew_history(offline=offline_skew)
        except Exception as exc:  # the study can run without rule 9
            log.warning("SKEW unavailable (%s); ^SKEW column will be absent", exc)
    bars = assemble_vix_bars(asset_bars, index_bars, skew, asset=asset)
    have = sorted(bars["symbol"].unique())
    log.info("vix dataset: %d sessions x %s", bars["timestamp"].nunique(), have)
    return bars


def coverage(bars: pd.DataFrame) -> pd.DataFrame:
    """First/last non-NaN close per symbol — a quick look at who starts when."""
    rows = []
    for sym, grp in bars.groupby("symbol"):
        ok = grp.dropna(subset=["close"])
        rows.append(
            {
                "symbol": sym,
                "first": ok["timestamp"].min().date() if not ok.empty else None,
                "last": ok["timestamp"].max().date() if not ok.empty else None,
                "n": int(len(ok)),
                "nan_share": float(np.isnan(grp["close"]).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")
