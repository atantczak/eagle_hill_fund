"""The leveraged-pair dataset: a plain ETF, its daily-leveraged cousin, and the VIX family.

Dad's question (ideas box, 2026-11-03): when the market turns bullish, does a 3x
fund such as TQQQ beat QQQ, and for how long should you hold it? Everything the
``lev_*`` strategies and the leverage study consume comes through this loader, so
the alignment and the synthetic-history rules live in one place.

  * **Pairs.** ``PAIRS`` names the look-alikes: the plain fund, the leveraged one,
    its multiple, and a plain-English name. Eight pairs ship (Nasdaq-100 at 3x and
    2x, S&P 500 at 3x and 2x, small caps, the Dow, semiconductors, technology).
  * **Sessions are the plain fund's.** The leveraged fund and the indices are
    reindexed onto them (bounded forward-fill), the synthetic ``CASH`` leg accrues
    the 13-week bill, exactly as ``data.vix`` does for SPY.
  * **Extended history.** The 3x funds were born in 2009–2010, so their own record
    is almost all bull market. A daily-leveraged fund is, by construction, the
    plain fund's daily return times the multiple, minus the cost of the borrowed
    money and the fee. ``synthetic_leveraged`` rebuilds that from the plain fund
    and the bill rate; ``fit_drag`` measures the real fund's drag over the years it
    exists so the rebuild uses the fund's own cost, not a textbook guess; and
    ``extend_leveraged`` splices the rebuild in front of the real fund's history
    under the symbol ``<LEV>.X`` (``TQQQ.X``). A run on ``TQQQ.X`` therefore
    covers 1999 onward, with the real fund wherever it existed.

Nothing here is stored in the price cache: synthetic bars are rebuilt on load,
from cached inputs, so they always reflect the inputs' latest back-adjustment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ehf.research.data.provider import DataProvider
from ehf.research.data.schema import BARS_COLUMNS, Timeframe, validate_bars
from ehf.research.data.vix import (
    CASH,
    INDEX_SYMBOLS,
    IRX,
    align_to_sessions,
    load_vix_bars,
)

log = logging.getLogger(__name__)

EXTENDED_SUFFIX = ".X"
PERIODS_PER_YEAR = 252.0


@dataclass(frozen=True)
class Pair:
    base: str
    lev: str
    leverage: float
    name: str
    expense_ratio: float = (
        0.0095  # the leveraged fund's annual fee (ProShares / Direxion ~0.9-1.0%)
    )

    @property
    def key(self) -> str:
        return f"{self.base}/{self.lev}"

    @property
    def extended(self) -> str:
        return self.lev + EXTENDED_SUFFIX


PAIRS: dict[str, Pair] = {
    p.key: p
    for p in (
        Pair("QQQ", "TQQQ", 3.0, "Nasdaq-100, 3x"),
        Pair("QQQ", "QLD", 2.0, "Nasdaq-100, 2x"),
        Pair("SPY", "UPRO", 3.0, "S&P 500, 3x"),
        Pair("SPY", "SSO", 2.0, "S&P 500, 2x"),
        Pair("IWM", "TNA", 3.0, "Russell 2000 small caps, 3x"),
        Pair("DIA", "UDOW", 3.0, "Dow Jones Industrials, 3x"),
        Pair("SOXX", "SOXL", 3.0, "Semiconductors, 3x"),
        Pair("XLK", "TECL", 3.0, "Technology sector, 3x"),
    )
}
DEFAULT_PAIR = "QQQ/TQQQ"


def pair_for(key: str) -> Pair:
    try:
        return PAIRS[key.upper()]
    except KeyError:
        raise KeyError(f"unknown pair {key!r}; known: {', '.join(PAIRS)}") from None


def pair_by_symbol(symbol: str) -> Pair | None:
    """The pair whose leveraged fund (real or ``.X``) this is, if any."""
    s = real_symbol(symbol)
    for pair in PAIRS.values():
        if pair.lev == s:
            return pair
    return None


def is_extended(symbol: str) -> bool:
    return symbol.upper().endswith(EXTENDED_SUFFIX)


def real_symbol(symbol: str) -> str:
    """``TQQQ.X`` -> ``TQQQ``; a plain symbol is returned as is."""
    s = symbol.upper()
    return s[: -len(EXTENDED_SUFFIX)] if is_extended(s) else s


# --- the synthetic fund ----------------------------------------------------------------


def _one(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = bars.loc[bars["symbol"] == symbol].sort_values("timestamp").set_index("timestamp")
    if out.empty:
        raise ValueError(f"no bars for {symbol}")
    return out


def daily_drag(
    irx_pct: pd.Series, leverage: float, expense_ratio: float, spread: float
) -> pd.Series:
    """Per-session cost of running the fund: fee, plus borrowing ``leverage - 1`` times the
    money at the bill rate plus a spread. ``irx_pct`` is the 13-week yield in percent,
    as of the previous session (nothing is known at the open that was not known at the
    prior close)."""
    y = irx_pct.clip(lower=0.0, upper=25.0).ffill().fillna(0.0).shift(1).fillna(0.0) / 100.0
    annual = expense_ratio + (leverage - 1.0) * (y + spread)
    return annual / PERIODS_PER_YEAR


def synthetic_leveraged(
    base: pd.DataFrame,
    irx_pct: pd.Series,
    *,
    leverage: float,
    symbol: str,
    expense_ratio: float = 0.0095,
    spread: float = 0.0,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Rebuild a daily-rebalanced leveraged fund from its plain fund's bars.

    Close-to-close: ``1 + L * r_base - drag``. The open applies the multiple to the
    plain fund's overnight gap, so a next-open fill on the synthetic behaves like
    one on the real thing. High/low are the open/close envelope (the engine never
    reads them). Returns canonical bars for ``symbol``.
    """
    b = base.sort_index()
    close_b = b["close"].astype(float)
    open_b = b["open"].astype(float)
    drag = daily_drag(irx_pct.reindex(b.index), leverage, expense_ratio, spread)
    r = close_b.pct_change().fillna(0.0)
    growth = (1.0 + leverage * r - drag).clip(lower=0.01)
    close = start_price * growth.cumprod()
    prev_close = close.shift(1)
    gap = (open_b / close_b.shift(1) - 1.0).fillna(0.0)
    open_ = (prev_close * (1.0 + leverage * gap)).fillna(close)
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": b.index,
            "open": open_.to_numpy(),
            "high": np.maximum(open_.to_numpy(), close.to_numpy()),
            "low": np.minimum(open_.to_numpy(), close.to_numpy()),
            "close": close.to_numpy(),
            "volume": 0.0,
        }
    )
    return out.reset_index(drop=True)


def fit_drag(
    real: pd.DataFrame, base: pd.DataFrame, irx_pct: pd.Series, *, leverage: float
) -> dict[str, Any]:
    """How the real fund tracked ``leverage`` times its plain fund, over the days both exist.

    Returns the realised beta of the fund's daily return on the plain fund's (its
    *effective* multiple, a shade under the nominal one because fund prices carry
    a little noise around net asset value), the correlation, the annualised drag
    beyond ``beta * base`` (fees, financing, tracking), the share of that drag the
    bill rate explains, and the ``spread`` (possibly negative) the synthetic must
    add to the bill rate so its drag equals the fund's, fee included. Everything is
    close-to-close.
    """
    idx = real.index.intersection(base.index)
    if len(idx) < 60:
        raise ValueError("fewer than 60 overlapping sessions")
    r_real = real.loc[idx, "close"].astype(float).pct_change().dropna()
    r_base = base.loc[idx, "close"].astype(float).pct_change().dropna()
    idx = r_real.index.intersection(r_base.index)
    r_real, r_base = r_real.loc[idx], r_base.loc[idx]
    beta = float(np.polyfit(r_base.to_numpy(), r_real.to_numpy(), 1)[0])
    corr = float(np.corrcoef(r_base.to_numpy(), r_real.to_numpy())[0, 1])
    resid = r_real - beta * r_base
    drag_annual = float(-resid.mean() * PERIODS_PER_YEAR)
    y = irx_pct.reindex(idx).clip(lower=0.0, upper=25.0).ffill().fillna(0.0) / 100.0
    financing = float((beta - 1.0) * y.mean())
    return {
        "leverage": leverage,
        "beta": beta,
        "corr": corr,
        "drag_annual": drag_annual,
        "financing_annual": financing,
        "sessions": float(len(idx)),
        "years": len(idx) / PERIODS_PER_YEAR,
        "start": str(idx[0].date()),
        "end": str(idx[-1].date()),
    }


def implied_spread(fit: dict[str, Any], expense_ratio: float) -> float:
    """The spread over the bill rate that makes the synthetic's drag equal the fund's."""
    return float(
        (fit["drag_annual"] - fit["financing_annual"] - expense_ratio)
        / max(fit["beta"] - 1.0, 1e-9)
    )


def extend_leveraged(
    base: pd.DataFrame,
    real: pd.DataFrame,
    irx_pct: pd.Series,
    *,
    leverage: float,
    symbol: str,
    expense_ratio: float = 0.0095,
    spread: float | None = None,
    effective: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The real fund's history with a synthetic rebuild spliced in front of it.

    The rebuild uses the fund's measured effective multiple (``effective``; the
    nominal one otherwise) and, unless ``spread`` is given, the drag ``fit_drag``
    measured, so the synthetic years carry the fund's own cost and bumpiness. It is
    scaled so its close on the fund's first real session equals the fund's own;
    before that day the series is synthetic, from that day on it is the fund.
    Returns the bars and the fit, with ``growth_real`` / ``growth_synthetic`` over
    the overlap as the validation (how far the rebuild lands from the real thing).
    """
    fit = fit_drag(real, base, irx_pct, leverage=leverage)
    mult = fit["beta"] if effective else leverage
    sp = implied_spread(fit, expense_ratio) if spread is None else spread
    fit.update({"multiple_used": mult, "spread_used": sp, "expense_ratio": expense_ratio})
    syn = synthetic_leveraged(
        base, irx_pct, leverage=mult, symbol=symbol, expense_ratio=expense_ratio, spread=sp
    ).set_index("timestamp")
    first = real.index[0]
    last = real.index[-1]
    fit["growth_real"] = float(real.loc[last, "close"] / real.loc[first, "close"])
    fit["growth_synthetic"] = float(syn.loc[last, "close"] / syn.loc[first, "close"])
    before = syn.loc[syn.index < first]
    scale = float(real.loc[first, "close"]) / float(syn.loc[first, "close"])
    before = before.copy()
    for col in ("open", "high", "low", "close"):
        before[col] = before[col] * scale
    after = real.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    out = pd.concat([before[["open", "high", "low", "close", "volume"]], after])
    out["symbol"] = symbol
    out.index.name = "timestamp"
    fit["synthetic_start"] = str(out.index[0].date())
    fit["real_start"] = str(first.date())
    return out.reset_index()[list(BARS_COLUMNS)], fit


# --- the dataset -----------------------------------------------------------------------


def assemble_pair_bars(
    vix_bars: pd.DataFrame,
    lev_bars: pd.DataFrame,
    pair: Pair,
    *,
    extend: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pure assembly: the VIX-style dataset built around ``pair.base`` plus the
    leveraged fund (aligned to the base's sessions) and, with ``extend``, its
    ``.X`` extension. Returns the bars and a facts dict (coverage, the drag fit)."""
    sessions = pd.DatetimeIndex(
        vix_bars.loc[vix_bars["symbol"] == pair.base, "timestamp"].sort_values().unique()
    )
    lev = lev_bars.loc[lev_bars["symbol"] == pair.lev]
    if lev.empty:
        raise ValueError(f"no bars for {pair.lev}")
    lev = align_to_sessions(lev, sessions)
    lev = lev.dropna(subset=["close"])  # before inception there is nothing to carry forward
    pieces = [vix_bars, lev]
    facts: dict[str, Any] = {
        "pair": pair.key,
        "base": pair.base,
        "lev": pair.lev,
        "leverage": pair.leverage,
        "base_start": str(sessions[0].date()),
        "lev_start": str(lev["timestamp"].min().date()),
        "end": str(sessions[-1].date()),
    }
    if extend:
        wide_irx = vix_bars.loc[vix_bars["symbol"] == IRX].set_index("timestamp")["close"]
        base = _one(vix_bars, pair.base)
        real = _one(lev, pair.lev)
        ext, fit = extend_leveraged(
            base,
            real,
            wide_irx.reindex(sessions),
            leverage=pair.leverage,
            symbol=pair.extended,
            expense_ratio=pair.expense_ratio,
        )
        pieces.append(ext)
        facts["fit"] = fit
        facts["extended"] = pair.extended
    out = pd.concat(pieces, ignore_index=True)[list(BARS_COLUMNS)]
    out["symbol"] = out["symbol"].astype("string").str.upper()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = out[col].astype("float64")
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out.attrs["adjusted"] = bool(vix_bars.attrs.get("adjusted", True))
    out.attrs["timeframe"] = str(vix_bars.attrs.get("timeframe", Timeframe.DAY.value))
    return validate_bars(out), facts


def load_pair_bars(
    provider: DataProvider,
    pair: Pair | str,
    start: str | date,
    end: str | date,
    *,
    extend: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch (cached) everything a leverage run needs and assemble it.

    Symbols in the result: the plain fund, ``CASH``, the VIX family (``^VIX``,
    ``^VIX3M``, ``^VIX9D``, ``^VVIX``, ``^IRX``), the leveraged fund, and with
    ``extend`` its ``.X`` extension. No SKEW (no rule here reads it).
    """
    p = pair if isinstance(pair, Pair) else pair_for(pair)
    vix_bars = load_vix_bars(provider, start, end, asset=p.base, include_skew=False)
    lev_bars = provider.get_bars([p.lev], start, end, Timeframe.DAY)
    bars, facts = assemble_pair_bars(vix_bars, lev_bars, p, extend=extend)
    log.info(
        "pair dataset %s: %d sessions x %s",
        p.key,
        bars["timestamp"].nunique(),
        sorted(bars["symbol"].unique()),
    )
    return bars, facts


def pair_universe(pair: Pair, *, extended: bool) -> tuple[str, ...]:
    """The symbols a leverage ``RunSpec`` names (what ``load_pair_bars`` emits, minus the
    leveraged series the run does not use)."""
    lev = pair.extended if extended else pair.lev
    return (pair.base, lev, CASH, *INDEX_SYMBOLS)


def select_symbols(bars: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Keep only ``symbols`` (a run should not see the other leveraged series)."""
    out = bars.loc[bars["symbol"].isin(symbols)].reset_index(drop=True)
    out.attrs.update(bars.attrs)
    return out
