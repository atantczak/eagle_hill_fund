"""Ten strategies that read FMP's non-price data (see ``data.altdata``).

Six pick S&P 500 stocks (long-only, equal weight, members on the day only):

  * ``pead``              earnings surprise drift: buy big beats, hold a few weeks
  * ``grades_momo``       net analyst upgrades over a window
  * ``fscore_value``      cheapest by earnings yield among high Piotroski F-scores
  * ``insider_cluster``   several insiders buying on the open market
  * ``congress_follow``   net buying disclosed by members of Congress
  * ``shareholder_yield`` most cash returned (dividends + net buybacks) / market cap

Two are in/out dials on an index ETF vs T-bills:

  * ``curve_dial``        the 10-year minus 3-month Treasury spread (inversions)
  * ``claims_dial``       initial jobless claims rising off their low

Two rotate between ETFs on a macro gauge:

  * ``cugo_rotation``     copper/gold ratio: cyclical sectors vs defensive sectors
  * ``dollar_em``         dollar index trend: overseas stocks vs the S&P 500

Plus ``member_ew``: equal weight across the S&P 500 members of the day, the
benchmark the stock pickers must beat. Feature columns are ``^<FEAT>.<SYM>``
pseudo-symbols that the engine never trades.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from ehf.research import signals
from ehf.research.backtest.panel import Panel
from ehf.research.data.altdata import CYCLICALS, DEFENSIVES
from ehf.research.strategies.base import Strategy, register
from ehf.research.strategies.dial import DIAL_DEFAULTS, DialStrategy, hysteresis_gate


def features(panel: Panel, name: str) -> pd.DataFrame:
    """Feature ``name`` as a frame with one column per stock symbol."""
    pre = f"^{name}."
    cols = [c for c in panel.close.columns if c.startswith(pre)]
    df = panel.close[cols].copy()
    df.columns = [c[len(pre) :] for c in cols]
    return df


def tradable(panel: Panel) -> list[str]:
    return [c for c in panel.close.columns if not str(c).startswith("^") and c != "CASH"]


def members(panel: Panel) -> pd.DataFrame:
    """True where a stock is in the S&P 500 that day and has a price."""
    syms = tradable(panel)
    mem = features(panel, "MEM").ffill().reindex(columns=syms).fillna(0.0) > 0.5
    return mem & panel.close[syms].notna()


class StockPicker(Strategy):
    """Equal weight across ``select(panel)`` (restricted to members of the day)."""

    defaults: dict[str, Any] = {"gross_exposure": 1.0}

    def select(self, panel: Panel) -> pd.DataFrame:
        raise NotImplementedError

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        mem = members(panel)
        pick = self.select(panel).reindex(index=mem.index, columns=mem.columns)
        return (pick.fillna(False).astype(bool) & mem).astype(float)


def _top_quantile(score: pd.DataFrame, eligible: pd.DataFrame, q: float) -> pd.DataFrame:
    s = score.where(eligible)
    ranks = s.rank(axis=1, pct=True, method="first")
    return ranks > 1.0 - q


@register("member_ew")
class MemberEqualWeight(StockPicker):
    """Equal weight across every S&P 500 member of the day (the stock-picking benchmark)."""

    def select(self, panel: Panel) -> pd.DataFrame:
        return members(panel)


@register("pead")
class EarningsDrift(StockPicker):
    """Buy right after an earnings beat of at least ``min_surprise``; hold ``hold`` days."""

    defaults = {**StockPicker.defaults, "min_surprise": 0.05, "hold": 40}

    def select(self, panel: Panel) -> pd.DataFrame:
        sue = features(panel, "SUE")
        hit = (sue >= float(self.p("min_surprise"))).astype(float)
        return hit.rolling(int(self.p("hold")), min_periods=1).max() > 0


@register("grades_momo")
class AnalystUpgrades(StockPicker):
    """Hold stocks whose net analyst upgrades over ``window`` days are at least ``min_net``."""

    defaults = {**StockPicker.defaults, "window": 63, "min_net": 2}

    def select(self, panel: Panel) -> pd.DataFrame:
        net = features(panel, "GRD").fillna(0.0).rolling(int(self.p("window")), min_periods=1).sum()
        return net >= float(self.p("min_net"))


@register("fscore_value")
class QualityValue(StockPicker):
    """Cheapest ``quantile`` by earnings yield among F-scores >= ``min_fscore``."""

    defaults = {**StockPicker.defaults, "min_fscore": 7, "quantile": 0.2}

    def select(self, panel: Panel) -> pd.DataFrame:
        fs = features(panel, "FS").ffill(limit=300)
        ey = features(panel, "EY").ffill(limit=45)
        mem = members(panel)
        fs, ey = fs.reindex_like(mem), ey.reindex_like(mem)
        ok = mem & (fs >= float(self.p("min_fscore"))) & ey.notna() & (ey > 0)
        return _top_quantile(ey, ok, float(self.p("quantile")))


@register("insider_cluster")
class InsiderBuying(StockPicker):
    """Buy when at least ``min_buyers`` open-market insider purchases land within ``window``
    days; hold ``hold`` days from the last trigger."""

    defaults = {**StockPicker.defaults, "window": 21, "min_buyers": 2, "hold": 63}

    def select(self, panel: Panel) -> pd.DataFrame:
        buys = (
            features(panel, "INS").fillna(0.0).rolling(int(self.p("window")), min_periods=1).sum()
        )
        trig = (buys >= float(self.p("min_buyers"))).astype(float)
        return trig.rolling(int(self.p("hold")), min_periods=1).max() > 0


@register("congress_follow")
class CongressFollow(StockPicker):
    """Hold stocks with net Congress buys >= ``min_net`` disclosed within ``window`` days."""

    defaults = {**StockPicker.defaults, "window": 63, "min_net": 2}

    def select(self, panel: Panel) -> pd.DataFrame:
        net = features(panel, "CNG").fillna(0.0).rolling(int(self.p("window")), min_periods=1).sum()
        return net >= float(self.p("min_net"))


@register("shareholder_yield")
class ShareholderYield(StockPicker):
    """Top ``quantile`` by dividends + net buybacks over market cap (F >= ``min_fscore``)."""

    defaults = {**StockPicker.defaults, "quantile": 0.2, "min_fscore": 0}

    def select(self, panel: Panel) -> pd.DataFrame:
        mem = members(panel)
        py = features(panel, "PY").ffill(limit=45).reindex_like(mem)
        ok = mem & py.notna() & (py > 0)
        if float(self.p("min_fscore")) > 0:
            fs = features(panel, "FS").ffill(limit=300).reindex_like(mem)
            ok &= fs >= float(self.p("min_fscore"))
        return _top_quantile(py, ok, float(self.p("quantile")))


# --- macro dials ------------------------------------------------------------------------


def _trend_ok(px: pd.Series, on: bool, window: int = 200) -> pd.Series:
    """With ``on``, a macro 'out' only counts when price is also under its average."""
    if not on:
        return pd.Series(True, index=px.index)
    return px < cast(pd.Series, signals.sma(px, window))


@register("curve_dial")
class CurveDial(DialStrategy):
    """Out of the asset around yield-curve inversions (10y minus 3m below ``threshold``)."""

    defaults = {
        **DIAL_DEFAULTS,
        "threshold": 0.0,
        "mode": "inverted",
        "memory": 252,
        "sma_confirm": False,
    }

    def exposure(self, panel: Panel) -> pd.Series:
        spread = panel.close["^T10Y3M"].ffill()
        thr = float(self.p("threshold"))
        recent = spread.rolling(int(self.p("memory")), min_periods=1).min() < thr
        mode = str(self.p("mode"))
        if mode == "inverted":
            danger = spread < thr
        elif mode == "after":  # the re-steepening after an inversion, when recessions start
            danger = recent & (spread >= thr)
        else:  # "both": inverted now or recently
            danger = recent
        px = panel.close[str(self.p("asset"))]
        out = danger & _trend_ok(px, bool(self.p("sma_confirm")))
        return (~out).astype(float).where(spread.notna(), 1.0)


@register("claims_dial")
class ClaimsDial(DialStrategy):
    """Out when 4-week jobless claims rise ``threshold`` above their ``lookback``-day low."""

    defaults = {
        **DIAL_DEFAULTS,
        "lookback": 252,
        "threshold": 0.15,
        "exit_frac": 0.5,
        "sma_confirm": False,
    }

    def exposure(self, panel: Panel) -> pd.Series:
        c = panel.close["^CLAIMS4W"].ffill()
        rise = c / c.rolling(int(self.p("lookback")), min_periods=20).min() - 1.0
        thr = float(self.p("threshold"))
        gate = hysteresis_gate(
            rise.fillna(0.0), out_above=thr, in_below=thr * float(self.p("exit_frac"))
        )
        px = panel.close[str(self.p("asset"))]
        if bool(self.p("sma_confirm")):
            gate = gate.where(~((gate < 0.5) & ~_trend_ok(px, True)), 1.0)
        return gate


# --- macro rotations -------------------------------------------------------------------------


def _equal(cols: list[str], mask: pd.Series, panel: Panel) -> pd.DataFrame:
    live = panel.close[cols].notna().astype(float)
    w = live.div(live.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return w.mul(mask.astype(float), axis=0)


@register("cugo_rotation")
class CopperGoldRotation(Strategy):
    """Cyclical sectors when copper is beating gold, defensive sectors when gold leads."""

    defaults = {"lookback": 63, "mode": "mom"}

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        r = panel.close["^CUGO"].ffill()
        n = int(self.p("lookback"))
        risk_on = (r / r.shift(n) - 1.0 > 0) if self.p("mode") == "mom" else (r > signals.sma(r, n))
        cyc = [c for c in CYCLICALS if c in panel.close.columns]
        dfn = [c for c in DEFENSIVES if c in panel.close.columns]
        w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
        w[cyc] = _equal(cyc, risk_on, panel)
        w[dfn] = _equal(dfn, ~risk_on, panel)
        return w

    def size_positions(self, signals_df: pd.DataFrame, panel: Panel) -> pd.DataFrame:
        return signals_df


@register("dollar_em")
class DollarTilt(Strategy):
    """Overseas stocks while the dollar is under its ``window``-day average, else SPY."""

    defaults = {"window": 100, "band": 0.0, "foreign": "EEM"}

    def generate_signals(self, panel: Panel) -> pd.DataFrame:
        dxy = panel.close["^DXY"].ffill()
        ma = cast(pd.Series, signals.sma(dxy, int(self.p("window"))))
        weak = hysteresis_gate(
            dxy / ma - 1.0, out_above=float(self.p("band")), in_below=-float(self.p("band"))
        )  # 1 while the dollar sits below its average
        f = str(self.p("foreign"))
        legs = ["EEM", "EFA"] if f == "both" else [f]
        w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
        for leg in legs:
            w[leg] = weak / len(legs)
        w["SPY"] = 1.0 - weak
        return w

    def size_positions(self, signals_df: pd.DataFrame, panel: Panel) -> pd.DataFrame:
        return signals_df
