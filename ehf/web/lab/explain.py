"""The short version of a run: what was tested, the rule in plain words, what happened.

``explain(payload)`` reads the JSON ``run_payload`` builds and returns a small block the
run page puts above the charts (the register of the VIX explainer: real numbers, every
term explained, the verdict first). Every string in the block is HTML built here from
trusted literals; anything that came from a spec (a symbol, a parameter) is escaped.

One entry per registered strategy, written against the strategy's own parameters so
the sentences carry the numbers that actually ran (``window=200`` becomes "its 200-day
average"). A walk-forward run names the grid the settings were re-chosen from, and the
gauntlet's verdict is spelled out check by check.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from html import escape as h
from typing import Any, TypeGuard

from ehf.research import strategies  # noqa: F401  (import registers strategies)
from ehf.research.data.leverage import is_extended, pair_by_symbol, real_symbol
from ehf.research.strategies.base import _REGISTRY
from ehf.research.strategies.dial import DialStrategy
from ehf.research.strategies.leverage import LeverageDial

NAMES: dict[str, str] = {
    "dial_hold": "Buy and hold",
    "dial_sma": "The 200-day line",
    "vix_term": "Fear curve flip",
    "vix_resid": "Unexplained fear",
    "vix_vrp": "Overpriced insurance",
    "vix_spike": "Spike and fade",
    "vix_sized": "Size by fear",
    "vix_surprise": "Surprise count",
    "vix_corr": "Broken link",
    "vix_vvix": "Fear of fear",
    "vix_skew": "Insurance mix",
    "vix_vote": "The vote",
    "lev_hold": "Hold the leveraged fund",
    "lev_sma": "Above the average",
    "lev_cross": "Golden cross",
    "lev_swing": "Bull and bear swings",
    "lev_breakout": "Breakout",
    "lev_vixfade": "After the storm",
    "lev_voltarget": "Volatility target",
    "tsmom": "Trend following",
    "tsmom_regime": "Trend following, regime-gated",
    "xsmom": "Winners minus losers",
    "bollinger_reversion": "Buy the deep dip",
    "rise_fall": "Buy small dips in an uptrend",
}
RULE_NUMBER: dict[str, int] = {
    "vix_term": 1,
    "vix_resid": 2,
    "vix_vrp": 3,
    "vix_spike": 4,
    "vix_sized": 5,
    "vix_surprise": 6,
    "vix_corr": 7,
    "vix_vvix": 8,
    "vix_skew": 9,
    "vix_vote": 10,
}

#: The gauntlet's checks, in words a reader who has never heard of them can follow.
CHECKS: dict[str, str] = {
    "windows": "too few test windows to judge",
    "oos_sharpe": "the out-of-sample Sharpe ratio is below the bar",
    "psr": "the Sharpe ratio could plausibly be luck",
    "dsr": "the Sharpe ratio does not survive a correction for how many settings were tried",
    "efficiency": "it did far worse out of sample than in the training windows",
    "pbo": "the chosen settings look overfit (high probability of backtest overfitting)",
    "cost_stress": "it stops making money once trading costs are doubled",
    "trades": "too few trades to trust the numbers",
    "windows_pos": "it lost money in too many of the test windows",
    "regimes": "it did not hold up across bull, bear and sideways markets",
}


# -- formatting ------------------------------------------------------------------------


def _fin(v: object) -> TypeGuard[float]:
    """A usable number: int or float, finite, not a bool (``None`` and NaN are not)."""
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(float(v))


def pct(v: Any, d: int = 1) -> str:
    return f"{float(v) * 100:.{d}f}%" if _fin(v) else "n/a"


def num(v: Any, d: int = 2) -> str:
    return f"{float(v):.{d}f}" if _fin(v) else "n/a"


def money(v: Any) -> str:
    return f"${float(v):,.0f}" if _fin(v) else "n/a"


def plain(v: Any) -> str:
    """A parameter value as it should read in a sentence: 0.05 -> 0.05, 200 -> 200."""
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:g}"
    return h(str(v))


_SPANS = {
    1: "one day",
    2: "two days",
    3: "three days",
    5: "a week",
    10: "two weeks",
    21: "a month",
    42: "two months",
    63: "three months",
    126: "six months",
    252: "a year",
    504: "two years",
}


def days(n: Any) -> str:
    """``63`` -> ``63 trading days (about three months)``."""
    try:
        k = int(n)
    except (TypeError, ValueError):
        return plain(n)
    if k in _SPANS:
        about = _SPANS[k]
        return (
            f"{k} trading days (about {about})"
            if k > 3
            else f"{k} trading day{'s' if k > 1 else ''}"
        )
    if k >= 42:
        return f"{k} trading days (about {k / 21:.0f} months)"
    if k >= 10:
        return f"{k} trading days (about {k / 5:.0f} weeks)"
    return f"{k} trading days"


def years_between(start: str, end: str) -> float | None:
    try:
        y0, m0, d0 = (int(x) for x in str(start)[:10].split("-"))
        y1, m1, d1 = (int(x) for x in str(end)[:10].split("-"))
    except ValueError:
        return None
    return (y1 - y0) + (m1 - m0) / 12.0 + (d1 - d0) / 365.0


def grid_text(grid: dict[str, Any]) -> str:
    """``{"window": [100, 200]}`` -> ``window: 100 or 200``."""
    bits = []
    for k, vals in grid.items():
        if isinstance(vals, list | tuple) and len(vals) > 1:
            shown = [plain(v) for v in vals]
            bits.append(f"<b>{h(k)}</b>: {', '.join(shown[:-1])} or {shown[-1]}")
    return "; ".join(bits)


def count(n: Any, noun: str) -> str:
    k = int(n)
    return f"{k:,} {noun}{'' if k == 1 else 's'}"


OBJECTIVES = {"sharpe": "Sharpe ratio", "calmar": "Calmar ratio", "cagr": "growth rate"}


# -- the per-strategy text ---------------------------------------------------------------


@dataclass
class Ctx:
    """What the sentences need: the names of the things being traded, escaped."""

    key: str
    p: dict[str, Any]  # defaults merged with the run's own parameters
    asset: str  # the thing bought: SPY for a dial, TQQQ for a leverage rule
    parking: str  # where the money sits otherwise: cash, or the plain fund
    symbols: list[str] = field(default_factory=list)
    leverage: float | None = None


@dataclass
class Text:
    what: str
    rules: list[tuple[str, str]] = field(default_factory=list)
    math: str | None = None


def _sub(x: str, s: str = "t") -> str:
    return f"{x}<sub>{s}</sub>"


def _dial_hold(c: Ctx) -> Text:
    return Text(
        what=(
            f"Buy {c.asset} on day one and never sell. There is no rule to get right or wrong; "
            f"this is the yardstick every timing rule has to beat, because a rule that steps "
            f"out of the market has to earn back whatever it misses while it is out."
        ),
        math=f"{_sub('e')} = 1 on every day",
    )


def _dial_sma(c: Ctx) -> Text:
    w, b = c.p["window"], float(c.p["band"])
    band = (
        f"a dead band of {pct(b, 0)} on either side of the line stops it flip-flopping when price hovers right at the average"
        if b > 0
        else "there is no dead band, so a single close on the wrong side of the line flips it"
    )
    return Text(
        what=(
            f"The classic trend filter. Each day, compare {c.asset}'s closing price with the "
            f"average of its last {w} closes. Above the line the money is in {c.asset}; below "
            f"it, the money sits in {c.parking}. Note that {band}."
        ),
        rules=[
            (
                f"Step out of {c.asset} when",
                f"it closes more than {pct(b, 0)} below its {w}-day average.",
            ),
            ("Step back in when", f"it closes more than {pct(b, 0)} above its {w}-day average."),
        ],
        math=(
            f"{_sub('d')} = {_sub('P')} / SMA<sub>{w}</sub>({_sub('P')}) − 1; "
            f"out when {_sub('d')} &lt; −{plain(b)}, in when {_sub('d')} &gt; +{plain(b)}, "
            f"otherwise keep yesterday's state"
        ),
    )


def _vix_term(c: Ctx) -> Text:
    x, b, n = float(c.p["out_above"]), float(c.p["band"]), int(c.p["confirm_bars"])
    confirm = f", and stays there for {n} days in a row" if n > 1 else ""
    return Text(
        what=(
            f"Compares two versions of the fear gauge: the VIX, which prices how much the market "
            f"is expected to swing over the next 30 days, and the VIX3M, the same thing for the "
            f"next three months. Normally the three-month number is higher, because more can go "
            f"wrong in three months than in one. When the 30-day number climbs above it, traders "
            f"are paying up for protection right now: the fear curve has flipped. The rule keeps "
            f"the money in {c.parking} while the curve is flipped and puts it back in {c.asset} "
            f"once the curve rights itself."
        ),
        rules=[
            (f"Step out of {c.asset} when", f"VIX ÷ VIX3M rises above {plain(x)}{confirm}."),
            ("Step back in when", f"VIX ÷ VIX3M falls back below {plain(round(x - b, 4))}."),
        ],
        math=(
            f"{_sub('r')} = {_sub('VIX')} / {_sub('VIX3M')}; "
            f"out when {_sub('r')} &gt; {plain(x)}, in when {_sub('r')} &lt; {plain(round(x - b, 4))}"
        ),
    )


def _vix_resid(c: Ctx) -> Text:
    w, z, cd = int(c.p["window"]), float(c.p["warn_z"]), int(c.p["cooldown"])
    return Text(
        what=(
            f"The VIX usually moves in step with the market: {c.asset} falls 1% and the VIX "
            f"jumps by a predictable amount. The rule fits that relationship over the last "
            f"{days(w)} and watches for days when the VIX jumped far more than {c.asset}'s own "
            f"move explains, which means hedgers are paying up for protection the tape does not "
            f"justify. On such a day it moves the money to {c.parking} for {days(cd)}, then "
            f"comes back."
        ),
        rules=[
            (
                f"Step out of {c.asset} when",
                f"today's unexplained VIX move is more than {plain(z)} standard deviations above normal (normal measured over the last {w} days).",
            ),
            ("Step back in when", f"{cd} trading days have passed since the last warning."),
        ],
        math=(
            f"{_sub('ε')} = Δlog {_sub('VIX')} − {_sub('β')} · {_sub('r')} (β fitted over {w} days); "
            f"{_sub('z')} = {_sub('ε')} / σ(ε); out for {cd} days whenever {_sub('z')} &gt; {plain(z)}"
        ),
    )


def _vix_vrp(c: Ctx) -> Text:
    w, lo, hi = int(c.p["rv_window"]), float(c.p["out_below"]), float(c.p["full_above"])
    return Text(
        what=(
            f"The VIX is the price of insurance. How much {c.asset} actually moved over the last "
            f"{days(w)} is what that insurance would have paid out. The difference, in volatility "
            f"points, is the premium. A fat premium means fear is overdone and the rule is fully "
            f"in {c.asset}; a premium that turns negative means real moves have exceeded what "
            f"insurance priced, the market is being caught out, and the rule moves to "
            f"{c.parking}; in between it holds half."
        ),
        rules=[
            (
                f"Fully in {c.asset} when",
                f"VIX minus realized volatility is above {plain(hi)} points.",
            ),
            ("Half in when", f"it sits between {plain(lo)} and {plain(hi)} points."),
            (f"Out (in {c.parking}) when", f"it drops below {plain(lo)} points."),
        ],
        math=(
            f"{_sub('VRP')} = {_sub('VIX')} − RV<sub>{w}</sub> (vol points); "
            f"{_sub('e')} = 1 if {_sub('VRP')} &gt; {plain(hi)}; 0 if {_sub('VRP')} &lt; {plain(lo)}; else ½"
        ),
    )


def _vix_spike(c: Ctx) -> Text:
    w, z, hold, base = (
        int(c.p["window"]),
        float(c.p["spike_z"]),
        int(c.p["hold"]),
        float(c.p["base"]),
    )
    return Text(
        what=(
            f"A buying rule rather than a selling one. Most of the time it holds {pct(base, 0)} "
            f"of the money in {c.asset} and the rest in {c.parking}. When the VIX spikes far "
            f"above its own recent norm (a {plain(z)}-sigma jump against the last {days(w)}), "
            f"fear is at a peak and markets have historically bounced, so the rule goes fully "
            f"into {c.asset} for {days(hold)} and then drops back to the base level."
        ),
        rules=[
            (
                f"Go fully into {c.asset} when",
                f"the VIX's z-score over the last {w} days exceeds {plain(z)}.",
            ),
            (
                f"Drop back to {pct(base, 0)} when",
                f"{hold} trading days have passed since the last spike.",
            ),
        ],
        math=(
            f"{_sub('z')} = ({_sub('VIX')} − mean<sub>{w}</sub>) / sd<sub>{w}</sub>; "
            f"{_sub('e')} = 1 for {hold} days after {_sub('z')} &gt; {plain(z)}, else {plain(base)}"
        ),
    )


def _vix_sized(c: Ctx) -> Text:
    t, b, w, band = (
        float(c.p["target_vol"]),
        float(c.p["blend"]),
        int(c.p["rv_window"]),
        float(c.p["band"]),
    )
    if b >= 1:
        forecast = "the VIX itself"
        formula = _sub("VIX")
    elif b <= 0:
        forecast = f"{c.asset}'s realized volatility over the last {days(w)}"
        formula = f"RV<sub>{w}</sub>"
    else:
        forecast = f"a {pct(b, 0)} / {pct(1 - b, 0)} blend of the VIX and {c.asset}'s realized volatility over the last {days(w)}"
        formula = f"{plain(b)}·{_sub('VIX')} + {plain(round(1 - b, 4))}·RV<sub>{w}</sub>"
    return Text(
        what=(
            f"A dial, not a switch. The rule tries to keep the portfolio's expected volatility at "
            f"{pct(t, 0)} a year. Its forecast of volatility is {forecast}. If the forecast is "
            f"{pct(2 * t, 0)}, it holds half of the money in {c.asset}; at {pct(4 * t, 0)}, a "
            f"quarter; when the forecast is below {pct(t, 0)} it holds everything, and never "
            f"more than everything. It only trades when the target position has drifted by more "
            f"than {pct(band, 0)}, so it is not churning every day."
        ),
        rules=[
            (
                "Hold more when",
                f"forecast volatility falls (a full position once it is below {pct(t, 0)}).",
            ),
            (
                "Hold less when",
                f"forecast volatility rises (half at {pct(2 * t, 0)}, a quarter at {pct(4 * t, 0)}).",
            ),
        ],
        math=(
            f"σ̂<sub>t</sub> = {formula}; {_sub('e')} = min(1, {plain(t)} / σ̂<sub>t</sub>), "
            f"rebalanced only when it drifts by more than {plain(band)}"
        ),
    )


def _vix_surprise(c: Ctx) -> Text:
    w, x, b = int(c.p["window"]), float(c.p["out_above"]), float(c.p["band"])
    return Text(
        what=(
            f"Each day the VIX implies a typical one-day move for {c.asset}: VIX ÷ 100 ÷ √252, "
            f"so a VIX of 20 means about 1.3%. The rule counts how many of the last {days(w)} "
            f"{c.asset} moved more than that. If the VIX is pricing things right, roughly one day "
            f"in five is a surprise. Surprises piling up means the market keeps catching the "
            f"insurance pricing off guard: the regime is shifting under it, and the rule steps "
            f"aside until surprises are rare again."
        ),
        rules=[
            (
                f"Out (in {c.parking}) when",
                f"more than {pct(x, 0)} of the last {w} days were surprises.",
            ),
            ("Half in when", f"between {pct(x - b, 0)} and {pct(x, 0)} were."),
            (f"Fully in {c.asset} when", f"fewer than {pct(x - b, 0)} were."),
        ],
        math=(
            f"{_sub('s')} = share of the last {w} days with |r| &gt; VIX<sub>t−1</sub> / 100 / √252; "
            f"{_sub('e')} = 0 if {_sub('s')} &gt; {plain(x)}; 1 if {_sub('s')} &lt; {plain(round(x - b, 4))}; else ½"
        ),
    )


def _vix_corr(c: Ctx) -> Text:
    w, x, b = int(c.p["window"]), float(c.p["out_above"]), float(c.p["band"])
    return Text(
        what=(
            f"Normally {c.asset} and the VIX move in opposite directions: stocks down, fear up. "
            f"The rule measures their correlation over the last {days(w)}. The usual reading is "
            f"deeply negative, around −0.7 to −0.8. When the two start moving together the "
            f"normal link is broken, something unusual is going on, and the rule steps aside."
        ),
        rules=[
            (f"Out (in {c.parking}) when", f"the {w}-day correlation rises above {plain(x)}."),
            ("Half in when", f"it sits between {plain(round(x - b, 4))} and {plain(x)}."),
            (
                f"Fully in {c.asset} when",
                f"it is below {plain(round(x - b, 4))} (the normal opposite relationship).",
            ),
        ],
        math=(
            f"{_sub('ρ')} = corr<sub>{w}</sub>({_sub('r')}, Δlog {_sub('VIX')}); "
            f"{_sub('e')} = 0 if {_sub('ρ')} &gt; {plain(x)}; 1 if {_sub('ρ')} &lt; {plain(round(x - b, 4))}; else ½"
        ),
    )


def _vix_vvix(c: Ctx) -> Text:
    w, z, cd, ratio = (
        int(c.p["window"]),
        float(c.p["out_z"]),
        int(c.p["cooldown"]),
        bool(c.p["use_ratio"]),
    )
    thing = (
        "the VVIX divided by the VIX (how jumpy fear is relative to its level)"
        if ratio
        else "the VVIX"
    )
    sym = "VVIX/VIX" if ratio else "VVIX"
    return Text(
        what=(
            f"The VVIX measures how jumpy the VIX itself is: the fear of fear. It often stirs "
            f"before the VIX does. The rule compares {thing} with its own last {days(w)}; when "
            f"it is stretched more than {plain(z)} standard deviations above normal, the money "
            f"moves to {c.parking} for {days(cd)}."
        ),
        rules=[
            (
                f"Step out of {c.asset} when",
                f"the {sym} z-score over the last {w} days exceeds {plain(z)}.",
            ),
            ("Step back in when", f"{cd} trading days have passed since the last trigger."),
        ],
        math=(
            f"{_sub('z')} = ({_sub(sym)} − mean<sub>{w}</sub>) / sd<sub>{w}</sub>; "
            f"out for {cd} days whenever {_sub('z')} &gt; {plain(z)}"
        ),
    )


def _vix_skew(c: Ctx) -> Text:
    w, z, b = int(c.p["window"]), float(c.p["out_z"]), float(c.p["band"])
    return Text(
        what=(
            f"The Cboe SKEW index measures how much more traders pay for crash protection (deep "
            f"out-of-the-money puts) than for upside bets. When that demand surges against its "
            f"norm over the last {days(w)}, the big players are bracing for a tail event. The "
            f"rule is out of {c.asset} while the surge lasts, fully in when the mix is normal, "
            f"and half in between."
        ),
        rules=[
            (
                f"Out (in {c.parking}) when",
                f"the SKEW z-score over the last {w} days exceeds {plain(z)}.",
            ),
            ("Half in when", f"it sits between {plain(round(z - b, 4))} and {plain(z)}."),
            (f"Fully in {c.asset} when", f"it is below {plain(round(z - b, 4))}."),
        ],
        math=(
            f"{_sub('z')} = ({_sub('SKEW')} − mean<sub>{w}</sub>) / sd<sub>{w}</sub>; "
            f"{_sub('e')} = 0 if {_sub('z')} &gt; {plain(z)}; 1 if {_sub('z')} &lt; {plain(round(z - b, 4))}; else ½"
        ),
    )


def _vix_vote(c: Ctx) -> Text:
    p = c.p
    return Text(
        what=(
            f"Three nearly independent readings of fear, averaged. Rule 1, the fear curve flip, "
            f"votes out when VIX ÷ VIX3M is above {plain(p['term_out_above'])}. Rule 2, "
            f"unexplained fear, votes out for {days(p['resid_cooldown'])} whenever the VIX jumps "
            f"more than {plain(p['resid_warn_z'])} standard deviations beyond what {c.asset}'s "
            f"own move explains. Rule 3, overpriced insurance, votes fully in when the VIX is more "
            f"than {plain(p['vrp_full_above'])} points above realized volatility, out when it is "
            f"below {plain(p['vrp_out_below'])}, and half in between. Each vote is a number from "
            f"0 to 1, and the share of the money in {c.asset} is their average: all three must "
            f"agree for the rule to be fully in or fully out."
        ),
        rules=[
            (f"Fully in {c.asset} when", "all three rules read calm."),
            (f"Fully out (in {c.parking}) when", "all three rules read danger."),
            ("Part way when", "they disagree: one-third per calm vote."),
        ],
        math=f"{_sub('e')} = ({_sub('e')}<sup>curve</sup> + {_sub('e')}<sup>unexplained</sup> + {_sub('e')}<sup>premium</sup>) / 3",
    )


def _lev_intro(c: Ctx) -> str:
    m = f"{c.leverage:g}×" if c.leverage else "leveraged"
    return (
        f"The money is always in one of two places: the plain fund {c.parking} or its {m} "
        f"cousin {c.asset}, which moves {m if c.leverage else 'several times'} as much as "
        f"{c.parking} every day, up and down. The only question is when to be in the leveraged "
        f"one, and the answer is read from {c.parking}'s own price, never from {c.asset}'s. "
    )


def _lev_hold(c: Ctx) -> Text:
    return Text(
        what=(
            f"Hold {c.asset} the whole time and never switch. This is what 'just hold the "
            f"leveraged fund' would have done: the record the timing rules have to beat, and the "
            f"reason the question is worth asking, since a leveraged fund held through a crash "
            f"loses far more than the plain fund does."
        ),
        math=f"{_sub('e')} = 1 on every day",
    )


def _lev_sma(c: Ctx) -> Text:
    w, b = int(c.p["window"]), float(c.p["band"])
    return Text(
        what=_lev_intro(c)
        + (
            f"This rule says the market has turned bullish while {c.parking} sits above the "
            f"average of its last {w} closes, with a {pct(b, 0)} cushion on either side so a "
            f"close that merely grazes the line does not trigger a switch."
        ),
        rules=[
            (
                f"Into {c.asset} when",
                f"{c.parking} closes more than {pct(b, 0)} above its {w}-day average.",
            ),
            (
                f"Back to {c.parking} when",
                f"{c.parking} closes more than {pct(b, 0)} below its {w}-day average.",
            ),
        ],
        math=(
            f"{_sub('d')} = {_sub('P')} / SMA<sub>{w}</sub>({_sub('P')}) − 1 for {c.parking}; "
            f"leveraged when {_sub('d')} &gt; +{plain(b)}, plain when {_sub('d')} &lt; −{plain(b)}"
        ),
    )


def _lev_cross(c: Ctx) -> Text:
    f, s, b = int(c.p["fast"]), int(c.p["slow"]), float(c.p["band"])
    cushion = f" by more than {pct(b, 1)}" if b > 0 else ""
    return Text(
        what=_lev_intro(c)
        + (
            f"This rule uses two averages of {c.parking}'s price, a fast one over {f} days and a "
            f"slow one over {s} days. The fast average climbing above the slow one is the "
            f"'golden cross' chartists watch for; dropping below it is the 'death cross'."
        ),
        rules=[
            (
                f"Into {c.asset} when",
                f"the {f}-day average rises above the {s}-day average{cushion} (golden cross).",
            ),
            (
                f"Back to {c.parking} when",
                f"the {f}-day average falls below the {s}-day average{cushion} (death cross).",
            ),
        ],
        math=(
            f"{_sub('d')} = SMA<sub>{f}</sub> / SMA<sub>{s}</sub> − 1; "
            f"leveraged when {_sub('d')} &gt; +{plain(b)}, plain when {_sub('d')} &lt; −{plain(b)}"
        ),
    )


def _lev_swing(c: Ctx) -> Text:
    up, down, n = float(c.p["up_pct"]), float(c.p["down_pct"]), int(c.p["lookback"])
    return Text(
        what=_lev_intro(c)
        + (
            f"This rule uses the popular definitions: a bull market is a rise of {pct(up, 0)} "
            f"from a recent low, a bear market a fall of {pct(down, 0)} from a recent high, both "
            f"measured against the last {days(n)}. It moves into the leveraged fund when a bull "
            f"market is declared and back to the plain one when a bear market is."
        ),
        rules=[
            (
                f"Into {c.asset} when",
                f"{c.parking} is up {pct(up, 0)} or more from its lowest close of the last {n} days.",
            ),
            (
                f"Back to {c.parking} when",
                f"{c.parking} is down {pct(down, 0)} or more from its highest close of the last {n} days.",
            ),
        ],
        math=(
            f"{_sub('P')} / min<sub>{n}</sub>(P) − 1 ≥ {plain(up)} → leveraged; "
            f"{_sub('P')} / max<sub>{n}</sub>(P) − 1 ≤ −{plain(down)} → plain; otherwise stay put"
        ),
    )


def _lev_breakout(c: Ctx) -> Text:
    w, m = int(c.p["window"]), int(c.p["exit_window"])
    return Text(
        what=_lev_intro(c)
        + (
            f"This rule treats a new high as the signal: {c.parking} closing above everything it "
            f"did over the last {days(w)} means the market has broken out, so move into the "
            f"leveraged fund. A close under its low of the last {days(m)} means the breakout has "
            f"failed, so move back."
        ),
        rules=[
            (f"Into {c.asset} when", f"{c.parking} closes at a new {w}-day high."),
            (f"Back to {c.parking} when", f"{c.parking} closes below its {m}-day low."),
        ],
        math=(
            f"{_sub('P')} ≥ max<sub>{w}</sub>(P<sub>t−1</sub>, …) → leveraged; "
            f"{_sub('P')} ≤ min<sub>{m}</sub>(P<sub>t−1</sub>, …) → plain"
        ),
    )


def _lev_vixfade(c: Ctx) -> Text:
    panic, calm, n, hold = (
        float(c.p["panic"]),
        float(c.p["calm"]),
        int(c.p["lookback"]),
        int(c.p["hold"]),
    )
    return Text(
        what=_lev_intro(c)
        + (
            f"This one reads the VIX, the market's fear gauge, instead of price. A reading above "
            f"{plain(panic)} within the last {days(n)} counts as a scare. The first day the VIX "
            f"then closes back below {plain(calm)}, the storm has passed and the big rebounds "
            f"tend to follow, so the rule moves into the leveraged fund for {days(hold)} and "
            f"then goes back to the plain one. A fresh scare and calm-down restarts the clock."
        ),
        rules=[
            (
                f"Into {c.asset} when",
                f"the VIX closes below {plain(calm)} for the first time after having been above {plain(panic)} in the last {n} days.",
            ),
            (f"Back to {c.parking} when", f"{hold} trading days have passed since that signal."),
        ],
        math=(
            f"trigger when max<sub>{n}</sub>(VIX<sub>t−1</sub>, …) &gt; {plain(panic)} and {_sub('VIX')} &lt; {plain(calm)} "
            f"for the first time; {_sub('e')} = 1 for {hold} days after a trigger, else 0"
        ),
    )


def _lev_voltarget(c: Ctx) -> Text:
    t, lev, w, band = (
        float(c.p["target_vol"]),
        float(c.p["leverage"]),
        int(c.p["rv_window"]),
        float(c.p["band"]),
    )
    return Text(
        what=_lev_intro(c)
        + (
            f"This rule answers how much rather than when. It aims to run the whole position at "
            f"{pct(t, 0)} volatility a year. With σ as {c.parking}'s realized volatility over "
            f"the last {days(w)}, a share s in the {lev:g}× fund makes the position's volatility "
            f"about σ·(1 + {lev - 1:g}·s), so the rule solves for the s that hits the target and "
            f"clips it between 0 and 1. Calm market: mostly {c.asset}. Rough market: mostly "
            f"{c.parking}. It only trades when the share has drifted by more than {pct(band, 0)}."
        ),
        rules=[
            (
                f"More {c.asset} when",
                f"{c.parking} has been calm (all of it once σ is at or below {pct(t, 0)}).",
            ),
            (
                f"More {c.parking} when",
                f"{c.parking} has been rough (none of the leveraged fund once σ reaches {pct(t * lev, 0)}).",
            ),
        ],
        math=f"{_sub('s')} = clip( ({plain(t)} / {_sub('σ')} − 1) / ({lev:g} − 1), 0, 1 ), σ over {w} days",
    )


def _basket(c: Ctx) -> str:
    k = len(c.symbols)
    return f"each of the {k} symbols" if k != 1 else "the symbol"


def _tsmom(c: Ctx) -> Text:
    n, t, w, ml = (
        int(c.p["lookback"]),
        float(c.p["target_vol"]),
        int(c.p["vol_window"]),
        float(c.p["max_leverage"]),
    )
    return Text(
        what=(
            f"Time-series momentum: the bet that what has been going up keeps going up for a "
            f"while, and what has been falling keeps falling. For {_basket(c)}, look at its "
            f"return over the last {days(n)}. Positive: hold it. Negative: short it. Each "
            f"position is sized so it contributes about {pct(t, 0)} volatility a year, measured "
            f"over the last {days(w)}, so calm assets get bigger positions and wild ones smaller, "
            f"with total exposure capped at {ml:g}× the money."
        ),
        rules=[
            ("Go long when", f"the asset's return over the last {n} trading days is positive."),
            ("Go short when", "that return is negative."),
        ],
        math=(
            f"w<sub>i,t</sub> = sign(r<sub>i,{n}</sub>) · {plain(t)} / σ<sub>i,{w}</sub>, "
            f"scaled down so Σ|w| ≤ {ml:g}"
        ),
    )


def _tsmom_regime(c: Ctx) -> Text:
    base = _tsmom(c)
    ew, pe, hw, hm = (
        int(c.p["entropy_window"]),
        float(c.p["pe_max"]),
        int(c.p["hurst_window"]),
        float(c.p["hurst_min"]),
    )
    base.what += (
        f" On top of that sits a regime gate: a position is only allowed while the asset's "
        f"price looks like it is trending. Two tests must both pass. Its permutation entropy "
        f"over the last {days(ew)} must be at most {plain(pe)} (the price path is not pure "
        f"noise), and its Hurst exponent over the last {days(hw)} must be at least {plain(hm)} "
        f"(moves tend to persist rather than reverse). Otherwise the position is zero."
    )
    base.rules.append(
        (
            "Stay flat when",
            f"the trend tests fail: entropy above {plain(pe)} or Hurst below {plain(hm)}.",
        )
    )
    base.math = (
        base.math or ""
    ) + f"; w = 0 unless PE<sub>{ew}</sub> ≤ {plain(pe)} and H<sub>{hw}</sub> ≥ {plain(hm)}"
    return base


def _xsmom(c: Ctx) -> Text:
    n, s, q, g = (
        int(c.p["lookback"]),
        int(c.p["skip"]),
        float(c.p["quantile"]),
        float(c.p["gross_exposure"]),
    )
    return Text(
        what=(
            f"Cross-sectional momentum, built to be market-neutral. Each day, rank the "
            f"{len(c.symbols)} symbols by their return from {days(n)} ago to {days(s)} ago, "
            f"skipping the most recent {days(s)} because very recent moves tend to reverse. Buy "
            f"the top {pct(q, 0)} in equal amounts and short the bottom {pct(q, 0)} in equal "
            f"amounts, {pct(g / 2, 0)} of the money on each side, so the book nets to zero and "
            f"the bet is purely that recent winners keep beating recent losers."
        ),
        rules=[
            ("Go long when", f"a symbol ranks in the top {pct(q, 0)} by that return."),
            ("Go short when", f"it ranks in the bottom {pct(q, 0)}."),
        ],
        math=(
            f"score<sub>i,t</sub> = P<sub>t−{s}</sub> / P<sub>t−{n}</sub> − 1; "
            f"long the top {plain(q)} of ranks, short the bottom {plain(q)}, {plain(g / 2)} per side"
        ),
    )


def _bollinger(c: Ctx) -> Text:
    w, k, tp, sp, hold = (
        int(c.p["window"]),
        float(c.p["n_std"]),
        float(c.p["target_pct"]),
        float(c.p["stop_pct"]),
        int(c.p["max_hold"]),
    )
    return Text(
        what=(
            f"Mean reversion on big dislocations. For {_basket(c)}, draw a band {plain(k)} "
            f"standard deviations below its {w}-day average. A close below that band is a "
            f"statistically extreme drop, and the rule buys it, betting the price snaps back "
            f"toward the average. It sells on a bounce of {pct(tp, 0)} from the entry (take "
            f"profit), a further fall of {pct(sp, 0)} (stop loss), or after {days(hold)}, "
            f"whichever comes first. The money is split equally across whatever is open."
        ),
        rules=[
            (
                "Buy when",
                f"a symbol closes below its {w}-day average minus {plain(k)} standard deviations.",
            ),
            (
                "Sell when",
                f"it is up {pct(tp, 0)}, down {pct(sp, 0)}, or {hold} trading days have passed.",
            ),
        ],
        math=(
            f"enter when {_sub('P')} &lt; SMA<sub>{w}</sub> − {plain(k)}·sd<sub>{w}</sub>; "
            f"exit when {_sub('P')}/P<sub>entry</sub> − 1 ≥ {plain(tp)} or ≤ −{plain(sp)} or after {hold} bars"
        ),
    )


def _rise_fall(c: Ctx) -> Text:
    p = c.p
    drop, rise, stop = float(p["drop_pct"]), float(p["rise_pct"]), float(p["stop_pct"])
    aw, am, tw, hold = (
        int(p["atr_window"]),
        float(p["atr_mult"]),
        int(p["trend_window"]),
        int(p["max_hold"]),
    )
    return Text(
        what=(
            f"Buy small dips inside an uptrend, sell on a slightly bigger rise. Only symbols "
            f"trading above their {tw}-day average are eligible. A one-day drop of at least "
            f"{pct(drop, 1)}, or {plain(am)} times the symbol's typical daily range over the last "
            f"{aw} days if that is larger, is the entry. The rule sells at a {pct(rise, 0)} gain "
            f"(target), a {pct(stop, 0)} loss (stop), after {days(hold)}, or as soon as the "
            f"symbol drops below its trend line. Equal weight across open names."
        ),
        rules=[
            (
                "Buy when",
                f"a symbol above its {tw}-day average falls at least max({pct(drop, 1)}, {plain(am)} × ATR<sub>{aw}</sub>) in one day.",
            ),
            (
                "Sell when",
                f"it is up {pct(rise, 0)}, down {pct(stop, 0)}, {hold} days have passed, or it drops below the {tw}-day average.",
            ),
        ],
        math=(
            f"enter when r<sub>1</sub> ≤ −max({plain(drop)}, {plain(am)}·ATR<sub>{aw}</sub>/P) and {_sub('P')} &gt; SMA<sub>{tw}</sub>; "
            f"exit at +{plain(rise)}, −{plain(stop)}, {hold} bars, or {_sub('P')} ≤ SMA<sub>{tw}</sub>"
        ),
    )


WRITERS: dict[str, Callable[[Ctx], Text]] = {
    "dial_hold": _dial_hold,
    "dial_sma": _dial_sma,
    "vix_term": _vix_term,
    "vix_resid": _vix_resid,
    "vix_vrp": _vix_vrp,
    "vix_spike": _vix_spike,
    "vix_sized": _vix_sized,
    "vix_surprise": _vix_surprise,
    "vix_corr": _vix_corr,
    "vix_vvix": _vix_vvix,
    "vix_skew": _vix_skew,
    "vix_vote": _vix_vote,
    "lev_hold": _lev_hold,
    "lev_sma": _lev_sma,
    "lev_cross": _lev_cross,
    "lev_swing": _lev_swing,
    "lev_breakout": _lev_breakout,
    "lev_vixfade": _lev_vixfade,
    "lev_voltarget": _lev_voltarget,
    "tsmom": _tsmom,
    "tsmom_regime": _tsmom_regime,
    "xsmom": _xsmom,
    "bollinger_reversion": _bollinger,
    "rise_fall": _rise_fall,
}


def _generic(c: Ctx) -> Text:
    cls = _REGISTRY.get(c.key)
    doc = (cls.__doc__ or "").strip().splitlines()[0] if cls and cls.__doc__ else ""
    knobs = ", ".join(
        f"{h(k)} = {plain(v)}" for k, v in c.p.items() if not isinstance(v, dict | list)
    )
    return Text(what=f"{h(doc)} Settings: {knobs}." if doc else f"Settings: {knobs}.")


# -- the block ---------------------------------------------------------------------------


def _kind(key: str) -> str:
    cls = _REGISTRY.get(key)
    if cls is None:
        return "portfolio"
    if issubclass(cls, LeverageDial):
        return "leverage"
    return "dial" if issubclass(cls, DialStrategy) else "portfolio"


def _params(p: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the parameters that ran, defaults filled in; the walk-forward grid if any)."""
    raw = p.get("params") or {}
    base = raw.get("base_params") if isinstance(raw.get("base_params"), dict) else None
    own = dict(base) if base is not None else dict(raw)
    g = raw.get("grid")
    grid: dict[str, Any] = dict(g) if isinstance(g, dict) else {}
    cls = _REGISTRY.get(str(p.get("strategy")))
    merged = {**(cls.defaults if cls else {}), **own}
    return merged, grid


def _time_in(exposure: list[Any] | None) -> float | None:
    vals = [min(1.0, abs(float(v))) for v in (exposure or []) if _fin(v)]
    return sum(vals) / len(vals) if vals else None


def _tidy(v: Any) -> str:
    """A check's number: 19 stays 19, 0.9945 becomes 0.99."""
    f = float(v)
    return f"{int(f):,}" if f == int(f) and abs(f) >= 1 else f"{f:.2f}"


def _cmp(a: Any, b: Any, better_high: bool = True, tol: float = 0.0) -> str:
    """'better than', 'about the same as', 'worse than' for a vs b."""
    if not (_fin(a) and _fin(b)):
        return "against"
    d = float(a) - float(b)
    if abs(d) <= tol:
        return "about the same as"
    good = d > 0 if better_high else d < 0
    return "better than" if good else "worse than"


def _context(
    key: str, params_raw: dict[str, Any], universe: list[str], asset: str | None
) -> tuple[Ctx, dict[str, Any], bool]:
    """(the names the sentences use, the walk-forward grid, whether the asset is extended)."""
    params, grid = _params({"strategy": key, "params": params_raw})
    symbols = [s for s in universe if isinstance(s, str) and not s.startswith("^") and s != "CASH"]
    kind = _kind(key)
    asset_raw = str(asset or params.get("asset") or (symbols[0] if symbols else "SPY"))
    extended = is_extended(asset_raw)
    asset_real = real_symbol(asset_raw) if extended else asset_raw
    leverage = None
    parking_raw = "cash"
    if kind == "leverage":
        parking_raw = str(params.get("base") or "QQQ")
        pair = pair_by_symbol(asset_raw)
        if pair is not None:
            leverage = pair.leverage
        elif _fin(params.get("leverage")):
            leverage = float(params["leverage"])
    c = Ctx(
        key=key,
        p=params,
        asset=h(asset_real),
        parking=h(parking_raw),
        symbols=symbols,
        leverage=leverage,
    )
    return c, grid, extended


def _write(c: Ctx) -> Text:
    try:
        text = WRITERS.get(c.key, _generic)(c)
    except Exception:  # a parameter this module did not expect; never break the page
        text = _generic(c)
    dwell = c.p.get("min_dwell")
    if _kind(c.key) in ("dial", "leverage") and _fin(dwell) and int(dwell) > 1:
        text.what += f" Any change of position is held for at least {days(int(dwell))} before the next one is allowed, a brake on whipsaws."
    return text


def _first_sentences(html: str, n: int = 1) -> str:
    parts = html.split(". ")
    out = ". ".join(parts[:n])
    return out if out.endswith(".") else out + "."


def blurb(
    strategy: str,
    params: dict[str, Any],
    *,
    universe: Sequence[str] | str = (),
    asset: str | None = None,
    metrics: dict[str, Any] | None = None,
    market: dict[str, Any] | None = None,
    out_of_sample: bool = False,
) -> dict[str, Any]:
    """One or two sentences for a hover card: the rule, then what it made.

    ``universe`` may be the stored comma-separated string; ``market`` the benchmark's
    numbers (``cagr``, ``max_drawdown``) when they are at hand. ``text`` is HTML."""
    key = str(strategy or "")
    uni = re.split(r"[,\s]+", universe) if isinstance(universe, str) else list(universe)
    c, _grid, _ext = _context(key, params or {}, [u for u in uni if u], asset)
    text = _write(c)
    if text.rules and key != "vix_vote":
        bits = []
        for i, (when, what) in enumerate(text.rules[:3]):
            lead = when if i == 0 else when[0].lower() + when[1:]
            bits.append(f"{lead} {what.rstrip('.')}")
        rule = "; ".join(bits) + "."
    else:
        rule = _first_sentences(text.what, 2 if key == "vix_vote" else 1)
    m = metrics or {}
    made = ""
    if _fin(m.get("cagr")):
        made = f"Made <b>{pct(m['cagr'])}</b> a year"
        if market and _fin(market.get("cagr")):
            made += f" against {pct(market['cagr'])} for the market"
        if _fin(m.get("max_drawdown")):
            made += f", with a worst fall of <b>{pct(m['max_drawdown'], 0)}</b>"
            if market and _fin(market.get("max_drawdown")):
                made += f" versus {pct(market['max_drawdown'], 0)}"
        if _fin(m.get("sharpe")):
            made += f"; Sharpe {num(m['sharpe'])}"
        made += " out of sample." if out_of_sample else "."
    return {
        "name": NAMES.get(key, key.replace("_", " ")),
        "tag": _tag(key),
        "text": rule + (" " + made if made else ""),
    }


def _tag(key: str) -> str:
    rn = RULE_NUMBER.get(key)
    return {
        "dial": f"VIX rule {rn}" if rn else "In or out of one asset",
        "leverage": "Leverage rule",
        "portfolio": "Portfolio strategy",
    }[_kind(key)]


def explain(p: dict[str, Any]) -> dict[str, Any]:
    """The short version of a run payload (see the module docstring). Never raises: a
    strategy this module has no words for gets its docstring and its settings."""
    key = str(p.get("strategy") or "")
    kind = _kind(key)
    uni = [s for s in ((p.get("universe") or {}).get("symbols") or []) if isinstance(s, str)]
    c, grid, extended = _context(key, p.get("params") or {}, uni, p.get("asset"))
    if not grid:  # the scorecard carries the grid when the spec does not
        g = (p.get("study") or {}).get("grid")
        grid = dict(g) if isinstance(g, dict) else {}
    symbols, asset_real, leverage = c.symbols, c.asset, c.leverage
    text = _write(c)

    # -- what it was tested on --------------------------------------------------------
    start, end = str(p.get("start") or "")[:10], str(p.get("end") or "")[:10]
    m = p.get("metrics") or {}
    yrs = float(m["n_bars"]) / 252.0 if _fin(m.get("n_bars")) else years_between(start, end)
    years_txt = f" ({yrs:.1f} years)" if yrs else ""
    setup: list[str] = [
        f"Tested from <b>{h(start)}</b> to <b>{h(end)}</b>{years_txt} on daily closes, starting with {money(p.get('initial_capital'))}."
    ]
    bname = h(str(p["benchmark"])) if p.get("benchmark") else asset_real
    if kind == "dial":
        setup.append(
            f"The money is either in {c.asset} or in cash earning the 3-month Treasury bill rate. The grey line to beat is buying and holding {bname}."
        )
    elif kind == "leverage":
        setup.append(
            f"The money is either in {c.asset} or in {c.parking}. The grey line to beat is buying and holding {bname}, the plain fund."
        )
        if extended:
            setup.append(
                f"{c.asset} here is the extended series: the real fund's record with a rebuilt version spliced in front of it for the years before the fund existed ({leverage:g}× {c.parking}'s daily return, less financing and fees)."
                if leverage
                else f"{c.asset} here is the extended series: the real fund's record with a rebuilt version spliced in front for the years before the fund existed."
            )
    else:
        shown = ", ".join(h(s) for s in symbols[:12]) + (
            f" and {len(symbols) - 12} more" if len(symbols) > 12 else ""
        )
        setup.append(f"The basket: {shown}. The grey line to beat is buying and holding {bname}.")
    costs = p.get("costs") or {}
    slip = costs.get("slippage_bps")
    fill = str(p.get("fill_model") or "next_open").replace("_", " ")
    cost_bits = []
    if _fin(slip):
        cost_bits.append(
            f"{plain(slip)} basis point{'s' if float(slip) != 1 else ''} of slippage ({money(float(slip) / 10_000 * 1_000_000)} per $1,000,000 traded)"
        )
    if _fin(costs.get("commission_per_share")) and float(costs["commission_per_share"]) > 0:
        cost_bits.append(f"${plain(costs['commission_per_share'])} a share in commission")
    if _fin(costs.get("borrow_bps_annual")) and float(costs["borrow_bps_annual"]) > 0:
        cost_bits.append(
            f"{plain(costs['borrow_bps_annual'])} basis points a year to borrow shorts"
        )
    if costs.get("frictionless"):
        cost_bits = ["no costs at all (a debugging run)"]
    setup.append(
        f"Every trade pays {' and '.join(cost_bits) if cost_bits else 'nothing'} and fills at the {h(fill)}, never at the price that triggered it."
    )

    study = p.get("study") or {}
    windows = p.get("windows") or []
    if windows and study:
        obj = str(study.get("objective") or "")
        by = f" (the cell of the grid with the best {OBJECTIVES.get(obj, h(obj))})" if obj else ""
        tried = grid_text(grid)
        setup.append(
            f"<b>Walk-forward.</b> This record is stitched from {len(windows)} out-of-sample windows. Before each one the settings were chosen on the previous {study.get('train_months')} months of data{by} and then traded, untouched, for the next {study.get('test_months')} months."
            + (
                f" Settings tried: {tried}. The sentences above use the defaults; the table below the charts shows what each window actually chose."
                if tried
                else " There was nothing to choose: the rule has no settings."
            )
        )
    elif study and study.get("full_hash") == p.get("hash"):
        setup.append(
            "This is the full-sample run with the default settings: descriptive only, since the settings were not chosen out of sample. The walk-forward run is the one the verdict is based on."
        )

    # -- what happened ------------------------------------------------------------------
    b = p.get("benchmark_metrics") or {}
    has_b = bool(b) and _fin(b.get("cagr"))
    cap = float(p.get("initial_capital") or 1_000_000.0)
    results: list[str] = []
    final = cap * (1.0 + float(m["total_return"])) if _fin(m.get("total_return")) else None
    bfinal = (
        cap * (1.0 + float(b["total_return"])) if has_b and _fin(b.get("total_return")) else None
    )
    if final is not None:
        line = f"<b>{money(cap)} became {money(final)}</b>, which is {pct(m.get('cagr'))} a year."
        if bfinal is not None:
            line += f" Buying and holding {bname} would have made {money(bfinal)} ({pct(b.get('cagr'))} a year), so the rule did {_cmp(m.get('cagr'), b.get('cagr'), tol=0.005)} the market."
        results.append(line)
    if _fin(m.get("max_drawdown")):
        line = f"The deepest fall from a peak was <b>{pct(m['max_drawdown'])}</b>"
        if has_b and _fin(b.get("max_drawdown")):
            line += f", against {pct(b['max_drawdown'])} for {bname}: {_cmp(m['max_drawdown'], b['max_drawdown'], tol=0.01)} holding the market when it comes to pain."
        else:
            line += "."
        results.append(line)
    dd_bits = []
    for yr, key_ in (("2008", "dd_2008"), ("2020", "dd_2020"), ("2022", "dd_2022")):
        if _fin(study.get(key_)):
            dd_bits.append(f"{pct(study[key_], 0)} in {yr}")
    if dd_bits and windows:
        results.append(
            "In the crashes it fell " + ", ".join(dd_bits) + " (peak to trough, out of sample)."
        )
    if _fin(m.get("sharpe")):
        line = f"Sharpe ratio <b>{num(m['sharpe'])}</b>: return per unit of risk taken, where 1 is good and 0.5 is ordinary"
        if has_b and _fin(b.get("sharpe")):
            line += f"; {bname} scored {num(b['sharpe'])}"
        line += f". Year-to-year swings of {pct(m.get('ann_vol'), 0)}"
        if has_b and _fin(b.get("ann_vol")):
            line += f" against {pct(b['ann_vol'], 0)} for {bname}"
        results.append(line + ".")
    series = p.get("series") or {}
    tim = _time_in(series.get("gross") if kind == "portfolio" else series.get("exposure"))
    n_tr = p.get("n_trades")
    if tim is not None:
        where = "in the leveraged fund" if kind == "leverage" else "in the market"
        line = f"{where.capitalize()} <b>{pct(tim, 0)}</b> of the time"
        if _fin(n_tr) and yrs:
            per = float(n_tr) / yrs
            line += (
                f", with {count(n_tr, 'trade')}, about {per:,.0f} a year"
                if per >= 1.5
                else f", with {count(n_tr, 'trade')} in all"
            )
        results.append(line + ".")
    yearly = [y for y in (p.get("yearly") or []) if _fin(y.get("strategy"))]
    if yearly:
        if has_b:
            beat = sum(
                1
                for y in yearly
                if _fin(y.get("benchmark")) and float(y["strategy"]) > float(y["benchmark"])
            )
            judged = sum(1 for y in yearly if _fin(y.get("benchmark")))
            worst = min(yearly, key=lambda y: float(y["strategy"]))
            results.append(
                f"Beat {bname} in <b>{beat} of {judged}</b> calendar years. Worst year: {pct(worst['strategy'])} in {worst['year']}."
            )
        else:
            worst = min(yearly, key=lambda y: float(y["strategy"]))
            results.append(f"Worst calendar year: {pct(worst['strategy'])} in {worst['year']}.")

    # -- the verdict --------------------------------------------------------------------
    verdict: dict[str, Any] | None = None
    if study and windows:
        status = str(study.get("verdict") or "")
        passed = status == "pass"
        failed = [str(f) for f in (study.get("failed") or [])]
        checks = {
            str(ch.get("name")): ch for ch in (study.get("checks") or []) if isinstance(ch, dict)
        }
        why: list[str] = []
        for f in failed:
            ch = checks.get(f) or {}
            reason = CHECKS.get(f, f)
            v, t = ch.get("value"), ch.get("threshold")
            if _fin(t) and _fin(v):
                detail = f" ({_tidy(v)} against {_tidy(t)} needed)"
            elif _fin(t):
                detail = " (it could not even be computed)"
            else:
                detail = ""
            why.append(reason + detail)
        text_v = (
            "It passed every check in the gauntlet: the out-of-sample record holds up after correcting for luck, the number of settings tried, doubled costs and different market regimes."
            if passed
            else "Killed by the gauntlet. "
            + ("It failed because " + "; ".join(why) + "." if why else "")
        )
        verdict = {"status": "pass" if passed else "killed", "html": text_v}

    headline = ""
    if _fin(m.get("cagr")):
        headline = f"{pct(m['cagr'])} a year"
        if has_b:
            headline += f" against {pct(b['cagr'])} for holding {bname}"
        if _fin(m.get("max_drawdown")):
            headline += f", with a worst fall of {pct(m['max_drawdown'], 0)}"
            if has_b and _fin(b.get("max_drawdown")):
                headline += f" versus {pct(b['max_drawdown'], 0)}"
        headline += "."

    return {
        "name": NAMES.get(key, key.replace("_", " ")),
        "tag": _tag(key),
        "kind": kind,
        "headline": headline,
        "what": text.what,
        "rules": [{"when": w, "html": t} for w, t in text.rules],
        "math": text.math,
        "setup": setup,
        "results": results,
        "verdict": verdict,
    }
