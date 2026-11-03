"""Build the plain-English write-up for the leverage study (Dad's TQQQ-vs-QQQ question).

Reads the study summaries in ``var/scorecards`` and the stored equity curves and
trade ledgers in ``var/results``; writes a PDF, its chart PNGs and a short markdown
body for the library page.

    python scripts/build_leverage_report.py OUT.pdf IMG_DIR [STAMP]

STAMP names ``leverage_sharpe_<STAMP>.json`` (the extended-history run over every
pair) and, when present, ``leverage_sharpe_<STAMP>-real.json`` (the same rules on
the funds' own history only). The markdown body lands next to the PDF as ``.md``.
At the end it prints the ``ops.py publish-report`` thread arguments.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ehf.research.analysis.dial import positions_from_trades
from ehf.research.data.leverage import PAIRS
from ehf.research.strategies.leverage import GRIDS
from matplotlib.ticker import NullFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "var" / "scorecards"
RESULTS = ROOT / "var" / "results"

OUT_PDF = Path(sys.argv[1])
IMG_DIR = Path(sys.argv[2])
STAMP = sys.argv[3] if len(sys.argv) > 3 else "2026-11-03"
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_MD = OUT_PDF.with_suffix(".md")

TITLE_TEXT = "Three Times the Market, Some of the Time"
HEAD = "QQQ/TQQQ"

# ---- palette (dataviz reference instance, light mode) -----------------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"  # QQQ, the plain fund
ORANGE = "#eb6834"  # TQQQ held
AQUA = "#1baf7a"  # rule 1, above the 200-day line
YELLOW = "#eda100"  # rule 3, up from the low
MAGENTA = "#e87ba4"  # rule 6, size by calm
PURPLE = "#7b61ff"  # rule 2, golden cross
GREY = "#8a8985"
NAVY = colors.HexColor("#1f3a5f")

NAMES = {
    "hold_base": "Hold QQQ",
    "lev_hold": "Hold TQQQ, always",
    "lev_sma": "1. Above the 200-day line",
    "lev_cross": "2. Golden cross",
    "lev_swing": "3. Up 20% from the low",
    "lev_breakout": "4. New six-month high",
    "lev_vixfade": "5. After the storm",
    "lev_voltarget": "6. Size by calm",
}
SHORT = {
    "hold_base": "QQQ",
    "lev_hold": "TQQQ",
    "lev_sma": "200-day",
    "lev_cross": "Cross",
    "lev_swing": "Swing",
    "lev_breakout": "Breakout",
    "lev_vixfade": "Storm",
    "lev_voltarget": "Calm",
}
COLOR = {
    "hold_base": BLUE,
    "lev_hold": ORANGE,
    "lev_sma": AQUA,
    "lev_cross": PURPLE,
    "lev_swing": YELLOW,
    "lev_breakout": GREY,
    "lev_vixfade": GREY,
    "lev_voltarget": MAGENTA,
}
RULES = ["lev_sma", "lev_cross", "lev_swing", "lev_breakout", "lev_vixfade", "lev_voltarget"]
ORDER = ["hold_base", "lev_hold", *RULES]
FEATURED = ["hold_base", "lev_hold", "lev_sma", "lev_cross", "lev_swing", "lev_voltarget"]

# ---- data --------------------------------------------------------------------
S = json.loads((SCORE / f"leverage_sharpe_{STAMP}.json").read_text())
real_path = SCORE / f"leverage_sharpe_{STAMP}-real.json"
SR = json.loads(real_path.read_text()) if real_path.exists() else None

FACTS = S["pairs"]
PAIR_ORDER = [k for k in PAIRS if k in FACTS]


def outcomes_of(summary: dict, extended: bool) -> dict[tuple[str, str], dict]:
    return {(o["pair"], o["strategy"]): o for o in summary["outcomes"] if o["extended"] is extended}


def rows_of(summary: dict, variant: str) -> dict[tuple[str, str], dict]:
    return {(r["pair"], r["strategy"]): r for r in summary["rows"] if r["variant"] == variant}


OUT = outcomes_of(S, True)
ROW = rows_of(S, "extended")
OUT_R = outcomes_of(SR, False) if SR else {}
ROW_R = rows_of(SR, "real") if SR else {}
HOLD = {k.split(":")[0]: v for k, v in S["holds"].items() if k.endswith(":extended")}
HOLD_R = {k.split(":")[0]: v for k, v in SR["holds"].items() if k.endswith(":real")} if SR else {}

head_rules = [s for s in RULES if (HEAD, s) in OUT]
if not head_rules or (HEAD, "lev_hold") not in OUT:
    sys.exit(f"summary {STAMP} has no {HEAD} outcomes; run the study first")


def _naive(ts: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.to_datetime(ts, utc=True)
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
    )


def load_equity(h: str) -> pd.Series:
    e = pd.read_parquet(RESULTS / h / "equity.parquet")
    return pd.Series(e["equity"].to_numpy(), index=_naive(e["timestamp"]))


def load_trades(h: str) -> pd.DataFrame:
    t = pd.read_parquet(RESULTS / h / "trades.parquet").copy()
    t["timestamp"] = _naive(t["timestamp"])
    return t


# Out-of-sample (stitched) curves for the headline pair, plus QQQ held over the same span.
EQ: dict[str, pd.Series] = {}
for s in ["lev_hold", *head_rules]:
    EQ[s] = load_equity(OUT[(HEAD, s)]["stitched_hash"])
oos_start, oos_end = EQ["lev_hold"].index[0], EQ["lev_hold"].index[-1]
base_full = load_equity(HOLD[HEAD]["hash"])
EQ["hold_base"] = base_full.loc[oos_start:oos_end]
FULL_LEV = load_equity(OUT[(HEAD, "lev_hold")]["full_hash"])  # TQQQ.X held from 2000
years_span = (oos_end - oos_start).days / 365.25
FEATURED = [s for s in FEATURED if s in EQ]


def exposure(s: str) -> pd.Series:
    o = OUT[(HEAD, s)]
    idx = EQ[s].index
    trades = load_trades(o["stitched_hash"])
    resets = []
    for d in o["windows"]:
        i = idx.searchsorted(pd.Timestamp(d["test_start"]))
        if i < len(idx):
            resets.append(idx[i])
    return positions_from_trades(trades, idx, o["lev"], resets).clip(0, 1)


EXPO = {s: exposure(s) for s in head_rules}


def stints(s: str) -> pd.DataFrame:
    """Every stretch the rule spent (mostly) in the leveraged fund, with what each fund did."""
    ex = (EXPO[s] >= 0.5).astype(int)
    edges = ex.diff().fillna(ex.iloc[0])
    starts = list(ex.index[edges == 1])
    ends = list(ex.index[edges == -1])
    if len(ends) < len(starts):
        ends.append(ex.index[-1])
    out = []
    for a, b in zip(starts, ends, strict=True):
        lev = EQ["lev_hold"].loc[a:b]
        base = EQ["hold_base"].loc[a:b]
        if len(lev) < 2 or len(base) < 2:
            continue
        out.append(
            {
                "start": a,
                "end": b,
                "days": len(lev),
                "lev": float(lev.iloc[-1] / lev.iloc[0] - 1),
                "base": float(base.iloc[-1] / base.iloc[0] - 1),
            }
        )
    return pd.DataFrame(out)


CRISES = {
    "2002 dot-com tail": ("2002-03-10", "2003-03-31"),
    "2008 crisis": ("2007-11-01", "2009-06-30"),
    "2011 debt ceiling": ("2011-04-01", "2011-12-31"),
    "Late 2018": ("2018-09-01", "2019-03-31"),
    "Covid 2020": ("2020-01-01", "2020-09-30"),
    "2022 rate shock": ("2021-11-01", "2023-03-31"),
    "April 2025 tariffs": ("2025-01-01", "2025-09-30"),
}


def dd_in(eq: pd.Series, a: str, b: str) -> float:
    sub = eq.loc[a:b]
    return float((sub / sub.cummax() - 1.0).min()) if len(sub) > 1 else float("nan")


def max_dd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1.0).min())


def cagr(eq: pd.Series) -> float:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1)


CRISIS_DD = {s: {k: dd_in(EQ[s], *v) for k, v in CRISES.items()} for s in FEATURED}


def pct(x: float | None, nd: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    v = round(x * 100, nd)
    if v == 0:
        v = 0.0  # never "-0%"
    return f"{v:+.{nd}f}%" if sign else f"{v:.{nd}f}%"


def pp(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def num(x: float | None, nd: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def money(v: float) -> str:
    if v >= 1_000_000:
        return f"${v / 1_000_000:,.1f}m".replace(".0m", "m")
    return f"${v / 1000:,.0f}k" if v >= 10_000 else f"${v:,.0f}"


def loss(x: float | None) -> str:
    """A drawdown as a size: -0.9997 -> '99.97%', -0.53 -> '53%'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    a = abs(x) * 100
    return f"{a:.2f}%" if a > 99 else f"{a:.0f}%"


R = {s: ROW[(HEAD, s)] for s in ["lev_hold", *head_rules]}
R["hold_base"] = {
    "oos_cagr": cagr(EQ["hold_base"]),
    "oos_max_dd": max_dd(EQ["hold_base"]),
    "oos_sharpe": None,
    "time_lev": 0.0,
    "switch_py": 0.0,
    "oos_dd_2008": dd_in(EQ["hold_base"], "2007-11-01", "2009-06-30"),
    "oos_dd_2020": dd_in(EQ["hold_base"], "2020-01-01", "2020-09-30"),
    "oos_dd_2022": dd_in(EQ["hold_base"], "2021-11-01", "2023-03-31"),
}
final = {s: EQ[s].iloc[-1] / EQ[s].iloc[0] * 100_000 for s in FEATURED}
best_rule = max(head_rules, key=lambda s: R[s]["oos_sharpe"])
best_cagr_rule = max(head_rules, key=lambda s: R[s]["oos_cagr"])
calmest_rule = min(head_rules, key=lambda s: abs(R[s]["oos_max_dd"]))
mode_params = {
    s: json.loads(
        Counter(json.dumps(p, sort_keys=True) for p in OUT[(HEAD, s)]["chosen_params"]).most_common(
            1
        )[0][0]
    )
    for s in head_rules
}
EV = {s: OUT[(HEAD, s)]["events"] for s in head_rules}
EV_ROWS = {s: {r["h"]: r for r in EV[s]["rows"]} for s in head_rules}
BASE_RATE = {r["h"]: r for r in EV[head_rules[0]]["rows"]}
HORIZONS = [r["h"] for r in EV[head_rules[0]]["rows"]]
HLABEL = {r["h"]: r["label"] for r in EV[head_rules[0]]["rows"]}


def edge(s: str, h: int) -> float | None:
    r = EV_ROWS[s].get(h, {})
    if not r.get("n"):
        return None
    return float(r["excess_median"] - r["base_rate_excess_median"])


fit = FACTS[HEAD]["fit"]
STINTS = {s: stints(s) for s in head_rules}
lead = "lev_sma" if "lev_sma" in head_rules else head_rules[0]

# ---- chart helpers ------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT2,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.titlelocation": "left",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8,
    }
)


def save(fig, name):
    path = IMG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def years_axis(ax, step=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def log_money_axis(ax, ticks):
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.set_yticklabels([money(t) for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())


# Chart 1: the whole history, QQQ vs the rebuilt TQQQ, from March 2000
fig, ax = plt.subplots(figsize=(7.2, 3.6))
b = base_full / base_full.iloc[0] * 100_000
lv = FULL_LEV / FULL_LEV.iloc[0] * 100_000
real_start = pd.Timestamp(fit["real_start"])
after_low = lv[lv.index > lv.idxmin()]
recovered = after_low[after_low >= 100_000]
recovery = (
    f"did not get back to $100,000 until {recovered.index[0]:%B %Y}"
    if len(recovered)
    else "has still not got back to $100,000"
)
ax.axvspan(b.index[0], real_start, color=GRID, alpha=0.6, linewidth=0)
ax.plot(b.index, b, color=BLUE, lw=1.5, label="QQQ")
ax.plot(lv.index, lv, color=ORANGE, lw=1.5, label="TQQQ (rebuilt before 2010)")
ax.annotate(
    "rebuilt from QQQ\n(TQQQ did not exist yet)",
    (b.index[0] + pd.Timedelta(days=200), 350_000),
    color=TEXT2,
    fontsize=7.5,
)
ax.annotate(
    f"low: {money(lv.min())}",
    (lv.idxmin(), lv.min()),
    xytext=(8, -2),
    textcoords="offset points",
    color=ORANGE,
    fontsize=7.5,
)
for ser, col, lab in ((b, BLUE, "QQQ"), (lv, ORANGE, "TQQQ")):
    ax.annotate(
        f"{lab} {money(ser.iloc[-1])}",
        (ser.index[-1], ser.iloc[-1]),
        xytext=(5, 0),
        textcoords="offset points",
        color=col,
        fontsize=7.5,
        va="center",
    )
log_money_axis(ax, [1_000, 10_000, 100_000, 1_000_000])
ax.set_xlim(b.index[0], b.index[-1] + pd.Timedelta(days=1200))
ax.set_title("$100,000 put into each fund in March 2000, at the top of the dot-com bubble")
ax.legend(loc="lower right")
years_axis(ax, 2)
img_history = save(fig, "lev_history.png")

# Chart 2: how long to hold it. Median TQQQ-minus-QQQ after each call, vs any day
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for s in head_rules:
    pth = OUT[(HEAD, s)]["excess_path"]
    if not pth.get("h"):
        continue
    ax.plot(
        pth["h"],
        np.array(pth["median"]) * 100,
        color=COLOR[s] if COLOR[s] != GREY else GREY,
        lw=1.5 if s in FEATURED else 1.0,
        ls="-" if s in FEATURED else ":",
        label=NAMES[s],
    )
ax.plot(
    HORIZONS,
    [BASE_RATE[h]["base_rate_excess_median"] * 100 for h in HORIZONS],
    color=TEXT,
    lw=0,
    marker="o",
    ms=4,
    label="any day (the base rate)",
)
ax.axhline(0, color=TEXT2, lw=0.8)
ax.set_xlabel('trading days after the rule said "bullish"')
ax.set_ylabel("TQQQ minus QQQ, median (pct points)")
ax.set_title("What the extra was worth, day by day after a call, against starting on any day")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
img_path = save(fig, "lev_excess_path.png")

# Chart 3: growth of $100k, out of sample
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ends = []
for s in FEATURED:
    e = EQ[s] / EQ[s].iloc[0] * 100_000
    ax.plot(
        e.index,
        e,
        color=COLOR[s],
        lw=1.6 if s in ("hold_base", "lev_hold") else 1.3,
        ls="--" if s == "lev_hold" else "-",
        label=NAMES[s],
    )
    ends.append((s, float(e.iloc[-1])))
ends.sort(key=lambda t: t[1])
ys = [np.log10(v) for _, v in ends]
for i in range(1, len(ys)):
    ys[i] = max(ys[i], ys[i - 1] + 0.17)
for (s, v), y in zip(ends, ys, strict=True):
    ax.annotate(
        f"{SHORT[s]} {money(v)}",
        (oos_end, 10**y),
        xytext=(5, 0),
        textcoords="offset points",
        color=TEXT2,
        fontsize=7.5,
        va="center",
    )
log_money_axis(ax, [10_000, 100_000, 1_000_000, 10_000_000])
ax.set_xlim(oos_start, oos_end + pd.Timedelta(days=1300))
ax.set_title(f"Growth of $100,000 from {oos_start:%B %Y}, every decision made out of sample")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
years_axis(ax, 2)
img_growth = save(fig, "lev_growth.png")

# Chart 4: drawdowns
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for s in ("hold_base", "lev_hold", lead, "lev_voltarget"):
    if s not in EQ:
        continue
    dd = EQ[s] / EQ[s].cummax() - 1.0
    if s == "lev_hold":
        ax.fill_between(dd.index, dd * 100, 0, color=ORANGE, alpha=0.15, linewidth=0)
    ax.plot(dd.index, dd * 100, color=COLOR[s], lw=1.2, label=NAMES[s])
ax.set_ylabel("% below the previous high")
ax.set_title("How far under water each approach was, day by day")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
years_axis(ax, 2)
img_dd = save(fig, "lev_drawdown.png")

# Chart 5: the crises
fig, axes = plt.subplots(
    2, 4, figsize=(7.2, 4.0), sharex=True, gridspec_kw={"hspace": 0.55, "wspace": 0.6}
)
for ax, (label, _) in zip(axes.flat, CRISES.items(), strict=False):
    vals = [CRISIS_DD[s][label] * 100 for s in FEATURED]
    ax.barh(range(len(FEATURED)), vals, color=[COLOR[s] for s in FEATURED], height=0.7)
    ax.set_yticks(range(len(FEATURED)))
    ax.set_yticklabels([SHORT[s] for s in FEATURED], fontsize=7)
    ax.invert_yaxis()
    ax.set_title(label, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        ax.annotate(
            f"{v:.0f}%",
            (0, i),
            xytext=(2, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.5,
            color=TEXT2,
        )
axes.flat[-1].axis("off")
for ax in axes[1][:3]:
    ax.set_xlabel("worst loss (%)", fontsize=8)
axes[0][0].set_xlim(-100, 32)
for ax in axes.flat:
    ax.set_xticks([-100, -50, 0])
img_crises = save(fig, "lev_crises.png")

# Chart 6: QQQ with the lead rule's "in TQQQ" stretches shaded
fig, ax = plt.subplots(figsize=(7.2, 3.2))
qe = EQ["hold_base"] / EQ["hold_base"].iloc[0] * 100_000
ex = EXPO[lead].reindex(qe.index).ffill().fillna(0)
ax.plot(qe.index, qe, color=BLUE, lw=1.1)
ax.fill_between(
    qe.index,
    qe.min() * 0.9,
    qe.max() * 1.05,
    where=ex >= 0.5,
    color=COLOR[lead],
    alpha=0.22,
    linewidth=0,
    label="rule said: hold TQQQ",
)
log_money_axis(ax, [100_000, 200_000, 400_000, 800_000, 1_600_000])
ax.set_ylim(qe.min() * 0.9, qe.max() * 1.05)
ax.set_title(f"QQQ (blue) with the days rule {NAMES[lead][:1]} held TQQQ shaded")
ax.legend(loc="upper left")
years_axis(ax, 2)
img_shade = save(fig, "lev_exposure.png")

# Chart 7: across the pairs, worst loss of the leveraged fund held vs the best rule
fig, ax = plt.subplots(figsize=(7.2, 3.0))
labels, held, ruled, basedd = [], [], [], []
for key in PAIR_ORDER:
    rules_here = [s for s in RULES if (key, s) in ROW]
    if (key, "lev_hold") not in ROW or not rules_here:
        continue
    bst = max(rules_here, key=lambda s: ROW[(key, s)]["oos_sharpe"])
    labels.append(key)
    held.append(ROW[(key, "lev_hold")]["oos_max_dd"] * 100)
    ruled.append(ROW[(key, bst)]["oos_max_dd"] * 100)
    basedd.append(ROW[(key, "lev_hold")]["mkt_max_dd"] * 100)
x = np.arange(len(labels))
ax.bar(x - 0.27, held, 0.26, color=ORANGE, label="leveraged fund, held")
ax.bar(x, ruled, 0.26, color=AQUA, label="leveraged fund, best rule")
ax.bar(x + 0.27, basedd, 0.26, color=BLUE, label="plain fund, held")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("worst loss (%)")
ax.set_title("Worst loss across the eight pairs, out of sample")
ax.legend(loc="lower right", ncol=3)
img_pairs = save(fig, "lev_pairs.png")

# ---- PDF ---------------------------------------------------------------------
ss = getSampleStyleSheet()
H1 = ParagraphStyle(
    "H1",
    parent=ss["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=17,
    textColor=NAVY,
    spaceBefore=14,
    spaceAfter=8,
    keepWithNext=1,
)
H2 = ParagraphStyle(
    "H2",
    parent=ss["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12.5,
    textColor=NAVY,
    spaceBefore=12,
    spaceAfter=5,
    keepWithNext=1,
)
BODY = ParagraphStyle(
    "Body",
    parent=ss["Normal"],
    fontName="Helvetica",
    fontSize=10.5,
    leading=15,
    spaceAfter=8,
    alignment=TA_LEFT,
)
SMALL = ParagraphStyle(
    "Small", parent=BODY, fontSize=9, leading=12, textColor=colors.HexColor(TEXT2)
)
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=4)
CAP = ParagraphStyle("Cap", parent=SMALL, spaceBefore=2, spaceAfter=12)
TITLE = ParagraphStyle(
    "Title",
    parent=ss["Title"],
    fontName="Helvetica-Bold",
    fontSize=24,
    leading=30,
    textColor=NAVY,
    spaceAfter=6,
    alignment=TA_LEFT,
)
SUB = ParagraphStyle(
    "Sub", parent=BODY, fontSize=12.5, leading=17, textColor=colors.HexColor(TEXT2), spaceAfter=18
)
BOX = ParagraphStyle(
    "Box",
    parent=BODY,
    backColor=colors.HexColor("#eef3fa"),
    borderPadding=8,
    spaceBefore=6,
    spaceAfter=14,
)
cell = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=10.5, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
cellr = ParagraphStyle("cellr", parent=cell, alignment=2)
cellrb = ParagraphStyle("cellrb", parent=cellr, fontName="Helvetica-Bold")
hdr = ParagraphStyle("hdr", parent=cell, fontName="Helvetica-Bold", textColor=NAVY)
hdrr = ParagraphStyle("hdrr", parent=hdr, alignment=2)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullet(t):
    return Paragraph(t, BUL, bulletText="•")


def fig_block(path, caption=None, width=6.6 * inch):
    im = Image(str(path))
    ratio = im.imageHeight / im.imageWidth
    im.drawWidth = width
    im.drawHeight = width * ratio
    return KeepTogether([im, P(caption, CAP)])


def table(data, widths, bold_rows=(), align_right_from=1):
    out = []
    for i, row in enumerate(data):
        cells = []
        for j, c in enumerate(row):
            if i == 0:
                st = hdrr if j >= align_right_from else hdr
            elif i in bold_rows:
                st = cellrb if j >= align_right_from else cellb
            else:
                st = cellr if j >= align_right_from else cell
            cells.append(Paragraph(str(c), st))
        out.append(cells)
    t = Table(out, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


# numbers used in the prose
qqq, tqqq, br = R["hold_base"], R["lev_hold"], R[best_rule]
ld = R[lead]
n_wins = len(OUT[(HEAD, "lev_hold")]["windows"])
ev_lead = EV[lead]
r1y = EV_ROWS[lead].get(252, {})
r3m = EV_ROWS[lead].get(63, {})
r2y = EV_ROWS[lead].get(504, {})
b1y, b3m, b2y = BASE_RATE[252], BASE_RATE[63], BASE_RATE[504]
edge_1y = edge(lead, 252)
edge_3m = edge(lead, 63)
hits_pos_1y = r1y.get("lev_positive")
st = STINTS[lead]
st_won = int((st["lev"] > st["base"]).sum()) if len(st) else 0
st_med_days = float(st["days"].median()) if len(st) else float("nan")
real_rows = {s: ROW_R.get((HEAD, s)) for s in ["lev_hold", *RULES]} if SR else {}

story = []

# ---- cover + short version -----------------------------------------------------
short_version = (
    f"<b>Our verdict: this is not a strategy we would run with real money.</b> Not holding TQQQ, and not "
    f"switching into it on a signal either. Here is why in four numbers. "
    f"<b>Holding TQQQ all the time</b> is a bet on never seeing a bear market: from March 2000 the rebuilt fund "
    f"lost {loss(max_dd(FULL_LEV))} of its value, and even over the {years_span:.0f} graded years since "
    f"{oos_start:%B %Y} it fell {loss(tqqq['oos_max_dd'])} at its worst, against {loss(qqq['oos_max_dd'])} for QQQ. "
    f"<b>Switching only when the market has turned</b> is better than holding, but not good enough. The best of "
    f"the six rules we tested, <b>{NAMES[best_rule][3:]}</b>, grew money at {pct(br['oos_cagr'])} a year against "
    f"{pct(qqq['oos_cagr'])} for QQQ, and on the way it lost <b>{loss(br['oos_max_dd'])}</b> of its value in one stretch. "
    f"The rule with the shallowest loss, {NAMES[calmest_rule][3:]}, still lost {loss(R[calmest_rule]['oos_max_dd'])}. "
    f"A plan that can lose two-thirds or more of your money is not a plan most people will stay with, and if you do not "
    f"stay with it you get the loss without the recovery. "
    f'<b>How long to hold:</b> after rule {NAMES[lead][:1]} said "bullish", TQQQ beat QQQ by a median of '
    f"{pct(r1y.get('excess_median'), 0, True)} over the next year, but buying on <i>any</i> day gave "
    f"{pct(b1y['base_rate_excess_median'], 0, True)}. The rule's own edge is about {pct(edge_1y, 0, True)} a year and "
    f"close to nothing over a few weeks; most of the extra is just leverage over a market that happened to rise. "
    f"The signals did not know the market was turning before it was obvious. They only knew, late, that it had."
)
story += [
    P(TITLE_TEXT, TITLE),
    P(
        "What a 3x fund did when the market turned bullish, how long it was worth holding, "
        "and whether the idea survives an honest test. Andy's question, run through the machine.",
        SUB,
    ),
    P(
        f"Prepared for The Eagle Hill Fund. Out-of-sample results run from {oos_start:%B %Y} through "
        f"{oos_end:%B %Y} on QQQ and TQQQ, with seven look-alike pairs checked alongside. Study stamp {STAMP}.",
        SMALL,
    ),
    Spacer(1, 10),
    P("The short version", H1),
    P(short_version, BOX),
    P(
        "The question that came in was in two parts: when the market is turning bullish, does a 3x fund like TQQQ beat its "
        "plain cousin QQQ, and how long should you hold it? Both parts have an answer below, and the answers are tied together, "
        'because much of "is the market turning bullish" is really "is this a bull market", and a 3x fund in a bull market is '
        "a wonderful thing until the bull market ends. The rules in this report are six different ways of asking that question "
        "of the price chart, and the test is whether any of them knew the answer before it was obvious."
    ),
]

# ---- Part 1: what a leveraged fund is -------------------------------------------
story += [
    PageBreak(),
    P("Part 1: What a 3x fund actually does", H1),
    P(
        "TQQQ promises three times the <i>daily</i> move of the Nasdaq-100. The word daily matters. If QQQ rises 1% today, "
        "TQQQ rises about 3%. If QQQ falls 1% tomorrow, TQQQ falls about 3%. Over a single day the promise holds. Over "
        "months it drifts, and the drift has a direction: against you, in proportion to how choppy the market is."
    ),
    P("Volatility drag, in one example", H2),
    P(
        "Say QQQ goes up 10% one day and down 10% the next. QQQ ends at 99% of where it started, a loss of 1%. TQQQ goes up 30% "
        "then down 30%: 1.30 times 0.70 is 0.91, a loss of 9%. The market round-tripped and the 3x fund lost nine times "
        "as much as the market did. That is the price of resetting the leverage every day. In a calm, steadily rising market "
        "the reset works <i>for</i> you (gains compound on gains) and the fund can deliver more than three times the market. "
        "In a choppy or falling one it works against you, hard. The fund also pays interest on the borrowed two-thirds and a "
        "fee of about 1% a year."
    ),
    P(
        f"Table 1 is the drag measured on TQQQ's own history, from its launch in February 2010 through {oos_end:%B %Y}. "
        f"The fund delivered {num(fit['beta'], 2)} times QQQ's daily move (not quite 3.0), tracked it with a correlation of "
        f"{num(fit['corr'], 3)}, and bled about {pct(fit['drag_annual'])} a year to fees and financing on top of the volatility "
        "drag the arithmetic above implies."
    ),
    table(
        [
            ["What we measured", "Value", "What it means"],
            ["Daily multiple", num(fit["beta"], 2), "the fund's move per 1% move in QQQ"],
            ["Correlation", num(fit["corr"], 3), "how faithfully it tracks, day to day"],
            [
                "Fees and financing",
                pct(fit["drag_annual"]) + " a year",
                "what leaves the fund before any market move",
            ],
            [
                "Growth, real fund",
                f"{fit['growth_real']:.0f}x",
                f"$1 became ${fit['growth_real']:.0f} from 2010 to {oos_end:%Y}",
            ],
            [
                "Growth, our rebuild",
                f"{fit['growth_synthetic']:.0f}x",
                "the same, from QQQ and interest rates alone (Appendix A)",
            ],
        ],
        [1.6 * inch, 1.2 * inch, 3.8 * inch],
        align_right_from=99,
    ),
    P("Table 1. TQQQ against QQQ on the fund's own history.", CAP),
    P("The long view", H2),
    P(
        f"TQQQ only launched in 2010, which means its entire recorded life has been one long bull market with a few sharp "
        f"interruptions. That is a flattering record to judge it on. To see what it would have done in a real bear market "
        f"we rebuilt it from QQQ and Treasury-bill rates back to 1999, using the multiple and drag measured on the real fund "
        f"(Appendix A checks the rebuild: it lands within {pct(abs(fit['growth_synthetic'] / fit['growth_real'] - 1), 0)} "
        f"of the real fund over sixteen years). Figure 1 is the result, and it is the single most important chart in this report."
    ),
    fig_block(
        img_history,
        f"Figure 1. QQQ and the rebuilt TQQQ from March 2000. The shaded years are before the fund existed. $100,000 in the 3x "
        f"fund fell to {money(lv.min())} by {lv.idxmin():%B %Y} and {recovery}.",
    ),
    P(
        f"Someone who bought the 3x fund at the top in March 2000 was down {loss(max_dd(FULL_LEV))} at the bottom and, even "
        f"after the longest technology bull market in history, ended with {money(lv.iloc[-1])} against {money(b.iloc[-1])} for "
        'plain QQQ. That is the whole case against "just hold TQQQ", and it is why the question is worth asking the way Andy '
        "asked it: hold it <i>when</i>, not hold it."
    ),
]

# ---- Part 2: the six rules ------------------------------------------------------
rule_rows = [["Rule", "Hold TQQQ while...", "Go back to QQQ when...", "Settings tried"]]
rule_desc = {
    "lev_sma": (
        "QQQ is above its moving average (100 to 250 days), by at least a small band",
        "it closes below the average by the same band",
        "average length, band",
    ),
    "lev_cross": (
        "the fast average (20 or 50 days) is above the slow one (100 to 200 days): a golden cross",
        "the fast average drops below the slow one: a death cross",
        "fast, slow, band",
    ),
    "lev_swing": (
        "QQQ has risen 10 to 20% from its low of the past six or twelve months",
        "it has fallen 10 to 20% from its high over the same window",
        "rise, fall, window",
    ),
    "lev_breakout": (
        "QQQ has just made a new three-, six- or twelve-month high",
        "it closes below its one-, two- or three-month low",
        "entry window, exit window",
    ),
    "lev_vixfade": (
        "the VIX has just settled back below 18 to 22 after a scare above 28 to 32 in the past quarter; "
        "stay for one, three or six months",
        "the holding period runs out",
        "panic, calm, hold",
    ),
    "lev_voltarget": (
        "QQQ is calm: the TQQQ share is whatever runs the whole position at a target volatility of 20 to 35%",
        "it never fully leaves; the share shrinks as the market gets rougher",
        "target volatility, lookback",
    ),
}
for s in RULES:
    d = rule_desc[s]
    rule_rows.append([NAMES[s], d[0], d[1], d[2]])
story += [
    P('Part 2: Six ways to say "the market has turned"', H1),
    P(
        "A rule has to be written down before it can be tested, so we wrote down six. Each one looks only at QQQ's own price "
        "(rule 5 also looks at the VIX, the market's fear gauge) and never at TQQQ, and each one answers a yes-or-no question "
        "every day: is this a market to be levered into? Yes means the money sits in TQQQ. No means it sits in QQQ, not in cash, "
        "because the question was never whether to be in the market, only how hard. Rule 6 is the odd one out: it answers "
        '"how much" instead of "whether", holding more TQQQ when the market is calm and less when it is rough.'
    ),
    table(rule_rows, [1.3 * inch, 2.5 * inch, 1.75 * inch, 1.05 * inch], align_right_from=9),
    P(
        "Table 2. The six rules. Every setting was chosen by the walk-forward test, never by us.",
        CAP,
    ),
    P(
        "Rules 1 to 4 are four dialects of the same idea, trend-following, and they will agree with each other most of the time. "
        'That is on purpose. Andy\'s intuition that "turning bullish" is really "being in a bull market" is right, and these '
        "are the standard ways of defining one. Rule 5 is a different idea (buy the calm after the storm) and rule 6 is "
        "different again (do not pick days, pick a size)."
    ),
]

# ---- Part 3: how we tested --------------------------------------------------------
story += [
    P("Part 3: How we tested it", H1),
    P(
        f"The method is the one from <i>The Fear Gauge, Put to the Test</i>, so briefly: every year, each rule was allowed to "
        f"choose its settings only from the two years before, and was then graded on a year it had never seen. That happened "
        f"{n_wins} times in a row, from {oos_start:%B %Y} through {oos_end:%B %Y}, and the graded years were stitched together "
        "into the records you see here. Every switch paid a small slippage cost, and every rule was stress-tested at double that "
        "cost. Where a rule's record is quoted below, it is this out-of-sample record unless it says otherwise."
    ),
    P(
        "Because TQQQ did not exist before 2010 the graded years before then use the rebuilt fund from Part 1, and the years "
        "after use the real one. As a check, the three most important pairs were also run on the real funds' history alone "
        "(Part 6). The rules were also run on seven look-alike pairs: the 2x version of the same bet (QLD), the S&P 500 at 3x "
        "and 2x (UPRO, SSO), small caps (TNA), the Dow (UDOW), semiconductors (SOXL) and the technology sector (TECL)."
    ),
]

# ---- Part 4: how long to hold it ---------------------------------------------------
hz_rows = [
    [
        "Held for",
        "Calls",
        "TQQQ, median",
        "QQQ, median",
        "Extra",
        "Ahead",
        "Extra, any day",
        "Ahead, any day",
        "Edge",
    ]
]
for h in HORIZONS:
    r = EV_ROWS[lead].get(h, {})
    bb = BASE_RATE[h]
    if not r.get("n"):
        continue
    hz_rows.append(
        [
            HLABEL[h],
            int(r["n"]),
            pct(r["lev_median"], 0, True),
            pct(r["base_median"], 0, True),
            pct(r["excess_median"], 0, True),
            pct(r["hit_rate"], 0),
            pct(bb["base_rate_excess_median"], 0, True),
            pct(bb["base_rate_hit_rate"], 0),
            pct(edge(lead, h), 0, True),
        ]
    )
edge_rows = [
    ["Rule", "Calls", "3 months", "6 months", "1 year", "2 years", "Ahead at 1 year", "Worst year"]
]
for s in head_rules:
    r = EV_ROWS[s]
    if not EV[s]["n_events"]:
        continue
    edge_rows.append(
        [
            NAMES[s],
            EV[s]["n_events"],
            pct(edge(s, 63), 0, True),
            pct(edge(s, 126), 0, True),
            pct(edge(s, 252), 0, True),
            pct(edge(s, 504), 0, True),
            pct(r.get(252, {}).get("hit_rate"), 0),
            pct(r.get(252, {}).get("lev_worst"), 0),
        ]
    )
story += [
    P("Part 4: How long should you hold it?", H1),
    P(
        f'Every time a rule flipped from "no" to "yes" we call that a <b>call</b>, and we looked at what TQQQ and QQQ did '
        f"over the week, month, quarter, year and two years that followed. Rule {NAMES[lead][:1]} ({NAMES[lead][3:]}) made "
        f"{ev_lead['n_events']} calls in {years_span:.0f} years. Table 3 is its record. The two columns on the right are the "
        "comparison that matters: what you got by buying TQQQ on <i>any</i> day and holding for the same time. A rule only "
        "earns its keep if its calls beat that."
    ),
    table(
        hz_rows,
        [
            0.7 * inch,
            0.45 * inch,
            0.8 * inch,
            0.8 * inch,
            0.6 * inch,
            0.55 * inch,
            0.85 * inch,
            0.85 * inch,
            0.55 * inch,
        ],
    ),
    P(
        f"Table 3. After each call by rule {NAMES[lead][:1]}: the median return of each fund, the median extra TQQQ earned "
        "over QQQ, how often it was ahead, and the same two numbers for a purchase on any day. Edge is the rule's extra minus "
        "the any-day extra.",
        CAP,
    ),
    P(
        f"Read the one-year row. After a call, TQQQ beat QQQ by {pct(r1y.get('excess_median'), 0, True)} in the median case and "
        f"was ahead {pct(r1y.get('hit_rate'), 0)} of the time. Those are good numbers, and they are mostly not the rule's doing: "
        f"any day gave {pct(b1y['base_rate_excess_median'], 0, True)} and {pct(b1y['base_rate_hit_rate'], 0)}, because QQQ went "
        f"up over these {years_span:.0f} years and a 3x fund in a rising market earns extra. The rule's own contribution, the "
        f"edge, is about {pct(edge_1y, 0, True)} a year. Over three months it is {pct(edge_3m, 0, True)}. The edge grows with "
        "the holding period in the table because long holds that start after a confirmed turn are the ones that miss the "
        "crashes, and that is the real finding: the value of the rule is in <i>when it says leave</i>, not in the day it says enter."
    ),
    fig_block(
        img_path,
        "Figure 2. The median extra TQQQ earned over QQQ, day by day after each rule's call, against the same thing measured "
        "from every day in the record (the dots). A line above the dots is a rule adding something; a line on the dots is "
        "leverage doing all the work.",
    ),
    table(
        edge_rows,
        [
            1.6 * inch,
            0.5 * inch,
            0.75 * inch,
            0.75 * inch,
            0.7 * inch,
            0.7 * inch,
            1.0 * inch,
            0.8 * inch,
        ],
    ),
    P(
        "Table 4. Each rule's edge (its median extra minus the any-day extra) by holding period, how often TQQQ was ahead a "
        "year after its calls, and the worst single year that followed one.",
        CAP,
    ),
    P(
        f"The last column is the one to remember before acting on any of this. The worst year after a call by rule "
        f"{NAMES[lead][:1]} was a loss of {loss(r1y.get('lev_worst'))} in TQQQ. A rule that is right {pct(r1y.get('hit_rate'), 0)} of the "
        "time is wrong the rest of the time, and when a 3x fund is wrong it is wrong by a lot."
    ),
]
if len(st):
    recent = st.tail(8)
    st_rows = [["Entered TQQQ", "Back to QQQ", "Days", "TQQQ", "QQQ", "Won"]]
    for _, r in recent.iterrows():
        st_rows.append(
            [
                f"{r['start']:%d %b %Y}",
                f"{r['end']:%d %b %Y}",
                int(r["days"]),
                pct(r["lev"], 0, True),
                pct(r["base"], 0, True),
                "yes" if r["lev"] > r["base"] else "no",
            ]
        )
    story += [
        P("What the stints looked like", H2),
        P(
            f'Another way to answer "how long" is to look at how long the rule itself stayed in. Rule {NAMES[lead][:1]} '
            f"took {len(st)} stints in TQQQ, the median one lasting {st_med_days:.0f} trading days, and TQQQ beat QQQ in "
            f"{st_won} of them. Table 5 is the most recent eight."
        ),
        table(st_rows, [1.1 * inch, 1.1 * inch, 0.6 * inch, 0.8 * inch, 0.8 * inch, 0.6 * inch]),
        P(
            f"Table 5. The last eight stretches rule {NAMES[lead][:1]} spent in TQQQ, and what each fund returned over "
            "the stretch.",
            CAP,
        ),
        fig_block(
            img_shade,
            f"Figure 3. QQQ, with the days rule {NAMES[lead][:1]} held TQQQ shaded. The gaps are the bear markets it sat out "
            "in QQQ.",
        ),
    ]

# ---- Part 5: the scoreboard ---------------------------------------------------------
sb_rows = [
    [
        "Approach",
        "Return / yr",
        "Worst loss",
        "Sharpe",
        "Time in TQQQ",
        "Switches / yr",
        "2008",
        "2020",
        "2022",
        "$100k became",
    ]
]
bold = []
for s in ORDER:
    if s not in R:
        continue
    r = R[s]
    if s == best_rule:
        bold.append(len(sb_rows))
    sb_rows.append(
        [
            NAMES[s],
            pct(r["oos_cagr"]),
            pct(r["oos_max_dd"], 0),
            pp(r.get("oos_sharpe")),
            pct(r["time_lev"], 0),
            num(r["switch_py"]),
            pct(r.get("oos_dd_2008"), 0),
            pct(r.get("oos_dd_2020"), 0),
            pct(r.get("oos_dd_2022"), 0),
            money(final[s]) if s in final else "",
        ]
    )
story += [
    PageBreak(),
    P("Part 5: The scoreboard", H1),
    P(
        f"Table 6 is every approach on the same {years_span:.0f} graded years. Return per year and worst loss are the two "
        "numbers to hold side by side; the three crisis columns are the worst loss inside each episode."
    ),
    table(
        sb_rows,
        [
            1.35 * inch,
            0.65 * inch,
            0.6 * inch,
            0.55 * inch,
            0.65 * inch,
            0.65 * inch,
            0.5 * inch,
            0.5 * inch,
            0.5 * inch,
            0.75 * inch,
        ],
        bold_rows=bold,
    ),
    P(
        f"Table 6. Out of sample, {oos_start:%B %Y} to {oos_end:%B %Y}, QQQ/TQQQ with the rebuilt history before 2010. "
        "Sharpe is the return above T-bills per unit of bumpiness. Hold QQQ has no Sharpe here because it was not walked forward.",
        CAP,
    ),
    fig_block(
        img_growth,
        f"Figure 4. Growth of $100,000, out of sample. Note the scale: each gridline is ten times the one below. "
        f"TQQQ held throughout ends at {money(final['lev_hold'])}, the best rule at {money(final[best_rule])}, QQQ at "
        f"{money(final['hold_base'])}.",
    ),
    P(
        f"Three things stand out. First, every rule beat holding TQQQ on worst loss, and {sum(R[s]['oos_cagr'] > tqqq['oos_cagr'] for s in head_rules)} "
        f"of the {len(head_rules)} beat it on return as well, because sitting out the worst stretches is worth more than the gains they cost. "
        f"Second, and this is the finding that decides the verdict, every rule still lost far more than QQQ at its worst. The "
        f"best rule by Sharpe, {NAMES[best_rule][3:]}, lost {loss(br['oos_max_dd'])} in one stretch; the shallowest, "
        f"{NAMES[calmest_rule][3:]}, lost {loss(R[calmest_rule]['oos_max_dd'])}; QQQ lost {loss(qqq['oos_max_dd'])}. "
        "Leverage that only steps aside after the market has already turned always eats the first leg of every decline, and "
        "the first leg of a 3x decline is enormous. "
        f"Third, the trend rules (1 to 4) are {num(min(R[s]['switch_py'] for s in ('lev_sma', 'lev_cross', 'lev_swing', 'lev_breakout') if s in R))} to "
        f"{num(max(R[s]['switch_py'] for s in ('lev_sma', 'lev_cross', 'lev_swing', 'lev_breakout') if s in R))} switches a year, "
        "which is a real-life tax bill as well as a slippage cost; rule 6 trades far more than that for a smoother ride."
    ),
    fig_block(
        img_dd,
        "Figure 5. How far under water each approach was. The orange shading is TQQQ held throughout.",
    ),
    fig_block(
        img_crises,
        "Figure 6. The worst loss inside each crisis. A rule that reacts to a slow decline (2008, 2022) looks much better than "
        "one that meets a crash from a standing start (Covid 2020, April 2025).",
    ),
    P(
        "The crisis panels show where the rules earn their keep and where they cannot. The slow bear markets of 2008 and 2022 "
        "gave them months of warning, and they stepped aside. The Covid crash took five weeks from top to bottom and no "
        "trend rule gets out of a 3x fund fast enough for that; only rule 6, which was already holding less TQQQ because the "
        "market had been calm for too long, fared noticeably better."
    ),
]

# ---- Part 6: other pairs and the real-history check ------------------------------------
pair_rows = [
    [
        "Pair",
        "What it is",
        "Plain held",
        "Lev. held",
        "Best rule",
        "Rule return",
        "Rule worst",
        "Lev. worst",
    ]
]
for key in PAIR_ORDER:
    rules_here = [s for s in RULES if (key, s) in ROW]
    if (key, "lev_hold") not in ROW or not rules_here:
        continue
    bst = max(rules_here, key=lambda s: ROW[(key, s)]["oos_sharpe"])
    pair_rows.append(
        [
            key,
            PAIRS[key].name,
            pct(ROW[(key, "lev_hold")]["mkt_cagr"]),
            pct(ROW[(key, "lev_hold")]["oos_cagr"]),
            NAMES[bst][3:],
            pct(ROW[(key, bst)]["oos_cagr"]),
            pct(ROW[(key, bst)]["oos_max_dd"], 0),
            pct(ROW[(key, "lev_hold")]["oos_max_dd"], 0),
        ]
    )
story += [
    P("Part 6: Does it carry to other funds?", H1),
    P(
        "The same six rules were run on seven other pairs, each on the plain fund's own history with the leveraged fund "
        "rebuilt before its launch. Table 7 shows holding each fund against the best rule for that pair, chosen by Sharpe. "
        "Figure 7 puts the worst losses side by side."
    ),
    table(
        pair_rows,
        [
            0.8 * inch,
            1.6 * inch,
            0.65 * inch,
            0.65 * inch,
            1.1 * inch,
            0.7 * inch,
            0.65 * inch,
            0.65 * inch,
        ],
        align_right_from=2,
    ),
    P(
        'Table 7. Return per year and worst loss across the pairs, out of sample. "Best rule" is the highest Sharpe of the '
        "six for that pair.",
        CAP,
    ),
    fig_block(
        img_pairs,
        "Figure 7. Worst loss by pair: leveraged fund held, leveraged fund under the best rule, plain fund held.",
    ),
]
if SR and real_rows.get("lev_hold"):
    rr_rows = [
        [
            "Approach",
            "Return, rebuilt",
            "Worst, rebuilt",
            "Return, real only",
            "Worst, real only",
            "Sharpe, real only",
        ]
    ]
    for s in ["lev_hold", *RULES]:
        rr = real_rows.get(s)
        if not rr or s not in R:
            continue
        rr_rows.append(
            [
                NAMES[s],
                pct(R[s]["oos_cagr"]),
                pct(R[s]["oos_max_dd"], 0),
                pct(rr["oos_cagr"]),
                pct(rr["oos_max_dd"], 0),
                pp(rr["oos_sharpe"]),
            ]
        )
    real_start_oos = (
        OUT_R[(HEAD, "lev_hold")]["windows"][0]["test_start"]
        if (HEAD, "lev_hold") in OUT_R
        else "2012"
    )
    story += [
        P("The real-history check", H2),
        P(
            f"Everything before 2010 above leans on our rebuild of TQQQ. To make sure the conclusions do not depend on it, "
            f"the QQQ/TQQQ rules were re-run on the real fund alone, graded from {pd.Timestamp(real_start_oos):%B %Y}. "
            "Table 8 puts the two records side by side. The real-only years are a bull market with interruptions, so every "
            "number is kinder, but the ordering is the same: the rules trade some return for a shallower worst loss, and none "
            "of them makes TQQQ safe."
        ),
        table(rr_rows, [1.6 * inch, 0.9 * inch, 0.9 * inch, 0.95 * inch, 0.95 * inch, 0.95 * inch]),
        P("Table 8. QQQ/TQQQ on the rebuilt history versus on the real fund's history only.", CAP),
    ]

# ---- Part 7: what it will not do -------------------------------------------------------
story += [
    P("Part 7: What this will not do", H1),
    bullet(
        "<b>It will not make a 3x fund safe.</b> Every rule here lost more than half its money at some point, out of sample. "
        "The rules shorten the bear markets you sit through; they do not remove them, and a one-day crash reaches you in full."
    ),
    bullet(
        "<b>It will not tell you the day to buy.</b> The edge from the entry call is small over short horizons. What a rule adds "
        "is a disciplined exit, and the willingness to be back in TQQQ after the exit was wrong."
    ),
    bullet(
        "<b>The history is partly rebuilt.</b> TQQQ has sixteen years of real data, all of them in a bull market. The years "
        "before are our reconstruction from QQQ, checked against the real fund (Appendix A) but still a model."
    ),
    bullet(
        "<b>Taxes are not in it.</b> A switch from TQQQ to QQQ is a sale. In a taxable account the trend rules' two to six "
        "switches a year turn long-term gains into short-term ones, which can cost more than slippage does."
    ),
    bullet(
        "<b>The formal gauntlet killed every rule.</b> As with the VIX dials, three of its checks are built for strategies "
        "that trade hundreds of times (the trade count, the deflated Sharpe, the overfitting test on tiny grids), and a dial "
        "that switches a few times a year fails them by construction. The checks it can pass (a positive out-of-sample Sharpe, "
        "consistency across the graded years, survival at double cost) it did pass. We report the verdict as it stands and "
        "will revisit the gauntlet's thresholds for dials separately."
    ),
    bullet(
        "<b>Rule 6 trades a lot.</b> Size-by-calm rebalances whenever its target share drifts by ten points, which is many "
        "times a year. Its smoother ride is real, and so is its cost in a taxable account."
    ),
    P("What we recommend", H1),
    P(
        f"<b>Do not run this.</b> The headline return is real ({pct(br['oos_cagr'])} a year for the best rule against "
        f"{pct(qqq['oos_cagr'])} for QQQ, which over {years_span:.0f} years is the difference between {money(final[best_rule])} and "
        f"{money(final['hold_base'])} from $100,000) and that is the case for it, stated fairly. But it was earned by sitting "
        f"through a {loss(br['oos_max_dd'])} loss, and the kindest rule in the study still asked for {loss(R[calmest_rule]['oos_max_dd'])}. "
        "Those are not numbers an investor sits through; they are numbers an investor sells at the bottom of, and selling at the "
        "bottom of a 3x fund is how the return in the table turns into a permanent loss in a real account. A strategy that only "
        "works if you never flinch is not a strategy, it is a dare."
    ),
    P(
        "<b>What the test did show.</b> Switching on a rule is better than holding TQQQ outright, by a lot. Trend rules get "
        "out of slow bear markets (2008, 2022) and cannot get out of fast ones (Covid, April 2025). And the entry signal adds "
        "little: the extra return after a call is mostly what any day would have given. If there is a version of this idea "
        "worth revisiting, it is one that fixes the fast-crash problem, which is why the next study adds the fear gauge as a "
        "second exit. That is queued, not started, and nothing in this report should be acted on before it reads."
    ),
    P(
        f"<b>If someone runs it anyway,</b> the least-bad version is rule {NAMES[lead][:1]}: hold TQQQ while QQQ is above its "
        f"200-day line, go back to QQQ when it closes below, about {num(ld['switch_py'], 0)} switches a year, with a share of the "
        "portfolio whose halving would not change any decision in your life. And hold it for exactly as long as the rule says, "
        'not through a close below the line because it "will probably come back". The whole record above was earned by leaving '
        "on the signal and coming back on the signal, including the times that was embarrassing."
    ),
]

# ---- Appendix ---------------------------------------------------------------------------
fit_rows = [
    [
        "Pair",
        "Multiple",
        "Correlation",
        "Drag / yr",
        "Real growth",
        "Rebuilt growth",
        "Fund from",
        "Rebuilt from",
    ]
]
for key in PAIR_ORDER:
    f = FACTS[key]["fit"]
    fit_rows.append(
        [
            key,
            num(f["beta"], 2),
            num(f["corr"], 3),
            pct(f["drag_annual"]),
            f"{f['growth_real']:.1f}x",
            f"{f['growth_synthetic']:.1f}x",
            f"{pd.Timestamp(f['real_start']):%b %Y}",
            f"{pd.Timestamp(f['synthetic_start']):%b %Y}",
        ]
    )
param_rows = [["Rule", "Settings chosen most often", "Grid size"]]
for s in head_rules:
    p = {k: v for k, v in mode_params[s].items() if k not in ("asset", "base", "benchmark")}
    n_cells = int(np.prod([len(v) for v in GRIDS[s].values()]))
    param_rows.append([NAMES[s], ", ".join(f"{k} {v}" for k, v in p.items()), n_cells])
story += [
    PageBreak(),
    P("Appendix A: Rebuilding the leveraged funds", H1),
    P(
        "Each day the rebuilt fund grows by the multiple times the plain fund's close-to-close return, minus a daily slice "
        "of the fund's fee and the cost of borrowing the extra exposure at the Treasury-bill rate plus a spread. The multiple "
        "and the spread are the values that make the rebuild match the real fund's own history as closely as possible, so "
        "the drag you see in Table 9 is measured, not assumed. The rebuilt series is spliced in front of the real one and the "
        "real one is used wherever it exists."
    ),
    table(
        fit_rows,
        [
            0.85 * inch,
            0.65 * inch,
            0.8 * inch,
            0.7 * inch,
            0.85 * inch,
            0.95 * inch,
            0.9 * inch,
            0.9 * inch,
        ],
    ),
    P(
        "Table 9. The fit for every pair. Real growth and rebuilt growth are $1 grown over the real fund's lifetime; "
        "the gap between them is the rebuild's error.",
        CAP,
    ),
    P("Appendix B: The settings the test chose", H1),
    P(
        f"Each rule re-chose its settings {n_wins} times. Table 10 lists the settings chosen most often and the number of "
        "combinations each rule had to pick from. Small grids were deliberate: the more settings a rule can try, the easier "
        "it is to fool."
    ),
    table(param_rows, [1.7 * inch, 3.9 * inch, 1.0 * inch], align_right_from=2),
    P("Table 10. Settings, QQQ/TQQQ, extended history.", CAP),
    P("Glossary", H1),
    bullet(
        "<b>Leveraged fund.</b> A fund that promises a multiple (2x, 3x) of an index's <i>daily</i> move, using borrowed money and derivatives."
    ),
    bullet(
        "<b>Volatility drag.</b> The loss a daily-reset leveraged fund suffers in a choppy market, from compounding the multiple of each day's move."
    ),
    bullet('<b>Call.</b> A day on which a rule flipped from "hold QQQ" to "hold TQQQ".'),
    bullet(
        "<b>Base rate.</b> What buying TQQQ on any day and holding for a given time delivered. The yardstick for a call."
    ),
    bullet(
        "<b>Edge.</b> A rule's median extra return minus the base rate's. The part of the result that the rule, not the leverage, produced."
    ),
    bullet(
        "<b>Walk-forward test.</b> A backtest where the rule's settings are chosen only from the past at every step, then graded on the year that follows. The graded years are <b>out of sample</b>."
    ),
    bullet(
        "<b>Compound annual growth rate (return per year).</b> The steady yearly rate that turns the start value into the end value."
    ),
    bullet("<b>Maximum drawdown (worst loss).</b> The deepest peak-to-trough fall."),
    bullet(
        "<b>Sharpe ratio.</b> Return above T-bills divided by the bumpiness of that return. Higher is smoother."
    ),
    bullet(
        "<b>Moving average.</b> The average price over the last N days; the 200-day line is the best known."
    ),
    bullet(
        "<b>Golden cross / death cross.</b> A short moving average crossing above / below a long one."
    ),
    bullet(
        "<b>VIX.</b> The market's fear gauge: the swing the options market expects over the next month, in annual percent."
    ),
    bullet(
        "<b>Slippage.</b> The small amount a real trade costs beyond the quoted price. Charged on every switch here, and doubled in a stress test."
    ),
]

doc = SimpleDocTemplate(
    str(OUT_PDF),
    pagesize=letter,
    leftMargin=0.9 * inch,
    rightMargin=0.9 * inch,
    topMargin=0.85 * inch,
    bottomMargin=0.85 * inch,
    title=TITLE_TEXT,
    author="The Eagle Hill Fund",
)


def footer(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(TEXT2))
    canvas.drawString(0.9 * inch, 0.55 * inch, f"The Eagle Hill Fund  |  {TITLE_TEXT}")
    canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, f"Page {doc_.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)

# ---- the markdown body for the library page ------------------------------------------------
md_short = (
    short_version.replace("<b>", "**")
    .replace("</b>", "**")
    .replace("<i>", "*")
    .replace("</i>", "*")
)
OUT_MD.write_text(
    f"""**The short version.** {md_short}

**How to read it.** Part 1 explains what a 3x fund does and shows what TQQQ would have done through the dot-com crash. \
Part 2 writes Andy's "turning bullish" down as six testable rules. Part 4 answers "how long to hold" with the record of \
every call, measured against buying on any day. Part 5 is the scoreboard, where the verdict comes from: look at the \
"worst loss" column before the "return" column. Part 6 is the seven look-alike pairs and the real-history check, \
Part 7 what the result will not do, and the recommendation at the end says plainly what we would and would not do.

**The backtests.** The cards below are the out-of-sample runs the charts are built from; open one to see its equity \
curve, trades and settings in the lab, or change a setting and re-run it in the playground. The study's full scorecard \
table is underneath them.
"""
)

# ---- what to publish ----------------------------------------------------------------------
print("wrote", OUT_PDF)
print("wrote", OUT_MD)
print("final", {k: round(v) for k, v in final.items()})
links = [
    (OUT[(HEAD, lead)]["stitched_hash"], f"{NAMES[lead][3:]} on QQQ/TQQQ, out of sample"),
    (OUT[(HEAD, best_rule)]["stitched_hash"], f"{NAMES[best_rule][3:]} on QQQ/TQQQ, out of sample"),
    (OUT[(HEAD, "lev_hold")]["stitched_hash"], "Hold TQQQ throughout, out of sample"),
    (OUT[(HEAD, "lev_hold")]["full_hash"], "Hold TQQQ from March 2000 (rebuilt history)"),
    (HOLD[HEAD]["hash"], "Hold QQQ from March 2000"),
]
if "lev_voltarget" in head_rules and best_rule != "lev_voltarget":
    links.append(
        (OUT[(HEAD, "lev_voltarget")]["stitched_hash"], "Size by calm on QQQ/TQQQ, out of sample")
    )
seen: set[str] = set()
args = []
for h, label in links:
    if h in seen:
        continue
    seen.add(h)
    args.append(f'--run {h}="{label}"')
print("publish:", " ".join(args))
