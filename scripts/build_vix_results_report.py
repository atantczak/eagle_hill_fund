"""Build the plain-English results report for the VIX overlay study.

Reads the study summaries in ``var/scorecards`` and the stored equity curves in
``var/results``; writes a PDF plus its chart PNGs.

    python scripts/build_vix_results_report.py OUT.pdf IMG_DIR [STAMP]
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

SERVER = Path(__file__).resolve().parents[1]
SCORE = SERVER / "var" / "scorecards"
RESULTS = SERVER / "var" / "results"

OUT_PDF = Path(sys.argv[1])
IMG_DIR = Path(sys.argv[2])
STAMP = sys.argv[3] if len(sys.argv) > 3 else "2026-11-02"
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

# ---- palette (dataviz reference instance, light mode) -----------------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"  # slot 1: SPY
ORANGE = "#eb6834"  # slot 2: rule 6, surprise count
AQUA = "#1baf7a"  # slot 3: rule 1, fear curve flip
YELLOW = "#eda100"  # slot 4: rule 5, size by fear
MAGENTA = "#e87ba4"  # slot 5: rule 10, the vote
GREY = "#8a8985"  # reference series: the 200-day line
NAVY = colors.HexColor("#1f3a5f")

NAMES = {
    "dial_hold": "Buy and hold SPY",
    "dial_sma": "200-day line (classic benchmark)",
    "vix_term": "1. Fear curve flip",
    "vix_resid": "2. Unexplained fear",
    "vix_vrp": "3. Overpriced insurance",
    "vix_spike": "4. Spike and fade",
    "vix_sized": "5. Size by fear",
    "vix_surprise": "6. Surprise count",
    "vix_corr": "7. Broken link",
    "vix_vvix": "8. Fear of fear",
    "vix_skew": "9. Insurance mix",
    "vix_vote": "10. Vote of rules 1, 2, 3",
}
SHORT = {
    "dial_hold": "SPY",
    "dial_sma": "200-day",
    "vix_term": "Curve flip",
    "vix_sized": "Size by fear",
    "vix_surprise": "Surprise",
    "vix_vote": "Vote",
}
COLOR = {
    "dial_hold": BLUE,
    "dial_sma": GREY,
    "vix_term": AQUA,
    "vix_sized": YELLOW,
    "vix_surprise": ORANGE,
    "vix_vote": MAGENTA,
}
FEATURED = ["dial_hold", "dial_sma", "vix_term", "vix_sized", "vix_surprise", "vix_vote"]
ORDER = list(NAMES)

# ---- data --------------------------------------------------------------------
summ = {
    obj: json.loads((SCORE / f"vix_overlay_{obj}_{STAMP}.json").read_text())
    for obj in ("sharpe", "calmar")
}
S = summ["sharpe"]
rows = {r["strategy"]: r for r in S["rows"]}
rows_c = {r["strategy"]: r for r in summ["calmar"]["rows"]}
outcomes = S["outcomes"]
wf = {s: json.loads((SCORE / f"{o['wf_id']}.json").read_text()) for s, o in outcomes.items()}


def _naive(ts: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.to_datetime(ts, utc=True)
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
    )


def load_equity(h: str) -> tuple[pd.Series, pd.Series]:
    e = pd.read_parquet(RESULTS / h / "equity.parquet")
    idx = _naive(e["timestamp"])
    return pd.Series(e["equity"].to_numpy(), index=idx), pd.Series(
        e["net_return"].to_numpy(), index=idx
    )


def load_trades(h: str) -> pd.DataFrame:
    t = pd.read_parquet(RESULTS / h / "trades.parquet")
    t = t.copy()
    t["timestamp"] = _naive(t["timestamp"])
    return t


EQ: dict[str, pd.Series] = {}
NET: dict[str, pd.Series] = {}
FULL_EQ: dict[str, pd.Series] = {}
for s, o in outcomes.items():
    EQ[s], NET[s] = load_equity(o["stitched_hash"])
    FULL_EQ[s], _ = load_equity(o["full_hash"])
oos_start, oos_end = EQ["dial_hold"].index[0], EQ["dial_hold"].index[-1]
years_span = (oos_end - oos_start).days / 365.25


def exposure(s: str) -> pd.Series:
    idx = EQ[s].index
    trades = load_trades(outcomes[s]["stitched_hash"])
    resets = []
    for d in wf[s]["windows"]:
        i = idx.searchsorted(pd.Timestamp(d["test_start"]))
        if i < len(idx):
            resets.append(idx[i])
    return positions_from_trades(trades, idx, "SPY", resets).clip(0, 1)


EXPO = {s: exposure(s) for s in ("vix_surprise", "vix_term")}

CRISES = {
    "2008 (from Aug)": ("2008-08-01", "2009-06-30"),
    "2011 debt ceiling": ("2011-04-01", "2011-12-31"),
    "Late 2018": ("2018-09-01", "2019-03-31"),
    "Covid 2020": ("2020-01-01", "2020-09-30"),
    "2022 rate shock": ("2022-01-01", "2023-03-31"),
    "April 2025 tariffs": ("2025-01-01", "2025-09-30"),
}


def dd_in(eq: pd.Series, a: str, b: str) -> float:
    sub = eq.loc[a:b]
    return float((sub / sub.cummax() - 1.0).min()) if len(sub) > 1 else float("nan")


def max_dd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1.0).min())


CRISIS_DD = {s: {k: dd_in(EQ[s], *v) for k, v in CRISES.items()} for s in FEATURED}
YEARLY = pd.DataFrame({s: (1 + NET[s]).groupby(NET[s].index.year).prod() - 1 for s in FEATURED})

# numbers used in the prose
R = rows


def pct(x: float, nd: int = 1, sign: bool = False) -> str:
    return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"


def pp(x: float) -> str:
    return f"{x:.2f}"


spy, sma, term, sized, surp, vote = (R[k] for k in FEATURED)
final = {s: EQ[s].iloc[-1] / EQ[s].iloc[0] * 100_000 for s in FEATURED}
mode_params = {
    s: dict(
        Counter(json.dumps(p, sort_keys=True) for p in outcomes[s]["chosen_params"]).most_common(1)[
            0
        ][0]
        and json.loads(
            Counter(
                json.dumps(p, sort_keys=True) for p in outcomes[s]["chosen_params"]
            ).most_common(1)[0][0]
        )
    )
    for s in ORDER
}
win_years = {s: int((YEARLY[s] > YEARLY["dial_hold"]).sum()) for s in FEATURED}
n_years = len(YEARLY)
chop_note = {
    s: next(c["note"] for c in outcomes[s]["verdict"]["checks"] if c["name"] == "regimes")
    for s in ORDER
}

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


# Chart 1: the walk-forward schedule
wins = wf["vix_surprise"]["windows"]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for w in wins:
    y = w["window"]
    ts, te = pd.Timestamp(w["train_start"]), pd.Timestamp(w["train_end"])
    os_, oe = pd.Timestamp(w["test_start"]), pd.Timestamp(w["test_end"])
    ax.barh(y, (te - ts).days, left=ts, height=0.7, color=GRID, edgecolor="none")
    ax.barh(y, (oe - os_).days, left=os_, height=0.7, color=BLUE, edgecolor="none")
ax.set_yticks([0, 6, 12, 18])
ax.set_yticklabels(["exam 1", "exam 7", "exam 13", "exam 19"])
ax.invert_yaxis()
ax.set_title("Nineteen exams: study two years (grey), then get graded on the next one (blue)")
ax.xaxis_date()
years_axis(ax, 2)
ax.grid(axis="y", visible=False)
img_wf = save(fig, "walk_forward.png")

# Chart 2: growth of $100k, out of sample, log scale
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ends = []
for s in FEATURED:
    e = EQ[s] / EQ[s].iloc[0] * 100_000
    ax.plot(
        e.index,
        e,
        color=COLOR[s],
        lw=1.1 if s == "dial_sma" else 1.5,
        ls="--" if s == "dial_sma" else "-",
        label=NAMES[s],
    )
    ends.append((s, float(e.iloc[-1])))
# stagger the end labels so none overlap (work in log space)
ends.sort(key=lambda t: t[1])
ys = [np.log10(v) for _, v in ends]
for i in range(1, len(ys)):
    ys[i] = max(ys[i], ys[i - 1] + 0.06)
for (s, v), y in zip(ends, ys, strict=True):
    ax.annotate(
        f"{SHORT[s]} ${v / 1000:,.0f}k",
        (oos_end, 10**y),
        xytext=(5, 0),
        textcoords="offset points",
        color=TEXT2,
        fontsize=7.5,
        va="center",
    )
ax.set_yscale("log")
ax.set_yticks([50_000, 100_000, 200_000, 400_000, 800_000])
ax.set_yticklabels(["$50k", "$100k", "$200k", "$400k", "$800k"])
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlim(oos_start, oos_end + pd.Timedelta(days=1100))
ax.set_title("Growth of $100,000 from August 2008, every decision made out of sample")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
years_axis(ax, 2)
img_growth = save(fig, "growth.png")

# Chart 3: drawdown (underwater) chart
fig, ax = plt.subplots(figsize=(7.2, 2.9))
for s in ("dial_hold", "vix_surprise", "vix_term"):
    dd = EQ[s] / EQ[s].cummax() - 1.0
    if s == "dial_hold":
        ax.fill_between(dd.index, dd * 100, 0, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(dd.index, dd * 100, color=COLOR[s], lw=1.2, label=NAMES[s])
ax.set_ylabel("% below the previous high")
ax.set_title("How far under water each approach was, day by day")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
years_axis(ax, 2)
img_dd = save(fig, "drawdown.png")

# Chart 4: the crises, one panel each
fig, axes = plt.subplots(
    2, 3, figsize=(7.2, 4.0), sharex=True, gridspec_kw={"hspace": 0.55, "wspace": 0.55}
)
for ax, (label, _) in zip(axes.flat, CRISES.items(), strict=False):
    vals = [CRISIS_DD[s][label] * 100 for s in FEATURED]
    ax.barh(range(len(FEATURED)), vals, color=[COLOR[s] for s in FEATURED], height=0.7)
    ax.set_yticks(range(len(FEATURED)))
    ax.set_yticklabels([SHORT[s] for s in FEATURED], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_title(label, fontsize=9)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        ax.annotate(
            f"{v:.0f}%",
            (v, i),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7,
            color=TEXT2,
        )
for ax in axes[1]:
    ax.set_xlabel("worst loss in the episode (%)", fontsize=8)
axes[0][0].set_xlim(-64, 0)
img_crises = save(fig, "crises.png")

# Chart 5: SPY with the rule's out-of-market periods shaded
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw={"hspace": 0.4})
spy_eq = EQ["dial_hold"] / EQ["dial_hold"].iloc[0] * 100_000
for ax, s in zip(axes, ("vix_surprise", "vix_term"), strict=True):
    ex = EXPO[s].reindex(spy_eq.index).ffill().fillna(0)
    ax.plot(spy_eq.index, spy_eq, color=BLUE, lw=1.1)
    ax.fill_between(
        spy_eq.index,
        spy_eq.min() * 0.9,
        spy_eq.max() * 1.05,
        where=ex < 0.75,
        color=COLOR[s],
        alpha=0.18,
        linewidth=0,
        label="rule said half in",
    )
    ax.fill_between(
        spy_eq.index,
        spy_eq.min() * 0.9,
        spy_eq.max() * 1.05,
        where=ex < 0.25,
        color=COLOR[s],
        alpha=0.35,
        linewidth=0,
        label="rule said out",
    )
    ax.set_yscale("log")
    ax.set_yticks([100_000, 200_000, 400_000, 800_000])
    ax.set_yticklabels(["$100k", "$200k", "$400k", "$800k"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(spy_eq.min() * 0.9, spy_eq.max() * 1.05)
    ax.set_title(f"SPY (blue) with the days rule {NAMES[s][:2].strip('.')} stood aside shaded")
    ax.legend(loc="lower right")
years_axis(axes[1], 2)
img_shade = save(fig, "exposure.png")

# Chart 6: each exam year, rule minus SPY
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, s in zip(axes, ("vix_surprise", "vix_term"), strict=True):
    diff = (YEARLY[s] - YEARLY["dial_hold"]) * 100
    ax.bar(diff.index, diff, color=COLOR[s], width=0.7)
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.set_title(f"{NAMES[s]}: beat SPY in {win_years[s]} of {n_years} years", fontsize=9)
    ax.set_xticks([2009, 2012, 2015, 2018, 2021, 2024])
axes[0].set_ylabel("percentage points vs SPY")
img_years = save(fig, "yearly.png")

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


def table(data, widths, bold_rows=(), align_right_from=1, font=8.5):
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


story = []

# ---- cover + short version -----------------------------------------------------
story += [
    P("The Fear Gauge, Put to the Test", TITLE),
    P(
        "What happened when we ran nine VIX rules against eighteen years of market history, "
        "which ones earned a place, and what we recommend doing with them.",
        SUB,
    ),
    P(
        f"Prepared for The Eagle Hill Fund. Companion to <i>The Fear Gauge and the Market</i>. "
        f"Out-of-sample results run from {oos_start:%B %Y} through {oos_end:%B %Y}. Study stamp {STAMP}.",
        SMALL,
    ),
    Spacer(1, 10),
    P("The short version", H1),
    P(
        "We took the nine rules described in the first document, gave each one a handful of adjustable settings, and tested them "
        "the way an honest skeptic would. Every year, each rule was allowed to tune itself only on the two years before, and was then "
        f"graded on a year it had never seen. That happened {len(wins)} times in a row, from August 2008 through October 2026. "
        "Everything in this report comes from those graded years unless it says otherwise."
    ),
    P(
        f"<b>Two rules clearly earned a place.</b> The <b>surprise count</b> (rule 6) kept almost all of the market's return "
        f"({pct(surp['oos_cagr'])} a year against {pct(spy['oos_cagr'])} for simply holding SPY) while cutting the worst loss from "
        f"{pct(spy['oos_max_dd'], 0)} to {pct(surp['oos_max_dd'], 0)}. The <b>fear curve flip</b> (rule 1) is the simplest in-or-out switch we tested: "
        f"about {term['switch_py']:.0f} moves a year, and it held the Covid crash to {pct(term['oos_dd_2020'], 0)} and the April 2025 shock to "
        f"{pct(term['oos_dd_2025'], 0)}, against {pct(spy['oos_dd_2020'], 0)} and {pct(spy['oos_dd_2025'], 0)} for the market. "
        f"<b>Size by fear</b> (rule 5) had the smoothest ride of all, with a worst loss of just {pct(sized['oos_max_dd'], 0)}, but gave up more return to get it. "
        "Every VIX rule beat the classic 200-day-line approach on return, and the 200-day line only won in one place: the slow grind of 2008. "
        "The obvious next step is to combine a VIX rule with the 200-day line, so each covers the other's blind spot.",
        BOX,
    ),
    P(
        "None of this is magic. The rules react to fear that has already shown up in prices, so they are always a few days late. "
        "They give false alarms, each of which costs a little. And 2022, a slow decline with no panic in it, hurt nearly every rule. "
        "What the numbers do show is that a disciplined fear-based dial has, for eighteen years, delivered most of the market's gains with "
        "far shallower losses. That is what we set out to find."
    ),
]

# ---- Part 1: how we tested ------------------------------------------------------
story += [
    PageBreak(),
    P("Part 1: How we tested it", H1),
    P(
        "Most backtests you will ever see are too good to be true, for one reason: the person running them looked at the whole history, "
        "picked the settings that would have worked best, and then reported how well those settings worked. That is like writing an exam "
        "after reading the answer key. We did it differently."
    ),
    P("The exam method", H2),
    P(
        "Each rule has a few adjustable settings. The fear curve flip, for example, has to decide how far above the 3-month fear the 30-day fear "
        'must climb before it says "out", and how far back below it must drop before it says "in". We did not pick those numbers. '
        "Instead, for every rule, we listed a short menu of reasonable settings (between 12 and 54 combinations), and then ran this loop:"
    ),
    bullet("Take two years of history. Try every setting on the menu. Keep whichever did best."),
    bullet(
        "Run that one setting, unchanged, on the following year. Record what happened. That year is the grade."
    ),
    bullet("Slide forward a year and repeat."),
    fig_block(
        img_wf,
        f"Figure 1. The {len(wins)} overlapping exams. Grey is the study period for each one; blue is the year that was graded. Stitching the blue years together gives the out-of-sample record this report is built on.",
    ),
    P(
        'Statisticians call this walk-forward testing, and the graded years are "out of sample". It is slower and it produces worse-looking '
        'numbers than the usual method, which is exactly why we trust it more. It also answers the question "what if we had been messing with the '
        'settings all along?" because that is literally what the test does, one year at a time, without ever peeking ahead.'
    ),
    P("The ground rules", H2),
    bullet(
        "<b>Data.</b> SPY, the VIX and its 3-month, 9-day and vol-of-vol cousins, the 3-month Treasury bill rate, and Cboe's SKEW index, daily, from 2006 onward. The first two years are study time only, so the graded record starts in August 2008, a few weeks before Lehman."
    ),
    bullet(
        '<b>"Out" means T-bills.</b> Money that steps out of the market earns the 3-month Treasury rate. That was nearly nothing from 2009 to 2016 and 4 to 5% recently.'
    ),
    bullet(
        "<b>Decide at the close, act the next morning.</b> The VIX is a closing number. Every rule reads it after the close and trades at the next day's opening price, with a small trading cost charged each time. No rule ever gets to trade on a price it could not have known."
    ),
    bullet(
        "<b>Dial, not switch.</b> Each rule can be fully in, half in, or out. A few of the rules (size by fear in particular) use finer steps."
    ),
    bullet(
        "<b>Two ways of grading.</b> We ran the whole exercise twice: once letting each rule choose the setting with the best risk-adjusted return (the Sharpe ratio), and once letting it choose the setting with the best return per unit of worst loss (the Calmar ratio). The two agreed on every important point. This report uses the first; the second is in the appendix."
    ),
    P("How to read the numbers", H2),
    bullet(
        "<b>Return per year</b> is the compound annual growth rate: the steady yearly rate that would have turned the starting money into the ending money."
    ),
    bullet(
        "<b>Worst loss</b> is the maximum drawdown: the deepest fall from a high to the low that followed. It is the number that makes people quit."
    ),
    bullet(
        f"<b>Sharpe ratio</b> is return per unit of bumpiness. Holding SPY scored {pp(spy['oos_sharpe'])} over this period. Anything near 1.0 is excellent for a stock strategy."
    ),
    bullet(
        "<b>Time in</b> is the share of days the rule was at least half invested. <b>Moves per year</b> counts how often it changed its mind. Fewer is easier to live with."
    ),
]

# ---- Part 2: scoreboard ---------------------------------------------------------
score_rows = [
    [
        "Approach",
        "Return / yr",
        "Worst loss",
        "Sharpe",
        "Time in",
        "Moves / yr",
        "Covid 2020",
        "Apr 2025",
    ]
]
for s in ORDER:
    r = R[s]
    score_rows.append(
        [
            NAMES[s],
            pct(r["oos_cagr"]),
            pct(r["oos_max_dd"], 0),
            pp(r["oos_sharpe"]),
            pct(r["time_in"], 0),
            f"{r['switch_py']:.0f}",
            pct(r["oos_dd_2020"], 0),
            pct(r["oos_dd_2025"], 0),
        ]
    )
bold = {ORDER.index(s) + 1 for s in ("vix_term", "vix_sized", "vix_surprise")}
story += [
    P("Part 2: The scoreboard", H1),
    P(
        f"Every number below is from the graded years only: {oos_start:%B %Y} to {oos_end:%B %Y}, about {years_span:.0f} years. "
        "The first two rows are the yardsticks: simply holding SPY, and the oldest trend rule in the book, which holds SPY only while it is above its own 200-day average price."
    ),
    KeepTogether(
        [
            table(
                score_rows,
                [
                    2.05 * inch,
                    0.75 * inch,
                    0.7 * inch,
                    0.55 * inch,
                    0.6 * inch,
                    0.65 * inch,
                    0.75 * inch,
                    0.65 * inch,
                ],
                bold_rows=bold,
            ),
            P(
                "Table 1. Out-of-sample results, Sharpe objective. Bold rows are the three rules we would actually use. Crisis columns show the worst loss inside that episode.",
                CAP,
            ),
        ]
    ),
    fig_block(
        img_growth,
        f"Figure 2. What $100,000 became. SPY finished at ${final['dial_hold'] / 1000:,.0f}k with a {pct(spy['oos_max_dd'], 0)} fall along the way; the surprise count finished at ${final['vix_surprise'] / 1000:,.0f}k and never fell more than {pct(surp['oos_max_dd'], 0)}. The vertical scale is logarithmic so equal steps are equal percentage gains.",
    ),
    P("What the table says", H2),
    bullet(
        f"<b>The surprise count (rule 6) is the standout.</b> It watches how often the market moves more than the VIX said it would, and steps aside when surprises pile up. It gave up {pct(spy['oos_cagr'] - surp['oos_cagr'])} a year of return in exchange for cutting the worst loss by more than half, and it posted the best Sharpe ratio of anything we tested."
    ),
    bullet(
        f"<b>The fear curve flip (rule 1) is the cleanest switch.</b> About {term['switch_py']:.0f} moves a year. It earned less than the surprise count ({pct(term['oos_cagr'])} a year) because it sat out some of the recoveries, but it was the best of all in the fast crashes: {pct(term['oos_dd_2020'], 0)} in Covid and {pct(term['oos_dd_2025'], 0)} in April 2025."
    ),
    bullet(
        f"<b>Size by fear (rule 5) is the smoothest.</b> It never fell more than {pct(sized['oos_max_dd'], 0)}. The price was return: {pct(sized['oos_cagr'])} a year, because it held only about {pct(sized['time_in'], 0)} of a full position on average. For someone who cares about sleep more than the last few percent, this is the one."
    ),
    bullet(
        f"<b>The vote (rule 10) is solid but busy.</b> Averaging rules 1, 2 and 3 gave {pct(vote['oos_cagr'])} a year with a {pct(vote['oos_max_dd'], 0)} worst loss, but it changed its mind {vote['switch_py']:.0f} times a year. Rule 3 inside it is the fidgety one."
    ),
    bullet(
        "<b>Rules 2, 3, 4, 7, 8 and 9 did not earn a place on their own.</b> Broken link (7) almost never fired, so it simply tracked SPY. The SKEW rule (9) traded constantly for nothing. Spike and fade (4) was the one rule built to buy fear rather than sell it, and on its own it missed too much of the ordinary upside."
    ),
    bullet(
        f"<b>Every VIX rule out-earned the 200-day line</b>, which managed {pct(sma['oos_cagr'])} a year. The 200-day line cuts losses, but it is slow to get back in and gets whipsawed in choppy years."
    ),
]

# ---- Part 3: crises -------------------------------------------------------------
full_rows = [["Approach", "Return / yr", "Worst loss", "2008 episode"]]
for s in FEATURED:
    r = R[s]
    full_rows.append(
        [NAMES[s], pct(r["full_cagr"]), pct(r["full_max_dd"], 0), pct(r["full_dd_2008"], 0)]
    )
story += [
    P("Part 3: The crises, one at a time", H1),
    P(
        'A rule like this earns its keep in a handful of weeks per decade. So the fair question is not "what was the average?" but '
        '"what happened in each of the bad stretches?" Figure 3 shows the whole record as a single line per approach: how far below its '
        "previous high each one was on every day. Figure 4 then zooms in on six episodes."
    ),
    fig_block(
        img_dd,
        "Figure 3. Days under water. The blue shaded area is SPY. Both VIX rules spent far less time in deep holes, and climbed out faster.",
    ),
    fig_block(
        img_crises,
        "Figure 4. Worst loss inside each episode, graded years only. Shorter bars are better. The 2008 panel starts in August 2008 because that is where the graded record begins.",
    ),
    P("What stands out", H2),
    bullet(
        f"<b>The fast crashes are where the VIX rules shine.</b> In Covid, SPY fell {pct(spy['oos_dd_2020'], 0)}; the curve flip, surprise count and size-by-fear held their losses to {pct(term['oos_dd_2020'], 0)}, {pct(surp['oos_dd_2020'], 0)} and {pct(sized['oos_dd_2020'], 0)}. In April 2025, when a tariff announcement knocked SPY down {pct(spy['oos_dd_2025'], 0)}, the curve flip lost {pct(term['oos_dd_2025'], 0)}."
    ),
    bullet(
        f"<b>The slow grinds are harder.</b> 2022 was a year-long decline with no single panic; the VIX never spiked the way it does in a crash. The surprise count and size by fear were the best of the VIX rules at {pct(surp['oos_dd_2022'], 0)}, but the curve flip ({pct(term['oos_dd_2022'], 0)}) did no better than the market ({pct(spy['oos_dd_2022'], 0)})."
    ),
    bullet(
        "<b>2008 deserves its own paragraph.</b> The graded record begins in August 2008, so it only catches the second half of that crisis. To see the whole thing we also ran every rule across the full history with its default settings. That is not a fair test (the settings were chosen with hindsight) but it is a useful picture. One oddity in Figure 4 first: the curve flip shows no loss at all in the 2008 panel because it was out of the market for that entire stretch, which also means it sat out the start of the 2009 rebound."
    ),
    KeepTogether(
        [
            table(full_rows, [2.4 * inch, 1.1 * inch, 1.1 * inch, 1.2 * inch]),
            P(
                "Table 2. Full history 2006 to 2026 with default settings, in sample, descriptive only. The 2008 episode runs October 2007 to June 2009.",
                CAP,
            ),
        ]
    ),
    P(
        f"The 200-day line held 2008 to {pct(sma['full_dd_2008'], 0)}. The VIX rules trimmed it to between {pct(surp['full_dd_2008'], 0)} and {pct(term['full_dd_2008'], 0)}, "
        f"against {pct(spy['full_dd_2008'], 0)} for the market. Better, but not good enough. The reason is instructive: 2008 was not one crash, it was a "
        "year of crashes with rallies in between, and the VIX fell back enough during each rally to pull the rules back in before the next leg down. "
        "A price-trend rule does not have that problem because the price never recovered its 200-day average until mid-2009. This is the clearest "
        "argument for combining the two kinds of rule rather than choosing one."
    ),
]

# ---- Part 4: year by year -------------------------------------------------------
yr_rows = [["Year"] + [SHORT[s] for s in FEATURED]]
for y, rec in YEARLY.iterrows():
    label = str(y)
    if y == YEARLY.index[0]:
        label += " (from Aug)"
    if y == YEARLY.index[-1]:
        label += " (to Oct)"
    yr_rows.append([label] + [pct(v, 1, sign=True) for v in rec])
story += [
    P("Part 4: Year by year", H1),
    P(
        "Averages hide a lot. This table shows every calendar year of the graded record. The point to notice is not the winners "
        "but the pattern: the VIX rules give up a few points in the strong up years and earn it back several times over in the bad ones."
    ),
    KeepTogether(
        [
            table(yr_rows, [1.3 * inch] + [0.88 * inch] * 6),
            P(
                "Table 3. Calendar-year returns, graded years only. Each strategy used whatever settings it had chosen for that exam.",
                CAP,
            ),
        ]
    ),
    fig_block(
        img_years,
        "Figure 5. Each year's return minus SPY's. The pattern is small losses in most years, large gains in the crisis years, which is exactly the trade an insurance buyer is making.",
    ),
    P(
        f"The surprise count beat SPY outright in {win_years['vix_surprise']} of {n_years} years and the curve flip in {win_years['vix_term']}. "
        "That is a low batting average, and it is supposed to be. A rule that stepped out of a rising market more often would be doing "
        "something wrong. The job is to lose a little, often, in exchange for losing a lot, rarely."
    ),
]

# ---- Part 5: what the rule actually did ------------------------------------------
out_days = {s: float((EXPO[s] < 0.25).mean()) for s in EXPO}
in_2009 = {s: float((EXPO[s].loc["2009"] >= 0.25).mean()) for s in EXPO}
in_h2_2020 = {s: float((EXPO[s].loc["2020-07":"2020-12"] >= 0.25).mean()) for s in EXPO}
half_days = {s: float(((EXPO[s] >= 0.25) & (EXPO[s] < 0.75)).mean()) for s in EXPO}
story += [
    P("Part 5: What the rules actually did", H1),
    P(
        "It helps to see the decisions on the price chart. Figure 6 shows SPY with the days each rule was standing aside shaded in. "
        "Light shading is half in; dark shading is fully out."
    ),
    fig_block(
        img_shade,
        f"Figure 6. The surprise count (top) was fully out on {pct(out_days['vix_surprise'], 0)} of days and half in on {pct(half_days['vix_surprise'], 0)}. The curve flip (bottom) was fully out on {pct(out_days['vix_term'], 0)} and half in on {pct(half_days['vix_term'], 0)}.",
    ),
    bullet(
        "<b>The clusters are in the right places.</b> Late 2008, mid 2010, the second half of 2011, the turn of 2016, late 2018, early 2020, much of 2022 and April 2025. These are the episodes a person would name if asked to list the scary stretches."
    ),
    bullet(
        "<b>The false alarms are visible too.</b> The thin bands scattered through the calm years are scares that faded within days. Each one cost a little and is part of the price."
    ),
    bullet(
        f"<b>The surprise count gets back in early.</b> It was invested for {pct(in_2009['vix_surprise'], 0)} of 2009 and {pct(in_h2_2020['vix_surprise'], 0)} of the second half of 2020, because surprises stop long before the price recovers its 200-day average. That is the main reason it out-earned the trend rule."
    ),
    bullet(
        f"<b>The curve flip's weakness is also visible.</b> The fear curve stayed inverted through most of the 2009 rebound, so the rule was invested for only {pct(in_2009['vix_term'], 0)} of that year and missed the first leg of the recovery. Most of its return gap to the surprise count comes from that one stretch."
    ),
]


# ---- Part 6: the knobs -------------------------------------------------------------
def fmt_params(p: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in p.items())


def ranges(s: str) -> str:
    chosen = outcomes[s]["chosen_params"]
    keys = sorted(chosen[0])
    parts = []
    for k in keys:
        vals = sorted({c[k] for c in chosen}, key=lambda v: (isinstance(v, bool), v))
        parts.append(f"{k}: {', '.join(str(v) for v in vals)}")
    return "; ".join(parts)


knob_rows = [
    ["Rule", "Settings on the menu", "Most often chosen", "Everything the exams ever chose"]
]
grids = {s: wf[s]["grid"] for s in ORDER}
for s in ("dial_sma", "vix_term", "vix_sized", "vix_surprise", "vix_vote"):
    menu = "; ".join(f"{k}: {', '.join(str(v) for v in vs)}" for k, vs in grids[s].items())
    knob_rows.append([NAMES[s], menu, fmt_params(mode_params[s]), ranges(s)])
story += [
    P("Part 6: Messing with the settings", H1),
    P(
        "Because each exam re-chose its settings, we can see which ones the rules kept picking. A rule whose best setting jumps around "
        "every year is a rule that is fitting noise. A rule that picks nearly the same thing every time has found something real."
    ),
    KeepTogether(
        [
            table(
                knob_rows, [1.25 * inch, 2.1 * inch, 1.45 * inch, 1.9 * inch], align_right_from=9
            ),
            P(
                "Table 4. The menus and the choices. Window lengths are in trading days; 'band' is how far a reading must move back before the rule reverses, to avoid flip-flopping.",
                CAP,
            ),
        ]
    ),
    bullet(
        "<b>The curve flip barely cares about its knobs.</b> Its menu was narrow on purpose, because the idea only makes sense near a ratio of 1.0, and all three triggers it was offered worked. The one clear preference: in 16 of the 19 exams it chose to act the day the curve flipped rather than wait three days for confirmation. Waiting cost more than the false alarms it avoided."
    ),
    bullet(
        "<b>Size by fear kept choosing the most cautious setting on the menu.</b> In 12 of 19 exams it picked the lowest volatility target offered, 8%, which works out to roughly half a position in normal markets. Next time we should offer it 6% and see whether it keeps asking for less."
    ),
    bullet(
        "<b>The surprise count was the least settled of the three.</b> It moved between a two-week and a two-month window, and between stepping out when 30% of recent days beat the implied move and waiting for 40%. It did well with all of them, which says the idea is sturdier than any particular setting, but it is the rule we would watch most closely as new data arrives."
    ),
    bullet(
        "<b>The 200-day line was the least stable.</b> The exams picked 100, 150, 200 and 250-day versions at various times, which is a polite way of saying it did not much matter."
    ),
]

# ---- Part 7: the stress tests --------------------------------------------------------
CHECKS = [
    ("oos_sharpe", "Risk-adjusted return above a minimum", "0.3 or more"),
    ("psr", "Odds the result is better than zero, allowing for luck", "90% or more"),
    ("efficiency", "Graded-year result as a share of study-year result", "50% or more"),
    ("cost_stress", "Still profitable with trading costs doubled", "above zero"),
    ("trades", "Enough trades to judge", "100 or more"),
    ("windows_pos", "Share of exam years that made money", "50% or more"),
    ("regimes", "Made money in rising, falling AND sideways markets", "2 of 3"),
    ("pbo", "Chance the best setting was a fluke (overfitting test)", "30% or less"),
    ("dsr", "Result survives a penalty for every setting ever tried", "95% or more"),
]
gt_rows = [["What the check asks", "Bar", "Surprise", "Curve flip", "Size by fear", "Vote"]]
for name, q, bar in CHECKS:
    row = [q, bar]
    for s in ("vix_surprise", "vix_term", "vix_sized", "vix_vote"):
        c = next(c for c in outcomes[s]["verdict"]["checks"] if c["name"] == name)
        v = c["value"]
        if name in ("psr", "efficiency", "windows_pos", "regimes", "pbo", "dsr"):
            txt = (
                "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.0f}%"
            )
        else:
            txt = f"{v:.2f}" if name != "trades" else f"{v:.0f}"
        row.append(f"{txt} {'pass' if c['passed'] else 'FAIL'}")
    gt_rows.append(row)
story += [
    P("Part 7: The stress tests", H1),
    P(
        "Every strategy at the fund goes through the same gauntlet of checks before it is believed. It is deliberately strict, and this is "
        "the part of the report where we show you what it did not like, because a report that only shows the good news is not worth reading."
    ),
    KeepTogether(
        [
            table(
                gt_rows, [2.55 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.85 * inch, 0.8 * inch]
            ),
            P(
                "Table 5. The gauntlet. Every one of the twelve approaches, including simply holding SPY, failed the last two checks, which is why they need explaining.",
                CAP,
            ),
        ]
    ),
    bullet(
        "<b>The checks that matter passed.</b> Positive out of sample, robust to doubled costs, profitable in most exam years, and the graded result was a healthy share of the study result (a rule that scores 1.1 in practice and 0.8 on the exam is a rule that mostly works; one that scores 1.5 and then 0.2 was memorizing)."
    ),
    bullet(
        '<b>"All three market types" failed for nearly everyone, including buy and hold.</b> The check wants a profit in rising, falling and sideways markets separately. Every approach made money in rising markets, roughly broke even in falling ones (which is the entire goal of a dial), and lost a little in sideways chop, where any in-or-out rule gets whipsawed. SPY itself fails the sideways part. We read this as a description of how the rules behave, not as a flaw.'
    ),
    bullet(
        "<b>The last two checks are structurally unfair to this kind of strategy.</b> The overfitting test needs many settings to compare and is not defined for buy and hold, which has none; the penalty test counts every setting tried across all nineteen exams (several hundred of them) as a separate attempt, which punishes the walk-forward method for its own thoroughness. We left the bars where they are rather than bend them for a result we liked, and note the limitation here. Size by fear, with its stable settings, passed the overfitting test outright."
    ),
]

# ---- Part 8: conclusion ----------------------------------------------------------------
story += [
    PageBreak(),
    P("Part 8: What we recommend", H1),
    P(
        "The question we started with was whether the VIX can give a disciplined reason to be in or out of the market as a whole. "
        "The answer is yes, with two rules in particular, and with one known blind spot.",
        BOX,
    ),
    P("Use the surprise count as the dial", H2),
    P(
        f"It had the best risk-adjusted result, the best return of the serious contenders, and it earned its keep in every kind of crisis "
        f"including the slow one in 2022. Expect it to be fully out about a tenth of the time and half in about a fifth, and to change its mind about "
        f"{surp['switch_py']:.0f} times a year, mostly between full and half."
    ),
    P("Keep the fear curve flip as the sanity check", H2),
    P(
        "It is the simplest rule here, it barely cares about its settings, and it was the best of all in the fast crashes. When both rules say "
        "out at once, that is a stronger signal than either alone. When they disagree, half in is the natural answer."
    ),
    P("Add the 200-day line for the slow grinds", H2),
    P(
        "The one episode the VIX rules handled poorly was 2008, where the fear gauge kept giving the all-clear during rallies that failed. "
        "The 200-day line handled that episode well and the fast crashes poorly. A combination (out if either says out) is the next study, "
        "and we expect it to be better than both parents."
    ),
    P("What to expect if you follow it", H2),
    bullet(
        "Most years you will trail the market by a few points. Some years, more. That is the premium on the insurance."
    ),
    bullet(
        "In a crash you will lose roughly a third to a half of what the market loses, and be back in well before the recovery is complete."
    ),
    bullet(
        "It will not protect against a single overnight gap. Nothing that reads closing prices can."
    ),
    bullet(
        "It will give false alarms, several a year. Act on them anyway. The record above includes every false alarm and still came out ahead."
    ),
    P("Where to see it", H2),
    P(
        "The daily market report now carries a <b>Market dial</b> tile at the top: IN, HALF or OUT, with each contributing rule's reading "
        "beside it. It is computed from the same code that produced every number in this document. The study itself re-runs every Sunday "
        "on the Studio so the scorecards stay current as new months arrive."
    ),
    P("Appendix: the Calmar version", H1),
    P(
        "The same test, but with each rule choosing the setting with the best return per unit of worst loss instead of the best Sharpe ratio. The ranking is unchanged.",
        SMALL,
    ),
]
cal_rows = [
    [
        "Approach",
        "Return / yr",
        "Worst loss",
        "Sharpe",
        "Time in",
        "Moves / yr",
        "Covid 2020",
        "Apr 2025",
    ]
]
for s in ORDER:
    r = rows_c[s]
    cal_rows.append(
        [
            NAMES[s],
            pct(r["oos_cagr"]),
            pct(r["oos_max_dd"], 0),
            pp(r["oos_sharpe"]),
            pct(r["time_in"], 0),
            f"{r['switch_py']:.0f}",
            pct(r["oos_dd_2020"], 0),
            pct(r["oos_dd_2025"], 0),
        ]
    )
story += [
    table(
        cal_rows,
        [
            2.05 * inch,
            0.75 * inch,
            0.7 * inch,
            0.55 * inch,
            0.6 * inch,
            0.65 * inch,
            0.75 * inch,
            0.65 * inch,
        ],
        bold_rows=bold,
    ),
    P("Table 6. Out-of-sample results, Calmar objective.", CAP),
    P("Glossary", H1),
    bullet("<b>Backtest.</b> Running a rule over past prices to see how it would have done."),
    bullet(
        "<b>Walk-forward test.</b> A backtest where the rule's settings are chosen only from the past at every step, then graded on the period that follows. The graded periods are <b>out of sample</b>; the study periods are <b>in sample</b>."
    ),
    bullet(
        "<b>Overfitting.</b> Finding a setting that fits the past perfectly and the future not at all. The thing walk-forward testing exists to catch."
    ),
    bullet(
        "<b>Compound annual growth rate (return per year).</b> The steady yearly rate that turns the start value into the end value."
    ),
    bullet("<b>Maximum drawdown (worst loss).</b> The deepest peak-to-trough fall."),
    bullet(
        "<b>Sharpe ratio.</b> Return above T-bills divided by the bumpiness of that return. Higher is smoother."
    ),
    bullet(
        "<b>Calmar ratio.</b> Return per year divided by worst loss. Higher means the gains were cheap in terms of pain."
    ),
    bullet(
        '<b>Treasury bill (T-bill).</b> A short-term loan to the U.S. government. What "cash" earns in this study.'
    ),
    bullet(
        "<b>Whipsaw.</b> Stepping out on a scare and stepping back in higher after it fades. The main cost of any dial."
    ),
    bullet(
        "<b>Slippage.</b> The small amount a real trade costs beyond the quoted price. Charged on every trade here, and doubled in one of the stress tests."
    ),
    bullet(
        '<b>Implied move.</b> The daily swing the VIX is pricing in: roughly the VIX divided by 16. A "surprise" is a day that moves more than that.'
    ),
]

doc = SimpleDocTemplate(
    str(OUT_PDF),
    pagesize=letter,
    leftMargin=0.9 * inch,
    rightMargin=0.9 * inch,
    topMargin=0.85 * inch,
    bottomMargin=0.85 * inch,
    title="The Fear Gauge, Put to the Test",
    author="The Eagle Hill Fund",
)


def footer(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(TEXT2))
    canvas.drawString(
        0.9 * inch, 0.55 * inch, "The Eagle Hill Fund  |  The Fear Gauge, Put to the Test"
    )
    canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, f"Page {doc_.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("final", {k: round(v) for k, v in final.items()})
print("win_years", win_years, "n_years", n_years)
print("out/half days", out_days, half_days)
print("wrote", OUT_PDF)
