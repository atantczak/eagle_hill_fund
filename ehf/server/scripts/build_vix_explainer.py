"""Build the plain-English VIX / SPY explainer PDF with real-data charts."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from ehf.server.research.data.fmp_provider import FMPProvider

OUT_PDF = Path(sys.argv[1])
IMG_DIR = Path(sys.argv[2])
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ---- palette (dataviz reference instance, light mode) -----------------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"  # slot 1: SPY
ORANGE = "#eb6834"  # slot 2: VIX
AQUA = "#1baf7a"  # slot 3: realized vol
NAVY = colors.HexColor("#1f3a5f")

# ---- data --------------------------------------------------------------------
p = FMPProvider()


def series(sym: str) -> pd.Series:
    df = p.get_bars([sym], "2004-01-01", "2026-11-01")
    s = df.set_index(pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None))["close"]
    return s.sort_index()


spy = series("SPY")
vix = series("^VIX")
vix3m = series("^VIX3M")
both = pd.concat({"spy": spy, "vix": vix}, axis=1).dropna()
both["ret"] = both["spy"].pct_change()
both["dvix"] = both["vix"].pct_change()
both["rv21"] = both["ret"].rolling(21).std() * np.sqrt(252) * 100
# realized vol over the NEXT 21 days (what the VIX was trying to price)
both["rv_fwd"] = both["ret"][::-1].rolling(21).std()[::-1] * np.sqrt(252) * 100
both["fwd63"] = both["spy"].shift(-63) / both["spy"] - 1

# ---- stats for the text -------------------------------------------------------
d = both.dropna(subset=["ret", "dvix"])
corr = d["ret"].corr(d["dvix"])
beta = np.polyfit(d["ret"], d["dvix"], 1)[0]
down = d[d["ret"] < -0.02]["dvix"].mean() * 100
up = d[d["ret"] > 0.02]["dvix"].mean() * 100
prem = (both["vix"] - both["rv_fwd"]).dropna()
prem_mean = prem.mean()
prem_pos = (prem > 0).mean() * 100
f = both.dropna(subset=["fwd63"])
fwd_hi = f[f["vix"] > 30]["fwd63"].mean() * 100
fwd_lo = f[f["vix"] < 15]["fwd63"].mean() * 100
fwd_all = f["fwd63"].mean() * 100
n_hi = int((f["vix"] > 30).sum())
ts = pd.concat({"vix": vix, "vix3m": vix3m}, axis=1).dropna()
ts["ratio"] = ts["vix"] / ts["vix3m"]
back_pct = (ts["ratio"] > 1).mean() * 100
start_year = both.index[0].year
end_date = both.index[-1].strftime("%B %Y")


def pct(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}%"


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
    }
)


def save(fig, name):
    path = IMG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# Chart 1: SPY and VIX as two stacked panels (one axis each, never dual-axis)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 3.5), sharex=True, gridspec_kw={"hspace": 0.45})
a1.plot(both.index, both["spy"], color=BLUE, lw=1.4)
a1.set_title("SPY price (the stock market)")
a1.set_ylabel("$")
a2.plot(both.index, both["vix"], color=ORANGE, lw=1.4)
a2.set_title("VIX level (the fear gauge)")
a2.set_ylabel("VIX")
for label, when in (("2008 crisis", "2008-11-20"), ("Covid", "2020-03-16"), ("Aug 2015", "2015-08-24"), ("Apr 2025", "2025-04-08")):
    t = pd.Timestamp(when)
    if t in both.index:
        a2.annotate(label, (t, both.loc[t, "vix"]), xytext=(6, 2), textcoords="offset points", color=TEXT2, fontsize=8)
a2.xaxis.set_major_locator(mdates.YearLocator(2))
a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
img1 = save(fig, "spy_vix.png")

# Chart 2: scatter of daily SPY move vs daily VIX move
fig, ax = plt.subplots(figsize=(7.2, 2.5))
ax.scatter(d["ret"] * 100, d["dvix"] * 100, s=5, color=BLUE, alpha=0.35, linewidths=0)
xs = np.linspace(-8, 8, 50)
ax.plot(xs, beta * xs, color=ORANGE, lw=1.6)
ax.set_xlim(-8, 8)
ax.set_ylim(-40, 80)
ax.set_title("Each dot is one trading day: when SPY falls, the VIX jumps")
ax.set_xlabel("SPY move that day (%)")
ax.set_ylabel("VIX move that day (%)")
ax.annotate(f"trend line: VIX moves about {abs(beta):.1f}x the opposite way", (-7.6, 60), color=TEXT2, fontsize=8)
img2 = save(fig, "scatter.png")

# Chart 3: VIX vs realized vol that followed (two series, legend + end labels)
sub = both.loc["2015":].dropna(subset=["rv_fwd"])
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.plot(sub.index, sub["vix"], color=ORANGE, lw=1.4, label="VIX (the insurance price)")
ax.plot(sub.index, sub["rv_fwd"], color=AQUA, lw=1.4, label="How much SPY actually moved over the next month")
ax.set_title("The insurance usually costs more than the storm that follows")
ax.set_ylabel("Annualized %")
ax.legend(frameon=False, loc="upper right", fontsize=8)
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
img3 = save(fig, "premium.png")

# Chart 4: term-structure ratio with the 1.0 line
sub = ts.loc["2015":]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(sub.index, sub["ratio"], color=ORANGE, lw=1.2)
ax.axhline(1.0, color=TEXT2, lw=0.9, ls="--")
ax.fill_between(sub.index, 1.0, sub["ratio"], where=sub["ratio"] > 1.0, color=ORANGE, alpha=0.25, linewidths=0)
ax.set_title("30-day VIX divided by 3-month VIX")
ax.set_ylabel("Ratio")
ax.annotate("danger zone: near-term fear above long-term fear", (pd.Timestamp("2021-03-01"), 1.30), color=TEXT2, fontsize=8)
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
img4 = save(fig, "term.png")

# ---- PDF ---------------------------------------------------------------------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17, textColor=NAVY, spaceBefore=14, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, textColor=NAVY, spaceBefore=12, spaceAfter=5)
BODY = ParagraphStyle("Body", parent=ss["Normal"], fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=8, alignment=TA_LEFT)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=9, leading=12, textColor=colors.HexColor(TEXT2))
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=4)
CAP = ParagraphStyle("Cap", parent=SMALL, spaceBefore=2, spaceAfter=12)
TITLE = ParagraphStyle("Title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=24, leading=30, textColor=NAVY, spaceAfter=6, alignment=TA_LEFT)
SUB = ParagraphStyle("Sub", parent=BODY, fontSize=12.5, leading=17, textColor=colors.HexColor(TEXT2), spaceAfter=18)
BOX = ParagraphStyle("Box", parent=BODY, backColor=colors.HexColor("#eef3fa"), borderPadding=8, spaceBefore=6, spaceAfter=14)


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


story = []
story += [
    P("The Fear Gauge and the Market", TITLE),
    P("A plain-English guide to the VIX, how it moves against the S&amp;P 500, and how we plan to use it to decide when to be in the market and when to step aside.", SUB),
    P(f"Prepared for The Eagle Hill Fund. Data runs from {start_year} through {end_date}.", SMALL),
    Spacer(1, 10),
    P("The short version", H1),
    P(
        "The VIX is the price of insurance on the stock market. When investors get nervous they buy protection, "
        "the price of that protection rises, and the VIX goes up. When they relax, it drifts back down. Because fear "
        "and prices move in opposite directions, the VIX tends to spike exactly when the market is falling.",
    ),
    P(
        "That sounds like it should make the VIX a crystal ball. It is not. But it has three reliable quirks, and "
        "each one can be turned into a rule for stepping in or out of the market. First, the insurance is usually "
        "overpriced, so extreme fear has historically been a better time to buy than to sell. Second, fear spikes fade. "
        "Third, when short-term fear climbs above long-term fear, trouble is often already under way. The rest of this "
        "document explains each of those ideas and the nine rules we are going to test against real history.",
        BOX,
    ),
]

story += [
    P("Part 1: What the VIX actually is", H1),
    P("Think of it as a hurricane-insurance premium", H2),
    P(
        "If you own a house in Florida, the cost of hurricane insurance goes up when a storm shows up on the radar. "
        "The premium does not tell you whether the storm will hit. It tells you how worried the people selling "
        "insurance are, and how much the people buying it are willing to pay for peace of mind."
    ),
    P(
        "The VIX is that same premium for the whole U.S. stock market. Big investors protect themselves against a "
        "crash by buying options (contracts that pay out if the market falls). The VIX is built from the prices of "
        "those options. More demand for protection means higher option prices, which means a higher VIX. "
        "The exchange that publishes it, the Chicago Board Options Exchange, updates it every few seconds during the trading day."
    ),
    P("How to read the number", H2),
    P(
        "The VIX is quoted as a percentage, and it means roughly this: \"over the next 30 days, the market expects "
        "the S&amp;P 500 to swing by about this much, on an annual basis.\" That is awkward to picture, so here is "
        "a simpler translation. Divide the VIX by about 16 and you get the size of a typical daily move the market is bracing for."
    ),
]

rows = [
    ["VIX level", "Typical daily swing it implies", "What it feels like"],
    ["12", "about 0.75%", "Calm. Quiet, steady markets."],
    ["20", "about 1.25%", "Normal. Long-run average is near here."],
    ["30", "about 1.9%", "Nervous. Headlines are scary."],
    ["50", "about 3.1%", "Crisis. Seen in 2008, 2020 and briefly in 2025."],
    ["80", "about 5%", "Panic. Only reached at the worst of 2008 and Covid."],
]
t = Table(rows, colWidths=[1.0 * inch, 2.0 * inch, 3.6 * inch])
t.setStyle(
    TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )
)
story[-2:] = [KeepTogether([story[-2], story[-1], t])]
story += [Spacer(1, 10)]

story += [
    P("Where the number comes from (optional detail)", H2),
    P(
        "You do not need this to use the VIX, but it helps to know it is not somebody's opinion. The exchange takes "
        "every option on the S&amp;P 500 that expires in roughly a month, weights each one by its price, and adds them "
        "up with a formula that produces the market's expected variance (variance is the square of volatility). "
        "The VIX is the square root of that, scaled to a yearly number. Three things fall out of the formula:"
    ),
    bullet("<b>It is a price, not a forecast.</b> It tells you what protection costs today, not what will happen tomorrow."),
    bullet("<b>Protection against falls dominates.</b> The formula gives far more weight to options that pay off in a crash than to ones that pay off in a rally, so the VIX is mostly a measure of how badly people want downside protection."),
    bullet("<b>It is a 30-day number.</b> The exchange also publishes 9-day, 3-month and 6-month versions. Comparing them tells you whether fear is about right now or about the longer future, which turns out to be very useful."),
]

story += [
    PageBreak(),
    P("Part 2: How the VIX and the market move together", H1),
    P(
        "The chart below shows the S&amp;P 500 (through the SPY fund, which simply owns all 500 stocks) on top, and the "
        "VIX underneath, over more than twenty years. Every big drop in the top panel lines up with a spike in the bottom one."
    ),
    fig_block(img1, "Figure 1. SPY (top) and the VIX (bottom). Market crashes and VIX spikes are the same events seen from two sides."),
    P(
        f"Day to day, the relationship is strong and lopsided. Since {start_year}, the daily moves of the two have a "
        f"correlation of about {corr:.2f} (where minus 1 would be perfect opposites). On average the VIX moves about "
        f"{abs(beta):.1f} times as much as SPY in percentage terms, and in the opposite direction. The lopsided part: "
        f"on days when SPY fell more than 2%, the VIX jumped {pct(abs(down))} on average, but on days when SPY rose more than 2% "
        f"it only fell {pct(abs(up))}. Fear arrives fast and leaves slowly."
    ),
    fig_block(img2, width=5.0 * inch, caption="Figure 2. Every trading day as a dot. The cloud slopes down and to the right, and stretches much further upward than downward."),
]

story += [
    PageBreak(),
    P("Part 3: Why there is something to exploit", H1),
    P("Quirk 1: The insurance is usually overpriced", H2),
    P(
        "Here is the single most important fact in this document. If you compare what the VIX said the market would do "
        "with what the market actually did over the following month, the VIX is too high most of the time. Investors "
        "consistently pay more for protection than the protection turns out to be worth, the same way most people pay more "
        "in home-insurance premiums than they ever collect. That gap is called the variance risk premium."
    ),
    P(
        f"Since {start_year}, the VIX has run on average about {prem_mean:.1f} points above the volatility that actually followed, "
        f"and it was above the actual outcome on {pct(prem_pos, 0)} of all days. The chart below shows the two lines for the last decade."
    ),
    fig_block(img3, "Figure 3. The VIX (orange) against how much the market really moved in the following month (green). Orange sits above green most of the time; the exceptions are the first days of a crisis."),
    P(
        "Why does this matter for getting in or out? Because when fear is extreme, the overpricing is extreme too. The VIX at 40 "
        "is pricing in a disaster that, historically, usually does not fully arrive. The numbers bear this out:"
    ),
]
hdr = ParagraphStyle("hdr", parent=BODY, fontName="Helvetica-Bold", fontSize=9.5, leading=12, textColor=NAVY, spaceAfter=0)
rows = [
    [Paragraph("Condition on the day you looked", hdr), Paragraph("Average SPY return over the next 3 months", hdr), Paragraph("Number of days", hdr)],
    ["VIX below 15 (calm)", pct(fwd_lo), str(int((f["vix"] < 15).sum()))],
    ["All days", pct(fwd_all), str(len(f))],
    ["VIX above 30 (scared)", pct(fwd_hi), str(n_hi)],
]
t = Table(rows, colWidths=[2.4 * inch, 2.8 * inch, 1.3 * inch])
t.setStyle(
    TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )
)
story += [
    KeepTogether([t, Spacer(1, 6)]),
    P(
        "Read that carefully, because it runs against instinct. The scariest days have, on average, been followed by the best "
        "three months. That does not mean buying every spike is safe (in 2008 the VIX stayed above 30 for months while the market kept falling). "
        "It means that a high VIX on its own is not a reason to sell. The rules in Part 4 are about telling the fast, fading scares apart from the slow, grinding ones.",
        SMALL,
    ),
    P("Quirk 2: Fear spikes fade", H2),
    P(
        "The VIX is tethered. It cannot stay at 50 for long because at some point the panic sellers are done, option sellers step in "
        "to collect the fat premiums, and the price of protection drifts back toward its long-run home near 20. "
        "Statisticians call this mean reversion. A spike that takes a week to build often takes a month to unwind, and the market tends to recover alongside it."
    ),
    P("Quirk 3: Near-term fear versus long-term fear", H2),
    P(
        "Normally the 3-month version of the VIX sits a little above the 30-day version. That makes sense: more can go wrong in "
        "three months than in one, so longer insurance costs more. When that flips, and 30-day fear climbs above 3-month fear, it means "
        "investors believe the trouble is right now rather than someday. Historically that flip has been one of the cleanest warnings "
        f"that a sell-off is under way. It has happened on about {pct(back_pct, 0)} of trading days since 2006."
    ),
    fig_block(img4, "Figure 4. The 30-day VIX divided by the 3-month VIX. Spikes above the dashed line are the periods when near-term fear exceeded longer-term fear."),
]

story += [
    PageBreak(),
    P("Part 4: The nine rules we are going to test", H1),
    P(
        "Each rule below watches one thing about the VIX and translates it into a simple instruction: stay in, lighten up, or step out. "
        "None of them is a prediction. Each is a tilt that has been right more often than wrong in the past. We will test every one "
        "against twenty years of history, and only keep the ones that hold up. Part 5 explains how they combine into a single in-or-out decision."
    ),
]
rules = [
    ["#", "Rule", "What it watches", "When it says step out", "When it says step back in"],
    ["1", "Fear curve flip", "30-day VIX vs 3-month VIX", "30-day climbs above 3-month", "30-day drops back below"],
    ["2", "Unexplained fear", "Did the VIX move more than the market's drop justifies?", "VIX jumps far more than SPY's fall explains", "SPY falls but the VIX barely reacts (a quiet dip)"],
    ["3", "Overpriced insurance", "VIX minus how much the market really moved lately", "The gap goes negative (real moves exceed the insurance price)", "The gap is unusually large (fear is overdone)"],
    ["4", "Spike and fade", "VIX compared to its own recent average", "Never on its own; this is a buying rule", "VIX is far above its recent norm"],
    ["5", "Size by fear", "The VIX as a dial, not a switch", "Hold less when VIX is high, more when low", "Automatically scales back up as VIX falls"],
    ["6", "Surprise count", "How often SPY moves more than the VIX said it would", "Surprises are piling up", "Surprises are rare again"],
    ["7", "Broken link", "Whether VIX and SPY are still moving in opposite directions", "They start rising together", "The normal opposite relationship returns"],
    ["8", "Fear of fear", "The VVIX, which measures how jumpy the VIX itself is", "VVIX spikes before the VIX does", "VVIX settles"],
    ["9", "Insurance mix", "Which share of the VIX comes from crash protection vs. rally bets", "Crash protection share surges", "The mix normalizes"],
]
t = Table(rules, colWidths=[0.3 * inch, 1.1 * inch, 1.75 * inch, 1.85 * inch, 1.6 * inch], repeatRows=1)
t.setStyle(
    TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (1, 1), (1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEADING", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )
)
# wrap cells in Paragraphs so long text flows
cell = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
wrapped = [[Paragraph(c, cellb if (r == 0 or j == 1) else cell) for j, c in enumerate(row)] for r, row in enumerate(rules)]
t = Table(wrapped, colWidths=[0.3 * inch, 1.1 * inch, 1.75 * inch, 1.85 * inch, 1.6 * inch], repeatRows=1)
t.setStyle(
    TableStyle(
        [
            ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )
)
story += [t, Spacer(1, 10)]
story += [
    P("A few of these deserve a sentence more.", H2),
    bullet("<b>Rule 2 is the one a professional trader would recognize.</b> The VIX normally moves a predictable amount for a given market drop. When it moves much more than that, hedgers know something the price does not show yet. When it moves much less, the drop is probably noise. The rule trades the difference."),
    bullet("<b>Rule 5 is not in-or-out at all.</b> It holds a position whose size shrinks as the VIX rises, so a VIX of 40 means holding half as much as a VIX of 20. It is the gentlest rule and often the hardest to beat."),
    bullet("<b>Rules 1, 2 and 3 watch different things,</b> so when two of the three agree the signal is much stronger than any one alone. The final version will likely be a vote among them."),
]

story += [
    PageBreak(),
    P("Part 5: Using this to decide when to be in or out", H1),
    P(
        "The goal is simple to state: own the stock market most of the time, because it goes up most of the time, but have a "
        "disciplined reason to stand aside during the stretches that do real damage. SPY is our stand-in for \"the market\" because "
        "it is the fund most people actually own and the VIX is built from options on the same 500 stocks."
    ),
    P("What a good version of this looks like", H2),
    bullet("<b>A dial, not a switch.</b> Fully in, half in, or out. Binary rules get whipsawed; they sell at the bottom and buy back higher. A three-position dial cuts that cost dramatically."),
    bullet("<b>\"Out\" means Treasury bills, not a mattress.</b> Cash that earns the short-term government rate makes stepping aside much cheaper, especially in the current rate environment."),
    bullet("<b>Judged on drawdown, not just return.</b> The point of this is not to beat buy-and-hold on raw return. It is to get most of the upside with meaningfully shallower losses, so that you can actually stay the course."),
    bullet("<b>Checked at the close, acted on the next morning.</b> The VIX is a closing number. Every rule is tested that way so the results are honest."),
    P("What it will not do", H2),
    bullet("It will not catch the top. Every rule here reacts to fear that has already shown up in prices, so it will always be a few days late."),
    bullet("It will not avoid every loss. A sudden one-day shock (the 1987 crash, or the April 2025 tariff gap) hits before any closing-price rule can respond."),
    bullet("It will give false alarms. Stepping out on a scare that fades costs a little each time. The backtests will tell us whether the big saves pay for the small misses."),
    P("The honest summary", H2),
    P(
        "The VIX will never tell you what the market will do. What it tells you, reliably, is how expensive fear has become and whether "
        "that fear is about today or about someday. Those two facts, used with discipline, have historically been enough to sidestep the "
        "worst of the big declines while staying invested for the recoveries. The next step is to prove, rule by rule, that this "
        "held up across twenty years of history and was not just a lucky stretch.",
        BOX,
    ),
    P("Glossary", H1),
    bullet("<b>S&amp;P 500.</b> The 500 largest U.S. companies, weighted by size. \"The market\" in everyday speech."),
    bullet("<b>SPY.</b> An exchange-traded fund that owns the S&amp;P 500. Buying it is buying the market."),
    bullet("<b>Option.</b> A contract that pays off if the market moves past a set level by a set date. Put options pay in a fall; they are the insurance the VIX measures."),
    bullet("<b>VIX.</b> The 30-day expected volatility of the S&amp;P 500, derived from option prices. Published by Cboe."),
    bullet("<b>VIX3M, VIX9D.</b> The same calculation for 3 months and 9 days ahead."),
    bullet("<b>VVIX.</b> The volatility of the VIX itself. How jumpy the fear gauge is."),
    bullet("<b>Volatility.</b> How much prices swing. Realized volatility is what actually happened; implied volatility (the VIX) is what the market is bracing for."),
    bullet("<b>Variance risk premium.</b> The habitual gap between what the VIX implies and what actually happens. Positive means insurance was overpriced."),
    bullet("<b>Backtest.</b> Running a rule over past prices to see how it would have done. Useful, and also where most bad ideas look good, which is why we test out-of-sample."),
    bullet("<b>Drawdown.</b> The fall from a peak to the next trough. The number that makes people sell at the bottom."),
    bullet("<b>Whipsaw.</b> Selling on a scare and buying back higher after it fades. The main cost of any in-or-out rule."),
]

doc = SimpleDocTemplate(
    str(OUT_PDF),
    pagesize=letter,
    leftMargin=0.9 * inch,
    rightMargin=0.9 * inch,
    topMargin=0.85 * inch,
    bottomMargin=0.85 * inch,
    title="The Fear Gauge and the Market",
    author="The Eagle Hill Fund",
)


def footer(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(TEXT2))
    canvas.drawString(0.9 * inch, 0.55 * inch, "The Eagle Hill Fund  |  The Fear Gauge and the Market")
    canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, f"Page {doc_.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("stats", dict(corr=corr, beta=beta, down=down, up=up, prem_mean=prem_mean, prem_pos=prem_pos, fwd_hi=fwd_hi, fwd_lo=fwd_lo, fwd_all=fwd_all, n_hi=n_hi, back_pct=back_pct))
print("wrote", OUT_PDF)
