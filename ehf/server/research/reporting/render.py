"""Render a ``Snapshot`` (+ optional narrative) to markdown, a standalone HTML page,
and a compact email body. (The PDF lives in ``pdf.py`` and consumes the same sections.)

All renderers consume the same intermediate ``Section`` list so the outputs never
drift apart. The markdown is the archival artifact; the HTML is the "open on
anything" web page (self-contained: no scripts, no external assets); the email is
a short, email-client-safe summary with links to the page and the PDF.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

from ehf.server.research.reporting.narrative import (
    IDEA_DISCLAIMER,
    IDEA_KEY,
    NARRATIVE_SECTIONS,
    split_narrative,
)
from ehf.server.research.reporting.screens import SCREEN_ORDER, Hit
from ehf.server.research.reporting.snapshot import Snapshot

Json = dict[str, Any]

REPORT_TITLE = "Daily Market Report"
BRAND = "Eagle Hill Fund"

SCREEN_TITLES = {
    "quality_dislocation": "Quality on sale — fell hard, fundamentals intact",
    "pead": "Post-earnings drift — surprise and reaction agree",
    "insider_cluster": "Insider cluster buying",
    "analyst_revision": "Analyst revision momentum",
    "breakout_52w": "52-week-high breakouts with volume",
    "outlier_move": "Abnormal one-day moves (explain, don't chase)",
}
SCREEN_SHORT = {
    "quality_dislocation": "Quality on sale",
    "pead": "Earnings drift",
    "insider_cluster": "Insider buying",
    "analyst_revision": "Revisions",
    "breakout_52w": "Breakouts",
    "outlier_move": "Outliers",
}
IDEA_TITLE = "One idea to chew on"
IDEA_PLACEHOLDER = (
    "The research assistant idea needs the narrative layer, which is off for this run "
    "(no Anthropic API key configured). The structured screens below are unaffected."
)

# Which metrics to show per screen, in order: (key, label, format)
SCREEN_COLUMNS: dict[str, list[tuple[str, str, str]]] = {
    "quality_dislocation": [
        ("close", "Px", "num"),
        ("ret_21d", "21d", "pct"),
        ("pct_from_high_252", "vs 52wH", "pct"),
        ("piotroski", "Pio", "int"),
        ("altman_z", "AltZ", "num1"),
        ("fcf_yield", "FCF yld", "pct"),
        ("roic", "ROIC", "pct"),
        ("net_debt_ebitda", "ND/EBITDA", "num1"),
        ("pe", "P/E", "num1"),
        ("rsi_14", "RSI", "int"),
    ],
    "pead": [
        ("close", "Px", "num"),
        ("report_date", "Reported", "str"),
        ("eps_surprise", "EPS surp", "pct"),
        ("revenue_surprise", "Rev surp", "pct"),
        ("reaction", "Reaction", "pct"),
        ("ret_21d", "21d", "pct"),
    ],
    "insider_cluster": [
        ("close", "Px", "num"),
        ("n_insiders", "Insiders", "int"),
        ("n_filings", "Filings", "int"),
        ("total_value", "$ bought", "money"),
        ("last_filing", "Last", "str"),
        ("ret_21d", "21d", "pct"),
        ("pct_from_high_252", "vs 52wH", "pct"),
    ],
    "analyst_revision": [
        ("close", "Px", "num"),
        ("upgrades", "Up", "int"),
        ("downgrades", "Down", "int"),
        ("n_price_targets", "PT notes", "int"),
        ("avg_pt_upside", "PT upside", "pct"),
        ("ret_5d", "5d", "pct"),
        ("ret_21d", "21d", "pct"),
    ],
    "breakout_52w": [
        ("close", "Px", "num"),
        ("mom_12_1", "12-1 mom", "pct"),
        ("vol_ratio_20", "Vol x", "num1"),
        ("ret_1d", "1d", "pct"),
        ("ret_21d", "21d", "pct"),
        ("rsi_14", "RSI", "int"),
    ],
    "outlier_move": [
        ("close", "Px", "num"),
        ("ret_1d", "1d", "pct"),
        ("z_1d", "Sigma", "num1"),
        ("vol_ratio_20", "Vol x", "num1"),
        ("ret_21d", "21d", "pct"),
        ("pct_from_high_252", "vs 52wH", "pct"),
    ],
}


@dataclass
class Table:
    columns: list[str]
    rows: list[list[str]]
    kinds: list[str] | None = None  # per-column format kind (drives +/- colouring)


@dataclass
class Section:
    title: str
    kind: str = "data"  # idea | narrative | market | screen | notes | data
    key: str = ""  # anchor id / screen key
    intro: str = ""
    bullets: list[str] = field(default_factory=list)
    table: Table | None = None
    hits: list[Hit] = field(default_factory=list)
    markdown: str = ""  # free-form markdown body (narrative)
    level: int = 2


@dataclass
class Tile:
    label: str
    value: str
    detail: str = ""
    tone: str = ""  # up | down | ""


def fmt(v: Any, kind: str) -> str:
    if v is None:
        return "–"
    try:
        if kind == "pct":
            return f"{float(v) * 100:+.1f}%"
        if kind == "pct0":  # unsigned: levels, shares, percentiles
            return f"{float(v) * 100:.1f}%"
        if kind == "num":
            return f"{float(v):,.2f}"
        if kind == "num1":
            return f"{float(v):,.1f}"
        if kind == "num2":
            return f"{float(v):,.2f}"
        if kind == "int":
            return f"{float(v):.0f}"
        if kind == "money":
            x = float(v)
            if x >= 1e12:
                return f"${x / 1e12:.2f}T"
            if x >= 1e9:
                return f"${x / 1e9:.1f}B"
            return f"${x / 1e6:.1f}M" if x >= 1e6 else f"${x / 1e3:.0f}k"
        if kind == "bp":  # input in percentage points (4.2 = 4.2%)
            return f"{float(v) * 100:+.0f}bp"
    except (TypeError, ValueError):
        return str(v)
    return str(v)


def tone_of(text: str, kind: str = "pct") -> str:
    """'up' / 'down' / '' for a formatted signed number (used for colouring)."""
    if kind not in {"pct", "bp"}:
        return ""
    if text.startswith("+") and not text.startswith("+0.0"):
        return "up"
    if text.startswith("-"):
        return "down"
    return ""


def regime_tone(regime: str) -> str:
    r = (regime or "").lower()
    if "risk-on" in r:
        return "on"
    if "risk-off" in r:
        return "off"
    return "mid"


def _is_nan(v: object) -> bool:
    return isinstance(v, float) and v != v


# --- pure section model ----------


def dial_tile(dial: Json) -> Tile | None:
    """The in/out dial as the first hero tile: label, exposure, the three readings."""
    if not dial or dial.get("exposure") is None:
        return None
    label = str(dial.get("label", "unknown"))
    tone = {"IN": "up", "OUT": "down"}.get(label, "")
    t, r, v = dial.get("term", {}), dial.get("resid", {}), dial.get("vrp", {})
    moved = " · changed today" if dial.get("changed") else f" · {dial.get('days_in_state', 0)}d"
    sub = (
        f"{fmt(dial.get('exposure'), 'pct0')} in{moved} · curve {fmt(t.get('ratio'), 'num2')} "
        f"· fear z {fmt(r.get('z'), 'num1')} · premium {fmt(v.get('points'), 'num1')}"
    )
    return Tile("Market dial", label, sub, tone)


def hero_tiles(market: Json) -> list[Tile]:
    idx, vix, br, tr = (
        market.get("index", {}),
        market.get("vix", {}),
        market.get("breadth", {}),
        market.get("treasury", {}),
    )
    tiles: list[Tile] = []
    dial = dial_tile(market.get("dial", {}))
    if dial is not None:
        tiles.append(dial)
    if idx:
        d1 = fmt(idx.get("ret_1d"), "pct")
        tiles.append(
            Tile(
                "S&P 500 (SPY)",
                d1,
                f"{fmt(idx.get('ret_5d'), 'pct')} 5d · {fmt(idx.get('ret_21d'), 'pct')} 21d",
                tone_of(d1),
            )
        )
        ytd = fmt(idx.get("ret_ytd"), "pct")
        tiles.append(
            Tile(
                "SPY YTD",
                ytd,
                f"{fmt(idx.get('pct_from_252d_high'), 'pct')} from 52w high",
                tone_of(ytd),
            )
        )
    if vix:
        chg = fmt((vix.get("change_pct") or 0) / 100, "pct")
        vix_tone = {"up": "down", "down": "up"}.get(tone_of(chg), "")  # rising VIX is "down"
        tiles.append(
            Tile(
                "VIX",
                fmt(vix.get("level"), "num1"),
                f"{chg} on the day · 50d avg {fmt(vix.get('avg_50d'), 'num1')}",
                vix_tone,
            )
        )
    if br:
        tiles.append(
            Tile(
                "Above 200dma",
                fmt(br.get("pct_above_sma200"), "pct0"),
                f"{fmt(br.get('pct_above_sma50'), 'pct0')} above 50dma",
            )
        )
        tiles.append(
            Tile(
                "52w highs / lows",
                f"{br.get('new_52w_highs')} / {br.get('new_52w_lows')}",
                f"{fmt(br.get('pct_up_1d'), 'pct0')} of names up on the day",
            )
        )
    if tr:
        tiles.append(
            Tile(
                "10y Treasury",
                f"{fmt(tr.get('10y'), 'num1')}%",
                f"10y–2y {fmt(tr.get('slope_10y_2y'), 'bp')} · 2y {fmt(tr.get('2y'), 'num1')}%",
            )
        )
    return tiles


def _rule_word(exposure: object) -> str:
    try:
        e = float(exposure)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    return "in" if e >= 0.75 else "out" if e <= 0.25 else "half"


def dial_bullet(dial: Json) -> str:
    t, r, v = dial.get("term", {}), dial.get("resid", {}), dial.get("vrp", {})
    days = dial.get("days_in_state", 0)
    moved = "Changed today." if dial.get("changed") else f"Unchanged {days} days."
    return (
        f"**Market dial: {dial.get('label')} ({fmt(dial.get('exposure'), 'pct0')} in SPY, "
        f"rest in bills).** {moved} Fear curve VIX/VIX3M {fmt(t.get('ratio'), 'num2')} "
        f"({_rule_word(t.get('exposure'))}; out above {fmt(t.get('out_above'), 'num2')}); "
        f"unexplained-fear z {fmt(r.get('z'), 'num1')} ({_rule_word(r.get('exposure'))}; "
        f"warns above {fmt(r.get('warn_z'), 'num1')}); insurance premium "
        f"{fmt(v.get('points'), 'num1')} pts ({_rule_word(v.get('exposure'))}; out below "
        f"{fmt(v.get('out_below'), 'num1')}, full above {fmt(v.get('full_above'), 'num1')}). "
        f"Read at the close, acted on at the next open."
    )


def _market_sections(market: Json) -> list[Section]:
    idx, vix, br, tr = (
        market.get("index", {}),
        market.get("vix", {}),
        market.get("breadth", {}),
        market.get("treasury", {}),
    )
    bullets = []
    dial = market.get("dial", {})
    if dial and dial.get("exposure") is not None:
        bullets.append(dial_bullet(dial))
    if idx:
        bullets.append(
            f"**Regime: {idx.get('regime', 'unknown')}.** "
            f"SPY {fmt(idx.get('ret_1d'), 'pct')} on the day, "
            f"{fmt(idx.get('ret_5d'), 'pct')} 5d, {fmt(idx.get('ret_21d'), 'pct')} 21d, "
            f"{fmt(idx.get('ret_ytd'), 'pct')} YTD; "
            f"{fmt(idx.get('pct_from_252d_high'), 'pct')} from its 52-week high. "
            f"20d realized vol {fmt(idx.get('realized_vol_20d'), 'pct0')} "
            f"(pctile over 1y: {fmt(idx.get('realized_vol_20d_pctile_1y'), 'pct0')})."
        )
    if vix:
        bullets.append(
            f"VIX {fmt(vix.get('level'), 'num1')} "
            f"({fmt((vix.get('change_pct') or 0) / 100, 'pct')} on the day), "
            f"50d avg {fmt(vix.get('avg_50d'), 'num1')}, "
            f"1y range {fmt(vix.get('year_low'), 'num1')}–{fmt(vix.get('year_high'), 'num1')}."
        )
    if br:
        bullets.append(
            f"Breadth ({br.get('n_symbols')} names): "
            f"{fmt(br.get('pct_above_sma200'), 'pct0')} above 200dma, "
            f"{fmt(br.get('pct_above_sma50'), 'pct0')} above 50dma; "
            f"new 52w highs {br.get('new_52w_highs')} vs lows {br.get('new_52w_lows')}; "
            f"{fmt(br.get('pct_up_1d'), 'pct0')} of names up on the day; "
            f"{fmt(br.get('pct_down_20_from_52w_high'), 'pct0')} are 20%+ below their 52w high."
        )
    if tr:
        bullets.append(
            f"Treasuries ({tr.get('date')}): 3m {fmt(tr.get('3m'), 'num1')}%, "
            f"2y {fmt(tr.get('2y'), 'num1')}%, 10y {fmt(tr.get('10y'), 'num1')}%; "
            f"10y–2y {fmt(tr.get('slope_10y_2y'), 'bp')}, "
            f"10y–3m {fmt(tr.get('slope_10y_3m'), 'bp')}."
        )
    sections = [Section(title="Market data", kind="market", key="market", bullets=bullets)]

    sectors = market.get("sectors") or []
    if sectors:
        sections.append(
            Section(
                title="Sectors (last session)",
                kind="market",
                key="sectors",
                table=Table(
                    ["Sector", "Avg change"],
                    [
                        [str(s["sector"]), fmt((s["avg_change_pct"] or 0) / 100, "pct")]
                        for s in sectors
                    ],
                    kinds=["str", "pct"],
                ),
            )
        )
    econ = market.get("econ_ahead") or []
    if econ:
        sections.append(
            Section(
                title="High-impact US data this week",
                kind="market",
                key="econ",
                table=Table(
                    ["When", "Release", "Prev", "Est"],
                    [
                        [
                            str(e["date"]),
                            str(e["event"]),
                            fmt(e["previous"], "str"),
                            fmt(e["estimate"], "str"),
                        ]
                        for e in econ[:15]
                    ],
                ),
            )
        )
    earn = market.get("earnings_ahead") or []
    if earn:
        sections.append(
            Section(
                title="Large-cap earnings ahead",
                kind="market",
                key="earnings",
                table=Table(
                    ["Date", "Symbol", "Name", "EPS est", "Mkt cap"],
                    [
                        [
                            e["date"],
                            e["symbol"],
                            str(e.get("name") or ""),
                            fmt(e["eps_estimate"], "num"),
                            fmt(e["market_cap"], "money"),
                        ]
                        for e in earn[:25]
                    ],
                ),
            )
        )
    return sections


def _hit_table(screen: str, hits: list[Hit]) -> Table:
    cols = SCREEN_COLUMNS.get(screen, [])
    header = ["Symbol", *[c[1] for c in cols]]
    rows = [
        [
            h.symbol,
            *[
                fmt(None if _is_nan(h.metrics.get(k)) else h.metrics.get(k), kind)
                for k, _, kind in cols
            ],
        ]
        for h in hits
    ]
    return Table(header, rows, kinds=["str", *[c[2] for c in cols]])


def narrative_sections(narrative: str | None) -> list[Section]:
    """The idea card first, then the model's other sections in their fixed order."""
    parts = split_narrative(narrative)
    out: list[Section] = []
    if narrative is None:
        out.append(Section(title=IDEA_TITLE, kind="idea", key="idea", intro=IDEA_PLACEHOLDER))
    elif IDEA_KEY in parts:
        out.append(Section(title=IDEA_TITLE, kind="idea", key="idea", markdown=parts[IDEA_KEY]))
    for key, heading in NARRATIVE_SECTIONS:
        if key != IDEA_KEY and key in parts:
            out.append(Section(title=heading, kind="narrative", key=key, markdown=parts[key]))
    leftovers = "\n\n".join(parts[k] for k in ("preamble", "other") if k in parts)
    if leftovers:
        out.append(
            Section(
                title="Research assistant notes", kind="narrative", key="notes", markdown=leftovers
            )
        )
    return out


def build_sections(snapshot: Snapshot, narrative: str | None) -> list[Section]:
    sections: list[Section] = narrative_sections(narrative)
    sections.extend(_market_sections(snapshot.market))
    for screen in SCREEN_ORDER:
        hits = snapshot.hits.get(screen, [])
        title = SCREEN_TITLES[screen]
        if not hits:
            sections.append(
                Section(title=title, kind="screen", key=screen, intro="Nothing fired today.")
            )
            continue
        sections.append(
            Section(
                title=title, kind="screen", key=screen, table=_hit_table(screen, hits), hits=hits
            )
        )
    foot = [
        f"Universe: {len(snapshot.universe)} symbols (liquid screener ∪ S&P 500). "
        f"Data as of {snapshot.as_of}. "
        f"FMP calls: {snapshot.api_calls}. Timings (s): "
        + ", ".join(f"{k} {v}" for k, v in snapshot.timings.items())
    ]
    foot.extend(f"Warning: {w}" for w in snapshot.warnings)
    sections.append(Section(title="Run notes", kind="notes", key="run-notes", bullets=foot))
    return sections


def hit_extra_line(h: Hit) -> str:
    ctx = h.context
    extra = []
    if ctx.get("pt_upside") is not None:
        extra.append(
            "consensus PT "
            f"{fmt(ctx.get('pt_consensus'), 'num')} ({fmt(ctx.get('pt_upside'), 'pct')})"
        )
    if ctx.get("revenue_growth_qoq_yoy") is not None:
        extra.append(f"last-qtr revenue growth {fmt(ctx.get('revenue_growth_qoq_yoy'), 'pct')}")
    if ctx.get("next_earnings"):
        extra.append(f"next earnings {ctx['next_earnings']}")
    return "; ".join(extra)


# --- markdown ----------


def _md_table(t: Table) -> str:
    out = ["| " + " | ".join(t.columns) + " |", "|" + "|".join("---" for _ in t.columns) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in t.rows)
    return "\n".join(out)


def _hit_notes_md(h: Hit) -> str:
    also = f"; also in: {', '.join(h.context['also_in'])}" if h.context.get("also_in") else ""
    lines = [f"- **{h.symbol}** — {h.headline} _(horizon: {h.horizon}; {h.direction}{also})_"]
    lines.extend(f"  - {r}" for r in h.reasons)
    if extra := hit_extra_line(h):
        lines.append(f"  - {extra}")
    for n in h.context.get("news", [])[:3]:
        lines.append(f"  - news {n.get('date')}: {n.get('title')}")
    return "\n".join(lines)


def render_markdown(snapshot: Snapshot, narrative: str | None) -> str:
    regime = snapshot.market.get("index", {}).get("regime", "")
    out = [
        f"# {BRAND} daily market report — {snapshot.run_date}",
        f"_Data as of {snapshot.as_of} close.{f' Regime: {regime}.' if regime else ''}_",
        "",
    ]
    for s in build_sections(snapshot, narrative):
        out.append(f"{'#' * s.level} {s.title}")
        if s.intro:
            out.append(s.intro)
        if s.markdown:
            out.append(s.markdown)
        if s.kind == "idea" and s.markdown:
            out.append(f"\n_{IDEA_DISCLAIMER}_")
        out.extend(f"- {b}" for b in s.bullets)
        if s.table:
            out.append(_md_table(s.table))
        if s.hits:
            out.append("")
            out.extend(_hit_notes_md(h) for h in s.hits)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# --- markdown subset -> html ----------


def _inline_md(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


_LABEL_RE = re.compile(r"^\s*\*\*[^*]+\*\*")


def md_to_html(md: str) -> str:
    """Tiny markdown subset: #/##/### headings, -/* bullets, 1. lists, **bold**, paragraphs."""
    out: list[str] = []
    para: list[str] = []
    list_tag: str | None = None

    def flush_para() -> None:
        if para:
            out.append("<p>" + _inline_md(" ".join(para)) + "</p>")
            para.clear()

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush_para()
            close_list()
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush_para()
            close_list()
            lvl = min(len(m.group(1)) + 1, 4)  # demote: report h1 is the page title
            out.append(f"<h{lvl}>{_inline_md(m.group(2))}</h{lvl}>")
            continue
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            flush_para()
            if list_tag != "ul":
                close_list()
                out.append("<ul>")
                list_tag = "ul"
            out.append(f"<li>{_inline_md(m.group(1))}</li>")
            continue
        m = re.match(r"^\s*\d+[.)]\s+(.*)$", line)
        if m:
            flush_para()
            if list_tag != "ol":
                close_list()
                out.append("<ol>")
                list_tag = "ol"
            out.append(f"<li>{_inline_md(m.group(1))}</li>")
            continue
        close_list()
        if para and _LABEL_RE.match(line):  # "**The idea.** ..." starts its own paragraph
            flush_para()
        para.append(line.strip())
    flush_para()
    close_list()
    return "\n".join(out)


# --- standalone html page ----------

_CSS = """
:root{--bg:#070b14;--card:rgba(17,25,43,.74);--border:rgba(148,163,184,.16);--text:#e6edf7;
--muted:#8b9bb4;--soft:#c7d2e3;--accent:#5eead4;--accent2:#a5b4fc;--up:#34d399;--down:#fb7185;
--warn:#fbbf24;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth;-webkit-text-size-adjust:100%}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,
sans-serif;font-size:14.5px;line-height:1.55;color:var(--text);background:var(--bg);
background-image:radial-gradient(1100px 560px at 8% -12%,rgba(94,234,212,.16),transparent 60%),
radial-gradient(900px 520px at 100% -5%,rgba(129,140,248,.2),transparent 55%),
radial-gradient(700px 500px at 50% 110%,rgba(94,234,212,.07),transparent 60%),
linear-gradient(180deg,#0b1222 0%,#070b14 55%);background-attachment:fixed}
.wrap{max-width:1080px;margin:0 auto;padding:26px 18px 60px}
header.hero{padding:30px 0 10px}
.eyebrow{font-size:11px;letter-spacing:.24em;text-transform:uppercase;color:var(--accent);
font-weight:700}
h1{font-size:36px;line-height:1.08;margin:8px 0 6px;font-weight:800;letter-spacing:-.025em;
background:linear-gradient(90deg,#fff 0%,#c9d4ff 60%,#5eead4 110%);-webkit-background-clip:text;
background-clip:text;color:transparent}
.sub{color:var(--muted);font-size:14px}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:700;
border:1px solid var(--border);background:rgba(255,255,255,.04);vertical-align:middle;
margin-left:8px;letter-spacing:.02em}
.pill.on{color:var(--up);border-color:rgba(52,211,153,.45);background:rgba(52,211,153,.1)}
.pill.off{color:var(--down);border-color:rgba(251,113,133,.45);background:rgba(251,113,133,.1)}
.pill.mid{color:var(--warn);border-color:rgba(251,191,36,.45);background:rgba(251,191,36,.1)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;
margin:20px 0 6px}
.tile{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:12px 14px;
backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.tile .k{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;
font-weight:600}
.tile .v{font-size:23px;font-weight:800;font-family:var(--mono);margin-top:3px;
font-variant-numeric:tabular-nums}
.tile .d{font-size:12px;color:var(--muted);margin-top:2px}
nav.toc{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:6px;padding:10px 0;
margin:14px 0 6px;background:linear-gradient(180deg,rgba(7,11,20,.96),rgba(7,11,20,.86));
backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border-bottom:1px solid var(--border)}
nav.toc a{color:var(--muted);text-decoration:none;font-size:12px;padding:4px 10px;
border-radius:999px;border:1px solid transparent;white-space:nowrap}
nav.toc a:hover{color:var(--text);border-color:var(--border);background:rgba(255,255,255,.04)}
nav.toc a .n{color:var(--accent);font-family:var(--mono);margin-left:5px}
nav.toc a.idea{color:var(--accent2)}
section.card{background:var(--card);border:1px solid var(--border);border-radius:16px;
padding:18px 20px;margin:16px 0;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
scroll-margin-top:56px}
section.card h2{font-size:18px;margin:0 0 10px;display:flex;align-items:center;gap:10px;
flex-wrap:wrap;letter-spacing:-.01em}
.count{font-family:var(--mono);font-size:12px;color:var(--accent);
background:rgba(94,234,212,.1);border-radius:999px;padding:2px 9px;font-weight:600}
section.idea{border:1px solid rgba(165,180,252,.5);
background:linear-gradient(135deg,rgba(129,140,248,.2),rgba(94,234,212,.09) 70%,rgba(17,25,43,.8));
box-shadow:0 0 0 1px rgba(129,140,248,.12),0 24px 70px rgba(0,0,0,.4)}
section.idea .eyebrow{color:var(--accent2)}
section.idea h2{font-size:24px;margin-top:4px;font-weight:800}
section.idea .narr p{font-size:15.5px;line-height:1.6}
section.idea .narr b{color:var(--accent2)}
.disclaimer{font-size:12px;color:var(--muted);border-top:1px dashed rgba(165,180,252,.35);
margin-top:14px;padding-top:9px}
.narr h3,.narr h4{font-size:14.5px;margin:14px 0 6px;color:var(--accent);
text-transform:uppercase;letter-spacing:.06em}
.narr p{margin:7px 0;color:var(--soft)}.narr ul,.narr ol{margin:4px 0 10px 20px;padding:0;
color:var(--soft)}.narr li{margin:4px 0}.narr b{color:var(--text)}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:6px 0 4px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 9px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--border);
font-variant-numeric:tabular-nums}
th{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;
font-weight:700}
td{font-family:var(--mono);font-size:12.5px;color:var(--soft)}
th:first-child,td:first-child{text-align:left}
td:first-child{font-family:inherit;font-weight:700;color:var(--text);font-size:13.5px}
tbody tr:hover{background:rgba(255,255,255,.035)}
.up{color:var(--up)}.down{color:var(--down)}.warn{color:var(--warn)}
.hits{margin-top:12px}
details.hit{margin:8px 0;border-left:2px solid rgba(94,234,212,.35);padding:3px 12px}
details.hit summary{cursor:pointer;list-style:none;line-height:1.5}
details.hit summary::-webkit-details-marker{display:none}
details.hit summary b{font-size:15px;margin-right:6px}
.tag{display:inline-block;font-size:10.5px;padding:1px 8px;border-radius:999px;
background:rgba(255,255,255,.06);color:var(--muted);margin-left:6px;vertical-align:middle;
font-weight:600;letter-spacing:.03em}
.tag.long{color:var(--up);background:rgba(52,211,153,.1)}
.tag.short{color:var(--down);background:rgba(251,113,133,.1)}
.tag.also{color:var(--accent2);background:rgba(165,180,252,.12)}
.hit ul{margin:6px 0 4px 18px;padding:0;color:var(--soft)}.hit li{margin:2px 0}
.hit li.muted{color:var(--muted)}
ul.b{margin:4px 0 0 18px;padding:0;color:var(--soft)}ul.b li{margin:5px 0}
p.muted{color:var(--muted);margin:4px 0}
footer{color:var(--muted);font-size:12px;margin-top:26px;text-align:center}
@media (max-width:640px){h1{font-size:27px}.wrap{padding:14px 12px 40px}
section.card{padding:14px 14px}.tile .v{font-size:20px}}
@media print{body{background:#fff;color:#111;background-image:none}
section.card,.tile{background:#fff;border-color:#ccc;break-inside:avoid;backdrop-filter:none}
h1{color:#111;background:none;-webkit-text-fill-color:#111}nav.toc{display:none}
td,th,.narr p,.narr ul,.hit ul,ul.b{color:#222}.tag{color:#444}section.idea{background:#f3f4ff}}
"""


def _html_table(t: Table) -> str:
    kinds = t.kinds or ["str"] * len(t.columns)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in t.columns)
    body_rows = []
    for r in t.rows:
        cells = []
        for i, c in enumerate(r):
            cls = tone_of(c, kinds[i] if i < len(kinds) else "str")
            attr = f" class={cls}" if cls else ""
            cells.append(f"<td{attr}>{html.escape(c)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<div class=tw><table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _hit_html(h: Hit) -> str:
    items = "".join(f"<li>{html.escape(r)}</li>" for r in h.reasons)
    if extra := hit_extra_line(h):
        items += f"<li>{html.escape(extra)}</li>"
    for n in h.context.get("news", [])[:3]:
        when, title = html.escape(str(n.get("date"))), html.escape(str(n.get("title")))
        items += f"<li class=muted>news {when}: {title}</li>"
    tags = [(h.direction, h.direction), (h.horizon, "")]
    tags += [(f"also: {SCREEN_SHORT.get(s, s)}", "also") for s in h.context.get("also_in", [])]
    tag_html = "".join(
        f"<span class='tag {cls}'>{html.escape(t)}</span>" for t, cls in tags
    ).replace("class='tag '", "class=tag")
    return (
        f"<details class=hit open><summary><b>{html.escape(h.symbol)}</b>"
        f"{html.escape(h.headline)}{tag_html}</summary><ul>{items}</ul></details>"
    )


def _section_html(s: Section) -> str:
    count = f"<span class=count>{len(s.hits)}</span>" if s.hits else ""
    eyebrow = "<div class=eyebrow>Research assistant</div>" if s.kind == "idea" else ""
    cls = "card idea" if s.kind == "idea" else "card"
    parts = [f"<section class='{cls}' id='{html.escape(s.key or s.title)}'>{eyebrow}"]
    parts.append(f"<h2>{html.escape(s.title)}{count}</h2>")
    if s.intro:
        parts.append(f"<p class=muted>{html.escape(s.intro)}</p>")
    if s.markdown:
        parts.append(f"<div class=narr>{md_to_html(s.markdown)}</div>")
    if s.kind == "idea" and s.markdown:
        parts.append(f"<div class=disclaimer>{html.escape(IDEA_DISCLAIMER)}</div>")
    if s.bullets:
        items = "".join(f"<li>{_inline_md(b)}</li>" for b in s.bullets)
        parts.append(f"<ul class=b>{items}</ul>")
    if s.table:
        parts.append(_html_table(s.table))
    if s.hits:
        parts.append("<div class=hits>" + "".join(_hit_html(h) for h in s.hits) + "</div>")
    parts.append("</section>")
    return "\n".join(parts)


def _nav_html(sections: list[Section]) -> str:
    links = []
    for s in sections:
        if s.kind == "idea":
            links.append(f"<a class=idea href='#{s.key}'>✦ Idea</a>")
        elif s.kind == "screen":
            n = f"<span class=n>{len(s.hits)}</span>" if s.hits else ""
            label = html.escape(SCREEN_SHORT.get(s.key, s.title))
            links.append(f"<a href='#{s.key}'>{label}{n}</a>")
        elif s.key == "market":
            links.append("<a href='#market'>Market</a>")
    return f"<nav class=toc>{''.join(links)}</nav>"


def render_html(snapshot: Snapshot, narrative: str | None) -> str:
    """A self-contained page: no scripts, no external assets, works offline and on phones."""
    sections = build_sections(snapshot, narrative)
    regime = str(snapshot.market.get("index", {}).get("regime", "") or "")
    pill = (
        f"<span class='pill {regime_tone(regime)}'>{html.escape(regime)}</span>" if regime else ""
    )
    tiles = "".join(
        f"<div class=tile><div class=k>{html.escape(t.label)}</div>"
        f"<div class='v {t.tone}'>{html.escape(t.value)}</div>"
        f"<div class=d>{html.escape(t.detail)}</div></div>"
        for t in hero_tiles(snapshot.market)
    )
    title = f"{BRAND} — {REPORT_TITLE} — {snapshot.run_date}"
    parts = [
        "<!doctype html><html lang=en><head><meta charset='utf-8'>",
        "<meta name=viewport content='width=device-width,initial-scale=1'>",
        "<meta name=color-scheme content='dark'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{_CSS}</style></head><body><div class=wrap>",
        "<header class=hero>",
        f"<div class=eyebrow>{html.escape(BRAND)}</div>",
        f"<h1>{REPORT_TITLE}</h1>",
        f"<div class=sub>{snapshot.run_date:%A, %B %-d, %Y} · data as of {snapshot.as_of} close"
        f"{pill}</div>",
        f"<div class=tiles>{tiles}</div>",
        "</header>",
        _nav_html(sections),
    ]
    parts.extend(_section_html(s) for s in sections)
    parts.append(
        f"<footer>{html.escape(BRAND)} · generated {snapshot.run_date} · "
        f"{len(snapshot.universe)} symbols screened · internal research, not investment advice"
        "</footer></div></body></html>"
    )
    return "\n".join(parts)


# --- email body (inline styles only; Gmail / Apple Mail / Outlook safe) ----------

_E_FONT = "font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;"
_E_CARD = (
    "background:#ffffff;border:1px solid #e3e8f0;border-radius:12px;padding:16px 18px;"
    "margin:0 0 14px 0;"
)
_E_H = "font-size:16px;font-weight:800;margin-bottom:4px"
_E_EYEBROW = "font-size:11px;letter-spacing:.18em;text-transform:uppercase;font-weight:700"
_E_IDEA_CARD = f"{_E_CARD}background:#f5f3ff;border-color:#c7d2fe"


def _email_md(md: str) -> str:
    """Narrative markdown -> HTML with inline styles (email clients drop <style> rules)."""
    out = md_to_html(md)
    h = (
        "<p style='margin:12px 0 4px;font-size:12px;letter-spacing:.06em;"
        "text-transform:uppercase;color:#0f766e;font-weight:700'>"
    )
    out = re.sub(r"<h[34]>", h, out)
    out = re.sub(r"</h[34]>", "</p>", out)
    out = out.replace("<p>", "<p style='margin:6px 0;line-height:1.5'>")
    out = out.replace("<ul>", "<ul style='margin:4px 0 8px 20px;padding:0'>")
    out = out.replace("<ol>", "<ol style='margin:4px 0 8px 20px;padding:0'>")
    return out.replace("<li>", "<li style='margin:3px 0;line-height:1.45'>")


def _email_tiles(snapshot: Snapshot) -> str:
    """Stat tiles, three per row so they survive a phone-width inbox."""
    tiles = hero_tiles(snapshot.market)[:6]
    cells = []
    for t in tiles:
        colour = {"up": "#059669", "down": "#e11d48"}.get(t.tone, "#0f172a")
        cells.append(
            "<td width='33%' style='padding:0 4px 8px 0;vertical-align:top'>"
            "<div style='background:#f6f8fb;border:1px solid #e3e8f0;border-radius:10px;"
            "padding:8px 10px'><div style='font-size:10px;color:#64748b;text-transform:uppercase;"
            f"letter-spacing:.08em'>{html.escape(t.label)}</div>"
            f"<div style='font-size:18px;font-weight:800;color:{colour}'>{html.escape(t.value)}"
            f"</div><div style='font-size:11px;color:#64748b'>{html.escape(t.detail)}</div>"
            "</div></td>"
        )
    rows = "".join(f"<tr>{''.join(cells[i : i + 3])}</tr>" for i in range(0, len(cells), 3))
    return (
        "<table role=presentation width='100%' cellpadding=0 cellspacing=0 "
        f"style='border-collapse:collapse;margin:0 0 10px'>{rows}</table>"
    )


def _email_buttons(links: dict[str, str], link_days: int) -> str:
    if not links:
        return ""
    btn = (
        "display:inline-block;padding:11px 18px;border-radius:999px;text-decoration:none;"
        "font-weight:700;font-size:14px;margin:0 8px 8px 0;"
    )
    out = ""
    if "html" in links:
        out += (
            f"<a href='{html.escape(links['html'])}' style='{btn}background:#0f172a;"
            "color:#5eead4'>Open the full report →</a>"
        )
    if "pdf" in links:
        out += (
            f"<a href='{html.escape(links['pdf'])}' style='{btn}background:#eef2ff;"
            "color:#3730a3'>Download PDF</a>"
        )
    note = f"Links work for {link_days} days; the PDF is also attached."
    return (
        f"<div style='margin:14px 0 4px'>{out}"
        f"<div style='font-size:11px;color:#94a3b8'>{note}</div></div>"
    )


def _email_fired(snapshot: Snapshot) -> str:
    lines = []
    for screen in SCREEN_ORDER:
        hits = snapshot.hits.get(screen, [])
        if not hits:
            continue
        arrows = {"long": " ↑", "short": " ↓"}
        syms = ", ".join(
            f"<b>{html.escape(h.symbol)}</b>{arrows.get(h.direction, '')}" for h in hits[:8]
        )
        more = f" +{len(hits) - 8} more" if len(hits) > 8 else ""
        label = html.escape(SCREEN_SHORT.get(screen, screen))
        lines.append(
            "<tr><td style='padding:6px 10px 6px 0;color:#64748b;font-size:12px;"
            f"white-space:nowrap;vertical-align:top'>{label}"
            f" <span style='color:#0f766e'>({len(hits)})</span></td>"
            f"<td style='padding:6px 0;font-size:13px;line-height:1.5'>{syms}{more}</td></tr>"
        )
    if not lines:
        return ""
    return (
        f"<div style='{_E_CARD}'><div style='{_E_H}'>What fired</div>"
        "<table role=presentation cellpadding=0 cellspacing=0 style='border-collapse:collapse'>"
        f"{''.join(lines)}</table><div style='font-size:12px;color:#64748b;margin-top:8px'>"
        "Tables, reasons, and news for every name are in the full report and the attached PDF."
        "</div></div>"
    )


def render_email_html(
    snapshot: Snapshot, narrative: str | None, links: dict[str, str], *, link_days: int = 7
) -> str:
    """Short summary for the inbox: idea, market read, top ideas, screen headlines, links."""
    parts = split_narrative(narrative)
    regime = str(snapshot.market.get("index", {}).get("regime", "") or "")
    pill_bg, pill_fg = {
        "on": ("#d1fae5", "#065f46"),
        "off": ("#ffe4e6", "#9f1239"),
        "mid": ("#fef3c7", "#92400e"),
    }[regime_tone(regime)]
    pill = (
        "<div style='margin-top:10px'><span style='display:inline-block;padding:4px 12px;"
        f"border-radius:999px;font-size:12px;font-weight:700;background:{pill_bg};"
        f"color:{pill_fg}'>{html.escape(regime)}</span></div>"
        if regime
        else ""
    )
    body: list[str] = [
        f"<div style='{_E_CARD}border:0;background:#0b1222;color:#fff;padding:22px 20px'>",
        f"<div style='{_E_EYEBROW}color:#5eead4'>{html.escape(BRAND)}</div>",
        f"<div style='font-size:26px;font-weight:800;margin:6px 0 4px'>{REPORT_TITLE}</div>",
        f"<div style='font-size:13px;color:#aab4c8'>{snapshot.run_date:%A, %B %-d, %Y} · "
        f"data as of {snapshot.as_of} close</div>{pill}{_email_buttons(links, link_days)}</div>",
        _email_tiles(snapshot),
    ]
    if narrative is None:
        body.append(
            f"<div style='{_E_IDEA_CARD}'><div style='{_E_EYEBROW}color:#4338ca'>{IDEA_TITLE}"
            f"</div><p style='margin:6px 0;color:#475569'>{html.escape(IDEA_PLACEHOLDER)}</p></div>"
        )
    elif IDEA_KEY in parts:
        body.append(
            f"<div style='{_E_IDEA_CARD}'><div style='{_E_EYEBROW}color:#4338ca'>{IDEA_TITLE}"
            f"</div>{_email_md(parts[IDEA_KEY])}<div style='font-size:11px;color:#64748b;"
            "border-top:1px dashed #c7d2fe;margin-top:10px;padding-top:8px'>"
            f"{html.escape(IDEA_DISCLAIMER)}</div></div>"
        )
    for key, heading in NARRATIVE_SECTIONS:
        if key in parts and key in {"market_read", "top_ideas"}:
            body.append(
                f"<div style='{_E_CARD}'><div style='{_E_H}'>{html.escape(heading)}</div>"
                f"{_email_md(parts[key])}</div>"
            )
    body.append(_email_fired(snapshot))
    if snapshot.warnings:
        body.append(
            f"<div style='{_E_CARD}background:#fffbeb;border-color:#fde68a;font-size:12px;"
            "color:#92400e'>" + "<br>".join(html.escape(w) for w in snapshot.warnings) + "</div>"
        )
    body.append(
        "<div style='font-size:11px;color:#94a3b8;text-align:center;margin:10px 0'>"
        f"{html.escape(BRAND)} · {len(snapshot.universe)} symbols screened · "
        "internal research, not investment advice</div>"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name=viewport content='width=device-width'>"
        f"<title>{html.escape(BRAND)} {REPORT_TITLE} {snapshot.run_date}</title></head>"
        f"<body style='margin:0;padding:16px 10px;background:#eef2f7;{_E_FONT}font-size:14px;"
        f"color:#0f172a'><div style='max-width:680px;margin:0 auto'>{''.join(body)}</div>"
        "</body></html>"
    )
