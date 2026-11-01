"""Render a ``Snapshot`` (+ optional narrative) to markdown and HTML.

Both renderers consume the same intermediate ``Section`` list so the two outputs
never drift apart. The markdown is the archival artifact; the HTML is the email.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

from ehf.server.research.reporting.screens import SCREEN_ORDER, Hit
from ehf.server.research.reporting.snapshot import Snapshot

Json = dict[str, Any]

SCREEN_TITLES = {
    "quality_dislocation": "Quality on sale — fell hard, fundamentals intact",
    "pead": "Post-earnings drift — surprise and reaction agree",
    "insider_cluster": "Insider cluster buying",
    "analyst_revision": "Analyst revision momentum",
    "breakout_52w": "52-week-high breakouts with volume",
    "outlier_move": "Abnormal one-day moves (explain, don't chase)",
}

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


@dataclass
class Section:
    title: str
    intro: str = ""
    bullets: list[str] = field(default_factory=list)
    table: Table | None = None
    hits: list[Hit] = field(default_factory=list)
    markdown: str = ""  # free-form markdown body (narrative)
    level: int = 2


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


def _is_nan(v: object) -> bool:
    return isinstance(v, float) and v != v


def _market_sections(market: Json, snapshot: Snapshot) -> list[Section]:
    idx, vix, br, tr = (
        market.get("index", {}),
        market.get("vix", {}),
        market.get("breadth", {}),
        market.get("treasury", {}),
    )
    bullets = []
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
    sections = [Section(title="Market read", bullets=bullets)]

    sectors = market.get("sectors") or []
    if sectors:
        sections.append(
            Section(
                title="Sectors (last session)",
                table=Table(
                    ["Sector", "Avg change"],
                    [
                        [str(s["sector"]), fmt((s["avg_change_pct"] or 0) / 100, "pct")]
                        for s in sectors
                    ],
                ),
            )
        )
    econ = market.get("econ_ahead") or []
    if econ:
        sections.append(
            Section(
                title="High-impact US data this week",
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
    return Table(header, rows)


def build_sections(snapshot: Snapshot, narrative: str | None) -> list[Section]:
    sections: list[Section] = []
    if narrative:
        sections.append(Section(title="Research assistant read", markdown=narrative))
    sections.extend(_market_sections(snapshot.market, snapshot))
    for screen in SCREEN_ORDER:
        hits = snapshot.hits.get(screen, [])
        title = SCREEN_TITLES[screen]
        if not hits:
            sections.append(Section(title=title, intro="Nothing fired today."))
            continue
        sections.append(Section(title=title, table=_hit_table(screen, hits), hits=hits))
    foot = [
        f"Universe: {len(snapshot.universe)} symbols (liquid screener ∪ S&P 500). "
        f"Data as of {snapshot.as_of}. "
        f"FMP calls: {snapshot.api_calls}. Timings (s): "
        + ", ".join(f"{k} {v}" for k, v in snapshot.timings.items())
    ]
    foot.extend(f"Warning: {w}" for w in snapshot.warnings)
    sections.append(Section(title="Run notes", bullets=foot))
    return sections


# --- markdown ----------


def _md_table(t: Table) -> str:
    out = ["| " + " | ".join(t.columns) + " |", "|" + "|".join("---" for _ in t.columns) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in t.rows)
    return "\n".join(out)


def _hit_notes_md(h: Hit) -> str:
    also = f"; also in: {', '.join(h.context['also_in'])}" if h.context.get("also_in") else ""
    lines = [f"- **{h.symbol}** — {h.headline} _(horizon: {h.horizon}; {h.direction}{also})_"]
    lines.extend(f"  - {r}" for r in h.reasons)
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
    if extra:
        lines.append("  - " + "; ".join(extra))
    for n in ctx.get("news", [])[:3]:
        lines.append(f"  - news {n.get('date')}: {n.get('title')}")
    return "\n".join(lines)


def render_markdown(snapshot: Snapshot, narrative: str | None) -> str:
    out = [
        f"# Eagle Hill daily market report — {snapshot.run_date}",
        f"_Data as of {snapshot.as_of} close._",
        "",
    ]
    for s in build_sections(snapshot, narrative):
        out.append(f"{'#' * s.level} {s.title}")
        if s.intro:
            out.append(s.intro)
        if s.markdown:
            out.append(s.markdown)
        out.extend(f"- {b}" for b in s.bullets)
        if s.table:
            out.append(_md_table(s.table))
        if s.hits:
            out.append("")
            out.extend(_hit_notes_md(h) for h in s.hits)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# --- html ----------

_CSS = "".join(
    (
        "body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;font-size:14px;",
        "line-height:1.45;color:#1b1f23;max-width:960px;margin:0 auto;padding:16px}",
        "h1{font-size:22px;margin:0 0 4px}",
        "h2{font-size:17px;margin:26px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}",
        "h3{font-size:15px;margin:16px 0 6px}",
        "table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}",
        "th,td{border:1px solid #e1e4e8;padding:4px 6px;text-align:right;white-space:nowrap}",
        "th:first-child,td:first-child,td.l{text-align:left}th{background:#f6f8fa}",
        ".muted{color:#6a737d}.hit{margin:8px 0 10px}.hit ul{margin:2px 0 0 18px;padding:0}",
        ".hit li{margin:1px 0}",
        ".tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:9px;",
        "background:#eef;color:#335;margin-left:6px}",
        ".narr p{margin:6px 0}.narr ul{margin:4px 0 8px 20px}",
    )
)


def _inline_md(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


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
        para.append(line.strip())
    flush_para()
    close_list()
    return "\n".join(out)


_TD_LEFT = ' class="l"'


def _html_table(t: Table) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in t.columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td{_TD_LEFT if i == 0 else ''}>{html.escape(c)}</td>" for i, c in enumerate(r))
        + "</tr>"
        for r in t.rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _hit_html(h: Hit) -> str:
    items = "".join(f"<li>{html.escape(r)}</li>" for r in h.reasons)
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
    if extra:
        items += f"<li>{html.escape('; '.join(extra))}</li>"
    for n in ctx.get("news", [])[:3]:
        when, title = html.escape(str(n.get("date"))), html.escape(str(n.get("title")))
        items += f"<li class=muted>news {when}: {title}</li>"
    tags = [h.direction, h.horizon, *(f"also: {s}" for s in ctx.get("also_in", []))]
    tag_html = "".join(f"<span class=tag>{html.escape(t)}</span>" for t in tags)
    return (
        f"<div class=hit><b>{h.symbol}</b> — {html.escape(h.headline)}{tag_html}"
        f"<ul>{items}</ul></div>"
    )


def render_html(snapshot: Snapshot, narrative: str | None) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Eagle Hill daily market report — {snapshot.run_date}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>Eagle Hill daily market report — {snapshot.run_date}</h1>",
        f"<div class=muted>Data as of {snapshot.as_of} close.</div>",
    ]
    for s in build_sections(snapshot, narrative):
        parts.append(f"<h{s.level}>{html.escape(s.title)}</h{s.level}>")
        if s.intro:
            parts.append(f"<p class=muted>{html.escape(s.intro)}</p>")
        if s.markdown:
            parts.append(f"<div class=narr>{md_to_html(s.markdown)}</div>")
        if s.bullets:
            parts.append("<ul>" + "".join(f"<li>{_inline_md(b)}</li>" for b in s.bullets) + "</ul>")
        if s.table:
            parts.append(_html_table(s.table))
        parts.extend(_hit_html(h) for h in s.hits)
    parts.append("</body></html>")
    return "\n".join(parts)
