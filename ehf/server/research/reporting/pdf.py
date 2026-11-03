"""PDF rendering of the report with reportlab (pure Python, so it runs inside the Lambda zip).

Consumes the same ``Section`` list as the markdown/HTML renderers. Layout: a dark
cover band with the regime and stat tiles, the idea card, the narrative, then the
market tables and one block per screen (table + per-name notes). Letter size,
light body for printing, page numbers in the footer.
"""

from __future__ import annotations

import io
import re
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from ehf.server.research.reporting.narrative import IDEA_DISCLAIMER
from ehf.server.research.reporting.render import (
    BRAND,
    REPORT_TITLE,
    SCREEN_SHORT,
    Section,
    build_sections,
    hero_tiles,
    hit_extra_line,
    regime_tone,
    tone_of,
)
from ehf.server.research.reporting.render import Table as RTable
from ehf.server.research.reporting.screens import Hit
from ehf.server.research.reporting.snapshot import Snapshot

NAVY = colors.HexColor("#0b1222")
TEAL = colors.HexColor("#14b8a6")
TEAL_DARK = colors.HexColor("#0f766e")
INDIGO = colors.HexColor("#4338ca")
INDIGO_LIGHT = colors.HexColor("#eef2ff")
INDIGO_LINE = colors.HexColor("#c7d2fe")
INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#64748b")
SOFT = colors.HexColor("#f6f8fb")
LINE = colors.HexColor("#e2e8f0")
UP = colors.HexColor("#059669")
DOWN = colors.HexColor("#e11d48")
WARN = colors.HexColor("#b45309")
PILL = {
    "on": (colors.HexColor("#d1fae5"), colors.HexColor("#065f46")),
    "off": (colors.HexColor("#ffe4e6"), colors.HexColor("#9f1239")),
    "mid": (colors.HexColor("#fef3c7"), colors.HexColor("#92400e")),
}

PAGE_W, PAGE_H = letter
MARGIN = 0.55 * inch
BAND_H = 2.15 * inch
AVAIL_W = PAGE_W - 2 * MARGIN

_SUBS = {
    "≥": ">=",
    "≤": "<=",
    "∪": "+",
    "→": "->",
    "←": "<-",
    "✦": "*",
    "σ": "sigma",
    "Σ": "sum",
    "≈": "~",
    "−": "-",
    "↑": "(up)",
    "↓": "(down)",
}


def _safe(text: str) -> str:
    """Standard PDF fonts are WinAnsi only: map the few symbols the report uses."""
    for k, v in _SUBS.items():
        text = text.replace(k, v)
    return text.encode("cp1252", "replace").decode("cp1252")


def _esc(text: str) -> str:
    return _safe(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline(text: str) -> str:
    """Markdown inline subset -> reportlab paragraph markup."""
    text = _esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)
    return re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=9.2, leading=13.2, textColor=INK)
    return {
        "body": base,
        "narr": ParagraphStyle("narr", base, fontSize=9.6, leading=14.2),
        "idea": ParagraphStyle("idea", base, fontSize=10.2, leading=15.2),
        "small": ParagraphStyle("small", base, fontSize=7.8, leading=10.5, textColor=MUTED),
        "label": ParagraphStyle(
            "label",
            base,
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=10,
            textColor=TEAL_DARK,
            spaceBefore=6,
            spaceAfter=1,
        ),
        "h2": ParagraphStyle(
            "h2", base, fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=INK
        ),
        "bullet": ParagraphStyle(
            "bullet", base, leftIndent=11, bulletIndent=1, spaceBefore=1.5, spaceAfter=1.5
        ),
        "nbullet": ParagraphStyle(
            "nbullet",
            base,
            fontSize=9.6,
            leading=14.2,
            leftIndent=11,
            bulletIndent=1,
            spaceBefore=2,
            spaceAfter=2,
        ),
        "hit": ParagraphStyle("hit", base, fontSize=9.4, leading=13.4, spaceBefore=5),
        "hitb": ParagraphStyle(
            "hitb", base, fontSize=8.2, leading=11.4, leftIndent=12, bulletIndent=3, textColor=INK
        ),
        "hitn": ParagraphStyle(
            "hitn", base, fontSize=7.8, leading=10.8, leftIndent=12, bulletIndent=3, textColor=MUTED
        ),
        "cell": ParagraphStyle("cell", base, fontSize=7.4, leading=9.4, alignment=TA_RIGHT),
        "cell_l": ParagraphStyle("cell_l", base, fontSize=7.4, leading=9.4, alignment=TA_LEFT),
        "cell_sym": ParagraphStyle(
            "cell_sym",
            base,
            fontName="Helvetica-Bold",
            fontSize=7.9,
            leading=9.6,
            alignment=TA_LEFT,
        ),
        "th": ParagraphStyle(
            "th",
            base,
            fontName="Helvetica-Bold",
            fontSize=6.6,
            leading=8.4,
            textColor=colors.white,
            alignment=TA_RIGHT,
        ),
        "th_l": ParagraphStyle(
            "th_l",
            base,
            fontName="Helvetica-Bold",
            fontSize=6.6,
            leading=8.4,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "tile_k": ParagraphStyle(
            "tile_k", base, fontName="Helvetica-Bold", fontSize=6.3, leading=8, textColor=MUTED
        ),
        "tile_v": ParagraphStyle(
            "tile_v", base, fontName="Helvetica-Bold", fontSize=13.5, leading=16, textColor=INK
        ),
        "tile_d": ParagraphStyle("tile_d", base, fontSize=6.8, leading=8.6, textColor=MUTED),
    }


_LABEL_RE = re.compile(r"^\s*\*\*[^*]+\*\*")


def _md_flowables(md: str, st: dict[str, ParagraphStyle], body: str = "narr") -> list[Flowable]:
    out: list[Flowable] = []
    para: list[str] = []
    n = 0

    def flush() -> None:
        if para:
            out.append(Paragraph(_inline(" ".join(para)), st[body]))
            para.clear()

    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            out.append(Paragraph(_inline(m.group(2)).upper(), st["label"]))
            n = 0
            continue
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            flush()
            out.append(Paragraph(_inline(m.group(1)), st["nbullet"], bulletText="•"))
            continue
        m = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)
        if m:
            flush()
            n += 1
            out.append(Paragraph(_inline(m.group(2)), st["nbullet"], bulletText=f"{n}."))
            continue
        if para and _LABEL_RE.match(line):  # "**The idea.** ..." starts its own paragraph
            flush()
        para.append(line.strip())
    flush()
    return out


def _heading(title: str, st: dict[str, ParagraphStyle], count: int | None = None) -> Flowable:
    label = _esc(title)
    if count:
        label += f'  <font size="8" color="#0f766e">{count}</font>'
    t = Table([[Paragraph(label, st["h2"])]], colWidths=[AVAIL_W])
    t.setStyle(
        TableStyle(
            [
                ("LINEBEFORE", (0, 0), (0, 0), 2.6, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return t


def _data_table(t: RTable, st: dict[str, ParagraphStyle]) -> Flowable:
    kinds = t.kinds or ["str"] * len(t.columns)
    ncol = len(t.columns)
    first_w = 0.62 * inch if ncol > 3 else 2.2 * inch
    widths = [first_w] + [(AVAIL_W - first_w) / max(ncol - 1, 1)] * (ncol - 1)
    head = [
        Paragraph(_esc(c), st["th_l"] if i == 0 or kinds[i] == "str" else st["th"])
        for i, c in enumerate(t.columns)
    ]
    rows: list[list[Any]] = [head]
    for r in t.rows:
        cells = []
        for i, c in enumerate(r):
            kind = kinds[i] if i < len(kinds) else "str"
            tone = tone_of(c, kind)
            text = _esc(c)
            if tone == "up":
                text = f'<font color="#059669">{text}</font>'
            elif tone == "down":
                text = f'<font color="#e11d48">{text}</font>'
            style = st["cell_sym"] if i == 0 else st["cell_l"] if kind == "str" else st["cell"]
            cells.append(Paragraph(text, style))
        rows.append(cells)
    tbl = Table(rows, colWidths=widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("ROUNDEDCORNERS", [4, 4, 0, 0]),
            ]
        )
    )
    return tbl


def _hit_flowables(h: Hit, st: dict[str, ParagraphStyle]) -> list[Flowable]:
    tags = [h.direction, h.horizon] + [
        f"also: {SCREEN_SHORT.get(s, s)}" for s in h.context.get("also_in", [])
    ]
    colour = {"long": "#059669", "short": "#e11d48"}.get(h.direction, "#64748b")
    head = (
        f"<b>{_esc(h.symbol)}</b>  {_esc(h.headline)}  "
        f'<font size="7" color="{colour}">[{_esc(" · ".join(tags))}]</font>'
    )
    out: list[Flowable] = [Paragraph(head, st["hit"])]
    out.extend(Paragraph(_esc(r), st["hitb"], bulletText="•") for r in h.reasons)
    if extra := hit_extra_line(h):
        out.append(Paragraph(_esc(extra), st["hitb"], bulletText="•"))
    for n in h.context.get("news", [])[:3]:
        out.append(
            Paragraph(_esc(f"news {n.get('date')}: {n.get('title')}"), st["hitn"], bulletText="•")
        )
    return [KeepTogether(out)]


def _tiles(snapshot: Snapshot, st: dict[str, ParagraphStyle]) -> Flowable | None:
    tiles = hero_tiles(snapshot.market)
    if not tiles:
        return None
    gap = 5
    n = len(tiles)
    w = (AVAIL_W - gap * (n - 1)) / n
    cells: list[Any] = []
    widths: list[float] = []
    for i, t in enumerate(tiles):
        colour = {"up": "#059669", "down": "#e11d48"}.get(t.tone, "#0f172a")
        inner = Table(
            [
                [
                    [
                        Paragraph(_esc(t.label).upper(), st["tile_k"]),
                        Paragraph(f'<font color="{colour}">{_esc(t.value)}</font>', st["tile_v"]),
                        Paragraph(_esc(t.detail), st["tile_d"]),
                    ]
                ]
            ],
            colWidths=[w],
        )
        inner.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                    ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                    ("ROUNDEDCORNERS", [6, 6, 6, 6]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        cells.append(inner)
        widths.append(w)
        if i < n - 1:
            cells.append("")
            widths.append(gap)
    outer = Table([cells], colWidths=widths)
    outer.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return outer


def _idea_card(s: Section, st: dict[str, ParagraphStyle]) -> Flowable:
    inner: list[Flowable] = [
        Paragraph(
            f'<font color="#4338ca"><b>RESEARCH ASSISTANT  ·  {_esc(s.title).upper()}</b></font>',
            st["tile_k"],
        ),
        Spacer(1, 4),
    ]
    if s.markdown:
        inner.extend(_md_flowables(s.markdown, st, body="idea"))
        inner.append(Spacer(1, 6))
        inner.append(Paragraph(_esc(IDEA_DISCLAIMER), st["small"]))
    else:
        inner.append(Paragraph(_esc(s.intro), st["narr"]))
    box = Table([[inner]], colWidths=[AVAIL_W])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), INDIGO_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.9, INDIGO_LINE),
                ("ROUNDEDCORNERS", [8, 8, 8, 8]),
                ("LEFTPADDING", (0, 0), (-1, -1), 13),
                ("RIGHTPADDING", (0, 0), (-1, -1), 13),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
            ]
        )
    )
    return box


def _section_flowables(s: Section, st: dict[str, ParagraphStyle]) -> list[Flowable]:
    if s.kind == "idea":
        return [_idea_card(s, st), Spacer(1, 10)]
    out: list[Flowable] = [Spacer(1, 6)]
    head: list[Flowable] = [_heading(s.title, st, len(s.hits) or None), Spacer(1, 4)]
    if s.intro:
        head.append(Paragraph(_esc(s.intro), st["small"]))
    if s.markdown:
        head.extend(_md_flowables(s.markdown, st)[:2])
        out.append(KeepTogether(head))
        out.extend(_md_flowables(s.markdown, st)[2:])
    elif s.table:
        head.append(_data_table(s.table, st))
        out.append(KeepTogether(head) if len(s.table.rows) <= 14 else head[0])
        if len(s.table.rows) > 14:
            out.extend(head[1:])
    else:
        out.append(KeepTogether(head))
    for b in s.bullets:
        style = st["small"] if s.kind == "notes" else st["bullet"]
        text = _inline(b)
        if b.startswith("Warning:"):
            text = f'<font color="#b45309">{text}</font>'
        out.append(Paragraph(text, style, bulletText="•"))
    for h in s.hits:
        out.extend(_hit_flowables(h, st))
    out.append(Spacer(1, 6))
    return out


def _draw_band(canvas: Any, snapshot: Snapshot) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - BAND_H, PAGE_W, BAND_H, stroke=0, fill=1)
    # soft glows, like the web page
    canvas.setFillColor(colors.Color(0.37, 0.92, 0.83, alpha=0.13))
    canvas.circle(PAGE_W * 0.92, PAGE_H + 10, 1.9 * inch, stroke=0, fill=1)
    canvas.setFillColor(colors.Color(0.5, 0.55, 0.97, alpha=0.16))
    canvas.circle(PAGE_W * 0.12, PAGE_H - BAND_H * 0.1, 1.5 * inch, stroke=0, fill=1)
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H - BAND_H, PAGE_W, 2.2, stroke=0, fill=1)

    x, top = MARGIN, PAGE_H - 0.62 * inch
    canvas.saveState()  # character spacing is graphics state: keep it off the title
    canvas.setFillColor(TEAL)
    eyebrow = canvas.beginText(x, top)
    eyebrow.setFont("Helvetica-Bold", 8.5)
    eyebrow.setCharSpace(2.2)
    eyebrow.textOut(BRAND.upper())
    canvas.drawText(eyebrow)
    canvas.restoreState()
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 27)
    canvas.drawString(x, top - 0.42 * inch, REPORT_TITLE)
    canvas.setFillColor(colors.HexColor("#aab4c8"))
    canvas.setFont("Helvetica", 10.5)
    canvas.drawString(
        x,
        top - 0.68 * inch,
        _safe(f"{snapshot.run_date:%A, %B %-d, %Y}  ·  data as of {snapshot.as_of} close"),
    )
    regime = str(snapshot.market.get("index", {}).get("regime", "") or "")
    if regime:
        bg, fg = PILL[regime_tone(regime)]
        canvas.setFont("Helvetica-Bold", 8.5)
        tw = canvas.stringWidth(regime, "Helvetica-Bold", 8.5)
        py = top - 1.02 * inch
        canvas.setFillColor(bg)
        canvas.roundRect(x, py - 4, tw + 18, 16, 8, stroke=0, fill=1)
        canvas.setFillColor(fg)
        canvas.drawString(x + 9, py, _safe(regime))
    canvas.restoreState()


def _draw_footer(canvas: Any, doc: Any, snapshot: Snapshot) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 0.48 * inch, PAGE_W - MARGIN, 0.48 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.2)
    canvas.drawString(
        MARGIN,
        0.33 * inch,
        _safe(
            f"{BRAND}  ·  {REPORT_TITLE}  ·  {snapshot.run_date}  ·  "
            "internal research, not investment advice"
        ),
    )
    canvas.drawRightString(PAGE_W - MARGIN, 0.33 * inch, f"Page {doc.page}")
    canvas.restoreState()


def render_pdf(snapshot: Snapshot, narrative: str | None) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=0.6 * inch,
        bottomMargin=0.65 * inch,
        title=f"{BRAND} {REPORT_TITLE} {snapshot.run_date}",
        author=BRAND,
        subject="Daily market screens and research notes",
    )
    first_frame = Frame(
        MARGIN,
        doc.bottomMargin,
        AVAIL_W,
        PAGE_H - BAND_H - doc.bottomMargin - 0.18 * inch,
        id="first",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    later_frame = Frame(
        MARGIN,
        doc.bottomMargin,
        AVAIL_W,
        PAGE_H - doc.topMargin - doc.bottomMargin,
        id="later",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )

    def on_first(canvas: Any, d: Any) -> None:
        _draw_band(canvas, snapshot)
        _draw_footer(canvas, d, snapshot)

    def on_later(canvas: Any, d: Any) -> None:
        _draw_footer(canvas, d, snapshot)

    doc.addPageTemplates(
        [
            PageTemplate(id="first", frames=[first_frame], onPage=on_first),
            PageTemplate(id="later", frames=[later_frame], onPage=on_later),
        ]
    )

    story: list[Flowable] = [NextPageTemplate("later")]
    tiles = _tiles(snapshot, st)
    if tiles is not None:
        story.extend([tiles, Spacer(1, 12)])
    for s in build_sections(snapshot, narrative):
        story.extend(_section_flowables(s, st))
    doc.build(story)
    return buf.getvalue()
