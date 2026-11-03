"""
Builds the single submission PDF the assignment asks for.

The brief requires one PDF holding, in this order:

    1. GitHub repository link
    2. Live Streamlit app link
    3. Screenshot of the assignment running on BITS Virtual Lab
    4. The full README content

Sections 1-3 are filled from SUBMISSION details below; section 4 is rendered
from README.md, so editing the README and re-running this script keeps the
PDF in step with the repository.

    python build_submission_pdf.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent
README = PROJECT_ROOT / "README.md"
OUTPUT_PDF = PROJECT_ROOT / "ML_Assignment_2_Submission.pdf"

# ---------------------------------------------------------------------------
# Fill these in before the final run.
# ---------------------------------------------------------------------------
SUBMISSION = {
    "student_name": "<YOUR NAME>",
    "student_id": "<YOUR BITS ID>",
    "course": "Machine Learning - Assignment 2",
    "programme": "M.Tech (AIML), WILP Division, BITS Pilani",
    "github_url": "<PASTE YOUR GITHUB REPOSITORY URL HERE>",
    "streamlit_url": "<PASTE YOUR STREAMLIT COMMUNITY CLOUD APP URL HERE>",
    # Path to the BITS Virtual Lab screenshot, e.g. "bits_lab_screenshot.png".
    # Leave as None to emit a labelled placeholder box instead.
    "screenshot_path": None,
}

INK = colors.HexColor("#1b2a4a")
ACCENT = colors.HexColor("#2f6f6b")
MUTED = colors.HexColor("#5c6b84")
RULE = colors.HexColor("#d8dde6")
BAND = colors.HexColor("#eef3f2")

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 16 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def build_styles() -> dict:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1f2733"),
        alignment=TA_LEFT,
        spaceAfter=5,
    )
    return {
        "body": body,
        "h1": ParagraphStyle("H1", parent=body, fontName="Helvetica-Bold",
                             fontSize=17, leading=21, textColor=INK,
                             spaceBefore=4, spaceAfter=8),
        "h2": ParagraphStyle("H2", parent=body, fontName="Helvetica-Bold",
                             fontSize=12.5, leading=16, textColor=INK,
                             spaceBefore=12, spaceAfter=5),
        "h3": ParagraphStyle("H3", parent=body, fontName="Helvetica-Bold",
                             fontSize=10.5, leading=14, textColor=ACCENT,
                             spaceBefore=9, spaceAfter=4),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=7.6, leading=10.2,
                               spaceAfter=0),
        "cell_head": ParagraphStyle("CellHead", parent=body, fontSize=7.6,
                                    leading=10.2, fontName="Helvetica-Bold",
                                    textColor=colors.white, spaceAfter=0),
        "code": ParagraphStyle("Code", parent=body, fontName="Courier",
                               fontSize=7.6, leading=10.4,
                               textColor=colors.HexColor("#243040"),
                               leftIndent=6, spaceBefore=2, spaceAfter=6),
        "quote": ParagraphStyle("Quote", parent=body, fontSize=9.5, leading=14,
                                leftIndent=8, textColor=INK,
                                fontName="Helvetica-Bold"),
        "bullet": ParagraphStyle("Bullet", parent=body, leftIndent=12,
                                 bulletIndent=2, spaceAfter=3),
        "meta": ParagraphStyle("Meta", parent=body, fontSize=9, leading=13,
                               textColor=MUTED),
        "label": ParagraphStyle("Label", parent=body, fontName="Helvetica-Bold",
                                fontSize=8, leading=11, textColor=MUTED),
        "link": ParagraphStyle("Link", parent=body, fontName="Courier",
                               fontSize=9.5, leading=13, textColor=ACCENT),
    }


# ---------------------------------------------------------------------------
# Inline markdown -> reportlab mini-HTML
# ---------------------------------------------------------------------------
# GitHub renders the box-drawing characters in the repository-tree block fine,
# but reportlab's built-in Courier has no glyphs for them and draws a filled
# rectangle instead, so they are transliterated to ASCII for the PDF only.
BOX_DRAWING = str.maketrans({
    "│": "|", "├": "|", "└": "`", "─": "-",
    "┬": "+", "┴": "+", "┼": "+", "┐": "+",
    "┌": "+", "┘": "+", "╶": "-", "╴": "-",
})


def ascii_box(text: str) -> str:
    return text.translate(BOX_DRAWING)


def inline(text: str) -> str:
    """Convert the inline markdown used in the README into reportlab markup."""
    text = html.escape(text, quote=False)
    # [label](url)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda m: f'<link href="{m.group(2)}" color="#2f6f6b">{m.group(1)}</link>',
        text,
    )
    # <https://autolink>
    text = re.sub(
        r"&lt;(https?://[^&\s]+)&gt;",
        lambda m: f'<link href="{m.group(1)}" color="#2f6f6b">{m.group(1)}</link>',
        text,
    )
    # `code`
    text = re.sub(
        r"`([^`]+)`",
        lambda m: f'<font face="Courier" size="8">{m.group(1)}</font>',
        text,
    )
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", text)
    return text


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def column_widths(rows: list[list[str]], total: float) -> list[float]:
    """Width proportional to the longest cell per column, with a sane floor."""
    count = len(rows[0])
    longest = [
        max((len(row[i]) for row in rows if i < len(row)), default=1)
        for i in range(count)
    ]
    # Compress very long cells so one essay column cannot starve the rest.
    weights = [max(w, 4) ** 0.62 for w in longest]
    scale = total / sum(weights)
    widths = [w * scale for w in weights]

    # A two-column table is the observations layout: one short label column
    # against one essay column. Without a generous floor the label column gets
    # squeezed until short names break mid-word ("Logistic R / egression").
    floor = {2: 30 * mm, 3: 18 * mm}.get(count, 11 * mm)
    for i, w in enumerate(widths):
        if w < floor:
            widths[i] = floor
    overflow = sum(widths) - total
    if overflow > 0:
        flexible = [i for i, w in enumerate(widths) if w > floor + 1]
        pool = sum(widths[i] - floor for i in flexible) or 1
        for i in flexible:
            widths[i] -= overflow * (widths[i] - floor) / pool
    return widths


def make_table(raw_rows: list[list[str]], styles: dict) -> Table:
    header, *body_rows = raw_rows
    data = [[Paragraph(inline(c), styles["cell_head"]) for c in header]]
    for row in body_rows:
        padded = row + [""] * (len(header) - len(row))
        data.append([Paragraph(inline(c), styles["cell"]) for c in padded[: len(header)]])

    table = Table(
        data,
        colWidths=column_widths(raw_rows, CONTENT_WIDTH),
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    return table


# ---------------------------------------------------------------------------
# Block-level markdown walk
# ---------------------------------------------------------------------------
def render_markdown(markdown_text: str, styles: dict) -> list:
    flow: list = []
    lines = markdown_text.replace("\r\n", "\n").split("\n")
    index = 0
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            flow.append(Paragraph(inline(" ".join(paragraph_buffer)), styles["body"]))
            paragraph_buffer.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        # Fenced code block
        if stripped.startswith("```"):
            flush_paragraph()
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            body = "<br/>".join(
                html.escape(ascii_box(b), quote=False).replace(" ", "&nbsp;")
                for b in block
            )
            holder = Table([[Paragraph(body, styles["code"])]],
                           colWidths=[CONTENT_WIDTH], hAlign="LEFT")
            holder.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f7fa")),
                ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            flow.extend([Spacer(1, 3), holder, Spacer(1, 6)])
            continue

        # Pipe table
        if stripped.startswith("|") and index + 1 < len(lines) and re.match(
            r"^\|[\s:|-]+\|?$", lines[index + 1].strip()
        ):
            flush_paragraph()
            raw_rows = [split_row(stripped)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                raw_rows.append(split_row(lines[index].strip()))
                index += 1
            table = make_table(raw_rows, styles)
            # Short tables (the comparison and confusion-matrix tables) read
            # badly with one orphan row on the next page, so keep them whole.
            # Long ones are left splittable with the header row repeated.
            if len(raw_rows) <= 8 and max(len(c) for r in raw_rows for c in r) < 120:
                flow.extend([Spacer(1, 3), KeepTogether(table), Spacer(1, 8)])
            else:
                flow.extend([Spacer(1, 3), table, Spacer(1, 8)])
            continue

        # Horizontal rule
        if re.fullmatch(r"-{3,}", stripped):
            flush_paragraph()
            flow.extend([
                Spacer(1, 5),
                HRFlowable(width="100%", thickness=0.6, color=RULE),
                Spacer(1, 7),
            ])
            index += 1
            continue

        # Headings
        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            style = styles[f"h{level}"]
            text = Paragraph(inline(heading.group(2)), style)
            flow.append(text if level == 1 else KeepTogether([text, Spacer(1, 1)]))
            if level == 1:
                flow.append(HRFlowable(width="100%", thickness=1.1, color=ACCENT,
                                       spaceBefore=1, spaceAfter=7))
            index += 1
            continue

        # Block quote
        if stripped.startswith(">"):
            flush_paragraph()
            quoted = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                piece = lines[index].strip().lstrip(">").strip()
                if piece:
                    quoted.append(piece)
                index += 1
            for piece in quoted:
                flow.append(Paragraph(inline(piece), styles["quote"]))
            flow.append(Spacer(1, 5))
            continue

        # Lists
        bullet = re.match(r"^(\d+)\.\s+(.*)$", stripped) or re.match(r"^[-*]\s+(.*)$",
                                                                    stripped)
        if bullet:
            flush_paragraph()
            while index < len(lines):
                current = lines[index].strip()
                ordered = re.match(r"^(\d+)\.\s+(.*)$", current)
                unordered = re.match(r"^[-*]\s+(.*)$", current)
                if not (ordered or unordered):
                    if current and not re.match(r"^(#{1,3}\s|\||>|```|-{3,}$)", current):
                        # continuation line of the previous bullet
                        previous = flow.pop()
                        flow.append(Paragraph(previous.text + " " + inline(current),
                                              styles["bullet"]))
                        index += 1
                        continue
                    break
                marker = f"{ordered.group(1)}." if ordered else "•"
                text = ordered.group(2) if ordered else unordered.group(1)
                flow.append(Paragraph(inline(text), styles["bullet"],
                                      bulletText=marker))
                index += 1
            flow.append(Spacer(1, 4))
            continue

        paragraph_buffer.append(stripped)
        index += 1

    flush_paragraph()
    return flow


# ---------------------------------------------------------------------------
# Submission front matter
# ---------------------------------------------------------------------------
def link_block(number: str, heading: str, url: str, styles: dict) -> KeepTogether:
    # A placeholder still contains angle brackets, which reportlab would read as
    # an unknown tag and silently drop, so escape it before it reaches Paragraph.
    is_placeholder = url.strip().startswith("<")
    body = Paragraph(
        html.escape(url) if is_placeholder
        else f'<link href="{html.escape(url, quote=True)}" color="#2f6f6b">'
             f'{html.escape(url)}</link>',
        styles["link"],
    )
    inner = Table(
        [[Paragraph(f"{number}. {heading}", styles["label"])], [body]],
        colWidths=[CONTENT_WIDTH],
        hAlign="LEFT",
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
    ]))
    return KeepTogether([inner, Spacer(1, 9)])


def screenshot_block(styles: dict) -> list:
    flow = [Paragraph("3. Screenshot of execution on BITS Virtual Lab",
                      styles["label"]), Spacer(1, 5)]
    path = SUBMISSION["screenshot_path"]
    if path and (PROJECT_ROOT / path).exists():
        source = PROJECT_ROOT / path
        from reportlab.lib.utils import ImageReader

        width_px, height_px = ImageReader(str(source)).getSize()
        draw_width = CONTENT_WIDTH
        draw_height = draw_width * height_px / width_px
        ceiling = 165 * mm
        if draw_height > ceiling:
            draw_height = ceiling
            draw_width = draw_height * width_px / height_px
        flow.append(Image(str(source), width=draw_width, height=draw_height))
    else:
        placeholder = Table(
            [[Paragraph(
                "Paste the BITS Virtual Lab screenshot here.<br/><br/>"
                "Set <font face='Courier' size='8'>screenshot_path</font> in "
                "build_submission_pdf.py to the image filename and re-run the "
                "script, or drop the image into this box in a PDF editor.",
                styles["meta"])]],
            colWidths=[CONTENT_WIDTH],
            rowHeights=[62 * mm],
            hAlign="LEFT",
        )
        placeholder.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.9, ACCENT),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfcfe")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ]))
        flow.append(placeholder)
    flow.append(Spacer(1, 10))
    return flow


def page_furniture(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 10 * mm,
                      f"{SUBMISSION['course']}  |  {SUBMISSION['student_name']}"
                      f"  |  {SUBMISSION['student_id']}")
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 10 * mm, f"Page {document.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 13 * mm, PAGE_WIDTH - MARGIN, 13 * mm)
    canvas.restoreState()


def main() -> None:
    styles = build_styles()
    document = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=14 * mm,
        bottomMargin=18 * mm,
        title="Machine Learning Assignment 2 - Submission",
        author=SUBMISSION["student_name"],
    )

    flow: list = [
        Paragraph(SUBMISSION["course"], styles["h1"]),
        HRFlowable(width="100%", thickness=1.1, color=ACCENT, spaceAfter=7),
        Paragraph(SUBMISSION["programme"], styles["meta"]),
        Paragraph(
            f"<b>{html.escape(SUBMISSION['student_name'])}</b> &nbsp;|&nbsp; "
            f"{html.escape(SUBMISSION['student_id'])}",
            styles["meta"],
        ),
        Paragraph(
            "Online Shoppers Purchasing Intention - five classification models "
            "and a deployed Streamlit application.",
            styles["meta"],
        ),
        Spacer(1, 13),
        Paragraph("Submission links", styles["h2"]),
        link_block("1", "GitHub repository link", SUBMISSION["github_url"], styles),
        link_block("2", "Live Streamlit app link", SUBMISSION["streamlit_url"], styles),
    ]
    flow.extend(screenshot_block(styles))
    flow.append(PageBreak())
    flow.append(Paragraph("4. GitHub README content", styles["h2"]))
    flow.append(Spacer(1, 4))
    flow.extend(render_markdown(README.read_text(encoding="utf-8"), styles))

    document.build(flow, onFirstPage=page_furniture, onLaterPages=page_furniture)
    print(f"Wrote {OUTPUT_PDF.name}")


if __name__ == "__main__":
    main()
