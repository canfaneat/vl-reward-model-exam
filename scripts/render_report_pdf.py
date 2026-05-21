#!/usr/bin/env python3
"""Render the Chinese Markdown report to a compact PDF.

This intentionally keeps the renderer small and local.  The assessment only
requires a PDF report, and this avoids pulling in a full TeX installation.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "docs" / "REPORT_FINAL_CN.md"
REPORT_PDF = ROOT / "docs" / "REPORT_FINAL_CN.pdf"
PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = RIGHT_MARGIN = 1.55 * cm
TOP_MARGIN = 1.35 * cm
BOTTOM_MARGIN = 1.35 * cm
AVAILABLE_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN


def clean_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return text


def is_table(lines: list[str], i: int) -> bool:
    return (
        i + 1 < len(lines)
        and lines[i].lstrip().startswith("|")
        and lines[i + 1].lstrip().startswith("|")
        and set(lines[i + 1].strip().replace("|", "").replace(" ", "")) <= {"-", ":"}
    )


def parse_table(lines: list[str], i: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        line = lines[i].strip()
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not (set(line.replace("|", "").replace(" ", "")) <= {"-", ":"}):
            rows.append(cells)
        i += 1
    return rows, i


def col_widths(rows: list[list[str]]) -> list[float]:
    n = max(len(r) for r in rows)
    header = [c.strip().lower() for c in rows[0]] if rows else []

    # Paper-style tables read better when method/description columns are wider
    # and metric columns stay compact.  The rules below are keyed by table shape
    # and header names rather than by a single table title, so later report edits
    # still keep sensible widths.
    if n == 2:
        return [AVAILABLE_WIDTH * 0.30, AVAILABLE_WIDTH * 0.70]
    if n == 3:
        return [AVAILABLE_WIDTH * 0.24, AVAILABLE_WIDTH * 0.42, AVAILABLE_WIDTH * 0.34]
    if n == 4:
        if any("accuracy" in h or "score" in h for h in header):
            return [
                AVAILABLE_WIDTH * 0.28,
                AVAILABLE_WIDTH * 0.26,
                AVAILABLE_WIDTH * 0.17,
                AVAILABLE_WIDTH * 0.29,
            ]
        return [AVAILABLE_WIDTH * 0.25, AVAILABLE_WIDTH * 0.25, AVAILABLE_WIDTH * 0.20, AVAILABLE_WIDTH * 0.30]
    if n == 5:
        return [
            AVAILABLE_WIDTH * 0.28,
            AVAILABLE_WIDTH * 0.18,
            AVAILABLE_WIDTH * 0.18,
            AVAILABLE_WIDTH * 0.18,
            AVAILABLE_WIDTH * 0.18,
        ]

    max_chars = [1] * n
    for row in rows:
        for idx, cell in enumerate(row):
            max_chars[idx] = max(max_chars[idx], min(len(cell), 40))

    # Give explanation columns a little more room without hard-coding tables.
    total = sum(max_chars)
    widths = [AVAILABLE_WIDTH * c / total for c in max_chars]
    min_w = 1.6 * cm
    if any(w < min_w for w in widths) and n * min_w <= AVAILABLE_WIDTH:
        deficit = sum(max(0, min_w - w) for w in widths)
        widths = [max(min_w, w) for w in widths]
        wide = [j for j, w in enumerate(widths) if w > min_w]
        wide_total = sum(widths[j] - min_w for j in wide)
        if wide_total > 0:
            for j in wide:
                widths[j] -= deficit * (widths[j] - min_w) / wide_total
    return widths


def make_table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    n = max(len(r) for r in rows)
    normalized = [r + [""] * (n - len(r)) for r in rows]
    data = []
    for ridx, row in enumerate(normalized):
        style = styles["table_header"] if ridx == 0 else styles["table_cell"]
        data.append([Paragraph(clean_inline(cell), style) for cell in row])
    table = Table(data, colWidths=col_widths(normalized), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4FAFF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#13293D")),
                ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.HexColor("#183B56")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.HexColor("#88C6DC")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor("#183B56")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FBFDFF")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
                ("TOPPADDING", (0, 0), (-1, -1), 4.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.0),
            ]
        )
    )
    return table


def make_image(md_line: str, base_dir: Path, styles: dict[str, ParagraphStyle]):
    match = re.match(r"!\[(.*?)\]\((.*?)\)", md_line.strip())
    if not match:
        return []
    caption, rel_path = match.groups()
    image_path = (base_dir / rel_path).resolve()
    if not image_path.exists():
        return [Paragraph(f"图像缺失：{html.escape(str(image_path))}", styles["body"])]

    with PILImage.open(image_path) as img:
        width, height = img.size
    max_width = AVAILABLE_WIDTH
    max_height = 10.5 * cm
    scale = min(max_width / width, max_height / height, 1.0)
    flowables = [
        Spacer(1, 0.15 * cm),
        Image(str(image_path), width=width * scale, height=height * scale, hAlign="CENTER"),
    ]
    if caption:
        flowables.append(Paragraph(clean_inline(caption), styles["caption"]))
    flowables.append(Spacer(1, 0.15 * cm))
    return [KeepTogether(flowables)]


def build_styles() -> dict[str, ParagraphStyle]:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    sample = getSampleStyleSheet()
    base = ParagraphStyle(
        "CNBase",
        parent=sample["Normal"],
        fontName="STSong-Light",
        fontSize=10.0,
        leading=15.2,
        wordWrap="CJK",
        alignment=TA_LEFT,
        spaceAfter=5,
    )
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base,
            fontSize=18.2,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=colors.HexColor("#0F2F44"),
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base,
            fontSize=14.2,
            leading=20,
            spaceBefore=12,
            spaceAfter=7,
            textColor=colors.HexColor("#0F5D73"),
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base,
            fontSize=12.0,
            leading=17,
            spaceBefore=8,
            spaceAfter=5,
            textColor=colors.HexColor("#226A73"),
        ),
        "body": base,
        "caption": ParagraphStyle(
            "Caption",
            parent=base,
            fontSize=8.5,
            leading=11.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#52616B"),
            spaceAfter=6,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base,
            fontSize=7.35,
            leading=10.4,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base,
            fontSize=7.45,
            leading=10.4,
            wordWrap="CJK",
            textColor=colors.HexColor("#13293D"),
        ),
        "code": ParagraphStyle(
            "Code",
            parent=sample["Code"],
            fontName="Courier",
            fontSize=7.0,
            leading=9.2,
            leftIndent=6,
            rightIndent=6,
            backColor=colors.HexColor("#F6FAFD"),
            borderColor=colors.HexColor("#D5EAF5"),
            borderWidth=0.4,
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=7,
        ),
    }


def render() -> None:
    styles = build_styles()
    lines = REPORT_MD.read_text(encoding="utf-8").splitlines()
    story = []
    base_dir = REPORT_MD.parent
    i = 0
    in_code = False
    code_lines: list[str] = []
    para_lines: list[str] = []

    def flush_para() -> None:
        nonlocal para_lines
        if para_lines:
            text = " ".join(line.strip() for line in para_lines if line.strip())
            if text:
                story.append(Paragraph(clean_inline(text), styles["body"]))
            para_lines = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["code"], maxLineLength=88))
                code_lines = []
                in_code = False
            else:
                flush_para()
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            flush_para()
            story.append(Spacer(1, 0.06 * cm))
            i += 1
            continue

        if stripped.startswith("# "):
            flush_para()
            story.append(Paragraph(clean_inline(stripped[2:]), styles["title"]))
            i += 1
            continue

        if stripped.startswith("## "):
            flush_para()
            if story:
                story.append(Spacer(1, 0.08 * cm))
            story.append(Paragraph(clean_inline(stripped[3:]), styles["h2"]))
            i += 1
            continue

        if stripped.startswith("### "):
            flush_para()
            story.append(Paragraph(clean_inline(stripped[4:]), styles["h3"]))
            i += 1
            continue

        if stripped.startswith("!["):
            flush_para()
            story.extend(make_image(stripped, base_dir, styles))
            i += 1
            continue

        if is_table(lines, i):
            flush_para()
            rows, i = parse_table(lines, i)
            story.append(make_table(rows, styles))
            story.append(Spacer(1, 0.18 * cm))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            flush_para()
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item_text = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                items.append(ListItem(Paragraph(clean_inline(item_text), styles["body"]), leftIndent=12))
                i += 1
            story.append(ListFlowable(items, bulletType="1", start="1", leftIndent=18))
            continue

        if stripped.startswith("- "):
            flush_para()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                item_text = lines[i].strip()[2:]
                items.append(ListItem(Paragraph(clean_inline(item_text), styles["body"]), leftIndent=12))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18))
            continue

        para_lines.append(line)
        i += 1

    flush_para()

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="InternVL2.5-2B 多模态奖励模型训练与数据选择",
        author="canfaneat",
    )
    doc.build(story)


if __name__ == "__main__":
    render()
    print(REPORT_PDF)
