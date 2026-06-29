"""span_df: sub-line granularity, on demand.

A line in `line_df` is the natural unit for text content. A span is finer: each
contiguous run of characters in a single font, size, weight, italic, and color.
A line that mixes bold and non-bold text expands to multiple spans.

This module is opt-in. `parse_pdf` returns an empty `span_df`; stages that need
typography (heading detection, listing aggregation, defined-term capture) call
`build_span_df(pdf_path)` themselves.
"""
from __future__ import annotations
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd


SPAN_DF_COLUMNS = [
    "page_num", "line_num", "span_id",
    "text", "x0", "y0", "x1", "y1",
    "font_name", "font_size", "is_bold", "is_italic", "color_rgb",
    "rotation",
]


def _dir_to_rotation(dx: float, dy: float) -> int:
    """Map PyMuPDF's writing-direction unit vector to a rotation in degrees.

    The `dir` field on a line is `(cos(theta), sin(theta))` in PDF user space
    where y grows downward. The four cardinal cases that matter in practice:
       ( 1,  0) -> 0    (normal horizontal)
       ( 0, -1) -> 90   (reads bottom-to-top, e.g. arXiv side stamp)
       (-1,  0) -> 180  (upside-down)
       ( 0,  1) -> 270  (reads top-to-bottom)
    """
    if abs(dx) > abs(dy):
        return 0 if dx >= 0 else 180
    return 90 if dy <= 0 else 270


def build_span_df(file_path: str | Path) -> pd.DataFrame:
    """Walk a PDF at span granularity, one row per typographic span."""
    doc = fitz.open(str(file_path))
    rows = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict").get("blocks", [])
        line_num = 0
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_num += 1
                dx, dy = line.get("dir", (1.0, 0.0))
                rotation = _dir_to_rotation(float(dx), float(dy))
                for span_id, span in enumerate(spans, start=1):
                    bbox = span.get("bbox", (0, 0, 0, 0))
                    flags = int(span.get("flags", 0))
                    color_int = int(span.get("color", 0))
                    rows.append({
                        "page_num":  page_num + 1,
                        "line_num":  line_num,
                        "span_id":   span_id,
                        "text":      span.get("text", ""),
                        "x0":        float(bbox[0]),
                        "y0":        float(bbox[1]),
                        "x1":        float(bbox[2]),
                        "y1":        float(bbox[3]),
                        "font_name": span.get("font", ""),
                        "font_size": float(span.get("size", 0.0)),
                        "is_bold":   bool(flags & 16),
                        "is_italic": bool(flags & 2),
                        "color_rgb": (
                            (color_int >> 16) & 0xFF,
                            (color_int >> 8) & 0xFF,
                            color_int & 0xFF,
                        ),
                        "rotation":  rotation,
                    })
    doc.close()
    if not rows:
        return pd.DataFrame(columns=SPAN_DF_COLUMNS)
    return pd.DataFrame(rows, columns=SPAN_DF_COLUMNS)
