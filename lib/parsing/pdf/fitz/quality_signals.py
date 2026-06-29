"""Cheap deterministic parsing-quality checks for the adaptive cascade.

Article 10 walks two stages of evaluation that fire BEFORE the LLM is
called :

  - Stage 1 (pre-parsing) : per-page and per-document metadata that PyMuPDF
    reports without doing any layout analysis. Char count per page, image
    count, the PDF producer string. Runs in milliseconds.

  - Stage 2 (parsing-time) : signals derived from the `line_df` that
    `fitz_pdf_to_line_df` just produced. Flat-table fingerprint, opaque
    figure region, column-count anomaly. Runs in O(n_lines) per page.

The escalation policy is encoded in the truth-or boolean returned by
`page_level_parsing_signals(...)['escalate']`. The pipeline reads it
before deciding to call retrieval/generation on a page, and routes flagged
pages to Azure DI (for flat tables / column anomalies) or to a vision LLM
(for opaque figures) instead.

These functions are deliberately CHEAP. They do not call any LLM, they do
not re-parse, they only read what PyMuPDF already produced.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz
import pandas as pd

from lib.parsing.pdf.fitz.columns import detect_columns_per_page

_SCANNER_PRODUCER_TOKENS = (
    "scan", "canon", "omnipage", "abbyy", "xerox", "epson",
)


def pre_parse_signals(pdf_path: str | Path) -> dict[str, Any]:
    """Per-page and per-document signals computable in O(milliseconds).

    Reads the PDF metadata (producer, creator) once, then sweeps every page
    to collect the text length, image count, and image bboxes. None of this
    requires any deeper parsing : the values are returned directly by
    PyMuPDF's own structures.

    Returns a dict with :
      - producer            : the PDF producer string (e.g. "pdfTeX-1.40.25").
      - creator             : the PDF creator string.
      - is_scanner_output   : True if the producer matches a known scanner.
      - pages               : DataFrame with one row per page, columns
                              [page_num, char_count, image_count, image_bboxes].
    """
    doc = fitz.open(str(pdf_path))
    meta = doc.metadata or {}
    producer = (meta.get("producer") or "").lower()
    pages_rows = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        image_info = page.get_image_info()
        pages_rows.append({
            "page_num": i + 1,
            "char_count": len(text),
            "image_count": len(image_info),
            "image_bboxes": [tuple(im["bbox"]) for im in image_info],
        })
    doc.close()
    return {
        "producer": meta.get("producer") or "",
        "creator":  meta.get("creator") or "",
        "is_scanner_output": any(tok in producer for tok in _SCANNER_PRODUCER_TOKENS),
        "pages": pd.DataFrame(pages_rows),
    }


def _line_overlaps_bbox(x0: float, y0: float, x1: float, y1: float,
                       bbox: tuple[float, float, float, float]) -> bool:
    bx0, by0, bx1, by1 = bbox
    return not (x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1)


_TABLE_CAPTION_RE = re.compile(r"^\s*Table\s+\d+[:.]", re.IGNORECASE)


def page_level_parsing_signals(
    line_df: pd.DataFrame,
    page_num: int,
    pre_signals: dict[str, Any] | None = None,
    *,
    flat_table_min_short_lines: int = 10,
    flat_table_max_short_text_len: int = 5,
    flat_table_max_short_bbox_width: float = 30.0,
    col_anomaly_threshold: int = 3,
) -> dict[str, Any]:
    """Three deterministic flags from line_df, computed on one page.

    Inputs :
      - line_df : the DataFrame returned by `fitz_pdf_to_line_df`. Must
        carry `page_num`, `text`, `x0`, `x1` (+ `y0`, `y1` if a bbox check
        is requested).
      - page_num : the 1-based page index to inspect.
      - pre_signals : the dict returned by `pre_parse_signals`. Optional.
        If absent, the opaque-figure check is skipped.

    Returns :
      - page_num             : echoed back.
      - flat_table           : True if the page looks like a flat-parsed table.
      - opaque_figure        : True if the page has an image bbox holding no
                               extractable text. Requires `pre_signals`.
      - col_anomaly          : True if PyMuPDF's column detector reports
                               `>= col_anomaly_threshold` columns.
      - reasons              : list of short strings naming the flags that fired.
      - escalate             : union of the three flags.
    """
    p = line_df[line_df.page_num == page_num]
    if len(p) == 0:
        return {
            "page_num": page_num,
            "flat_table": False, "opaque_figure": False, "col_anomaly": False,
            "reasons": ["empty_page"], "escalate": True,
        }

    has_caption = bool(p["text"].astype(str).str.match(_TABLE_CAPTION_RE).any())
    short_lines = (
        (p["text"].astype(str).str.len() < flat_table_max_short_text_len)
        & ((p["x1"] - p["x0"]) < flat_table_max_short_bbox_width)
    ).sum()
    is_flat_table = bool(has_caption and short_lines >= flat_table_min_short_lines)

    has_opaque_figure = False
    if pre_signals is not None:
        inventory = pre_signals["pages"]
        row = inventory[inventory.page_num == page_num]
        if len(row) > 0:
            image_bboxes = row.iloc[0]["image_bboxes"]
            for bbox in image_bboxes:
                covers_text = any(
                    _line_overlaps_bbox(r.x0, r.y0, r.x1, r.y1, bbox)
                    for _, r in p.iterrows()
                )
                if not covers_text:
                    has_opaque_figure = True
                    break

    cols_df = detect_columns_per_page(p)
    page_cols = int(cols_df.iloc[0]["n_columns"]) if len(cols_df) else 1
    is_col_anomaly = page_cols >= col_anomaly_threshold

    reasons = []
    if is_flat_table:    reasons.append("flat_table")
    if has_opaque_figure: reasons.append("opaque_figure")
    if is_col_anomaly:   reasons.append(f"col_anomaly_{page_cols}")

    return {
        "page_num": page_num,
        "flat_table": is_flat_table,
        "opaque_figure": has_opaque_figure,
        "col_anomaly": is_col_anomaly,
        "reasons": reasons,
        "escalate": is_flat_table or has_opaque_figure or is_col_anomaly,
    }
