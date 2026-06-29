"""Orchestrator : run EasyOCR on a PDF and return a partial parse dict.

Unlike `parse_pdf_fitz` and the layout-aware parsers (`parse_pdf_docling`,
`parse_pdf_azure_layout`), EasyOCR only feeds the `line_df` slot. There is no
layout model behind it : no TOC, no image_df, no object_registry. The dict
keeps the same key set as `parse_pdf` so downstream code can drop in EasyOCR
without conditional handling — every absent table is just empty.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib.parsing.pdf.easyocr.line_df import (
    PARSING_METHOD,
    easyocr_pdf_to_line_df,
)

_EMPTY_TABLES = ("page_df", "image_df", "toc_df", "span_df",
                 "object_registry", "cross_ref_df")


def parse_pdf_easyocr(
    pdf_path: str | Path,
    *,
    languages: tuple[str, ...] = ("en",),
    render_scale: float = 2.0,
    gpu: bool = False,
    confidence_threshold: float = 0.0,
) -> dict:
    """Run EasyOCR on `pdf_path` and return a dict with the same shape
    as `parse_pdf` (the canonical fitz orchestrator).

    Filled :
        line_df          - one row per detected text box
        parsing_summary  - method, page count, line count

    Empty (returned as DataFrames with the right columns) :
        page_df, image_df, toc_df, span_df, object_registry, cross_ref_df

    Downstream consumers (retrieval, generation, annotation) that need only
    `line_df` work as-is. Consumers that need TOC or image lookups have to
    enrich the result with a layout-aware pass (Article 5ter / Article 5bis).
    """
    line_df = easyocr_pdf_to_line_df(
        pdf_path,
        languages=languages,
        render_scale=render_scale,
        gpu=gpu,
        confidence_threshold=confidence_threshold,
    )
    n_pages = int(line_df["page_num"].max()) if len(line_df) else 0
    parsing_summary = {
        "pdf_path": str(pdf_path),
        "parsing_method": PARSING_METHOD,
        "n_pages": n_pages,
        "n_lines": int(len(line_df)),
        "languages": list(languages),
        "render_scale": render_scale,
    }
    out: dict = {"line_df": line_df, "parsing_summary": parsing_summary}
    for k in _EMPTY_TABLES:
        out[k] = pd.DataFrame()
    return out
