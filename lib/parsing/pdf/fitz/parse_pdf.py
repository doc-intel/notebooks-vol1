"""Unified entry point: one PDF in, a dictionary of relational tables out."""
from __future__ import annotations

from pathlib import Path

import fitz
import pandas as pd

from lib.parsing.pdf._shared import normalize_pages
from lib.parsing.pdf.doc_context import build_semantic_fields, build_summary
from lib.parsing.pdf.fitz.columns import (
    assign_column_positions,
    detect_columns_per_page,
)
from lib.parsing.pdf.fitz.line_df import fitz_pdf_to_line_df
from lib.parsing.pdf.fitz.native_metadata import read_native_metadata
from lib.parsing.pdf.fitz.objects import build_object_registry
from lib.parsing.pdf.fitz.page_df import build_page_df
from lib.parsing.pdf.fitz.references import build_cross_ref_df
from lib.parsing.pdf.fitz.source import detect_source_software
from lib.parsing.pdf.fitz.toc import build_toc_df


IMAGE_DF_COLUMNS = [
    "page_num", "image_num", "x0", "y0", "x1", "y1",
    "width_px", "height_px", "image_hash",
]


def _build_image_df(doc: fitz.Document, page_filter: set[int] | None = None) -> pd.DataFrame:
    rows = []
    for page_num in range(len(doc)):
        if page_filter is not None and (page_num + 1) not in page_filter:
            continue
        page = doc[page_num]
        for image_num, info in enumerate(page.get_image_info(), start=1):
            bbox = info.get("bbox", (0, 0, 0, 0))
            rows.append({
                "page_num": page_num + 1,
                "image_num": image_num,
                "x0": float(bbox[0]),
                "y0": float(bbox[1]),
                "x1": float(bbox[2]),
                "y1": float(bbox[3]),
                "width_px": int(info.get("width", 0)),
                "height_px": int(info.get("height", 0)),
                "image_hash": info.get("digest", b"").hex() if info.get("digest") else "",
            })
    if not rows:
        return pd.DataFrame(columns=IMAGE_DF_COLUMNS)
    return pd.DataFrame(rows, columns=IMAGE_DF_COLUMNS)


def _build_parsing_summary(
    pdf_path: Path,
    doc: fitz.Document,
    line_df: pd.DataFrame,
    image_df: pd.DataFrame,
    toc_df: pd.DataFrame,
    object_registry: pd.DataFrame,
    cross_ref_df: pd.DataFrame,
) -> dict:
    meta = doc.metadata or {}
    return {
        "pdf_path": str(pdf_path),
        "n_pages": len(doc),
        "pdf_version": getattr(doc, "pdf_version", lambda: None)() if callable(getattr(doc, "pdf_version", None)) else None,
        "source_software": detect_source_software(doc),
        "creator_raw": meta.get("creator") or "",
        "producer_raw": meta.get("producer") or "",
        "native_metadata": read_native_metadata(doc),
        "n_lines": int(len(line_df)),
        "n_images": int(len(image_df)),
        "n_toc_entries": int(len(toc_df)),
        "n_named_objects": int(len(object_registry)),
        "n_cross_refs": int(len(cross_ref_df)),
        "is_encrypted": bool(doc.is_encrypted),
        "needs_pass": bool(doc.needs_pass),
        "has_toc": bool(len(toc_df) > 0),
        # A page-bearing PDF with zero extracted text lines is image-only
        # (a scan with no text layer). Downstream (the V6 app) reads this flag
        # to offer OCR via azure_layout / docling instead of showing an error.
        "is_image_only": bool(len(line_df) == 0 and len(doc) > 0),
    }


def parse_pdf_fitz(
    pdf_path: str | Path,
    *,
    pages: list[int] | tuple[int, ...] | int | str | None = None,
    summarize: bool = True,
    summary_client=None,
    summary_model: str | None = None,
    summary_prompt: str | None = None,
    summary_project_id: str | None = None,
) -> dict:
    """Parse a PDF into a small relational database of DataFrames via fitz.

    Returns a dictionary with the following keys:
      - line_df:         one row = one text line (with column_position)
      - image_df:        one row = one embedded image
      - page_df:         one row = one page
      - span_df:         (placeholder) one row = one typographic span
      - toc_df:          one row = one TOC entry
      - object_registry: one row = one named object (figure / table / annex)
      - cross_ref_df:    one row = one body-text mention of a named object
      - parsing_summary: doc-level technical synthesis (dict)

    The PDF is opened once with fitz, every helper consumes the resulting
    state, and the document is closed before returning.

    ``pages`` restricts the parse to a 1-based subset (``[6, 9]``, ``6``,
    ``"3-5,8"``, or ``None`` for the whole document). Excluded pages are
    skipped at the fitz layer for line_df + image_df (real CPU saving).
    The ``toc_df`` is NOT filtered — TOC entries describe the document
    structure regardless of which pages were parsed ; ``parsing_summary
    .n_pages`` still reflects the full document. The caller filters TOC
    further if needed.

    This is the fitz-pinned sub-function. The top-level `parse_pdf`
    dispatcher (in `lib.parsing.pdf`) calls this when `method="fitz"`.
    """
    pdf_path = Path(pdf_path)
    page_filter = normalize_pages(pages)
    line_df = fitz_pdf_to_line_df(pdf_path, pages=pages)
    line_df = assign_column_positions(line_df)
    page_df = build_page_df(line_df) if not line_df.empty else pd.DataFrame()
    object_registry = build_object_registry(line_df) if not line_df.empty else pd.DataFrame()
    cross_ref_df = build_cross_ref_df(line_df) if not line_df.empty else pd.DataFrame()

    with fitz.open(str(pdf_path)) as doc:
        image_df = _build_image_df(doc, page_filter=page_filter)
        toc_df = build_toc_df(doc)
        n_columns_df = detect_columns_per_page(line_df) if not line_df.empty else pd.DataFrame()
        parsing_summary = _build_parsing_summary(
            pdf_path, doc, line_df, image_df, toc_df, object_registry, cross_ref_df,
        )

    if not page_df.empty and not n_columns_df.empty:
        page_df = page_df.merge(n_columns_df, on="page_num", how="left")

    # Provenance column on every per-row table. Symmetric with parse_pdf_azure_layout
    # so adaptive-parsing pipelines that mix engines on the same document can
    # filter by parsing_method or compare row-by-row across engines.
    for df in (line_df, image_df, page_df, toc_df, object_registry, cross_ref_df):
        if not df.empty:
            df["parsing_method"] = "fitz"
    parsing_summary["parsing_method"] = "fitz"

    # Semantic-zone fields for parsing_summary : doc_type / typical_fields /
    # summary. Heuristic-only at this stage ; ``summary`` stays None until a
    # summary builder lands. The question parser later pulls this subset
    # out of parsing_summary to scope ambiguous wording. Merging into the
    # existing dict keeps the doc-level synthesis a single artefact.
    semantic = build_semantic_fields(line_df, page_df, pdf_path=pdf_path)
    parsing_summary.update(semantic)

    # The one LLM-derived field of the semantic zone : a short factual blurb
    # built from the first pages (plus the TOC when present). Run-once per
    # document and cached ; degrades to None offline or on error. On by
    # default so the question parser always has a document context to read.
    if summarize:
        parsing_summary["summary"] = build_summary(
            line_df, toc_df=toc_df, client=summary_client, model=summary_model,
            prompt=summary_prompt, project_id=summary_project_id,
        )

    return {
        "line_df": line_df,
        "image_df": image_df,
        "page_df": page_df,
        "span_df": pd.DataFrame(),
        "toc_df": toc_df,
        "object_registry": object_registry,
        "cross_ref_df": cross_ref_df,
        "parsing_summary": parsing_summary,
    }
