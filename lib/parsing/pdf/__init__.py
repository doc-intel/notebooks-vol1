"""PDF parsing — Brick 1 for native PDF documents.

Engine-explicit by design : the per-engine details live under
`parsing.pdf.fitz` and `parsing.pdf.azure_layout`. This top-level module
exposes only the **engine-agnostic surface** plus the dispatcher and the
two engine-pinned full parsers.

Public surface :

    parse_pdf(pdf_path, method=...)        # dispatcher, routes by method
    parse_pdf_fitz(pdf_path)               # engine-pinned full parser
    parse_pdf_azure_layout(pdf_path)       # engine-pinned full parser (lazy import)
    fitz_pdf_to_line_df(pdf_path)          # Article 1 minimal entry (line_df only)

    # Engine-agnostic builders (read line_df only).
    build_page_df(line_df)                 # one row per page
    build_cross_ref_df(line_df)            # body-text mentions of figures/tables/annexes
    OBJECT_PATTERNS, REFERENCE_PATTERNS    # shared regex tables

    # Engine-agnostic profile probe.
    DocumentProfile
    detect_document_type(pdf_path)

For everything else (fitz's `build_toc_df`, `build_span_df`,
`assign_column_positions`, `detect_source_software`, `build_object_registry`,
…), import engine-explicit :

    from lib.parsing.pdf.fitz import build_toc_df, build_span_df, …
    from lib.parsing.pdf.azure_layout import build_toc_df_azure_layout, …

The format is part of the contract : `parsing/__init__.py` does not re-export,
and `parsing/pdf/__init__.py` does not re-export per-engine helpers either.
"""
from __future__ import annotations

from pathlib import Path

# Dispatcher and engine-pinned full parsers.
from lib.parsing.pdf.fitz.parse_pdf import parse_pdf_fitz

# Article 1 minimal entry — line_df only. Engine: fitz. Kept at the top level
# because Article 1's pedagogical narrative imports it as the simplest possible
# parsing surface ; engine-explicit form `from lib.parsing.pdf.fitz import
# fitz_pdf_to_line_df` is equivalent.
from lib.parsing.pdf.fitz import fitz_pdf_to_line_df

# Engine-agnostic helpers (read line_df only).
from lib.parsing.pdf._shared import (
    OBJECT_PATTERNS,
    REFERENCE_PATTERNS,
    build_cross_ref_df,
    build_page_df,
)

# Engine-agnostic profile probe.
from lib.parsing.pdf.profile import (
    DocumentProfile,
    detect_document_type,
)

# Adaptive parsing — Article 10's targeted re-parse helpers.
from lib.parsing.pdf.enrich import (
    enrich,
    get_or_enrich,
)

# Semantic-zone fields for parsing_summary — doc_type + typical_fields +
# summary, merged into the parsing_summary dict the engine builds. The
# question parser then pulls the semantic subset out of parsing_summary to
# scope ambiguous questions ("what is the name?" on a CV) without an
# operator-side hint.
from lib.parsing.pdf.doc_context import (
    build_semantic_fields,
    build_summary,
)


def parse_pdf(
    pdf_path: str | Path,
    *,
    method: str = "fitz",
    client=None,
    pages: list[int] | tuple[int, ...] | int | str | None = None,
    summarize: bool = True,
    summary_client=None,
    summary_model: str | None = None,
    summary_prompt: str | None = None,
    summary_project_id: str | None = None,
) -> dict:
    """Top-level parser dispatcher. Returns the canonical dict of tables.

    Routes to a `parse_pdf_<engine>` sub-function based on ``method``.
    Every per-row table in the returned dict carries a `parsing_method`
    column = ``method``. New engines plug in by adding a key here.

    Parameters
    ----------
    pdf_path : path to the PDF.
    method   : engine name. Today: ``"fitz"`` (default), ``"azure_layout"``,
               ``"docling"``.
    client   : forwarded to engines that need it (Azure). Ignored by fitz and
               docling.
    pages    : 1-based page subset (``[6, 9]``, ``6``, ``"3-5,8"``, or ``None``
               for the whole document). Honoured by every engine, but the
               cost model differs :

                 - ``fitz``         : skips ``page.get_text("dict")`` for
                   excluded pages (real CPU saving on large PDFs).
                 - ``azure_layout`` : passes the subset to the Azure API
                   (real billing saving).
                 - ``docling``      : converts the whole document, then post-
                   filters the per-row tables (NO compute saving — use fitz
                   if you need page-level cost control).

               ``toc_df`` and ``object_registry`` are kept whole regardless of
               ``pages`` because they describe document structure.
    summarize : when True (default), fill ``parsing_summary["summary"]`` with a
               short LLM-written document blurb (one cached call against the
               first pages plus the TOC). Degrades to None offline or on error.
               Pass False to skip the LLM call entirely.
    summary_client / summary_model : chat LLM client + model for the summary
               call. Both default to the wrapper's env resolution. These are
               NOT the engine ``client`` (the Azure layout client).
    summary_prompt / summary_project_id : override the summary system prompt
               (``document_summary``) for this call, or resolve the project
               override. ``None`` → catalogue default. The system prompt is
               always a parameter, never hard-coded.

    Returns
    -------
    dict with these keys: ``line_df, page_df, image_df, toc_df, span_df,
    object_registry, cross_ref_df, parsing_summary``.
    """
    summary_kw = {
        "summarize": summarize,
        "summary_client": summary_client,
        "summary_model": summary_model,
        "summary_prompt": summary_prompt,
        "summary_project_id": summary_project_id,
    }
    if method == "fitz":
        return parse_pdf_fitz(pdf_path, pages=pages, **summary_kw)
    if method == "azure_layout":
        # Imported lazily so callers that never use Azure don't trip on the
        # azure-sdk dependency at package import time.
        from lib.parsing.pdf.azure_layout import parse_pdf_azure_layout
        return parse_pdf_azure_layout(pdf_path, client=client, pages=pages, **summary_kw)
    if method == "docling":
        # Imported lazily so callers that never use Docling don't pay the
        # heavy docling/torch import at package load time.
        from lib.parsing.pdf.docling import parse_pdf_docling
        return parse_pdf_docling(pdf_path, pages=pages, **summary_kw)
    raise ValueError(
        f"Unknown parse_pdf method: {method!r}. "
        "Known engines: 'fitz', 'azure_layout', 'docling'."
    )


__all__ = [
    # Dispatcher + engine-pinned full parsers
    "parse_pdf",
    "parse_pdf_fitz",
    # Article 1 minimal entry
    "fitz_pdf_to_line_df",
    # Engine-agnostic builders + regex tables
    "build_page_df",
    "build_cross_ref_df",
    "OBJECT_PATTERNS",
    "REFERENCE_PATTERNS",
    # Engine-agnostic profile probe
    "DocumentProfile",
    "detect_document_type",
    # Adaptive parsing (Article 10)
    "enrich",
    "get_or_enrich",
    # Semantic-zone builder for parsing_summary
    "build_semantic_fields",
    "build_summary",
]
