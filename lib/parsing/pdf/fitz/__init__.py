"""Fitz (PyMuPDF) parser — the cheap, fast, native-PDF path.

Produces the full set of relational tables used by the rest of the pipeline:
line_df, page_df, span_df, toc_df, columns, object registry, cross-references,
and the source-software classifier. Best on born-digital PDFs; falls short on
scans (no OCR) and complex tables (column structure flattened — that's where
azure/, camelot/, docling/ take over).
"""
from lib.parsing.pdf.fitz.line_df import fitz_pdf_to_line_df
from lib.parsing.pdf.fitz.page_df import build_page_df
from lib.parsing.pdf.fitz.source import detect_source_software
from lib.parsing.pdf.fitz.columns import (
    assign_column_positions,
    detect_columns_per_page,
    detect_n_columns,
)
from lib.parsing.pdf.fitz.toc import build_toc_df
from lib.parsing.pdf.fitz.objects import build_object_registry
from lib.parsing.pdf.fitz.references import build_cross_ref_df
from lib.parsing.pdf.fitz.span_df import build_span_df
from lib.parsing.pdf.fitz.parse_pdf import parse_pdf_fitz
from lib.parsing.pdf.fitz.quality_signals import (
    pre_parse_signals,
    page_level_parsing_signals,
)

__all__ = [
    "fitz_pdf_to_line_df",
    "build_page_df",
    "detect_source_software",
    "assign_column_positions",
    "detect_columns_per_page",
    "detect_n_columns",
    "build_toc_df",
    "build_object_registry",
    "build_cross_ref_df",
    "build_span_df",
    "parse_pdf_fitz",
    "pre_parse_signals",
    "page_level_parsing_signals",
]
