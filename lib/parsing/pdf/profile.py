"""Document profiling for the parsing brick — Article 10.

`detect_document_type(pdf_path)` is the cheap probe the orchestrator
(Article 14) runs before deciding which retrieval / parsing patterns to
activate. It opens the PDF, reads a few signals (page count, TOC presence,
first-page text density), and returns a `DocumentProfile`.

Cheap means: one fitz.Document open, no parsing of the body. Adaptive
parsing (Article 10's full pipeline) is what runs *after* if the profile
flags a strategy beyond native_text.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class DocumentProfile:
    """Cheap profile read by the dispatcher (Article 14).

    Fields:
      total_pages: PyMuPDF page count.
      has_usable_toc: True when the PDF carries a non-empty TOC. The
        dispatcher uses this to enable TOC retrieval.
      is_likely_scanned: True when the first page has almost no extractable
        text. The dispatcher activates adaptive parsing (OCR path) when this
        fires.
      suggested_strategy: a coarse routing label. Values used by the
        dispatcher:
          'native_text_only'           — body is plain text, cheap parse OK
          'native_with_table_extraction' — body has tables/figures that may
                                            need richer parsing on demand
          'scan_with_ocr'              — needs OCR before any retrieval
    """
    total_pages: int
    has_usable_toc: bool
    is_likely_scanned: bool
    suggested_strategy: str


# A page below this character count is treated as scanned (no extractable
# text). Native PDFs hit ~1000+ chars per body page; scans return ~0.
_SCAN_TEXT_THRESHOLD_CHARS = 100

# A TOC of 4+ entries is treated as usable. Some PDFs ship a trivial
# 1-entry TOC ("Title") that isn't useful for section retrieval.
_MIN_USABLE_TOC_ENTRIES = 4


def detect_document_type(pdf_path: str | Path) -> DocumentProfile:
    """Open `pdf_path`, read the cheap signals, return a `DocumentProfile`.

    Used by the orchestrator (Article 14) as step 3 of the composite
    pipeline. Does not parse the body; that's Article 10's job, gated on
    this profile.
    """
    doc = fitz.open(str(pdf_path))
    try:
        total_pages = doc.page_count
        toc = doc.get_toc() or []
        has_usable_toc = len(toc) >= _MIN_USABLE_TOC_ENTRIES

        # Scan detection: read the first page's plain text. If it's almost
        # empty, the document is likely an image-only scan needing OCR.
        first_page_text = doc[0].get_text("text") if total_pages else ""
        is_likely_scanned = len(first_page_text.strip()) < _SCAN_TEXT_THRESHOLD_CHARS

        if is_likely_scanned:
            strategy = "scan_with_ocr"
        elif _looks_table_heavy(doc):
            strategy = "native_with_table_extraction"
        else:
            strategy = "native_text_only"
    finally:
        doc.close()

    return DocumentProfile(
        total_pages=total_pages,
        has_usable_toc=has_usable_toc,
        is_likely_scanned=is_likely_scanned,
        suggested_strategy=strategy,
    )


def _looks_table_heavy(doc) -> bool:
    """Heuristic: does the document carry table-like structures?

    Probes the first 5 pages via `page.find_tables()`. Returns True if any
    of them surfaces a non-empty table. This catches research papers
    (Transformer Table 3), standards (NIST Table 1), contracts with
    schedule grids; misses tables rendered as bitmaps.
    """
    sample = min(5, doc.page_count)
    for i in range(sample):
        try:
            tables = doc[i].find_tables()
            if list(tables):
                return True
        except Exception:
            # `find_tables` is recent in PyMuPDF; old versions raise.
            return False
    return False


__all__ = ["DocumentProfile", "detect_document_type"]
