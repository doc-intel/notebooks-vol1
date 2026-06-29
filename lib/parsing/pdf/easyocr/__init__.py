"""EasyOCR parser — basic OCR for scanned PDFs.

The simplest OCR-based parser in the series. Renders each PDF page to a raster
image (via PyMuPDF), runs EasyOCR's pre-trained text detector + recogniser on
each image, and returns the detected text boxes as `line_df` rows.

What EasyOCR gives us:
  - Per-page text boxes with bounding boxes and per-box confidence.
  - Works on scanned PDFs that fitz returns empty on (no embedded text layer).
  - Runs locally, no API key, no per-page billing.

What EasyOCR does NOT give us:
  - No TOC (no PDF metadata reach, no heading detection).
  - No image_df (sees the whole page as one image ; individual embedded
    figures inside a scanned page cannot be isolated without a layout model).
  - No object_registry, no cross-references, no font / span info.
  - No reading-order beyond top-to-bottom y-coordinate sort.

That makes the public surface intentionally small : `line_df` only.

Public surface:

    parse_pdf_easyocr(pdf_path) -> dict        # 1-key dict {'line_df': ...}
    easyocr_pdf_to_line_df(pdf_path) -> DataFrame
    get_easyocr_reader(languages=...) -> easyocr.Reader

Layout (one file per builder, mirror of `lib.parsing.pdf.fitz`):

    _client.py              - get_easyocr_reader (lazy singleton)
    line_df.py              - easyocr_pdf_to_line_df (per-page render + OCR + concat)
    parse_pdf_easyocr.py    - orchestrator returning the 1-key dict
"""
from lib.parsing.pdf.easyocr._client import get_easyocr_reader
from lib.parsing.pdf.easyocr.line_df import easyocr_pdf_to_line_df
from lib.parsing.pdf.easyocr.parse_pdf_easyocr import parse_pdf_easyocr

__all__ = [
    "parse_pdf_easyocr",
    "easyocr_pdf_to_line_df",
    "get_easyocr_reader",
]
