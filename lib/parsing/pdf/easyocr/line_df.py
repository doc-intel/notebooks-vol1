"""EasyOCR -> `line_df`.

The pipeline per page :

  1. Render the PDF page to a raster image at `render_scale` (default 2.0 =
     ~144 DPI). Higher scale = better OCR on small fonts, more memory + time.
  2. Pass the image to `easyocr.Reader.readtext()`. EasyOCR returns one row
     per detected text box : `(quad, text, confidence)` where `quad` is the
     four corners of the polygon around the text.
  3. Convert the polygon to an axis-aligned bbox in PDF coordinates (divide
     pixel coords by `render_scale`).
  4. Sort top-to-bottom (by y0) within each page and assign `line_num`.

Every row carries `parsing_method="easyocr"` so mixed-engine pipelines can
filter or de-duplicate by provenance, matching the convention of the other
parser subpackages.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
import pandas as pd

from lib.parsing.pdf.easyocr._client import get_easyocr_reader

PARSING_METHOD = "easyocr"

_LINE_COLS = [
    "page_num", "line_num", "text",
    "x0", "y0", "x1", "y1",
    "character_count", "confidence", "parsing_method",
]


def _quad_to_bbox(quad: list[list[float]]) -> tuple[float, float, float, float]:
    """Tight axis-aligned bbox enclosing the four EasyOCR polygon corners."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def easyocr_pdf_to_line_df(
    pdf_path: str | Path,
    *,
    languages: tuple[str, ...] = ("en",),
    render_scale: float = 2.0,
    gpu: bool = False,
    confidence_threshold: float = 0.0,
) -> pd.DataFrame:
    """Run EasyOCR on every page of a PDF and return a `line_df`.

    Each detected text box becomes one row. `line_num` resets per page and
    follows top-to-bottom reading order based on the box's top y-coordinate.
    Coordinates are converted back to PDF units (the bbox you'd see in a PDF
    viewer at 100% zoom), not pixel units.

    Pass `confidence_threshold > 0` to drop low-confidence detections (useful
    on heavily degraded scans). The raw confidence is kept on every row that
    survives the filter.
    """
    reader = get_easyocr_reader(languages=tuple(languages), gpu=gpu)
    doc = fitz.open(str(pdf_path))
    try:
        rows: list[dict] = []
        for page_idx, page in enumerate(doc, start=1):
            # Render the page to a pixmap, then a numpy array EasyOCR can read.
            matrix = fitz.Matrix(render_scale, render_scale)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            detections = reader.readtext(img)
            page_rows = []
            for quad, text, conf in detections:
                if conf < confidence_threshold:
                    continue
                px0, py0, px1, py1 = _quad_to_bbox(quad)
                page_rows.append({
                    "page_num": page_idx,
                    "text": text,
                    "x0": px0 / render_scale,
                    "y0": py0 / render_scale,
                    "x1": px1 / render_scale,
                    "y1": py1 / render_scale,
                    "character_count": len(text),
                    "confidence": float(conf),
                    "parsing_method": PARSING_METHOD,
                })
            # Sort top-to-bottom within the page, then left-to-right as tie-break.
            page_rows.sort(key=lambda r: (round(r["y0"], 1), r["x0"]))
            for line_idx, row in enumerate(page_rows, start=1):
                row["line_num"] = line_idx
                rows.append(row)
    finally:
        doc.close()

    if not rows:
        return pd.DataFrame(columns=_LINE_COLS)
    return pd.DataFrame(rows)[_LINE_COLS]
