"""line_df: extract every text line of a PDF with its bounding box."""
from __future__ import annotations
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

# Canonical column order for line_df. Used to build an empty-but-typed frame
# when a PDF has no extractable text layer (a scanned page), so downstream
# code can read line_df["text"] without a KeyError on the zero-row case.
LINE_DF_COLUMNS = [
    "page_num", "line_num", "text", "x0", "y0", "x1", "y1", "character_count",
]


def fitz_pdf_to_line_df(
    file_path: str | Path,
    *,
    pages: list[int] | tuple[int, ...] | int | str | None = None,
) -> pd.DataFrame:
    """Extract text line by line from each page of a PDF.

    Returns a DataFrame with columns: page_num, line_num, text, x0, y0, x1, y1,
    character_count.

    ``pages`` restricts the work to a 1-based subset (``[6, 9]``, ``6``,
    ``"3-5,8"``). ``None`` (default) parses every page. Pages outside the
    PDF's range are silently ignored. The expensive ``page.get_text("dict")``
    call is skipped for excluded pages — real CPU saving on large documents.
    """
    from lib.parsing.pdf._shared import normalize_pages

    page_filter = normalize_pages(pages)
    doc = fitz.open(str(file_path))
    rows = []
    for page_num in range(len(doc)):
        if page_filter is not None and (page_num + 1) not in page_filter:
            continue
        page = doc[page_num]
        blocks = page.get_text("dict").get("blocks", [])
        line_num = 0
        for block in blocks:
            if block.get("type") != 0:  # skip image/non-text blocks
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                span_text = "".join(s.get("text", "") for s in spans)
                if not span_text.strip():
                    continue
                rect = fitz.Rect(spans[0]["bbox"])
                for span in spans[1:]:
                    rect |= fitz.Rect(span["bbox"])
                rows.append({
                    "page_num": page_num + 1,
                    "line_num": line_num + 1,
                    "text": span_text,
                    "x0": float(rect.x0),
                    "y0": float(rect.y0),
                    "x1": float(rect.x1),
                    "y1": float(rect.y1),
                })
                line_num += 1
    df = pd.DataFrame(rows, columns=LINE_DF_COLUMNS)
    if not df.empty:
        # Invariant: line_df.text is always a non-null string. The parser logic
        # above already skips empty spans, but a parquet round-trip can convert
        # empty strings to NaN (pyarrow + pandas combo). We re-establish the
        # invariant here so every consumer downstream (retrieval, generation,
        # FTS) can call .astype(str) / .str.* / join() without surprise.
        df["text"] = df["text"].fillna("").astype(str)
        df["character_count"] = df["text"].str.len()
    return df
