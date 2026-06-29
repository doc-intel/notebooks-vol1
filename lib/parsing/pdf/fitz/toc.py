"""Build a TOC DataFrame from a PDF's native outline.

`toc_df` materializes one row per TOC entry. Two columns carry the page span:

  - `start_page` : where the section begins. The PDF's native outline gives this
                   directly (`doc.get_toc()` triples are `[level, title, page]`).
  - `end_page`   : where the section ends. The PDF NEVER writes this — TOCs
                   declare beginnings, not endings. We compute it: for each row,
                   `end_page` is the `start_page` of the next row at the same
                   level or shallower (i.e. the next peer or ancestor), or
                   `doc.page_count` for the last section. The convention keeps a
                   one-page overlap by design so Article 8's next-page peek
                   (`peek_next_section_change`) is a single lookup.

`start_y` carries the destination y-coordinate that the PDF outline ships with
the entry (the `to: Point(x, y)` of each bookmark). Same coordinate-orientation
caveat as `Page.get_links()`: LaTeX-generated PDFs return bottom-up y, viewer
tools return top-down y. Stored verbatim ; callers join against `line_df` with
a normalizer when they need the precise landing line.
"""
from __future__ import annotations

import fitz
import pandas as pd


TOC_COLUMNS = [
    "toc_idx", "level", "parent_idx", "title",
    "start_page", "end_page", "start_y", "breadcrumb",
]


def build_toc_df(doc: fitz.Document) -> pd.DataFrame:
    """Extract the native outline as a DataFrame.

    Returns columns: toc_idx, level, parent_idx, title, start_page, end_page,
    start_y, breadcrumb. Empty DataFrame if the document has no native TOC.
    """
    raw = doc.get_toc(simple=False) or []
    if not raw:
        return pd.DataFrame(columns=TOC_COLUMNS)
    total_pages = doc.page_count
    rows = []
    stack: list[tuple[int, int]] = []  # (level, idx)
    breadcrumbs: list[str] = []
    for idx, entry in enumerate(raw):
        level, title, page_num, dest = entry[0], entry[1], entry[2], entry[3]
        while stack and stack[-1][0] >= level:
            stack.pop()
            if breadcrumbs:
                breadcrumbs.pop()
        parent_idx = stack[-1][1] if stack else None
        breadcrumb = " > ".join(breadcrumbs + [title])
        to_pt = dest.get("to") if isinstance(dest, dict) else None
        start_y = float(to_pt.y) if to_pt is not None else None
        rows.append({
            "toc_idx": idx,
            "level": level,
            "parent_idx": parent_idx,
            "title": title,
            "start_page": page_num,
            "end_page": None,        # filled in the lookback pass below
            "start_y": start_y,
            "breadcrumb": breadcrumb,
        })
        stack.append((level, idx))
        breadcrumbs.append(title)

    # Lookback pass: for each row, end_page = start_page of the next entry at
    # the same level or shallower (peer or ancestor). Fallback = total_pages.
    for i, row in enumerate(rows):
        end = total_pages
        for j in range(i + 1, len(rows)):
            if rows[j]["level"] <= row["level"]:
                end = rows[j]["start_page"]
                break
        row["end_page"] = end

    return pd.DataFrame(rows, columns=TOC_COLUMNS)
