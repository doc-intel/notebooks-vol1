"""Engine-agnostic helpers for `parsing.pdf`.

Everything here reads only `line_df` (the canonical per-line table any engine
produces) and emits the same shape regardless of which parser produced the
input. Both `fitz/` and `azure_layout/` reuse these directly, so the helpers
sit at the engine-neutral level instead of being duplicated.

Contents :
    - `OBJECT_PATTERNS` : anchored regex for figure / table / annex caption
      detection. Used by `fitz/objects.build_object_registry` and by
      `azure_layout/objects.build_object_registry_azure_layout` to extract
      the `(object_type, object_id)` join key from the caption text.
    - `REFERENCE_PATTERNS` : unanchored regex for body-text mentions of the
      same objects ("see Figure 2", "refer to Annex B"). Used by
      `build_cross_ref_df`.
    - `build_page_df(line_df)` : aggregate to one row per page. Engine-agnostic.
    - `build_cross_ref_df(line_df)` : extract every body-text mention of a
      named object. Engine-agnostic.

Both `build_page_df` and `build_cross_ref_df` were previously hosted under
`fitz/` and called cross-engine from `azure_layout/parse_pdf_azure_layout.py`.
Hoisting them here removes that engine-to-engine coupling.
"""
from __future__ import annotations

import re

import pandas as pd


def normalize_pages(pages: list[int] | tuple[int, ...] | str | None) -> set[int] | None:
    """Coerce a page-subset argument to a ``set[int] | None`` (1-based).

    Engines accept ``pages=`` in several shapes for ergonomic reasons :

      - ``None``                  → parse all pages (the default everywhere)
      - ``list[int]`` / ``tuple`` → explicit page numbers, e.g. ``[6, 9]``
      - ``int``                   → single page, ``6`` is shorthand for ``[6]``
      - ``str``                   → range form, e.g. ``"3-5,8"`` (Azure-style)

    Page numbers are **1-based** to match the convention used in every
    output DataFrame (``line_df.page_num``, ``image_df.page_num``, ...).

    Returns ``None`` when the input is ``None``, otherwise a ``set[int]`` the
    engine can membership-test in a loop. Each engine then chooses what
    "honour pages" means for its cost model :

      - fitz    : skip ``page.get_text("dict")`` for excluded pages (real CPU
                  saving on large PDFs)
      - azure   : pass the page subset to the API (real billing saving)
      - docling : full conversion, post-filter the per-row tables (NO CPU
                  saving — docling pays for the whole document either way)
    """
    if pages is None:
        return None
    if isinstance(pages, int):
        return {pages}
    if isinstance(pages, str):
        out: set[int] = set()
        for chunk in pages.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                lo, hi = chunk.split("-", 1)
                out.update(range(int(lo), int(hi) + 1))
            else:
                out.add(int(chunk))
        return out
    # list / tuple / set / any other iterable of ints
    return {int(p) for p in pages}


def filter_df_by_pages(df: pd.DataFrame, pages: set[int] | None) -> pd.DataFrame:
    """Drop rows whose ``page_num`` is outside ``pages``. Identity when None.

    Used by every engine's orchestrator to keep image_df / line_df / page_df
    consistent with the requested subset. Returns ``df`` unchanged when
    ``pages`` is None or the DataFrame is empty / has no ``page_num`` column.
    """
    if pages is None or df.empty or "page_num" not in df.columns:
        return df
    return df[df["page_num"].isin(pages)].reset_index(drop=True)


OBJECT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*(?:Figure|Fig\.?)\s+(\d+)\b", re.IGNORECASE), "figure"),
    (re.compile(r"^\s*Table\s+(\d+)\b", re.IGNORECASE), "table"),
    (re.compile(r"^\s*(?:Annex|Appendix)\s+([A-Z0-9]+)\b", re.IGNORECASE), "annex"),
]

REFERENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:Figure|Fig\.?)\s+(\d+)\b", re.IGNORECASE), "figure"),
    (re.compile(r"\bTable\s+(\d+)\b", re.IGNORECASE), "table"),
    (re.compile(r"\b(?:Annex|Appendix)\s+([A-Z0-9]+)\b", re.IGNORECASE), "annex"),
]


def build_page_df(line_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line_df into one row per page with full page text concatenated."""
    return (
        line_df.groupby("page_num", as_index=False)
        .agg(text=("text", lambda s: "\n".join(s)))
        .sort_values("page_num")
        .reset_index(drop=True)
    )


def build_cross_ref_df(line_df: pd.DataFrame) -> pd.DataFrame:
    """Extract cross-reference SOURCES from body text.

    Returns columns: ref_type, ref_id, page_num, line_num, context.

    The `context` column carries ~30 characters of surrounding text so a reader
    can verify the mention without re-opening the PDF. The same target may
    appear multiple times: Figure 2 referenced on pages 3, 5, and 7 produces
    three rows with the same (ref_type, ref_id) and different (page_num,
    line_num) — that is the point.

    Caption lines (the ones that satisfy the anchored `OBJECT_PATTERNS`) are
    excluded so we don't double-count the line that defines the target with one
    that mentions it.
    """
    if line_df.empty:
        return pd.DataFrame(columns=["ref_type", "ref_id", "page_num", "line_num", "context"])

    caption_keys: set[tuple[int, int]] = set()
    for _, line in line_df.iterrows():
        text = str(line.get("text", ""))
        for pattern, _ in OBJECT_PATTERNS:
            if pattern.match(text):
                caption_keys.add((int(line["page_num"]), int(line["line_num"])))
                break

    rows = []
    for _, line in line_df.iterrows():
        key = (int(line["page_num"]), int(line["line_num"]))
        if key in caption_keys:
            continue
        text = str(line.get("text", ""))
        for pattern, ref_type in REFERENCE_PATTERNS:
            for m in pattern.finditer(text):
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 30)
                context = text[start:end].strip()
                rows.append({
                    "ref_type": ref_type,
                    "ref_id": m.group(1),
                    "page_num": key[0],
                    "line_num": key[1],
                    "context": context,
                })
    return pd.DataFrame(rows)
