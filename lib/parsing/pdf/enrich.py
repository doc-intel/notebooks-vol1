"""Adaptive PDF parsing — Article 10's targeted re-parse helpers.

The cheap-first cascade flags pages where `fitz` is not enough (flat
tables, opaque figures, scrambled OCR). When a flag fires, the pipeline
calls `enrich(pdf_path, line_df, page_df, pages, method)` to re-parse
just those pages with a deeper parser (`azure_layout` for tables,
`vision_gpt4o` for figures, …) and **append** the new rows to the
existing `line_df` / `page_df`. The original rows stay — the audit trail
is the point.

`get_or_enrich` adds a cache layer : if `page_df` already shows a row for
`(page_num, parsing_method)`, skip that page. The data model IS the cache.

Both functions return ``(line_df, page_df)`` as a tuple so callers can
chain enrichments without juggling a dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _enrich_one_method(
    pdf_path: str | Path,
    pages: list[int],
    method: str,
) -> pd.DataFrame:
    """Produce new `line_df` rows for ``pages`` using ``method``."""
    if method == "azure_layout":
        from lib.parsing.pdf.azure_layout import azure_layout_pdf_to_line_df
        new_lines = azure_layout_pdf_to_line_df(pdf_path, pages=pages)
    elif method == "docling":
        # Docling converts the whole document then filters the returned
        # tables to ``pages`` ; no per-page CPU saving but the audit-trail
        # semantics of `enrich` are identical (the new rows come pre-tagged
        # with parsing_method="docling" by the builders).
        from lib.parsing.pdf.docling.parse_pdf_docling import parse_pdf_docling
        parsed = parse_pdf_docling(pdf_path, pages=pages)
        new_lines = parsed["line_df"]
    elif method == "fitz":
        # Symmetry case — re-parse with fitz on a subset. Rare in practice (the
        # cascade typically calls fitz first and escalates), but supported.
        from lib.parsing.pdf.fitz import fitz_pdf_to_line_df
        all_lines = fitz_pdf_to_line_df(pdf_path)
        new_lines = all_lines[all_lines["page_num"].isin(pages)].copy()
        if "parsing_method" not in new_lines.columns:
            new_lines["parsing_method"] = "fitz"
    else:
        raise NotImplementedError(
            f"enrich method not yet wired in `docintel`: {method!r}. "
            "Known methods today: 'fitz', 'azure_layout', 'docling'. "
            "Article 10 also describes 'vision_gpt4o' inline ; promote it here "
            "when the vision-LLM parser lands."
        )
    return new_lines


def _build_enriched_page_rows(
    new_lines: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Aggregate new line_df rows into per-page rows tagged with the method."""
    if new_lines.empty:
        return pd.DataFrame(columns=["page_num", "text", "n_lines", "parsing_method"])
    return (
        new_lines.groupby("page_num", as_index=False)
        .agg(
            text=("text", lambda s: "\n".join(s.astype(str))),
            n_lines=("text", "count"),
        )
        .assign(parsing_method=method)
    )


def enrich(
    pdf_path: str | Path,
    line_df: pd.DataFrame,
    page_df: pd.DataFrame,
    pages: Iterable[int],
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-parse ``pages`` with ``method`` and append the new rows to the tables.

    The existing rows stay in ``line_df`` and ``page_df`` so the audit trail
    holds : a page may carry two `line_df` rows (one per method), and one
    `page_df` row per method. Downstream queries that want a specific
    method filter on the ``parsing_method`` column.

    Returns ``(line_df, page_df)`` — both with the new rows appended.
    """
    pages_list = sorted(int(p) for p in pages)
    new_lines = _enrich_one_method(pdf_path, pages_list, method)
    line_df_out = pd.concat([line_df, new_lines], ignore_index=True)
    new_page_rows = _build_enriched_page_rows(new_lines, method)
    page_df_out = pd.concat([page_df, new_page_rows], ignore_index=True)
    return line_df_out, page_df_out


def get_or_enrich(
    pdf_path: str | Path,
    line_df: pd.DataFrame,
    page_df: pd.DataFrame,
    pages: Iterable[int],
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`enrich` with a cache layer keyed on (page_num, parsing_method).

    Looks up which of ``pages`` already have a ``page_df`` row with the
    matching ``parsing_method`` and skips them. Re-parses only the missing
    ones. When nothing is missing, returns the tables unchanged.

    The cache lives in ``page_df`` itself ; persisting ``page_df`` (Article
    19's storage discipline) persists the cache.
    """
    pages_list = sorted(int(p) for p in pages)
    if "parsing_method" not in page_df.columns:
        missing = pages_list
    else:
        done_pages = set(
            page_df.loc[page_df["parsing_method"] == method, "page_num"]
            .astype(int)
            .tolist()
        )
        missing = [p for p in pages_list if p not in done_pages]
    if not missing:
        return line_df, page_df
    return enrich(pdf_path, line_df, page_df, missing, method)
