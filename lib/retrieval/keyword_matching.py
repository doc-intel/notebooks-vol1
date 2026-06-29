"""Retrieval method: keyword matching.

Implements the method walked in Article 1, §2.4.b. The score for a page is the
number of distinct keywords (from the parsed question) that appear in its text,
case-insensitive substring match. Pages with zero matches are dropped — we don't
pad the filter with unrelated pages. Transparent to the user: the table returned
shows exactly which keywords landed on which page, no opaque score to interpret.

Article 6 (Question Parsing) develops the keyword extraction step in depth
(domain glossaries, synonym expansion, etc.).
"""
from __future__ import annotations

import pandas as pd


def collapse_letter_spacing(text: str) -> str:
    """Collapse typographic letter-spacing on heading-like lines.

    Creatively-typeset documents render section headers with a space between
    every character (``E D U C ATION``, ``TE A C H IN G``). The parser keeps
    those spaces, so a keyword search for ``education`` never matches. This
    rebuilds the un-spaced form **for matching only**.

    Operates line by line (page text is ``\\n``-joined) and never crosses a line
    boundary. A line is treated as letter-spaced when single-character alpha
    tokens are the strict majority of its tokens (and there are at least three) ;
    its whitespace is then removed. Every other line (normal prose) is returned
    unchanged. ``line_df.text`` / ``page_df.text`` are left verbatim — this only
    builds an extra match form, so citations stay exact.
    """
    out: list[str] = []
    for line in text.split("\n"):
        tokens = line.split()
        singles = [t for t in tokens if len(t) == 1 and t.isalpha()]
        if len(singles) >= 3 and len(singles) * 2 > len(tokens):
            out.append("".join(tokens))      # un-space the letter-spaced word(s)
        else:
            out.append(line)
    return "\n".join(out)


def find_matching_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return the subset of `keywords` that appear (case-insensitive) in `text`.

    Matches against the text and, when a line is letter-spaced, against its
    collapsed form too (so ``education`` finds ``E D U C ATION``).
    """
    text_lower = text.lower()
    collapsed = collapse_letter_spacing(text_lower)
    if collapsed == text_lower:
        return [kw for kw in keywords if kw.lower() in text_lower]
    return [
        kw for kw in keywords
        if kw.lower() in text_lower or kw.lower() in collapsed
    ]


def retrieve_pages(
    page_df: pd.DataFrame,
    line_df: pd.DataFrame,
    keywords: list[str],
    top_k: int = 3,
    pages_hint: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Page-level retrieval by keyword match count.

    For each page, count how many of `keywords` appear in the page text. Drop
    pages with zero matches (we don't want to pad the filter with unrelated
    pages). Keep the top-k of the rest, ranked by match count descending. Then
    pull every line from those pages out of `line_df`.

    When ``pages_hint`` is set (e.g. user said *"on page 3"*, *"pages 5 to 7"*,
    *"page 2 and page 9"*), the search space is first restricted to those
    pages : ``page_df`` and ``line_df`` are filtered to
    ``page_num.isin(pages_hint)`` before scoring. Hinted pages are kept even
    when no keyword matches them — the user pinned them explicitly, so a
    zero-keyword page is still a deliberate answer surface. This is the
    short-doc / pinned-page pattern documented in Article 8 § 8.8.

    Returns (retrieved_pages_df, filtered_line_df). retrieved_pages_df is sorted
    by `match_count` descending and carries `matched_keywords` (the actual
    keywords that landed on the page) and `match_count` columns. May contain
    fewer than `top_k` rows if not enough pages match; may be empty if none do
    (or if ``pages_hint`` lists pages that do not exist in the document).
    """
    if pages_hint is not None and len(pages_hint) > 0:
        page_df = page_df[page_df["page_num"].isin(pages_hint)]
        line_df = line_df[line_df["page_num"].isin(pages_hint)]
    scored_pages = page_df.copy()
    scored_pages["matched_keywords"] = scored_pages["text"].apply(
        lambda text: find_matching_keywords(text, keywords)
    )
    scored_pages["match_count"] = scored_pages["matched_keywords"].apply(len)
    if not pages_hint:
        # Default behaviour : drop pages with zero hits to avoid padding the
        # filter with unrelated pages.
        scored_pages = scored_pages[scored_pages["match_count"] > 0]
    # When pages_hint is set, keep the hinted pages even with zero matches —
    # the user pinned them, that's the answer surface.
    retrieved_pages_df = (
        scored_pages.nlargest(top_k, "match_count")
        .reset_index(drop=True)
    )
    kept_pages = retrieved_pages_df["page_num"].tolist()
    filtered_line_df = (
        line_df[line_df["page_num"].isin(kept_pages)]
        .sort_values(["page_num", "line_num"])
        .copy()
    )
    return retrieved_pages_df, filtered_line_df
