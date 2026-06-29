"""Retrieval method: embeddings + cosine similarity.

Implements the method walked in Article 1, §2.4.a. Pages are turned into vectors
by `lib.core.embed_page_df` (one embedding per page); the question is
embedded the same way; cosine similarity ranks the pages; the top-k are kept.

Article 2 develops the method in depth (where embeddings shine, where they
quietly fail, how to tune chunking and choice of model).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from openai import OpenAI

from lib.core.embeddings import get_embedding


def cosine_sim(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between `query_vec` and each row of `doc_matrix`."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    d = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-12)
    return d @ q


def retrieve_pages_by_similarity(
    page_df: pd.DataFrame,
    line_df: pd.DataFrame,
    question: str,
    top_k: int = 3,
    client: OpenAI | None = None,
    pages_hint: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the top-k pages by cosine similarity and the corresponding lines.

    `page_df` must already have an `embedding` column (run `embed_page_df` first).
    Returns (retrieved_pages_df, filtered_line_df) sorted by `similarity` desc.

    When ``pages_hint`` is set, the candidate pool is first restricted to those
    pages — similarity is computed on the question vector against the full
    page matrix, but only hinted pages are kept for ranking. Same scoping
    semantic as :func:`retrieve_pages`.
    """
    if "embedding" not in page_df.columns:
        raise ValueError(
            "page_df must contain an 'embedding' column. Run embed_page_df first."
        )

    if pages_hint is not None and len(pages_hint) > 0:
        page_df = page_df[page_df["page_num"].isin(pages_hint)]
        line_df = line_df[line_df["page_num"].isin(pages_hint)]
        if page_df.empty:
            empty_pages = page_df.assign(similarity=pd.Series(dtype=float))
            return empty_pages, line_df.iloc[0:0].copy()

    query_vec = get_embedding(question, client=client)
    doc_matrix = np.vstack(page_df["embedding"].values)
    similarities = cosine_sim(query_vec, doc_matrix)

    scored_pages = page_df.copy()
    scored_pages["similarity"] = similarities
    retrieved_pages_df = (
        scored_pages.nlargest(top_k, "similarity")
        .reset_index(drop=True)
    )

    kept_pages = retrieved_pages_df["page_num"].tolist()
    filtered_line_df = (
        line_df[line_df["page_num"].isin(kept_pages)]
        .sort_values(["page_num", "line_num"])
        .copy()
    )
    return retrieved_pages_df, filtered_line_df


def score_lines_by_question(
    line_df: pd.DataFrame,
    page_num: int,
    question: str,
    *,
    client: OpenAI | None = None,
) -> pd.DataFrame:
    """Article 7B line-level scoring : embed every line on `page_num`, score
    cosine vs the question, return a DataFrame with the per-line score.

    Article 7B §3 introduces this as the per-page heatmap : when page-level
    embedding ranks a page middle of the pack but the relevant content is
    a handful of lines, line-level cosine surfaces those lines explicitly.
    The same primitive drives the interactive PDF.js viewer's "heatmap by
    embedding similarity" overlay in the shipai demo : every line of the
    visible page is tinted by its cosine to the question.

    The result row carries everything the overlay / heatmap needs :
    ``line_num`` + ``x0, y0, x1, y1`` (bbox in PDF coords) + ``score``
    (cosine in [-1, 1], typically [0.5, 0.9] for an English text).

    Parameters
    ----------
    line_df
        Parsed line_df (columns ``page_num``, ``line_num``, ``text``,
        ``x0``, ``y0``, ``x1``, ``y1``).
    page_num
        1-indexed page to score.
    question
        The user's question.
    client
        OpenAI client (defaults to the env-configured one).

    Returns
    -------
    pandas DataFrame sorted by line_num ; empty if the page has no lines.
    Columns : ``page_num, line_num, text, x0, y0, x1, y1, score``.
    """
    from lib.core.embeddings import embed_page_df  # local : avoid hard cycle

    plines = (
        line_df[line_df["page_num"] == int(page_num)]
        .sort_values("line_num")
        .copy()
    )
    if plines.empty:
        return plines.assign(score=pd.Series(dtype=float))
    q_vec = get_embedding(question, client=client)
    line_emb = embed_page_df(plines[["text"]].copy(), client=client)
    mat = np.vstack(line_emb["embedding"].tolist())
    sims = cosine_sim(q_vec, mat)
    plines["score"] = sims
    return plines.reset_index(drop=True)
