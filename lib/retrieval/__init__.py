"""Retrieval brick (public Vol.1 surface, trimmed).

Only the symbols the Vol.1 notebooks import are re-exported, so the
public package ships no later-volume code.
"""
from lib.retrieval.embedding_similarity import (
    cosine_sim,
    retrieve_pages_by_similarity,
)
from lib.retrieval.keyword_matching import retrieve_pages

__all__ = ["cosine_sim", "retrieve_pages_by_similarity", "retrieve_pages"]
