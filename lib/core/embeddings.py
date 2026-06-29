"""Embedding helpers — used by retrieval, question parsing, and any module
that needs to embed text. Lives in core because it's a cross-cutting primitive.

Every call routes through the SQLite cache in `lib.storage.sqlite_store`,
so a given (text, model) pair is embedded once and reused across articles
and across runs. Identical text in two different documents (boilerplate
disclaimers, common cover pages, repeated headers) costs one API call total.
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd
from openai import OpenAI

from lib.storage.sqlite_store import (
    embed_texts_cached,
    get_embedding_cached,
)


def _resolve_model(model: str | None) -> str:
    return model or os.getenv("MODEL_EMBED", "text-embedding-3-small")


def _resolve_client(client: OpenAI | None) -> OpenAI:
    if client is not None:
        return client
    return OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))


def get_embedding(
    text: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> np.ndarray:
    """Return the embedding for `text`. Cache hit = no API call."""
    return get_embedding_cached(
        _resolve_client(client),
        text,
        model=_resolve_model(model),
    )


def embed_page_df(
    page_df: pd.DataFrame,
    client: OpenAI | None = None,
    text_col: str = "text",
    model: str | None = None,
) -> pd.DataFrame:
    """Add an `embedding` column to `page_df`.

    Cache-aware : known (text, model) pairs are loaded from SQLite in one
    batch SELECT ; misses are batched to the embedding API. Re-running on
    the same DataFrame is free.
    """
    out = page_df.copy()
    texts = out[text_col].astype(str).tolist()
    vectors = embed_texts_cached(
        _resolve_client(client),
        texts,
        model=_resolve_model(model),
    )
    out["embedding"] = vectors
    return out
