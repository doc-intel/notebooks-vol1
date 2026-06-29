"""Core primitives — cross-cutting helpers used by multiple modules."""

from lib.core.embeddings import get_embedding, embed_page_df
from lib.core.llm import (
    PromptTooLargeError,
    analyze_image,
    combine_ca_bundle,
    default_client,
    default_model,
    llm_chat,
    llm_parse,
    resolve_ca_bundle,
)
from lib.core.llm_clients import azure_client, openai_compatible_client

__all__ = [
    "get_embedding",
    "embed_page_df",
    "PromptTooLargeError",
    "analyze_image",
    "combine_ca_bundle",
    "default_client",
    "default_model",
    "llm_chat",
    "llm_parse",
    "resolve_ca_bundle",
    "azure_client",
    "openai_compatible_client",
]
