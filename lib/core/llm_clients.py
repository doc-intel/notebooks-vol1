"""Optional helpers for constructing OpenAI-compatible clients.

The lib accepts a pre-built ``OpenAI`` and never branches on provider — that
stays a backend responsibility (V5 Streamlit, V6 Electron bridge, V7
FastAPI dependency injection). These helpers cover the common cases so a
backend doesn't have to recompute the URL pattern from scratch.

OpenAI-compatible providers (Anthropic via proxy, Together, Groq, vLLM,
LiteLLM, llama.cpp, ...) all work with :func:`openai_compatible_client` —
just point ``base_url`` at the right endpoint. Azure has its own URL shape,
hence the dedicated helper.

Ollama : ``openai_compatible_client(api_key="ollama", base_url="http://localhost:11434/v1/")``.
"""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


def openai_compatible_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 180.0,
    http_client: Any | None = None,
) -> OpenAI:
    """Build a vanilla OpenAI-compatible client.

    Defaults : ``api_key`` ← ``$API_KEY``, ``base_url`` ← ``$BASE_URL``.
    Same shape as ``OpenAI(...)`` but with a sane 180s timeout matching the
    rest of the package.

    Pass ``http_client=httpx.Client(verify=False)`` for self-hosted setups
    with self-signed certificates (typical Ollama / on-prem deployments).
    """
    kwargs: dict[str, Any] = {
        "api_key": api_key if api_key is not None else os.getenv("API_KEY"),
        "timeout": timeout,
    }
    resolved_base = base_url if base_url is not None else os.getenv("BASE_URL")
    if resolved_base:
        kwargs["base_url"] = resolved_base
    if http_client is not None:
        kwargs["http_client"] = http_client
    return OpenAI(**kwargs)


def azure_client(
    *,
    resource: str,
    api_key: str,
    api_version: str = "2024-10-21",
    timeout: float = 180.0,
) -> OpenAI:
    """Build an Azure-OpenAI client behind the OpenAI SDK.

    The SDK routes ``model=<deployment>`` to
    ``https://<resource>.openai.azure.com/openai/v1/...`` once we set the
    base URL. Pass the model name (which is also the Azure deployment name)
    when you call ``llm_parse(model=...)``.

    Parameters
    ----------
    resource
        The Azure resource name (the ``X`` in ``https://X.openai.azure.com``).
    api_key
        Resolved API key (no env var name lookup — the backend resolves
        secrets and passes the literal).
    api_version
        Azure API version. Defaults to a recent stable release.
    """
    base_url = f"https://{resource}.openai.azure.com/openai/v1/"
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        default_query={"api-version": api_version},
    )


__all__ = [
    "openai_compatible_client",
    "azure_client",
]
