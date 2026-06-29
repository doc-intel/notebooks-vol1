"""Shape-agnostic helpers for LLM responses.

Two API surfaces produce two payload shapes :

  - ``responses.parse`` returns ``{"output": [{"content": [{"type":
    "output_text", "text": ...}]}], "usage": {"input_tokens",
    "output_tokens", "total_tokens"}}``
  - ``chat.completions`` returns ``{"choices": [{"message": {"content":
    ...}}], "usage": {"prompt_tokens", "completion_tokens",
    "total_tokens"}}``

Plus reasoning models (deepseek-r1, QwQ, ...) prefix their answer with
``<think>...</think>`` content that pollutes JSON parsing. These helpers
normalise both quirks so the rest of the package never branches on the
provider.

Ported from ``ia_package.generation.chat`` with the multi-provider routing
removed (the lib accepts a pre-built client ; routing is a backend concern,
see ``project_docintel_pure_library``).
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel


_THINK_RE = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def extract_think_content(text: str) -> tuple[str | None, str]:
    """Split a reasoning model's response into ``(think, content)``.

    Returns ``(None, text.strip())`` when no ``<think>...</think>`` block
    is present. ``think`` is the raw reasoning (may be useful for an audit
    trail) ; ``content`` is what downstream code parses as the answer.
    """
    m = _THINK_RE.search(text)
    if m:
        think = m.group(1).strip() or None
        content = m.group(2).strip()
        return think, content
    return None, text.strip()


def strip_json_markdown(text: str) -> str:
    """Remove ``` and ```json fences that some models add around JSON."""
    return _JSON_FENCE_RE.sub("", text).strip()


def normalize_usage(raw: dict[str, Any]) -> dict[str, int]:
    """Return a unified ``{prompt_tokens, completion_tokens, total_tokens}``.

    Handles both ``responses.*`` (``input_tokens`` / ``output_tokens``) and
    ``chat.completions`` (``prompt_tokens`` / ``completion_tokens``) wire
    formats. Missing fields default to 0.
    """
    usage = raw.get("usage") or {}
    return {
        "prompt_tokens":
            int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "completion_tokens":
            int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "total_tokens":
            int(usage.get("total_tokens") or 0),
    }


def parse_structured_from_text(
    text: str,
    schema: type[BaseModel],
    *,
    strip_think: bool = True,
) -> BaseModel:
    """Parse JSON out of a free-form text response and validate it with ``schema``.

    The fallback path when the provider does not support ``responses.parse``
    or ``response_format=json_schema`` : the model returned text that
    *claims* to be JSON (often inside a ``` fence, sometimes after a
    ``<think>`` block). This helper cleans and validates it.

    Raises :class:`pydantic.ValidationError` on schema mismatch and
    :class:`json.JSONDecodeError` when the cleaned text isn't valid JSON.
    """
    if strip_think:
        _, text = extract_think_content(text)
    cleaned = strip_json_markdown(text)
    data = json.loads(cleaned)
    return schema.model_validate(data)


__all__ = [
    "extract_think_content",
    "strip_json_markdown",
    "normalize_usage",
    "parse_structured_from_text",
]
