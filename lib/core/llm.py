"""Single LLM call wrapper — retry, cache, token-limit detection, structured-output fallback.

Every ``generation/*`` and ``question/*`` and ``corpus/*`` site that used
to do ::

    client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
                    timeout=180.0)
    resp = client.responses.parse(model=os.getenv("MODEL_CHAT", "gpt-4.1"),
                                  input=..., text_format=Schema)
    return Schema.model_validate_json(resp.output_text)

becomes ::

    from lib.core.llm import llm_parse
    return llm_parse(input=..., text_format=Schema)

The wrapper handles :

  - **Client construction** : if ``client=None``, builds one from env vars
    ``API_KEY`` / ``BASE_URL`` (180s timeout). Pass a pre-built client to
    reuse a notebook session or hit Azure / Ollama.
  - **Model defaulting** : ``model=None`` resolves to ``$MODEL_CHAT``
    (fallback ``"gpt-4.1"``).
  - **Cache via SQLAlchemy engine** : pass ``engine=`` (from ``db_io.default_engine()``)
    and a hit returns instantly. Schema-aware key.
  - **Retry on transient errors** : exponential backoff + jitter, honours
    ``Retry-After`` headers / messages, bounded by ``max_retries``.
  - **Token limit → typed exception** : ``context_length_exceeded`` raises
    :class:`PromptTooLargeError` instead of remoting a generic API error
    so the pipeline can shrink the prompt and retry at a higher level.
  - **Structured-output fallback** : when the provider rejects
    ``json_schema``, retries via ``chat.completions`` with the schema
    inlined into the system prompt and parses JSON from the text.
  - **Reasoning models** : strips ``<think>...</think>`` before parsing.

Design follows ``project_docintel_pure_library`` : the lib accepts a pre-built
``OpenAI`` client (backend builds it), does NOT route between providers.
"""
from __future__ import annotations

import logging
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from openai import OpenAI
from pydantic import BaseModel

from lib.core.llm_normalize import (
    extract_think_content,
    parse_structured_from_text,
)


logger = logging.getLogger(__name__)


# --- Token usage tracking -------------------------------------------------
#
# pdf_qa / corpus_pdf_qa build a per-question pipeline_trace.json that
# documents every brick's output. Token usage is part of that trace : how
# many input + output tokens each LLM call consumed, which ones hit the
# cache (no tokens billed), and which provider+model produced them.
#
# The mechanism is a ContextVar-scoped list. Open a ``capture_llm_usage()``
# block, run any code that calls ``llm_parse`` / ``llm_chat``, and read the
# accumulated records. The wrappers themselves do the appending ; call-sites
# never touch the var directly.

@dataclass
class LLMCallRecord:
    """One LLM call's bookkeeping. Fresh calls carry token counts ;
    cache hits carry zeros + ``cached=True`` so the trace shows they were
    free."""
    label: str
    model: str
    schema_name: str  # text_format.__name__ or ``__llm_chat__``
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


_TOKEN_USAGE_LOG: ContextVar[list[LLMCallRecord] | None] = ContextVar(
    "_TOKEN_USAGE_LOG", default=None,
)


@contextmanager
def capture_llm_usage() -> Iterator[list[LLMCallRecord]]:
    """Context manager that captures every ``llm_parse`` / ``llm_chat`` call
    inside its scope into a fresh list.

    Usage ::

        with capture_llm_usage() as log:
            answer = pdf_qa(...)
        # log is a list of LLMCallRecord, one per LLM call inside pdf_qa

    Nested ``capture_llm_usage()`` blocks each get their own list ; calls
    only appear in the **innermost** active log (typical contextvar
    semantics). When no block is active, the wrappers silently skip the
    record — token tracking is opt-in.
    """
    log: list[LLMCallRecord] = []
    token = _TOKEN_USAGE_LOG.set(log)
    try:
        yield log
    finally:
        _TOKEN_USAGE_LOG.reset(token)


def _record_llm_call(
    label: str, model: str, schema_name: str,
    input_tokens: int = 0, output_tokens: int = 0, cached: bool = False,
) -> None:
    """Append one record to the active usage log, if any. No-op otherwise."""
    log = _TOKEN_USAGE_LOG.get()
    if log is not None:
        log.append(LLMCallRecord(
            label=label, model=model, schema_name=schema_name,
            input_tokens=int(input_tokens), output_tokens=int(output_tokens),
            cached=cached,
        ))


def _extract_usage(resp: Any) -> tuple[int, int]:
    """Best-effort extraction of (input_tokens, output_tokens) from an SDK
    response. Returns (0, 0) when the SDK does not expose usage (some
    providers omit it ; the cache only stores the result, not the usage)."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    # OpenAI responses API : input_tokens / output_tokens.
    # OpenAI chat.completions : prompt_tokens / completion_tokens.
    input_tok = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_tokens", None)
        or 0
    )
    output_tok = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None)
        or 0
    )
    return int(input_tok), int(output_tok)


# --- Exceptions -----------------------------------------------------------

class PromptTooLargeError(Exception):
    """Raised when the model rejects the prompt for ``context_length_exceeded``.

    Distinct from generic ``APIStatusError`` / ``RateLimitError`` so the
    pipeline can react specifically (shrink ``top_k``, chunk the input,
    fail fast with a user-friendly hint). NEVER retried internally — a
    retry with the same prompt would hit the same wall.
    """


# --- Retry classification -------------------------------------------------

_RETRIABLE_EXCEPTION_NAMES = frozenset({
    "APIConnectionError", "APITimeoutError", "ConnectError",
    "RateLimitError", "APIStatusError",
    "ServiceUnavailableError", "InternalServerError",
})

_RETRIABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})

_RETRIABLE_MSG_PATTERNS = (
    "rate limit", "rate_limit", "too many requests",
    "server error", "service unavailable", "overloaded",
    "timeout", "connection reset", "connection error",
    "temporarily", "please retry after",
)

# Errors that look retriable by HTTP code but mean "your prompt is too big".
# Detected BEFORE the retriable check so they never enter the backoff loop.
_CONTEXT_LENGTH_PATTERNS = (
    "context_length_exceeded",
    "maximum context length",
    "context length",
    "prompt is too long",
    "string too long",       # some providers wrap it this way
    "reduce the length",
)

# Errors that should NOT retry even when the HTTP code is retriable
# (typically schema-level problems that won't get better on the second try).
_NON_RETRIABLE_MSG_PATTERNS = (
    "invalid body: failed to parse json value",
    "invalid_request_error",
)

# Errors that mean "this provider doesn't support structured outputs". We
# catch these to switch to the chat.completions fallback path.
_JSON_SCHEMA_UNSUPPORTED_PATTERNS = (
    "json_schema",
    "response_format",
    "text_format",
    "structured output",
)


def is_context_length_error(e: Exception) -> bool:
    """True when the error reads as ``context_length_exceeded``."""
    msg = str(e).lower()
    return any(p in msg for p in _CONTEXT_LENGTH_PATTERNS)


def is_json_schema_unsupported(e: Exception) -> bool:
    """True when the provider rejected the structured-output payload."""
    msg = str(e).lower()
    return any(p in msg for p in _JSON_SCHEMA_UNSUPPORTED_PATTERNS)


def is_retriable(e: Exception) -> bool:
    """True for transient errors (429, 5xx, timeouts, connection resets)."""
    msg = str(e).lower()
    if any(p in msg for p in _NON_RETRIABLE_MSG_PATTERNS):
        return False
    if is_context_length_error(e):
        return False  # PromptTooLargeError path
    if type(e).__name__ in _RETRIABLE_EXCEPTION_NAMES:
        return True
    code = re.search(r"\b(\d{3})\b", msg)
    if code and int(code.group(1)) in _RETRIABLE_HTTP_CODES:
        return True
    return any(p in msg for p in _RETRIABLE_MSG_PATTERNS)


def extract_retry_after(e: Exception) -> float | None:
    """Parse a server-suggested wait from the error message.

    Honours both ``Please retry after N seconds`` and the more generic
    ``retry-after: N`` / ``try again in N`` / ``wait N``. Returns None
    when the message carries no such hint.
    """
    msg = str(e)
    m = re.search(r"please retry after\s+(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 1.0
    m = re.search(
        r"(?:retry[-_ ]?after|try again in|wait)\s*[:=]?\s*(\d+(?:\.\d+)?)",
        msg, re.IGNORECASE,
    )
    if m:
        return float(m.group(1))
    return None


def _compute_delay(
    attempt: int,
    *,
    base: float = 2.0,
    factor: float = 2.0,
    cap: float = 60.0,
    jitter: float = 0.3,
    retry_after: float | None = None,
) -> float:
    """Exponential backoff with jitter. ``retry_after`` sets a floor."""
    raw = min(base * (factor ** attempt), cap)
    delay = raw * (1.0 + jitter * random.random())
    if retry_after is not None:
        delay = max(delay, retry_after)
    return round(delay, 2)


# --- Client + model defaults ---------------------------------------------

def resolve_ca_bundle(
    ca_bundle: str | None = None,
    verify: "str | bool | None" = None,
):
    """Resolve the TLS verify setting for the LLM HTTP client.

    Priority, first set wins:
      1. explicit ``verify`` (a CA-bundle path, ``False`` to disable, ``True``
         for the default trust store) ;
      2. explicit ``ca_bundle`` path ;
      3. the ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` environment variables ;
      4. ``None`` -> the HTTP client's own default (certifi).

    Behind a TLS-MITM proxy (Netskope, Zscaler, …) point any of these at a
    bundle that concatenates certifi's public roots AND the proxy's injected CA
    (see :func:`combine_ca_bundle`). Feeding the proxy CA alone drops the public
    roots and every other TLS call breaks.
    """
    if verify is not None:
        return verify
    if ca_bundle:
        return str(ca_bundle)
    return os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE") or None


def combine_ca_bundle(*enterprise_ca_paths: str, out_path: str | None = None) -> str:
    """Write certifi's public roots followed by the enterprise CA(s) into one
    PEM file and return its path.

    Behind a TLS-MITM proxy the client must trust BOTH the public roots (for
    everything else) and the proxy's injected CA. Combine them once here, then
    pass the result as ``default_client(ca_bundle=...)`` (or export it as
    ``SSL_CERT_FILE``), so a front supplies the CA a single time instead of
    patching the environment of every spawned process.
    """
    import certifi  # ships with httpx / openai

    parts = [Path(certifi.where()).read_text(encoding="utf-8")]
    parts.extend(Path(ca).read_text(encoding="utf-8") for ca in enterprise_ca_paths)
    bundle = "".join(part.strip() + "\n" for part in parts)
    if out_path is None:
        import tempfile
        out_path = str(Path(tempfile.gettempdir()) / "docintel-ca-bundle.pem")
    Path(out_path).write_text(bundle, encoding="utf-8")
    return out_path


def default_client(
    *,
    ca_bundle: str | None = None,
    verify: "str | bool | None" = None,
) -> OpenAI:
    """Build an OpenAI-compatible client from env vars.

    Reads ``API_KEY`` (alias ``OPENAI_API_KEY``) and ``BASE_URL`` (alias
    ``OPENAI_BASE_URL``). The aliases exist because some call-sites pre-date
    the ``API_KEY`` / ``BASE_URL`` convention ; honouring both keeps every
    historical .env file working.

    180s timeout matches what every existing call-site sets manually — long
    enough for structured-output retries on the SDK side, short enough that
    a wedged connection eventually fails.

    TLS behind an enterprise proxy : when ``ca_bundle`` / ``verify`` is passed,
    or ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` is set, the underlying httpx
    client is built to verify against that bundle (see :func:`resolve_ca_bundle`
    and :func:`combine_ca_bundle`). A TLS-MITM proxy (Netskope) then no longer
    yields ``APIConnectionError`` ; the front supplies the CA once rather than
    patching the environment per process.
    """
    kwargs: dict[str, Any] = {
        "api_key": os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        "timeout": 180.0,
    }
    resolved = resolve_ca_bundle(ca_bundle, verify)
    if resolved is not None:
        import httpx
        if isinstance(resolved, str):
            # httpx 0.28 deprecates verify=<path>; pass an SSL context built
            # from the bundle instead.
            import ssl
            verify_arg: Any = ssl.create_default_context(cafile=resolved)
        else:
            verify_arg = resolved
        kwargs["http_client"] = httpx.Client(verify=verify_arg, timeout=180.0)
    return OpenAI(**kwargs)


def default_model() -> str:
    """``$MODEL_CHAT`` with fallback to ``gpt-4.1`` (matches the existing convention)."""
    return os.getenv("MODEL_CHAT", "gpt-4.1")


# Sentinel ``schema_name`` for plain-text ``llm_chat`` outputs persisted in
# the shared ``llm_calls`` table. Avoids a parallel ``llm_chats`` table for V1.
_CHAT_SCHEMA_NAME = "__llm_chat__"


def _default_cache_engine() -> Any | None:
    """Resolve the canonical cache engine via ``storage.paths.output_root``.

    Returns the SQLAlchemy engine for ``<output_root>/storage.sqlite``, or
    ``None`` when the engine can't be built (no SQLAlchemy, unreachable disk,
    sandbox without write access). A miss here MUST NOT crash the LLM call ;
    the worst case is "this call is not cached", not "the pipeline blows up".

    Re-resolved on every call to honour env-var changes between tests / runs.
    SQLAlchemy engine construction is lazy — no connection until the first
    SQL statement runs — so the cost is one ``Path`` call plus a dict lookup.
    """
    try:
        from lib.storage.db_io import default_engine
        return default_engine()
    except Exception as e:
        logger.warning(
            "[llm cache] could not resolve default engine "
            "(this call is not cached) : %s", e,
        )
        return None


# --- Main entry point ----------------------------------------------------

def llm_parse(
    *,
    input: str | list[dict[str, Any]],
    text_format: type[BaseModel],
    client: OpenAI | None = None,
    model: str | None = None,
    label: str = "default",
    max_retries: int = 6,
    engine: Any | None = None,
    cache: bool = True,
    fallback_to_chat_completions: bool = True,
    strip_think: bool = True,
) -> BaseModel:
    """Call ``client.responses.parse`` with retry + transparent cache.

    Caching is **on by default** : when ``engine=None`` and ``cache=True``,
    the wrapper resolves a default SQLAlchemy engine via
    ``lib.storage.db_io.default_engine()`` (which itself resolves
    ``$DOCINTEL_OUTPUT_DIR`` or ``~/.docintel/output``). Identical calls
    across processes / renders / PCs that share the cache file hit instantly
    and pay nothing. Pass ``cache=False`` to bypass for tests or one-off
    fresh calls.

    Parameters
    ----------
    input
        Either a single prompt string (sent as user message) or the
        responses-API ``input`` list (``[{"role": ..., "content": ...}, ...]``).
    text_format
        Pydantic ``BaseModel`` subclass for the structured output.
    client
        OpenAI-compatible client. ``None`` → :func:`default_client`.
    model
        Model name. ``None`` → :func:`default_model`.
    label
        Tag stored on cached rows / used in retry logs. Pick something
        recognisable per call-site (``"qa.answer"``, ``"corpus.cascade"``,
        …) so the audit query ``SELECT label, COUNT(*) FROM llm_calls``
        is informative.
    max_retries
        Attempts before giving up on transient errors.
    engine
        SQLAlchemy engine. When ``None`` and ``cache=True``, auto-resolved
        via :func:`_default_cache_engine`. Pass an explicit engine to write
        into a non-default location (per-project DB, in-memory engine, …).
    cache
        Enable the persistent cache for this call. Default ``True`` (the
        invariant : *identical call → one paid execution, ever*). Pass
        ``False`` to disable both read and write — useful for tests that
        want to count network calls or for callers that explicitly want a
        fresh answer.
    fallback_to_chat_completions
        On ``json_schema`` rejection, retry via ``chat.completions`` with
        the schema inlined into the system prompt. Useful for Ollama and
        legacy providers.
    strip_think
        Strip ``<think>...</think>`` from the response text before parsing
        (deepseek-r1, QwQ).

    Returns
    -------
    Validated ``text_format`` instance.

    Raises
    ------
    PromptTooLargeError
        Prompt exceeds the model's context window.
    Exception
        The last error after ``max_retries`` attempts, or any non-retriable
        error encountered immediately.
    """
    if client is None:
        client = default_client()
    if model is None:
        model = default_model()
    messages = _coerce_input(input)
    cache_key_input = _cache_key_input(messages)

    if engine is None and cache:
        engine = _default_cache_engine()

    # --- cache hit ?
    if engine is not None:
        from lib.storage.db_io import load_llm_call
        cached = load_llm_call(
            model=model, schema_name=text_format.__name__,
            input_text=cache_key_input, engine=engine,
        )
        if cached is not None:
            _record_llm_call(
                label=label, model=model,
                schema_name=text_format.__name__,
                cached=True,
            )
            return text_format.model_validate_json(cached)

    # --- retry loop
    last_exc: Exception | None = None
    fell_back = False
    for attempt in range(max_retries):
        try:
            if fell_back:
                parsed, usage = _call_chat_completions_fallback(
                    client, model=model, messages=messages, text_format=text_format,
                    strip_think=strip_think,
                )
            else:
                parsed, usage = _call_responses_parse(
                    client, model=model, messages=messages, text_format=text_format,
                    strip_think=strip_think,
                )
            if engine is not None:
                from lib.storage.db_io import save_llm_call
                save_llm_call(
                    model=model, schema_name=text_format.__name__,
                    input_text=cache_key_input,
                    output_json=parsed.model_dump_json(),
                    label=label, engine=engine,
                )
            _record_llm_call(
                label=label, model=model,
                schema_name=text_format.__name__,
                input_tokens=usage[0], output_tokens=usage[1],
                cached=False,
            )
            return parsed

        except Exception as e:
            last_exc = e
            if is_context_length_error(e):
                raise PromptTooLargeError(
                    f"prompt exceeds model context window "
                    f"(model={model}, schema={text_format.__name__}, "
                    f"approx {len(cache_key_input)} chars) : {e}"
                ) from e
            if (
                fallback_to_chat_completions
                and not fell_back
                and is_json_schema_unsupported(e)
            ):
                logger.info(
                    "[llm_parse %s] provider rejected json_schema → "
                    "fallback to chat.completions", label,
                )
                fell_back = True
                continue  # retry immediately on the fallback path, no backoff
            if not is_retriable(e):
                raise
            if attempt == max_retries - 1:
                break
            delay = _compute_delay(attempt, retry_after=extract_retry_after(e))
            sys.stderr.write(
                f"[llm_parse {label}] {type(e).__name__} "
                f"attempt {attempt + 1}/{max_retries} ; retrying in {delay}s\n"
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


# --- internals ------------------------------------------------------------

def _coerce_input(input: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept either a raw prompt string or a messages list."""
    if isinstance(input, str):
        return [{"role": "user", "content": input}]
    return input


def _cache_key_input(messages: list[dict[str, Any]]) -> str:
    """Stable text representation of ``messages`` for the cache key.

    The cache uses ``sha256(model + schema + input_text)`` so the input
    needs to be deterministic. JSON with ``sort_keys=True`` would also
    work ; we use a flat repr because the existing cache (sqlite_store)
    already keyed off ``str(input)`` and we want hits across both layers.
    """
    import json
    return json.dumps(messages, sort_keys=True, default=str, ensure_ascii=False)


def _call_responses_parse(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    text_format: type[BaseModel],
    strip_think: bool,
) -> tuple[BaseModel, tuple[int, int]]:
    """Primary path : ``client.responses.parse(...).output_text`` → validate.

    Returns ``(parsed, (input_tokens, output_tokens))``. The token tuple
    feeds the pipeline-trace token log ; both ints are 0 when the SDK
    does not expose ``response.usage``.
    """
    resp = client.responses.parse(
        model=model,
        input=messages,
        text_format=text_format,
        store=False,
    )
    text = resp.output_text
    if strip_think:
        _, text = extract_think_content(text)
    return text_format.model_validate_json(text), _extract_usage(resp)


def _call_chat_completions_fallback(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    text_format: type[BaseModel],
    strip_think: bool,
) -> tuple[BaseModel, tuple[int, int]]:
    """Fallback path : ``chat.completions`` with schema inlined in system prompt.

    Used when the provider rejects ``response_format=json_schema`` (Ollama,
    older Azure deployments, some local models). The schema is appended to
    the system message as an example dict so the model knows the shape.
    Quality is materially lower than native structured output — only used
    after the primary path fails.

    Two-level cascade inside this fallback :

    - **Mode B** : ``response_format={"type": "json_object"}`` + schema hint
      in the system prompt. Works on most OpenAI-compatible servers.
    - **Mode C** : plain ``chat.completions`` (no ``response_format`` at all)
      + schema hint in the system prompt + ``parse_structured_from_text``.
      Last-ditch path for providers that reject ``response_format`` entirely
      (older Ollama builds, llama.cpp ``--no-grammar`` servers, some local
      bridges). The model returns JSON-shaped text ; we strip fences /
      ``<think>`` and validate against the schema.
    """
    enriched = _inline_schema_in_system_prompt(messages, text_format)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=enriched,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        if not is_json_schema_unsupported(e):
            raise
        logger.info(
            "[llm_parse] provider also rejected response_format=json_object → "
            "fallback to plain chat.completions (Mode C)"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=enriched,
            temperature=0,
        )
    text = resp.choices[0].message.content or ""
    parsed = parse_structured_from_text(text, text_format, strip_think=strip_think)
    return parsed, _extract_usage(resp)


def _inline_schema_in_system_prompt(
    messages: list[dict[str, Any]],
    text_format: type[BaseModel],
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with the schema appended to the system role.

    If there is no system message, prepend one. The schema is rendered as a
    flat JSON example (one ``<type>`` placeholder per field) — enough hint
    for the model without the full JSON Schema verbosity.
    """
    import json as _json
    props = text_format.model_json_schema().get("properties", {})
    example = {k: f"<{v.get('type', 'value')}>" for k, v in props.items()}
    schema_hint = (
        "Reply ONLY with valid JSON, no surrounding text, with exactly these fields :\n"
        + _json.dumps(example, ensure_ascii=False)
    )
    out: list[dict[str, Any]] = []
    seen_system = False
    for m in messages:
        if m.get("role") == "system" and not seen_system:
            content = m.get("content") or ""
            if isinstance(content, list):
                # responses-API content shape ; flatten to a string
                content = " ".join(
                    str(c.get("text", "")) for c in content
                    if isinstance(c, dict)
                )
            out.append({"role": "system", "content": f"{content}\n\n{schema_hint}"})
            seen_system = True
        else:
            out.append(m)
    if not seen_system:
        out.insert(0, {"role": "system", "content": schema_hint})
    return out


def llm_chat(
    *,
    input: str | list[dict[str, Any]],
    client: OpenAI | None = None,
    model: str | None = None,
    label: str = "default",
    max_retries: int = 6,
    temperature: float = 0.0,
    strip_think: bool = True,
    engine: Any | None = None,
    cache: bool = True,
) -> str:
    """Call ``client.chat.completions.create`` for plain-text output.

    Sibling of :func:`llm_parse` for call-sites that need raw text (translation,
    summarisation rewriting, anything that doesn't fit a Pydantic schema). Same
    retry + token-limit discipline AND the same transparent cache : identical
    ``(model, messages)`` returns the previously-saved text without a network
    call. Temperature > 0 disables the cache automatically (a non-deterministic
    output would poison the slot for everyone).

    Pass ``cache=False`` to opt out (creative generation, A/B comparisons,
    tests that count network calls).

    Returns the model's text content. ``<think>...</think>`` blocks are stripped
    by default so reasoning-model outputs look like normal completions.
    """
    if client is None:
        client = default_client()
    if model is None:
        model = default_model()
    messages = _coerce_input(input)

    # Non-deterministic temperature poisons a shared cache (next reader gets
    # someone else's "creative" sample). Disable for any non-zero setting.
    cache_active = bool(cache) and temperature == 0.0
    if engine is None and cache_active:
        engine = _default_cache_engine()

    cache_key_input = _cache_key_input(messages) if engine is not None else None

    if engine is not None and cache_active:
        from lib.storage.db_io import load_llm_call
        cached = load_llm_call(
            model=model, schema_name=_CHAT_SCHEMA_NAME,
            input_text=cache_key_input, engine=engine,
        )
        if cached is not None:
            _record_llm_call(
                label=label, model=model,
                schema_name=_CHAT_SCHEMA_NAME, cached=True,
            )
            return cached

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            text = resp.choices[0].message.content or ""
            if strip_think:
                _, text = extract_think_content(text)
            if engine is not None and cache_active:
                from lib.storage.db_io import save_llm_call
                save_llm_call(
                    model=model, schema_name=_CHAT_SCHEMA_NAME,
                    input_text=cache_key_input, output_json=text,
                    label=label, engine=engine,
                )
            in_tok, out_tok = _extract_usage(resp)
            _record_llm_call(
                label=label, model=model,
                schema_name=_CHAT_SCHEMA_NAME,
                input_tokens=in_tok, output_tokens=out_tok,
                cached=False,
            )
            return text

        except Exception as e:
            last_exc = e
            if is_context_length_error(e):
                raise PromptTooLargeError(
                    f"prompt exceeds model context window (model={model}) : {e}"
                ) from e
            if not is_retriable(e):
                raise
            if attempt == max_retries - 1:
                break
            delay = _compute_delay(attempt, retry_after=extract_retry_after(e))
            sys.stderr.write(
                f"[llm_chat {label}] {type(e).__name__} "
                f"attempt {attempt + 1}/{max_retries} ; retrying in {delay}s\n"
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


class _ImageAnalysis(BaseModel):
    """One-field carrier so the vision call returns plain text to the caller."""
    text: str


def analyze_image(
    image_url: str,
    prompt: str,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    label: str = "analyze_image",
    **kwargs: Any,
) -> str:
    """Analyze one image with the LLM and return free text. Vision sibling of :func:`llm_chat`.

    The simplest possible "look at this picture and tell me about it" call:
    pass a ``data:image/...;base64,...`` URL (or an https image URL) plus a
    prompt, get the model's text back. Use it for a logo, a chart, a photo, a
    table-as-image, any region a text parser can't read.

    Goes through :func:`llm_parse` (a one-field schema under the hood) so it
    inherits the retry, token-limit handling and transparent cache: the same
    ``(model, image, prompt)`` returns the previous answer without a network
    call. ``model=None`` resolves to ``$MODEL_CHAT`` which is vision-capable in
    this series (gpt-4o-mini / gpt-4.1). Extra ``kwargs`` pass through to
    ``llm_parse`` (``engine=``, ``cache=``, ``max_retries=``).

    The structured-output fallback is disabled because the responses-API image
    content shape does not survive a ``chat.completions`` retry.
    """
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": image_url},
    ]
    result = llm_parse(
        input=[{"role": "user", "content": content}],
        text_format=_ImageAnalysis,
        client=client,
        model=model,
        label=label,
        fallback_to_chat_completions=False,
        **kwargs,
    )
    return result.text


__all__ = [
    "PromptTooLargeError",
    "llm_parse",
    "llm_chat",
    "analyze_image",
    "default_client",
    "default_model",
    "is_retriable",
    "is_context_length_error",
    "is_json_schema_unsupported",
    "extract_retry_after",
]
