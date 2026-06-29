"""Intermediate-results storage — save once per PDF, reload instead of recomputing.

Convention is path-driven : every PDF at `data/<subdir>/<stem>.pdf` stores its
intermediate results at `output/<subdir>/<stem>/`, mirroring the input layout.
From the PDF path alone, every downstream step knows where to read the cache.

    output/<subdir>/<stem>/
        parsing/
            line_df.parquet       one .parquet per parsing table
            page_df.parquet
            image_df.parquet
            toc_df.parquet
            span_df.parquet
            object_registry.parquet
            cross_ref_df.parquet
            parsing_summary.json
        questions/
            <question_slug>/
                parsed_question.json
                retrieved_pages.parquet
                answer.json

Parquet is the default tabular format since 2026-05-30 (replaced .xlsx, which
choked on PDF NUL characters and required openpyxl). Parquet handles full
Unicode, is faster to read/write, and writes a smaller file. Article 22 swaps
in SQLite for the production version. Legacy .xlsx caches written by older
versions are still read by load_parsed (parquet first, xlsx fallback) so V1
notebooks don't re-parse.

A missing cache returns None — callers do `load_parsed(p) or parse_pdf(p)`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from lib.storage.paths import output_root as _output_root_func

# Back-compat re-export : older callers imported the bare ``OUTPUT_ROOT``
# constant. New code should call ``output_root()`` so the destination honours
# the env var / explicit override. The constant is recomputed lazily on
# attribute access via ``__getattr__`` below so the value tracks env changes.
def __getattr__(name: str):
    if name == "OUTPUT_ROOT":
        return _output_root_func()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Parsing-engine names that may appear as a ``_<engine>`` suffix on cache
# files (e.g. ``line_df_docling.parquet``). When reading, files whose stem
# ends in any of these suffixes are recognised as engine-specific. New
# engines should be added here so the loader doesn't misread them as a
# generic table name. Save uses :func:`_engine_suffix` directly and does
# not consult this set.
_KNOWN_ENGINE_SUFFIXES: tuple[str, ...] = (
    "fitz",
    "docling",
    "azure_layout",
    "easyocr",
    "vision_gpt4o",
)


def intermediate_dir(
    pdf_path: str | Path,
    *,
    output_dir: str | os.PathLike[str] | Path | None = None,
    output_root: str | os.PathLike[str] | Path | None = None,  # legacy alias
) -> Path:
    """Resolve ``<output_root>/<subdir>/<stem>/`` for a PDF.

    The cache root is resolved via :py:func:`lib.storage.paths.output_root`
    (``output_dir`` arg → ``$DOCINTEL_OUTPUT_DIR`` → ``~/.docintel/output``),
    so the cache lives OUTSIDE the repo by default. When the PDF path
    contains a ``data/`` segment, the relative layout after ``data/`` is
    preserved under the cache root (``data/paper/foo.pdf`` →
    ``<root>/paper/foo/``) so siblings stay grouped. PDFs without a ``data/``
    segment land at ``<root>/<stem>/``.

    The legacy ``output_root=`` keyword is still accepted for back-compat.
    """
    p = Path(pdf_path).resolve() if Path(pdf_path).is_absolute() else Path(pdf_path)
    explicit = output_dir if output_dir is not None else output_root
    root = _output_root_func(explicit)
    parts = p.parts
    try:
        i = parts.index("data")
    except ValueError:
        return root / p.stem
    return root.joinpath(*parts[i + 1:-1], p.stem)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# --- Brick 1 : parsing -----------------------------------------------------

def _write_table(df: pd.DataFrame, target_without_ext: Path) -> Path:
    """Write a DataFrame to parquet (default since 2026-05-30).

    Parquet handles full Unicode (including NUL characters present in some
    PDF-parsed text), is faster than xlsx, produces a smaller file, and does
    not need the openpyxl runtime dep beyond what pandas requires.
    """
    target = target_without_ext.with_suffix(".parquet")
    df.to_parquet(target, index=False)
    return target


def _read_table(folder: Path, name: str) -> pd.DataFrame | None:
    """Read a cached table. Tries parquet first (current), xlsx fallback (legacy V1 caches)."""
    parquet_path = folder / f"{name}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    xlsx_path = folder / f"{name}.xlsx"
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    return None


def _engine_suffix(engine: str | None) -> str:
    """Return the cache-file suffix for an engine.

    fitz is the historical default and stays unsuffixed: line_df.parquet,
    toc_df.parquet, parsing_summary.json. Every consumer that lived before
    multi-engine was introduced keeps working without change.

    Non-default engines (docling) get suffixed: line_df_docling.parquet,
    etc., so they cohabit with fitz on disk and the user can compare.
    """
    if engine is None or engine == "" or engine == "fitz":
        return ""
    return f"_{engine}"


def save_parsed(
    pdf_path: str | Path,
    parsed: dict[str, Any],
    *,
    engine: str | None = None,
) -> Path:
    """Write the parsing tables (.parquet) and `parsing_summary` (.json).

    The parsed dict is **introspected** : every value that is a non-empty
    pandas DataFrame is persisted as ``<key><suffix>.parquet``. The special
    key ``parsing_summary`` is serialised to ``parsing_summary<suffix>.json``.
    Other keys are ignored.

    This format-agnostic shape means the cache supports every parser the
    package ships : PDF (``line_df``, ``page_df``, ``image_df``, ``toc_df``,
    ``span_df``, ``object_registry``, ``cross_ref_df``), DOCX
    (``paragraph_df``, ``runs_df``, ``section_df``, ``table_df``, ``header_df``,
    ``footer_df``, ``footnote_df``), XLSX (``cell_df``, ``sheet_df``,
    ``dataset_df``, ``merged_df``, ``named_range_df``), PPTX (``slide_df``,
    ``notes_df``), mail (``message_df``, ``thread_df``), and any new table a
    future parser adds — no hardcoded allow-list to maintain.

    When ``engine`` is given, each table is suffixed with the engine name
    (e.g. ``line_df_fitz.parquet``, ``line_df_docling.parquet``) so the two
    engines' artefacts cohabit on disk. The legacy unsuffixed layout
    (``line_df.parquet``) is still produced when engine is None for back
    compatibility with V1 notebook caches.

    Returns the ``parsing/`` folder so the caller can log the location.
    """
    out = intermediate_dir(pdf_path) / "parsing"
    out.mkdir(parents=True, exist_ok=True)
    suffix = _engine_suffix(engine)
    for name, value in parsed.items():
        if name == "parsing_summary":
            continue
        if not isinstance(value, pd.DataFrame):
            continue
        if value.empty:
            continue
        _write_table(value, out / f"{name}{suffix}")
    summary = parsed.get("parsing_summary")
    if summary is not None:
        _write_json(out / f"parsing_summary{suffix}.json", summary)
    return out


def _strip_known_engine_suffix(stem: str) -> tuple[str, str | None]:
    """Split a cache filename stem into ``(table_name, engine)``.

    ``"line_df_docling"`` → ``("line_df", "docling")``.
    ``"line_df"`` → ``("line_df", None)``.
    Only strips suffixes listed in :data:`_KNOWN_ENGINE_SUFFIXES` so a table
    named ``cell_df`` (no engine) is not misread as ``cell_df`` + nonsense.
    """
    for eng in _KNOWN_ENGINE_SUFFIXES:
        marker = f"_{eng}"
        if stem.endswith(marker):
            return stem[: -len(marker)], eng
    return stem, None


def load_parsed(
    pdf_path: str | Path,
    *,
    engine: str | None = None,
) -> dict[str, Any] | None:
    """Reload what `save_parsed` wrote. Returns None if the cache is missing.

    The set of tables to read is **discovered from the filesystem** (every
    ``*.parquet`` in the ``parsing/`` folder), not from a hardcoded list, so
    every parser format is supported : PDF, DOCX, XLSX, PPTX, mail, future
    parsers. The set of engine suffixes recognised on read is
    :data:`_KNOWN_ENGINE_SUFFIXES`.

    Lookup priority per table :

    1. ``<name>_<engine>.parquet`` (when ``engine`` is set, preferred for
       that engine).
    2. ``<name>.parquet`` (engine-agnostic, V1 notebook compatible).
    3. ``<name>.xlsx`` (legacy V1 cache pre-parquet, only when no parquet
       sibling exists).
    """
    folder = intermediate_dir(pdf_path) / "parsing"
    if not folder.exists():
        return None
    suffix = _engine_suffix(engine)
    parsed: dict[str, Any] = {}

    # Phase 1 : preferred engine-specific files (only when an engine is set).
    if suffix:
        for path in folder.glob(f"*{suffix}.parquet"):
            name, eng = _strip_known_engine_suffix(path.stem)
            if eng == engine:
                parsed[name] = pd.read_parquet(path)

    # Phase 2 : every unsuffixed parquet that we haven't already loaded for
    # this engine.
    for path in folder.glob("*.parquet"):
        name, eng = _strip_known_engine_suffix(path.stem)
        if eng is None and name not in parsed:
            parsed[name] = pd.read_parquet(path)

    # Phase 3 : legacy V1 xlsx fallback for tables that don't have a parquet
    # sibling (very old caches).
    for path in folder.glob("*.xlsx"):
        if path.stem not in parsed:
            parsed[path.stem] = pd.read_excel(path)

    if not parsed:
        return None  # no usable cache

    # Re-establish the "text is non-null string" invariant after parquet
    # round-trip (pyarrow + pandas can turn empty strings into NaN). Applies
    # whenever a table has a ``text`` column — covers PDF ``line_df``, XLSX
    # ``cell_df``, DOCX ``runs_df``, etc.
    for name, df in parsed.items():
        if isinstance(df, pd.DataFrame) and not df.empty and "text" in df.columns:
            df["text"] = df["text"].fillna("").astype(str)

    # Engine-specific summary first, fall back to legacy unsuffixed.
    summary_path = (
        folder / f"parsing_summary{suffix}.json"
        if suffix and (folder / f"parsing_summary{suffix}.json").exists()
        else folder / "parsing_summary.json"
    )
    parsed["parsing_summary"] = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else None
    )
    return parsed


# --- Bricks 2 / 3 / 4 : per-question artefacts -----------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _question_slug(question: str, *, max_len: int = 40) -> str:
    """Short, deterministic folder name : readable slug + 8-hex hash."""
    cleaned = _SLUG_RE.sub("_", question.lower()).strip("_")
    short = cleaned[:max_len].rstrip("_")
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
    return f"{short}_{digest}" if short else digest


def question_dir(pdf_path: str | Path, question: str) -> Path:
    return intermediate_dir(pdf_path) / "questions" / _question_slug(question)


def save_parsed_question(
    pdf_path: str | Path, question: str, parsed_question: BaseModel
) -> Path:
    out = question_dir(pdf_path, question)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "parsed_question.json"
    target.write_text(parsed_question.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_parsed_question(
    pdf_path: str | Path, question: str, model_cls: type[BaseModel]
) -> BaseModel | None:
    path = question_dir(pdf_path, question) / "parsed_question.json"
    if not path.exists():
        return None
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def save_retrieved_pages(
    pdf_path: str | Path, question: str, retrieved_pages_df: pd.DataFrame
) -> Path:
    """Save the page-level retrieval result (page_num + diagnostics).

    The filtered_line_df is intentionally not cached — re-derive it from
    `line_df` + page_num on the fly. Keeps the cache small and matches the
    brick contract (retrieval = page selection).
    """
    out = question_dir(pdf_path, question)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "retrieved_pages.parquet"
    retrieved_pages_df.to_parquet(target, index=False)
    return target


def load_retrieved_pages(pdf_path: str | Path, question: str) -> pd.DataFrame | None:
    folder = question_dir(pdf_path, question)
    path = folder / "retrieved_pages.parquet"
    if path.exists():
        return pd.read_parquet(path)
    legacy = folder / "retrieved_pages.xlsx"
    if legacy.exists():
        return pd.read_excel(legacy)
    return None


def save_answer(pdf_path: str | Path, question: str, answer: BaseModel) -> Path:
    out = question_dir(pdf_path, question)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "answer.json"
    target.write_text(answer.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_answer(
    pdf_path: str | Path, question: str, model_cls: type[BaseModel]
) -> BaseModel | None:
    path = question_dir(pdf_path, question) / "answer.json"
    if not path.exists():
        return None
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def save_generation_meta(
    pdf_path: str | Path, question: str, meta: dict
) -> Path:
    """Persist the GenerationResult.meta side-channel (schema_used,
    fragments_applied, template_version, raw_response) alongside answer.json."""
    out = question_dir(pdf_path, question)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "meta.json"
    target.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return target


def load_generation_meta(pdf_path: str | Path, question: str) -> dict | None:
    path = question_dir(pdf_path, question) / "meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --- V5 commands : translate / summarize / classify / redact -----------------
#
# Each command has its own sub-folder under <cache>/<command>/ so caches are
# isolated and can be cleared independently. The key is a small slug capturing
# the parameters that change the output (target_lang for translate, length+
# focus for summarize, etc.) so different parameter combinations get distinct
# cache files.


def _safe_slug(s: str, *, max_len: int = 40) -> str:
    cleaned = _SLUG_RE.sub("_", str(s).lower()).strip("_")
    short = cleaned[:max_len].rstrip("_")
    digest = hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:6]
    return f"{short}_{digest}" if short else digest






# Summary cache : <cache>/summaries/<length>__<focus>.json
def save_summary(
    pdf_path: str | Path, length: str, focus: str, payload: BaseModel
) -> Path:
    out = intermediate_dir(pdf_path) / "summaries"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{length}__{focus}.json"
    target.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_summary(
    pdf_path: str | Path, length: str, focus: str, model_cls: type[BaseModel]
) -> BaseModel | None:
    path = intermediate_dir(pdf_path) / "summaries" / f"{length}__{focus}.json"
    if not path.exists():
        return None
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


# Classification cache : <cache>/classification/<model_slug>.json
def save_classification(
    pdf_path: str | Path, model_used: str, payload: BaseModel
) -> Path:
    out = intermediate_dir(pdf_path) / "classification"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{_safe_slug(model_used)}.json"
    target.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_classification(
    pdf_path: str | Path, model_used: str, model_cls: type[BaseModel]
) -> BaseModel | None:
    path = intermediate_dir(pdf_path) / "classification" / f"{_safe_slug(model_used)}.json"
    if not path.exists():
        return None
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


# Redaction-detection cache : <cache>/redactions/<categories_hash>.json
def save_redaction_detection(
    pdf_path: str | Path, categories_key: str, payload: BaseModel
) -> Path:
    out = intermediate_dir(pdf_path) / "redactions"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"detect__{_safe_slug(categories_key)}.json"
    target.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_redaction_detection(
    pdf_path: str | Path, categories_key: str, model_cls: type[BaseModel]
) -> BaseModel | None:
    path = (
        intermediate_dir(pdf_path)
        / "redactions"
        / f"detect__{_safe_slug(categories_key)}.json"
    )
    if not path.exists():
        return None
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


# --- Invalidation cascade ---------------------------------------------------
#
# When parse.run is called with force=True (the source PDF changed or the user
# explicitly wants a fresh parse), every cache that depended on the old parse
# is stale. invalidate_dependents() removes them. Each command should also
# accept its own force flag so the user can invalidate just one cache.


def invalidate_dependents(pdf_path: str | Path) -> dict[str, int]:
    """Remove every cache that depended on a previous parse of this PDF.

    Returns a count of removed entries per sub-folder, useful for the audit
    log on force=True parse.run calls. The parsing/ folder itself is NOT
    removed — the caller is about to overwrite it.

    Sub-folders cleared: questions/, translations/, summaries/, classification/,
    redactions/, llm_cache/, embeddings/ (if present).
    """
    root = intermediate_dir(pdf_path)
    removed: dict[str, int] = {}
    for sub in (
        "questions",
        "translations",
        "summaries",
        "classification",
        "redactions",
        "llm_cache",
        "embeddings",
    ):
        folder = root / sub
        if not folder.exists():
            continue
        n = sum(1 for _ in folder.rglob("*") if _.is_file())
        import shutil as _shutil
        _shutil.rmtree(folder)
        removed[sub] = n
    return removed


def invalidate_question(pdf_path: str | Path, question: str) -> int:
    """Remove the per-question cache (retrieval + answer + meta) for one question.

    Useful when only the retrieval params change (top_k, method) so we don't
    want to invalidate the whole parse cache, just the question's stale outputs.
    Returns the number of files removed.
    """
    folder = question_dir(pdf_path, question)
    if not folder.exists():
        return 0
    n = sum(1 for _ in folder.rglob("*") if _.is_file())
    import shutil as _shutil
    _shutil.rmtree(folder)
    return n


# --- LLM call cache --------------------------------------------------------
# Per repo policy : before every paid API call (LLM `responses.parse` /
# `chat.completions.parse` / embedding), check disk first. Store responses as
# JSON keyed by (model, input, schema) hash so re-runs are free.

def _llm_cache_path(pdf_path: str | Path, label: str, key: str) -> Path:
    return intermediate_dir(pdf_path) / "llm_cache" / f"{label}__{key}.json"


def cached_llm_parse(
    client,
    *,
    pdf_path: str | Path,
    model: str,
    input: str,
    text_format: type[BaseModel],
    label: str = "default",
) -> BaseModel:
    """Drop-in replacement for `client.responses.parse(...).output_parsed`
    with a disk cache. Cache key = sha256(model + schema_name + input)."""
    key_material = f"{model}|||{text_format.__name__}|||{input}"
    key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:16]
    cache_path = _llm_cache_path(pdf_path, label, key)

    if cache_path.exists():
        return text_format.model_validate_json(cache_path.read_text(encoding="utf-8"))

    parsed = client.responses.parse(
        model=model, input=input, text_format=text_format,
    ).output_parsed
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
    return parsed


# --- Embeddings cache ------------------------------------------------------
# Embedding API calls are the second-largest LLM cost after generation. We
# cache embedded DataFrames as parquet (numpy arrays serialise efficiently)
# keyed by the model deployment name so a model change invalidates the cache.

def _embedded_df_path(pdf_path: str | Path, df_name: str, model: str) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", model)
    return intermediate_dir(pdf_path) / "embeddings" / f"{df_name}__{safe_model}.parquet"


def save_embedded_df(
    pdf_path: str | Path,
    df_name: str,
    df: pd.DataFrame,
    *,
    model: str,
) -> Path:
    """Persist a DataFrame whose `embedding` column holds vectors (np.ndarray)
    so the next run can reload instead of re-calling the embedding API."""
    target = _embedded_df_path(pdf_path, df_name, model)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Convert embedding column to list for parquet compatibility (some engines
    # don't serialise ndarray cells directly).
    to_save = df.copy()
    if "embedding" in to_save.columns:
        to_save["embedding"] = to_save["embedding"].apply(
            lambda v: v.tolist() if hasattr(v, "tolist") else list(v)
        )
    to_save.to_parquet(target, index=False)
    return target


def load_embedded_df(
    pdf_path: str | Path,
    df_name: str,
    *,
    model: str,
) -> pd.DataFrame | None:
    """Reload what `save_embedded_df` wrote. Returns None if the cache is
    missing or was written for a different embedding model."""
    import numpy as np  # local import to keep module-level imports tight
    path = _embedded_df_path(pdf_path, df_name, model)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "embedding" in df.columns:
        df["embedding"] = df["embedding"].apply(lambda v: np.asarray(v, dtype=np.float32))
    return df


# --- Corpus / grouping context (V2 T_v2.corpus_context_auto_description) -----
#
# A CorpusContext is the deterministic aggregated view of a corpus or grouping
# subtree. The producer (lib.corpus.build_corpus_context) writes it once
# at build time ; UI / pipeline consumers read it many times. Persistence
# layout :
#
#     <output_root>/corpus/<corpus_id>/corpus_context.json
#     <output_root>/corpus/<corpus_id>/groupings/<grouping_id>/grouping_context.json


def _corpus_root(corpus_id: str) -> Path:
    return _output_root_func() / "corpus" / corpus_id


def save_corpus_context(corpus_id: str, context: BaseModel) -> Path:
    out = _corpus_root(corpus_id)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "corpus_context.json"
    target.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_corpus_context(corpus_id: str):
    from lib.core.schemas.corpus_context import CorpusContext
    path = _corpus_root(corpus_id) / "corpus_context.json"
    if not path.exists():
        return None
    return CorpusContext.model_validate_json(path.read_text(encoding="utf-8"))


def save_grouping_context(corpus_id: str, grouping_id: str, context: BaseModel) -> Path:
    out = _corpus_root(corpus_id) / "groupings" / grouping_id
    out.mkdir(parents=True, exist_ok=True)
    target = out / "grouping_context.json"
    target.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_grouping_context(corpus_id: str, grouping_id: str):
    from lib.core.schemas.corpus_context import CorpusContext
    path = _corpus_root(corpus_id) / "groupings" / grouping_id / "grouping_context.json"
    if not path.exists():
        return None
    return CorpusContext.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "OUTPUT_ROOT",
    "intermediate_dir",
    "question_dir",
    "save_parsed",
    "load_parsed",
    "save_parsed_question",
    "load_parsed_question",
    "save_retrieved_pages",
    "load_retrieved_pages",
    "save_answer",
    "load_answer",
    "save_embedded_df",
    "load_embedded_df",
    "cached_llm_parse",
    "save_corpus_context",
    "load_corpus_context",
    "save_grouping_context",
    "load_grouping_context",
]