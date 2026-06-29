"""Unified SQLite storage — one file, many tables, one connection.

⚠ **Legacy module.** New code should use `lib.storage.db_io` instead.
That module rewrites every save/load as a modular per-brick function (no
`pdf_path` derivation, no implicit brick logic) on top of a SQLAlchemy
engine so the same code runs on SQLite / PostgreSQL / MySQL.

This module is kept around so Article 7 keeps rendering during the migration.
The schema is shared (both modules talk to the same `output/storage.sqlite`),
so data persisted via either path is readable by the other.

Migration map (old → new) :

    sqlite_store.embed_texts_cached(...)     → db_io.{load,save}_embeddings_batch
    sqlite_store.cached_llm_parse(...)       → db_io.{load,save}_llm_call
    sqlite_store.save_parsed(pdf_path, ...)  → db_io.save_line_df / save_toc_df / ...
                                                  + db_io.save_document
    sqlite_store.load_parsed(pdf_path)       → db_io.load_line_df / load_toc_df / ...
    sqlite_store.save_parsed_question(...)   → db_io.save_parsed_question_row
    sqlite_store.save_retrieved_pages(...)   → db_io.save_retrieved_pages_df
    sqlite_store.save_answer(...)            → db_io.save_answer_row

Started 2026-05-27 with the embeddings table and grown through phases ; full
4-phase coverage delivered on 2026-05-28. Refactored into the modular
`db_io` design on 2026-05-30 after the user re-emphasised the modularity
constraint and SQLAlchemy portability.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np

from lib.storage.paths import output_root

# Default DB lives at ``<output_root>/storage.sqlite`` — resolved via
# :py:func:`lib.storage.paths.output_root` (env var ``DOCINTEL_OUTPUT_DIR``,
# fallback ``~/.docintel/output``). NOT anchored to the repo any more so an
# in-process CLI bridge running with cwd=rag doesn't litter the source tree.
def _default_db_path() -> Path:
    return output_root() / "storage.sqlite"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS embeddings (
    text_hash  TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    text       TEXT NOT NULL,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_hash   TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    input       TEXT NOT NULL,
    output_json TEXT NOT NULL,
    label       TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_label ON llm_calls(label);
CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON llm_calls(model);

-- Parsing tables --
-- doc_id : path-derived, e.g. 'paper/1706.03762v7' for data/paper/1706.03762v7.pdf
-- All parsing tables are keyed on doc_id ; deleting from documents would
-- cascade (no FK constraint enforced, but delete_doc_parsing() removes rows
-- from every table for a given doc_id at once).

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    parsed_at       REAL NOT NULL,
    parsing_summary TEXT
);

CREATE TABLE IF NOT EXISTS lines (
    doc_id          TEXT NOT NULL,
    page_num        INTEGER NOT NULL,
    line_num        INTEGER NOT NULL,
    text            TEXT,
    x0              REAL,
    y0              REAL,
    x1              REAL,
    y1              REAL,
    character_count INTEGER,
    column_position TEXT,
    parsing_method  TEXT,
    PRIMARY KEY (doc_id, page_num, line_num)
);
CREATE INDEX IF NOT EXISTS idx_lines_doc ON lines(doc_id);

CREATE TABLE IF NOT EXISTS pages (
    doc_id         TEXT NOT NULL,
    page_num       INTEGER NOT NULL,
    text           TEXT,
    n_columns      INTEGER,
    parsing_method TEXT,
    PRIMARY KEY (doc_id, page_num)
);

CREATE TABLE IF NOT EXISTS toc (
    doc_id         TEXT NOT NULL,
    toc_idx        INTEGER NOT NULL,
    level          INTEGER,
    parent_idx     INTEGER,
    title          TEXT,
    start_page     INTEGER,
    end_page       INTEGER,
    start_y        REAL,
    breadcrumb     TEXT,
    parsing_method TEXT,
    PRIMARY KEY (doc_id, toc_idx)
);

CREATE TABLE IF NOT EXISTS images (
    doc_id         TEXT NOT NULL,
    page_num       INTEGER NOT NULL,
    image_num      INTEGER NOT NULL,
    x0             REAL,
    y0             REAL,
    x1             REAL,
    y1             REAL,
    width_px       INTEGER,
    height_px      INTEGER,
    image_hash     REAL,
    parsing_method TEXT,
    PRIMARY KEY (doc_id, page_num, image_num)
);

CREATE TABLE IF NOT EXISTS objects (
    doc_id         TEXT NOT NULL,
    object_type    TEXT NOT NULL,
    object_id      INTEGER NOT NULL,
    title          TEXT,
    page_num       INTEGER,
    line_num       INTEGER,
    parsing_method TEXT,
    PRIMARY KEY (doc_id, object_type, object_id)
);

CREATE TABLE IF NOT EXISTS cross_refs (
    doc_id         TEXT NOT NULL,
    rowid_         INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_type       TEXT NOT NULL,
    ref_id         INTEGER,
    page_num       INTEGER,
    line_num       INTEGER,
    context        TEXT,
    parsing_method TEXT
);
CREATE INDEX IF NOT EXISTS idx_cross_refs_doc ON cross_refs(doc_id);

-- Per-question pipeline artefacts --
-- question_id : sha1(question_text)[:16], stable across docs ; same text
-- asked of different docs reuses the questions row but creates separate
-- parsed_questions / retrieved_pages / answers rows per (question_id, doc_id).

CREATE TABLE IF NOT EXISTS questions (
    question_id   TEXT PRIMARY KEY,
    question_text TEXT NOT NULL,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS parsed_questions (
    question_id  TEXT NOT NULL,
    doc_id       TEXT NOT NULL,
    schema_name  TEXT NOT NULL,
    parsed_json  TEXT NOT NULL,
    created_at   REAL NOT NULL,
    PRIMARY KEY (question_id, doc_id)
);

CREATE TABLE IF NOT EXISTS retrieved_pages (
    question_id      TEXT NOT NULL,
    doc_id           TEXT NOT NULL,
    page_num         INTEGER NOT NULL,
    rank             INTEGER,
    score            REAL,
    similarity       REAL,
    match_count      INTEGER,
    matched_keywords TEXT,
    methods          TEXT,
    text             TEXT,
    PRIMARY KEY (question_id, doc_id, page_num)
);
CREATE INDEX IF NOT EXISTS idx_retrieved_pages_q ON retrieved_pages(question_id);

CREATE TABLE IF NOT EXISTS answers (
    question_id  TEXT NOT NULL,
    doc_id       TEXT NOT NULL,
    schema_name  TEXT NOT NULL,
    answer_json  TEXT NOT NULL,
    model_used   TEXT,
    created_at   REAL NOT NULL,
    PRIMARY KEY (question_id, doc_id)
);

CREATE TABLE IF NOT EXISTS generation_meta (
    question_id  TEXT NOT NULL,
    doc_id       TEXT NOT NULL,
    meta_json    TEXT NOT NULL,
    created_at   REAL NOT NULL,
    PRIMARY KEY (question_id, doc_id)
);
"""

# DataFrame name in the `parsed` dict → SQL table name. Used by save_parsed /
# load_parsed to round-trip the parser output.
_PARSING_TABLE_MAP: tuple[tuple[str, str], ...] = (
    ("line_df",         "lines"),
    ("page_df",         "pages"),
    ("toc_df",          "toc"),
    ("image_df",        "images"),
    ("object_registry", "objects"),
    ("cross_ref_df",    "cross_refs"),
)


def default_db_path() -> Path:
    """Return the canonical storage.sqlite path under
    :py:func:`lib.storage.paths.output_root`."""
    return _default_db_path()


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open the storage connection, bootstrapping the schema if needed."""
    path = Path(db_path) if db_path is not None else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA_SQL)
    return conn


def _hash_text(text: str, model: str) -> str:
    """Cache key : sha256(model + text). Same text under a different model
    gets a distinct entry because the vectors differ."""
    return hashlib.sha256(f"{model}|||{text}".encode("utf-8")).hexdigest()


def get_embedding_cached(
    client,
    text: str,
    *,
    model: str,
    db_path: Path | str | None = None,
) -> np.ndarray:
    """Return the embedding for `text` under `model`, hitting disk cache when
    possible. Misses call the API and persist the result immediately."""
    h = _hash_text(text, model)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT vector FROM embeddings WHERE text_hash = ?", (h,)
        ).fetchone()
        if row is not None:
            return np.frombuffer(row[0], dtype=np.float32)
        resp = client.embeddings.create(model=model, input=text)
        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        conn.execute(
            "INSERT INTO embeddings "
            "(text_hash, model, text, vector, dim, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (h, model, text, vec.tobytes(), len(vec), time.time()),
        )
        conn.commit()
        return vec
    finally:
        conn.close()


def embed_texts_cached(
    client,
    texts: list[str],
    *,
    model: str,
    db_path: Path | str | None = None,
    batch_size: int = 100,
) -> list[np.ndarray]:
    """Batch lookup-or-compute. Returns vectors in input order.

    Hits the cache for known texts in a single SQL query, batches the API
    calls for misses (OpenAI embeddings accepts up to ~2k inputs per call
    ; we keep batches conservative at 100 to keep SQL parameter lists sane).
    """
    if not texts:
        return []
    hashes = [_hash_text(t, model) for t in texts]
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(hashes))
        rows = conn.execute(
            f"SELECT text_hash, vector FROM embeddings WHERE text_hash IN ({placeholders})",
            hashes,
        ).fetchall()
        cached: dict[str, np.ndarray] = {
            h: np.frombuffer(v, dtype=np.float32) for h, v in rows
        }
        miss_indices = [i for i, h in enumerate(hashes) if h not in cached]
        for start in range(0, len(miss_indices), batch_size):
            chunk = miss_indices[start : start + batch_size]
            miss_texts = [texts[i] for i in chunk]
            resp = client.embeddings.create(model=model, input=miss_texts)
            insert_rows = []
            for idx, item in zip(chunk, resp.data):
                vec = np.asarray(item.embedding, dtype=np.float32)
                cached[hashes[idx]] = vec
                insert_rows.append(
                    (hashes[idx], model, texts[idx],
                     vec.tobytes(), len(vec), time.time())
                )
            conn.executemany(
                "INSERT OR IGNORE INTO embeddings "
                "(text_hash, model, text, vector, dim, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                insert_rows,
            )
            conn.commit()
        return [cached[h] for h in hashes]
    finally:
        conn.close()


def embeddings_count(*, model: str | None = None,
                     db_path: Path | str | None = None) -> int:
    """Diagnostic helper : how many embeddings the cache currently holds
    (optionally filtered to one model)."""
    conn = _connect(db_path)
    try:
        if model:
            row = conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return int(row[0])
    finally:
        conn.close()


# --- LLM call cache --------------------------------------------------------

def _hash_llm_call(model: str, schema_name: str, input_str: str) -> str:
    """Cache key : first 16 hex chars of sha256(model + schema + input).
    Schema-aware so a prompt change with a different schema reissues the
    call. The 16-char truncation matches the legacy `intermediate.cached_llm_parse`
    convention so a one-shot migration of existing JSON files into SQLite
    delivers cache hits to this helper without any reprocessing."""
    return hashlib.sha256(
        f"{model}|||{schema_name}|||{input_str}".encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Retry / backoff layer (inspired by ia_package.generation.chat_rate_limit).
# Wraps the OpenAI-compatible client.responses.parse(...) call so an
# APITimeoutError / APIConnectionError / 429 / 5xx on the first attempt does
# not abandon the whole article. The cache lives outside the loop : once the
# call returns 200, the row is committed and future renders skip the API.
# ---------------------------------------------------------------------------

_RETRIABLE_EXCEPTION_NAMES = {
    "APIConnectionError", "APITimeoutError", "ConnectError",
    "RateLimitError", "APIStatusError",
    "ServiceUnavailableError", "InternalServerError",
}
_RETRIABLE_MSG_PATTERNS = (
    "rate limit", "rate_limit", "too many requests",
    "server error", "service unavailable", "overloaded",
    "timeout", "connection", "temporarily",
    "please retry after",
)
_RETRIABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def _is_retriable(e: Exception) -> bool:
    if type(e).__name__ in _RETRIABLE_EXCEPTION_NAMES:
        return True
    msg = str(e).lower()
    import re as _re
    code = _re.search(r"\b(\d{3})\b", msg)
    if code and int(code.group(1)) in _RETRIABLE_HTTP_CODES:
        return True
    return any(p in msg for p in _RETRIABLE_MSG_PATTERNS)


def _backoff_delay(attempt: int, *, base: float = 2.0,
                   factor: float = 2.0, cap: float = 60.0,
                   jitter: float = 0.3) -> float:
    import random as _random
    raw = min(base * (factor ** attempt), cap)
    return round(raw * (1.0 + jitter * _random.random()), 2)


def _call_with_retry(fn, *, max_retries: int = 6, label: str = ""):
    """Invoke `fn()` with exponential backoff on retriable errors.
    Non-retriable exceptions propagate immediately. Raises the last
    exception after `max_retries` attempts."""
    import sys as _sys
    import time as _time
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_retriable(e):
                raise
            if attempt == max_retries - 1:
                break
            delay = _backoff_delay(attempt)
            _sys.stderr.write(
                f"  [cached_llm_parse{f' label={label!r}' if label else ''}] "
                f"{type(e).__name__} attempt {attempt+1}/{max_retries} ; "
                f"retrying in {delay}s\n"
            )
            _time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def cached_llm_parse(
    client,
    *,
    model: str,
    input: str,
    text_format,
    label: str = "default",
    db_path: Path | str | None = None,
    max_retries: int = 6,
):
    """Drop-in replacement for `client.responses.parse(...).output_parsed`
    that hits a SQLite cache first. Same signature minus `pdf_path` (the
    cache is corpus-wide, not per-PDF). Misses call the API and persist
    the parsed response as JSON.

    Transient errors on the API call (APITimeoutError, APIConnectionError,
    429, 5xx) trigger exponential-backoff retry with jitter up to
    `max_retries` attempts. Once any attempt succeeds the row is committed
    to the cache and future renders skip the API entirely.

    `text_format` should be a Pydantic `BaseModel` subclass."""
    key = _hash_llm_call(model, text_format.__name__, input)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT output_json FROM llm_calls WHERE call_hash = ?", (key,)
        ).fetchone()
        if row is not None:
            return text_format.model_validate_json(row[0])
        parsed = _call_with_retry(
            lambda: client.responses.parse(
                model=model, input=input, text_format=text_format,
            ).output_parsed,
            max_retries=max_retries,
            label=label,
        )
        conn.execute(
            "INSERT OR REPLACE INTO llm_calls "
            "(call_hash, model, schema_name, input, output_json, label, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, model, text_format.__name__, input,
             parsed.model_dump_json(), label, time.time()),
        )
        conn.commit()
        return parsed
    finally:
        conn.close()


def llm_calls_count(*, label: str | None = None,
                    db_path: Path | str | None = None) -> int:
    """Diagnostic helper : how many LLM calls are cached
    (optionally filtered to one call site label)."""
    conn = _connect(db_path)
    try:
        if label:
            row = conn.execute(
                "SELECT COUNT(*) FROM llm_calls WHERE label = ?", (label,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()
        return int(row[0])
    finally:
        conn.close()


# --- Parsing tables (documents / lines / pages / toc / images / ...) -------

def doc_id_from_path(pdf_path: "Path | str") -> str:
    """Stable, path-derived doc_id : everything between `data/` and the file
    stem, joined by '/'. Examples :

        data/paper/1706.03762v7.pdf  -> 'paper/1706.03762v7'
        data/insurance/abc.pdf       -> 'insurance/abc'

    Falls back to the bare stem when the path has no `data/` segment."""
    p = Path(pdf_path)
    parts = p.parts
    try:
        i = parts.index("data")
    except ValueError:
        return p.stem
    return "/".join(list(parts[i + 1 : -1]) + [p.stem])


def portable_source_path(pdf_path: "Path | str") -> str:
    """Repo-relative path with forward slashes, suitable for cross-machine
    portability. Stores 'data/paper/1706.03762v7.pdf' instead of an absolute
    Windows / Unix path so the SQLite store stays portable when the repo is
    moved, shared, or opened on a different machine."""
    p = Path(pdf_path)
    parts = p.parts
    try:
        i = parts.index("data")
    except ValueError:
        return p.name
    return "/".join(parts[i:])


def save_parsed(
    pdf_path: "Path | str",
    parsed: dict,
    *,
    db_path: "Path | str | None" = None,
) -> str:
    """Persist a parsed dict (line_df, page_df, toc_df, image_df,
    object_registry, cross_ref_df, parsing_summary) to SQLite. Returns the
    doc_id used. Idempotent : re-saving the same PDF replaces its rows."""
    import json as _json
    doc_id = doc_id_from_path(pdf_path)
    conn = _connect(db_path)
    try:
        summary = parsed.get("parsing_summary")
        conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, source_path, parsed_at, parsing_summary) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, portable_source_path(pdf_path), time.time(),
             _json.dumps(summary, default=str) if summary else None),
        )
        for df_name, sql_name in _PARSING_TABLE_MAP:
            df = parsed.get(df_name)
            if df is None or len(df) == 0:
                conn.execute(f"DELETE FROM {sql_name} WHERE doc_id = ?", (doc_id,))
                continue
            conn.execute(f"DELETE FROM {sql_name} WHERE doc_id = ?", (doc_id,))
            df_to_insert = df.copy()
            df_to_insert["doc_id"] = doc_id
            df_to_insert.to_sql(sql_name, conn, if_exists="append", index=False)
        conn.commit()
        return doc_id
    finally:
        conn.close()


def load_parsed(
    pdf_path: "Path | str",
    *,
    db_path: "Path | str | None" = None,
) -> dict | None:
    """Reload what `save_parsed` wrote. Returns None when the document has
    not been saved yet, so callers can fall back to running the parser."""
    import json as _json
    import pandas as _pd
    doc_id = doc_id_from_path(pdf_path)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT parsing_summary FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        summary = _json.loads(row[0]) if row[0] else None
        out: dict = {"parsing_summary": summary}
        for df_name, sql_name in _PARSING_TABLE_MAP:
            df = _pd.read_sql(
                f"SELECT * FROM {sql_name} WHERE doc_id = ?",
                conn, params=(doc_id,),
            )
            if len(df) == 0:
                out[df_name] = None
                continue
            # Drop the doc_id column (it's an internal foreign key, not part of
            # the original parser output) and the rowid_ surrogate if present.
            drop_cols = [c for c in ("doc_id", "rowid_") if c in df.columns]
            out[df_name] = df.drop(columns=drop_cols) if drop_cols else df
        return out
    finally:
        conn.close()


def parsed_docs_count(*, db_path: "Path | str | None" = None) -> int:
    """How many documents have parsing rows in the store."""
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0])
    finally:
        conn.close()


# --- Per-question artefacts (Phase 4) --------------------------------------

def question_id_from_text(question: str) -> str:
    """sha1(question_text)[:16] : stable across docs and runs. Use this as the
    PK on questions / parsed_questions / retrieved_pages / answers."""
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


def _ensure_question(conn: "sqlite3.Connection", question: str) -> str:
    qid = question_id_from_text(question)
    conn.execute(
        "INSERT OR IGNORE INTO questions (question_id, question_text, created_at) "
        "VALUES (?, ?, ?)",
        (qid, question, time.time()),
    )
    return qid


def save_parsed_question(
    pdf_path: "Path | str",
    question: str,
    parsed_question,
    *,
    db_path: "Path | str | None" = None,
) -> str:
    """Persist the ParsedQuestion (any Pydantic BaseModel) per (question, doc)."""
    from pydantic import BaseModel
    if not isinstance(parsed_question, BaseModel):
        raise TypeError(f"parsed_question must be a Pydantic BaseModel, got {type(parsed_question)}")
    doc_id = doc_id_from_path(pdf_path)
    conn = _connect(db_path)
    try:
        qid = _ensure_question(conn, question)
        conn.execute(
            "INSERT OR REPLACE INTO parsed_questions "
            "(question_id, doc_id, schema_name, parsed_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (qid, doc_id, type(parsed_question).__name__,
             parsed_question.model_dump_json(), time.time()),
        )
        conn.commit()
        return qid
    finally:
        conn.close()


def load_parsed_question(
    pdf_path: "Path | str",
    question: str,
    model_cls,
    *,
    db_path: "Path | str | None" = None,
):
    """Reload the ParsedQuestion for (question, doc). Returns None on miss."""
    doc_id = doc_id_from_path(pdf_path)
    qid = question_id_from_text(question)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT parsed_json FROM parsed_questions "
            "WHERE question_id = ? AND doc_id = ?",
            (qid, doc_id),
        ).fetchone()
        return model_cls.model_validate_json(row[0]) if row else None
    finally:
        conn.close()


# Columns the retrieved_pages table accepts ; anything beyond these is dropped.
_RETRIEVED_PAGES_COLS = (
    "question_id", "doc_id", "page_num", "rank", "score", "similarity",
    "match_count", "matched_keywords", "methods", "text",
)


def save_retrieved_pages(
    pdf_path: "Path | str",
    question: str,
    retrieved_pages_df,
    *,
    db_path: "Path | str | None" = None,
) -> str:
    """Persist a retrieved-pages DataFrame per (question, doc).

    The DataFrame may carry any subset of the columns the table defines :
    page_num is required, the rest (rank, score, similarity, match_count,
    matched_keywords, methods, text) are optional and stored as-is. Columns
    not in the schema are dropped silently.
    """
    import pandas as _pd
    doc_id = doc_id_from_path(pdf_path)
    conn = _connect(db_path)
    try:
        qid = _ensure_question(conn, question)
        conn.execute(
            "DELETE FROM retrieved_pages WHERE question_id = ? AND doc_id = ?",
            (qid, doc_id),
        )
        df = retrieved_pages_df.copy()
        df["question_id"] = qid
        df["doc_id"] = doc_id
        # Drop any columns the table doesn't know about (e.g. local helpers
        # like embedding vectors, fancy similarity matrices).
        keep = [c for c in df.columns if c in _RETRIEVED_PAGES_COLS]
        df = df[keep]
        # Normalize list-valued cells to comma-joined strings (matched_keywords)
        if "matched_keywords" in df.columns:
            df["matched_keywords"] = df["matched_keywords"].apply(
                lambda v: ", ".join(v) if isinstance(v, list) else v
            )
        df.to_sql("retrieved_pages", conn, if_exists="append", index=False)
        conn.commit()
        return qid
    finally:
        conn.close()


def load_retrieved_pages(
    pdf_path: "Path | str",
    question: str,
    *,
    db_path: "Path | str | None" = None,
):
    """Reload the retrieved-pages DataFrame for (question, doc). None on miss."""
    import pandas as _pd
    doc_id = doc_id_from_path(pdf_path)
    qid = question_id_from_text(question)
    conn = _connect(db_path)
    try:
        df = _pd.read_sql(
            "SELECT * FROM retrieved_pages WHERE question_id = ? AND doc_id = ? "
            "ORDER BY rank IS NULL, rank, page_num",
            conn, params=(qid, doc_id),
        )
        if len(df) == 0:
            return None
        return df.drop(columns=[c for c in ("question_id", "doc_id") if c in df.columns])
    finally:
        conn.close()


def save_answer(
    pdf_path: "Path | str",
    question: str,
    answer,
    *,
    model_used: str | None = None,
    db_path: "Path | str | None" = None,
) -> str:
    """Persist the GenerationResult (any Pydantic BaseModel) per (question, doc)."""
    from pydantic import BaseModel
    if not isinstance(answer, BaseModel):
        raise TypeError(f"answer must be a Pydantic BaseModel, got {type(answer)}")
    doc_id = doc_id_from_path(pdf_path)
    conn = _connect(db_path)
    try:
        qid = _ensure_question(conn, question)
        conn.execute(
            "INSERT OR REPLACE INTO answers "
            "(question_id, doc_id, schema_name, answer_json, model_used, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (qid, doc_id, type(answer).__name__,
             answer.model_dump_json(), model_used, time.time()),
        )
        conn.commit()
        return qid
    finally:
        conn.close()


def load_answer(
    pdf_path: "Path | str",
    question: str,
    model_cls,
    *,
    db_path: "Path | str | None" = None,
):
    """Reload the answer for (question, doc). Returns None on miss."""
    doc_id = doc_id_from_path(pdf_path)
    qid = question_id_from_text(question)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT answer_json FROM answers WHERE question_id = ? AND doc_id = ?",
            (qid, doc_id),
        ).fetchone()
        return model_cls.model_validate_json(row[0]) if row else None
    finally:
        conn.close()


def save_generation_meta(
    pdf_path: "Path | str",
    question: str,
    meta: dict,
    *,
    db_path: "Path | str | None" = None,
) -> str:
    """Persist GenerationResult.meta side-channel (schema_used, fragments_applied,
    raw_response, ...) per (question, doc)."""
    import json as _json
    doc_id = doc_id_from_path(pdf_path)
    conn = _connect(db_path)
    try:
        qid = _ensure_question(conn, question)
        conn.execute(
            "INSERT OR REPLACE INTO generation_meta "
            "(question_id, doc_id, meta_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (qid, doc_id, _json.dumps(meta, default=str), time.time()),
        )
        conn.commit()
        return qid
    finally:
        conn.close()


def load_generation_meta(
    pdf_path: "Path | str",
    question: str,
    *,
    db_path: "Path | str | None" = None,
) -> dict | None:
    import json as _json
    doc_id = doc_id_from_path(pdf_path)
    qid = question_id_from_text(question)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT meta_json FROM generation_meta WHERE question_id = ? AND doc_id = ?",
            (qid, doc_id),
        ).fetchone()
        return _json.loads(row[0]) if row else None
    finally:
        conn.close()


def questions_count(*, db_path: "Path | str | None" = None) -> int:
    conn = _connect(db_path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0])
    finally:
        conn.close()


__all__ = [
    "default_db_path",
    "get_embedding_cached",
    "embed_texts_cached",
    "embeddings_count",
    "cached_llm_parse",
    "llm_calls_count",
    "doc_id_from_path",
    "portable_source_path",
    "save_parsed",
    "load_parsed",
    "parsed_docs_count",
    "question_id_from_text",
    "save_parsed_question",
    "load_parsed_question",
    "save_retrieved_pages",
    "load_retrieved_pages",
    "save_answer",
    "load_answer",
    "save_generation_meta",
    "load_generation_meta",
    "questions_count",
]
