"""Semantic-zone fields for the parsing brick's `parsing_summary`.

`parsing_summary` (the doc-level synthesis returned by every PDF engine
alongside the per-row tables) ships four zones of deterministic mechanics :
identity, content, structure, routing. This module adds a fifth, **semantic**
zone : doc_type (resume / contract / paper / ...), typical_fields (per
doc_type), and a short LLM-written ``summary`` left None until a summary
builder is wired.

`build_semantic_fields(line_df, page_df, *, pdf_path) -> dict` is the cheap
heuristic that runs at the tail of the parsing brick. Filename + first-page
text drive the doc_type ; the typical_fields are looked up in the table
below. The result is merged into the parsing_summary dict the parser is
already building. No separate artefact, no separate JSON file.

Producers
---------

* PDF : called from `parse_pdf_fitz`, `parse_pdf_azure_layout`,
  `parse_pdf_docling`.
* DOCX / XLSX / PPTX / mail : analogous helpers land alongside the parsers
  in their own subpackages when Volume 2 wires them.

Consumers
---------

* ``lib.question.parse_question`` accepts ``parsing_summary=`` and
  pulls the semantic-zone subset into the system prompt.
* ``lib.pipeline.qa.pdf.pdf_qa`` reads
  ``parsed["parsing_summary"]`` and forwards it to the question parser.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typical-fields table
# ---------------------------------------------------------------------------
# Maps ``doc_type`` to the field names questions about a document of this
# type are most likely to target. Lives in code (not Pydantic) so the table
# can grow without a schema migration. Lookup is one dict access ; no LLM
# involved.
TYPICAL_FIELDS_BY_TYPE: dict[str, list[str]] = {
    "resume":             ["name", "email", "phone", "experience", "education",
                           "skills", "languages"],
    "contract":           ["policyholder", "insured", "premium", "deductible",
                           "coverage", "exclusions", "renewal_date"],
    "contract_amendment": ["effective_date", "diff_from_previous_version"],
    "invoice":            ["total_amount", "currency", "due_date", "vendor",
                           "invoice_number"],
    "academic_paper":     ["title", "authors", "abstract", "method", "results"],
    "memo":               ["author", "date", "subject", "audience"],
    "annual_report":      ["fiscal_year", "revenue", "net_income", "auditor"],
    "unknown":            [],
}


# ---------------------------------------------------------------------------
# doc_type heuristics — filename then first-page text
# ---------------------------------------------------------------------------
# A custom token boundary ``(?<![a-z0-9])`` + ``(?![a-z0-9])`` is used in
# place of ``\b`` because ``_`` is a word character in Python regex — so
# ``\bcv\b`` does NOT match in ``jane_smith_cv``. Treating underscores,
# dashes, dots, and digits as separators gives the right behaviour on
# typical filename stems.
_TOKEN_LEFT  = r"(?<![a-z0-9])"
_TOKEN_RIGHT = r"(?![a-z0-9])"

# Order matters : amendment must be checked before the generic ``contract``
# pattern can swallow an ``avenant_contrat_2024`` stem.
_FILENAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(_TOKEN_LEFT + r"(avenant|amendment)" + _TOKEN_RIGHT),
     "contract_amendment"),
    (re.compile(_TOKEN_LEFT + r"(cv|resume|curriculum)" + _TOKEN_RIGHT),
     "resume"),
    (re.compile(_TOKEN_LEFT + r"(invoice|facture)" + _TOKEN_RIGHT),
     "invoice"),
    (re.compile(_TOKEN_LEFT + r"(contract|policy|police)" + _TOKEN_RIGHT),
     "contract"),
    (re.compile(_TOKEN_LEFT + r"memo" + _TOKEN_RIGHT),
     "memo"),
    (re.compile(_TOKEN_LEFT + r"(annual[\s_-]?report|10[-_]?k|20[-_]?f)" + _TOKEN_RIGHT),
     "annual_report"),
    # arXiv preprint IDs match the academic-paper signature.
    (re.compile(r"\d{4}\.\d{4,5}(v\d+)?"),
     "academic_paper"),
]

# First-page text signatures. First match wins.
_FIRST_PAGE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcurriculum\s+vitae\b|\bresume\b|\bprofile\s+summary\b",
                re.IGNORECASE),                                "resume"),
    (re.compile(r"\babstract\b\s*\n|\barxiv\b",
                re.IGNORECASE),                                "academic_paper"),
    (re.compile(r"\binvoice\s+(no|number|#)|\bbill\s+to\b",
                re.IGNORECASE),                                "invoice"),
    (re.compile(r"\binsurance\s+policy|\bpolicyholder\b|\bpremium\b",
                re.IGNORECASE),                                "contract"),
    (re.compile(r"\bavenant\b|\bamendment\s+to\s+the\b",
                re.IGNORECASE),                                "contract_amendment"),
    (re.compile(r"\bannual\s+report\b|\bfiscal\s+year\b",
                re.IGNORECASE),                                "annual_report"),
]


def detect_doc_type(pdf_path: Path, first_page_text: str) -> str:
    """Return a doc_type from filename then first-page text. ``"unknown"`` falls
    out when nothing matches.
    """
    stem = pdf_path.stem.lower()
    for pattern, label in _FILENAME_PATTERNS:
        if pattern.search(stem):
            return label
    for pattern, label in _FIRST_PAGE_PATTERNS:
        if pattern.search(first_page_text):
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_semantic_fields(
    line_df: pd.DataFrame,
    page_df: pd.DataFrame,
    *,
    pdf_path: str | Path,
) -> dict:
    """Assemble the semantic-zone fields for ``parsing_summary``.

    Returns a dict with three keys :

    * ``doc_type`` — coarse document family (``resume`` / ``contract`` /
      ``academic_paper`` / ``invoice`` / ``memo`` / ``annual_report`` /
      ``contract_amendment`` / ``unknown``).
    * ``typical_fields`` — field names a question about a document of this
      type is most likely to target. Looked up in
      :data:`TYPICAL_FIELDS_BY_TYPE` from the resolved doc_type.
    * ``summary`` — short LLM-written paragraph describing the document.
      Always ``None`` from this deterministic builder ; a future summary
      builder (one LLM call against the first one or two pages) populates
      it. Kept as a key so consumers can rely on its presence.

    The caller merges this dict into the parsing_summary it is building.
    """
    pdf_path = Path(pdf_path)
    # First-page text : pull every line on page 1, join with spaces. Cheap
    # and avoids re-opening the PDF.
    if line_df is not None and not line_df.empty and "page_num" in line_df.columns:
        first_page = line_df[line_df["page_num"] == 1]
        first_page_text = " ".join(first_page["text"].astype(str).tolist())
    else:
        first_page_text = ""

    doc_type = detect_doc_type(pdf_path, first_page_text)
    return {
        "doc_type": doc_type,
        "typical_fields": list(TYPICAL_FIELDS_BY_TYPE.get(doc_type, [])),
        "summary": None,
    }


# ---------------------------------------------------------------------------
# LLM summary builder — the only LLM-derived field of the semantic zone
# ---------------------------------------------------------------------------
# Kept separate from ``build_semantic_fields`` (which stays deterministic and
# offline) so a parse that wants the ``summary`` opts into one LLM call while
# the heuristic doc_type / typical_fields path keeps costing nothing. The
# parsers call this when ``summarize=True`` and merge the result into
# ``parsing_summary["summary"]``.

class DocSummary(BaseModel):
    """Structured output for the document-summary LLM call."""

    summary: str


# First N pages fed to the summarizer. A document-level blurb only needs the
# opening : title page, abstract / profile, first section. Short docs (CV,
# invoice, memo) fit entirely ; long docs are described from their head plus
# the table of contents, which carries the rest of the structure.
_SUMMARY_MAX_PAGES = 2
# Cap the TOC titles and the assembled text so the prompt stays small and the
# call stays cheap regardless of document size.
_SUMMARY_MAX_TOC_ENTRIES = 40
_SUMMARY_MAX_CHARS = 6000

# Canonical default lives in the prompt catalogue (id ``document_summary``) so
# the app's prompt editor can surface and override it (e.g. "render as a Markdown
# table"). Callers pass their own ``prompt`` / ``project_id`` to override; None
# falls back to this default. Single source of truth.
from lib.prompts import get_default as _get_default

_SUMMARY_SYSTEM = _get_default("document_summary").default


def _summary_input_text(
    line_df: pd.DataFrame,
    toc_df: pd.DataFrame | None,
    max_pages: int,
) -> str:
    """Assemble the summarizer input : first ``max_pages`` pages plus the TOC.

    The table of contents (when present) is prepended as a compact map of the
    document's structure ; the first pages carry the title, subject, and the
    fields a short document spells out. Returns an empty string when there is
    nothing to feed.
    """
    parts: list[str] = []

    if toc_df is not None and not toc_df.empty and "title" in toc_df.columns:
        titles = [str(t).strip() for t in toc_df["title"].tolist() if str(t).strip()]
        if titles:
            parts.append(
                "TABLE OF CONTENTS:\n" + "\n".join(titles[:_SUMMARY_MAX_TOC_ENTRIES])
            )

    if (
        line_df is not None
        and not line_df.empty
        and "page_num" in line_df.columns
        and "text" in line_df.columns
    ):
        head = line_df[line_df["page_num"] <= max_pages]
        page_text = " ".join(head["text"].astype(str).tolist()).strip()
        if page_text:
            parts.append(f"FIRST {max_pages} PAGE(S):\n{page_text}")

    return "\n\n".join(parts).strip()[:_SUMMARY_MAX_CHARS]


def build_summary(
    line_df: pd.DataFrame,
    *,
    toc_df: pd.DataFrame | None = None,
    client=None,
    model: str | None = None,
    max_pages: int = _SUMMARY_MAX_PAGES,
    prompt: str | None = None,
    project_id: str | None = None,
) -> str | None:
    """Return a short factual document summary, or ``None`` when unavailable.

    One LLM call against the first ``max_pages`` pages (plus the table of
    contents when present), run-once per document and cached. The call is
    wrapped so a missing LLM client, an offline run, or a provider error
    degrades to ``None`` rather than breaking the parsing brick. ``client`` /
    ``model`` default to the wrapper's resolution from environment ; they are
    NOT the engine client (the Azure layout client, say), they are the chat
    LLM client.

    ``prompt`` overrides the system instruction for this call (a user-edited
    prompt, e.g. "render as a Markdown table"); ``None`` resolves the project
    override for ``document_summary`` (``project_id``) then the catalogue
    default. The system prompt is always a parameter, never hard-coded.
    """
    text = _summary_input_text(line_df, toc_df, max_pages)
    if not text:
        return None
    from lib.prompts import resolve_prompt

    system_prompt = prompt or resolve_prompt(project_id, "document_summary")
    try:
        from lib.core.llm import llm_parse

        result = llm_parse(
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            text_format=DocSummary,
            client=client,
            model=model,
            label="parsing.doc_summary",
            cache=True,
        )
        summary = (result.summary or "").strip()
        return summary or None
    except Exception as exc:  # noqa: BLE001 — never let summarization break a parse
        logger.warning("doc summary build failed, leaving summary=None: %s", exc)
        return None


__all__ = [
    "TYPICAL_FIELDS_BY_TYPE",
    "detect_doc_type",
    "build_semantic_fields",
    "DocSummary",
    "build_summary",
]
