"""Read the descriptive metadata a PDF already declares about itself.

A PDF carries a document-information dictionary (title, author, subject,
keywords, creation/modification dates) written by whatever produced it. fitz
exposes it as ``doc.metadata``. This is the *native* metadata : it is what the
file says about itself, not something inferred from the body text. Reading it
costs nothing and never needs an LLM, so it is the canonical source for the
title / author / dates shown next to a document.

Many real-world PDFs leave these fields blank (LaTeX exports, scans). A blank
field comes back as ``None`` here, never an empty string, so a consumer can tell
"absent" from "present but empty".
"""
from __future__ import annotations

import re

import fitz

# PDF date strings look like "D:20240410211143+02'00'" or "D:20240410211143Z".
# Group out the calendar parts ; the timezone tail is optional and ignored.
_PDF_DATE_RE = re.compile(
    r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?"
)


def _clean(value: str | None) -> str | None:
    """Empty / whitespace-only strings become None ; otherwise stripped text."""
    if not value:
        return None
    stripped = str(value).strip()
    return stripped or None


def parse_pdf_date(value: str | None) -> str | None:
    """Convert a PDF date string to an ISO 8601 string, or None.

    "D:20240410211143Z" -> "2024-04-10T21:11:43". A bare date with no time
    component -> "2024-04-10". Anything unparseable -> None.
    """
    cleaned = _clean(value)
    if not cleaned:
        return None
    match = _PDF_DATE_RE.match(cleaned)
    if not match:
        return None
    year, month, day, hour, minute, second = match.groups()
    date_part = f"{year}-{month or '01'}-{day or '01'}"
    if hour is None:
        return date_part
    return f"{date_part}T{hour}:{minute or '00'}:{second or '00'}"


def read_native_metadata(doc: fitz.Document) -> dict:
    """Project ``doc.metadata`` to a clean, display-ready block.

    Returns a dict with the descriptive fields a documents table wants :
    title, author, subject, keywords, creation_date, mod_date, format. Blank
    fields are ``None``. Technical producer/creator strings stay in the flat
    ``parsing_summary`` (creator_raw / producer_raw) ; this block is the
    human-facing metadata.
    """
    meta = doc.metadata or {}
    return {
        "title": _clean(meta.get("title")),
        "author": _clean(meta.get("author")),
        "subject": _clean(meta.get("subject")),
        "keywords": _clean(meta.get("keywords")),
        "creation_date": parse_pdf_date(meta.get("creationDate")),
        "mod_date": parse_pdf_date(meta.get("modDate")),
        "format": _clean(meta.get("format")),
    }


__all__ = ["read_native_metadata", "parse_pdf_date"]
