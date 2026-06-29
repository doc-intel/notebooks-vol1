"""Detect the producing software of a PDF from its metadata."""
from __future__ import annotations

import fitz


def detect_source_software(doc: fitz.Document) -> str:
    """Classify the producing software using Creator/Producer metadata.

    Branches are ordered by parsing-difficulty bucket (easiest first):
      1. Office authoring tools (Word, PowerPoint, LibreOffice)
      2. Document processors (LaTeX, Pandoc)
      3. Design and publishing tools (InDesign)
      4. Print pipelines and recompressors (browsers, Ghostscript)
      5. Scanner software (OCR mandatory)

    Returns a string label; unknown producers fall back to "unknown_source".
    """
    meta = doc.metadata or {}
    creator = (meta.get("creator") or "").lower()
    producer = (meta.get("producer") or "").lower()
    combined = f"{creator} {producer}"

    # Bucket 1 — office authoring tools
    if "microsoft" in combined and "word" in combined:
        return "word_export"
    # Acrobat PDFMaker exports from Word ("Acrobat PDFMaker 23 for Word")
    if "pdfmaker" in combined and "word" in combined:
        return "word_export"
    if "powerpoint" in combined:
        return "powerpoint_export"
    if any(s in combined for s in ("libreoffice", "openoffice")):
        return "libreoffice_export"

    # Bucket 2 — document processors
    if any(s in combined for s in ("pdftex", "xetex", "luatex")):
        return "latex_export"
    if "pandoc" in combined:
        return "pandoc_export"

    # Bucket 3 — design and publishing tools
    if "indesign" in combined:
        return "indesign_export"

    # Bucket 4 — print pipelines and recompressors
    if "ghostscript" in combined:
        return "ghostscript"
    if any(s in combined for s in ("chrome", "safari", "firefox")):
        return "browser_print"

    # Bucket 5 — scanner software (OCR mandatory)
    if any(s in combined for s in (
        "kofax", "abbyy", "adobe scan", "scansnap", "camscanner",
    )):
        return "scanner_software"

    return "unknown_source"
