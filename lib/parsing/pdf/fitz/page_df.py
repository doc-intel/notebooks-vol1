"""page_df: aggregate line_df into one row per page.

Engine-agnostic helper hoisted to `parsing/pdf/_shared.py` (it only reads
`line_df`). Re-exported here so existing imports
`from lib.parsing.pdf.fitz.page_df import build_page_df` keep working.
"""
from lib.parsing.pdf._shared import build_page_df

__all__ = ["build_page_df"]
