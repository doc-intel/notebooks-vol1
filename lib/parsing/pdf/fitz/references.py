"""cross_ref_df: extract body-text mentions of figures, tables, annexes.

Engine-agnostic helper hoisted to `parsing/pdf/_shared.py` (it only reads
`line_df`). Re-exported here so existing imports
`from lib.parsing.pdf.fitz.references import build_cross_ref_df` keep working.

`object_registry` (the symmetric *target* table) is fitz-specific because it
relies on `line_df`'s line start, so it stays in `fitz/objects.py`. Azure's
equivalent lives in `azure_layout/objects.py` and uses Azure paragraph roles.
"""
from lib.parsing.pdf._shared import (
    REFERENCE_PATTERNS,
    build_cross_ref_df,
)

__all__ = ["REFERENCE_PATTERNS", "build_cross_ref_df"]
