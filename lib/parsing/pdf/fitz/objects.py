"""Build a registry of named objects (figures, tables, equations, annexes)."""
from __future__ import annotations

import pandas as pd

from lib.parsing.pdf._shared import OBJECT_PATTERNS  # shared with Azure


def build_object_registry(line_df: pd.DataFrame) -> pd.DataFrame:
    """Extract figure/table/annex captions from line_df.

    Each line is matched against caption patterns anchored at the start of
    the line. The first occurrence of each (object_type, object_id) pair is
    kept (typically the actual caption); later body-text mentions like
    "as shown in Figure 2" don't match because the regex is anchored.

    Returns columns: object_type, object_id, title, page_num, line_num.
    """
    rows = []
    for _, line in line_df.iterrows():
        text = str(line.get("text", ""))
        for pattern, obj_type in OBJECT_PATTERNS:
            m = pattern.match(text)
            if m:
                rows.append({
                    "object_type": obj_type,
                    "object_id": m.group(1),
                    "title": text.strip(),
                    "page_num": int(line["page_num"]),
                    "line_num": int(line["line_num"]),
                })
                break
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(
            subset=["object_type", "object_id"], keep="first"
        )
    return df.reset_index(drop=True)
