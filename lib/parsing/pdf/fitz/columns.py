"""Detect the column layout of each page from line bounding boxes."""
from __future__ import annotations

import pandas as pd


def _cluster_x0(x0_values: list[float], gap_threshold: float) -> list[list[float]]:
    """Group sorted x0 values into clusters separated by gaps."""
    if not x0_values:
        return []
    sorted_x = sorted(x0_values)
    clusters: list[list[float]] = [[sorted_x[0]]]
    for x in sorted_x[1:]:
        if x - clusters[-1][-1] > gap_threshold:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    return clusters


def _significant_clusters(
    clusters: list[list[float]], total: int, min_cluster_fraction: float
) -> list[list[float]]:
    """Keep clusters with at least min_cluster_fraction of total members.

    A real column carries a meaningful share of the page's lines. Outliers
    (a single figure caption indented differently, a stray equation) form
    one-line "clusters" that should not be counted as columns.
    """
    threshold = max(1, int(total * min_cluster_fraction))
    return [c for c in clusters if len(c) >= threshold]


def detect_n_columns(
    x0_values: list[float],
    gap_threshold: float = 80.0,
    min_cluster_fraction: float = 0.10,
) -> int:
    """Number of significant column clusters detected from x0 values.

    A page where every x0 falls in a single populous band is single-column.
    A page with two populous bands (one near x=50, one near x=300) is
    two-column. Single-line outliers don't count as their own column.
    """
    if not x0_values:
        return 1
    clusters = _cluster_x0(x0_values, gap_threshold)
    sig = _significant_clusters(clusters, len(x0_values), min_cluster_fraction)
    return max(1, len(sig))


def detect_columns_per_page(
    line_df: pd.DataFrame,
    gap_threshold: float = 80.0,
    min_cluster_fraction: float = 0.10,
) -> pd.DataFrame:
    """Return one row per page with a ``n_columns`` count."""
    rows = []
    for page_num, sub in line_df.groupby("page_num"):
        rows.append({
            "page_num": int(page_num),
            "n_columns": detect_n_columns(
                sub["x0"].tolist(), gap_threshold, min_cluster_fraction,
            ),
        })
    return pd.DataFrame(rows).sort_values("page_num").reset_index(drop=True)


def assign_column_positions(
    line_df: pd.DataFrame,
    gap_threshold: float = 80.0,
    min_cluster_fraction: float = 0.10,
) -> pd.DataFrame:
    """Add a ``column_position`` field to each line.

    Values:
      - "single": page has a single column.
      - "left" / "right": page has two columns; the line falls in the left
        or right cluster (split at the midpoint between the two cluster
        centres, so figure-caption lines that span columns get assigned to
        whichever side they sit closer to).
      - "multi": page has three or more columns; reliable left/right
        labelling breaks down so we just flag the line as multi.

    The motivation is practical. On invoices, addresses sit in a specific
    half of the page ("the address is in the bottom-left"). Asking the
    retriever or the user to point at "the left column" is much more
    natural than asking for an exact bbox. Two columns is the regime where
    this label is genuinely useful; once a page has three or more columns,
    left/right loses meaning and we mark it as multi.
    """
    out = line_df.copy()
    out["column_position"] = "single"
    # Image-only / empty parses produce a line_df with no rows (and sometimes
    # no columns). Nothing to cluster; return with the column_position field
    # present so the schema stays stable and the groupby below never sees a
    # missing 'page_num'.
    if line_df.empty or "page_num" not in line_df.columns:
        return out
    for _, sub in line_df.groupby("page_num"):
        x0_values = sub["x0"].tolist()
        if not x0_values:
            continue
        clusters = _cluster_x0(x0_values, gap_threshold)
        sig = _significant_clusters(
            clusters, len(x0_values), min_cluster_fraction,
        )
        n_cols = max(1, len(sig))
        if n_cols == 1:
            continue
        if n_cols == 2:
            c1_center = sum(sig[0]) / len(sig[0])
            c2_center = sum(sig[1]) / len(sig[1])
            split = (c1_center + c2_center) / 2
            left_idx = sub.index[sub["x0"] < split]
            right_idx = sub.index[sub["x0"] >= split]
            out.loc[left_idx, "column_position"] = "left"
            out.loc[right_idx, "column_position"] = "right"
        else:
            out.loc[sub.index, "column_position"] = "multi"
    return out
