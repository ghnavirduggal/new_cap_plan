"""Group an already-computed scope balance by a custom dimension (Phase 2).

Pure / dimension-agnostic: it takes the per-scope rows the capacity engine
already produces (``workforce_preview``'s ``scope_balance``, each enriched with a
``dimensions`` map) and aggregates them by the value of one chosen dimension.
The capacity math is unchanged — this only re-buckets finished results, so it
can never alter a staffing number. See docs/FLEXIBLE_DIMENSIONS_DESIGN.md.
"""
from __future__ import annotations

from typing import Any

_UNASSIGNED = "(unassigned)"

# Numeric measures summed per bucket, mapped to their scope_balance row keys.
_MEASURES = {
    "required_fte": "required_fte_est",
    "supply_fte": "supply_fte_est",
    "gap_fte": "gap_fte",
    "shortfall_fte": "shortfall_fte",
    "surplus_fte": "surplus_fte",
}


def _num(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f else 0.0  # drop NaN


def rollup_by_dimension(rows: Any, dim_key: str) -> dict:
    """Aggregate scope_balance rows by ``row['dimensions'][dim_key]``.

    Rows missing a value for the dimension fall into an "(unassigned)" bucket.
    Returns ``{"dimension": key, "groups": [...], "scope_count": N}`` where each
    group has the dimension value, a scope count, and summed FTE measures
    (rounded). Groups are sorted by shortfall desc then value.
    """
    key = str(dim_key or "").strip().lower()
    buckets: dict[str, dict] = {}
    total_scopes = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        total_scopes += 1
        dims = row.get("dimensions") or {}
        value = str(dims.get(key) or "").strip() if isinstance(dims, dict) else ""
        label = value or _UNASSIGNED
        bucket = buckets.get(label)
        if bucket is None:
            bucket = {"value": label, "scope_count": 0}
            for measure in _MEASURES:
                bucket[measure] = 0.0
            buckets[label] = bucket
        bucket["scope_count"] += 1
        for measure, src in _MEASURES.items():
            bucket[measure] += _num(row.get(src))

    groups = list(buckets.values())
    for bucket in groups:
        for measure in _MEASURES:
            bucket[measure] = round(bucket[measure], 2)
    groups.sort(key=lambda g: (-g.get("shortfall_fte", 0.0), g.get("value") == _UNASSIGNED, g.get("value", "")))
    return {"dimension": key, "groups": groups, "scope_count": total_scopes}
