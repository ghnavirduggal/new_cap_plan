"""Hiring-plan solver.

The plan already computes a weekly required-vs-supply curve (the "FTE Over/Under"
row). This turns new-hire classes from a manual *input* into a recommended
*output*: given the projected shortfall, the training+nesting lead time, and an
optional attrition erosion, it proposes class start weeks and sizes that close the
gap — and flags weeks that physically can't be covered in time.

Deterministic greedy heuristic (not an LP): walk weeks forward; whenever a residual
shortfall remains, schedule a class early enough to be productive that week
(clamped to the first week if the lead time doesn't allow it), credit its
(attrition-eroded) contribution to all later weeks, and continue.
"""
from __future__ import annotations

import math
from typing import Any, Optional


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if not math.isnan(value) else None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "-", "—"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _find_metric_row(upper_rows: list[dict], *needles: str) -> Optional[dict]:
    wanted = [n.lower() for n in needles]
    for row in upper_rows or []:
        if not isinstance(row, dict):
            continue
        metric = str(row.get("metric") or "").strip().lower()
        if all(n in metric for n in wanted):
            return row
    return None


def _week_ids(row: dict) -> list[str]:
    ids = [k for k in row.keys() if k != "metric"]
    # Week column ids are ISO Mondays, which sort chronologically as strings.
    return sorted(ids, key=lambda k: str(k))


def solve_hiring_plan(upper_rows: list[dict], params: Optional[dict] = None) -> dict:
    """Recommend a hiring plan from a plan's weekly upper-summary rows.

    Uses the 'FTE Over/Under vs MTP' row (negative = shortfall). Returns the
    recommended classes, the before/after coverage curves, and a summary.
    """
    params = params or {}
    ramp_weeks = max(0, int(params.get("ramp_weeks", 6) or 0))
    buffer_pct = float(params.get("buffer_pct", 0) or 0)
    attrition_weekly_pct = max(0.0, float(params.get("attrition_weekly_pct", 0) or 0))
    tolerance = float(params.get("tolerance", 0.5) or 0.5)
    max_per_class = params.get("max_per_class")
    max_per_class = int(max_per_class) if max_per_class else None

    ou_row = _find_metric_row(upper_rows, "over", "mtp") or _find_metric_row(upper_rows, "over/under")
    if not ou_row:
        return {"status": "no_data", "reason": "No FTE Over/Under row found.", "classes": [], "weeks": []}

    weeks = _week_ids(ou_row)
    if not weeks:
        return {"status": "no_data", "reason": "No weekly columns found.", "classes": [], "weeks": []}

    n = len(weeks)
    over_under = [(_num(ou_row.get(w)) or 0.0) for w in weeks]
    shortfall = [max(0.0, -v) for v in over_under]

    supply_row = _find_metric_row(upper_rows, "projected supply")
    supply = [(_num(supply_row.get(w)) if supply_row else None) for w in weeks]

    a = attrition_weekly_pct / 100.0
    added = [0.0] * n
    raw_classes: list[dict] = []

    for i in range(n):
        gap = shortfall[i] - added[i]
        if gap <= tolerance:
            continue
        need = gap * (1.0 + buffer_pct / 100.0)
        start_idx = i - ramp_weeks
        uncoverable = start_idx < 0
        start_idx = max(0, start_idx)
        production_idx = start_idx + ramp_weeks
        # Erosion of this class's contribution by week i (0 if it lands after i).
        if production_idx <= i:
            erosion_at_i = (1.0 - a) ** (i - production_idx)
        else:
            erosion_at_i = 1.0  # lands after week i; can't help week i
        grads = math.ceil(need / max(erosion_at_i, 1e-6))
        capped = False
        if max_per_class and grads > max_per_class:
            grads = max_per_class
            capped = True
        if grads <= 0:
            continue
        for j in range(production_idx, n):
            added[j] += grads * ((1.0 - a) ** (j - production_idx))
        raw_classes.append(
            {
                "start_week": weeks[start_idx],
                "production_week": weeks[min(production_idx, n - 1)],
                "grads_needed": int(grads),
                "addresses_week": weeks[i],
                "uncoverable_in_time": bool(uncoverable and production_idx > i),
                "size_capped": capped,
            }
        )

    # Merge classes that start the same week into a single recommended class.
    merged: dict[str, dict] = {}
    for c in raw_classes:
        key = c["start_week"]
        if key in merged:
            m = merged[key]
            m["grads_needed"] += c["grads_needed"]
            m["uncoverable_in_time"] = m["uncoverable_in_time"] or c["uncoverable_in_time"]
            m["size_capped"] = m["size_capped"] or c["size_capped"]
        else:
            merged[key] = dict(c)
    classes = [merged[k] for k in sorted(merged.keys())]

    residual = [max(0.0, shortfall[i] - added[i]) for i in range(n)]
    projected_after = []
    for i in range(n):
        if supply[i] is not None:
            projected_after.append(round(supply[i] + added[i], 1))
        else:
            projected_after.append(None)

    weeks_short_before = sum(1 for s in shortfall if s > tolerance)
    weeks_short_after = sum(1 for r in residual if r > tolerance)
    first_uncoverable = next((c["addresses_week"] for c in classes if c["uncoverable_in_time"]), None)

    return {
        "status": "ok",
        "params": {
            "ramp_weeks": ramp_weeks,
            "buffer_pct": buffer_pct,
            "attrition_weekly_pct": attrition_weekly_pct,
            "max_per_class": max_per_class,
            "tolerance": tolerance,
        },
        "weeks": weeks,
        "shortfall_before": [round(s, 1) for s in shortfall],
        "added_supply": [round(x, 1) for x in added],
        "residual_after": [round(r, 1) for r in residual],
        "supply": [None if s is None else round(s, 1) for s in supply],
        "projected_after": projected_after,
        "classes": classes,
        "summary": {
            "total_grads": int(sum(c["grads_needed"] for c in classes)),
            "num_classes": len(classes),
            "weeks_short_before": weeks_short_before,
            "weeks_short_after": weeks_short_after,
            "peak_shortfall_before": round(max(shortfall) if shortfall else 0.0, 1),
            "peak_shortfall_after": round(max(residual) if residual else 0.0, 1),
            "first_uncoverable_week": first_uncoverable,
        },
    }
