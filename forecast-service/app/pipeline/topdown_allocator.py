"""Top-down target allocation.

The plan rolls bottom-up (child scopes sum to a BA total). This is the inverse:
given an org/BA-level target (a hiring budget, an FTE cap, a headcount number),
spread it down to the child scopes by a chosen basis — proportional to required
FTE, current shortfall, supply, or split equally. Whole-number allocations use
the largest-remainder method so the integer parts sum exactly to the target.

Pure helper — it only reshapes the per-scope balance the plan already computes.
"""
from __future__ import annotations

import math
from typing import Any, Optional

# basis -> the scope-balance field that drives each scope's share.
_BASIS_FIELDS = {
    "required": "required_fte_est",
    "supply": "supply_fte_est",
    "shortfall": "shortfall_fte",
    "surplus": "surplus_fte",
    "gap": "gap_fte",
}
VALID_BASES = sorted(set(_BASIS_FIELDS) | {"equal"})


def _num(v: Any) -> float:
    try:
        f = float(v)
        return f if f == f else 0.0
    except (TypeError, ValueError):
        return 0.0


def _largest_remainder(weights: list[float], total_int: int) -> list[int]:
    """Round fractional shares to integers summing exactly to total_int."""
    wsum = sum(weights)
    if wsum <= 0 or total_int <= 0:
        return [0] * len(weights)
    raw = [total_int * w / wsum for w in weights]
    floors = [int(math.floor(x)) for x in raw]
    remainder = total_int - sum(floors)
    # Hand out the leftover to the largest fractional parts.
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for k in range(remainder):
        floors[order[k % len(order)]] += 1
    return floors


def allocate(
    scope_balance: list[dict],
    target: float,
    basis: str = "required",
    *,
    integer: bool = False,
) -> dict:
    """Spread `target` across the scopes by `basis`. Returns per-scope allocations."""
    basis = (basis or "required").strip().lower()
    if basis not in VALID_BASES:
        basis = "required"
    rows = [s for s in (scope_balance or []) if isinstance(s, dict) and str(s.get("scope") or "").strip()]
    if not rows:
        return {"status": "no_data", "basis": basis, "target": float(target or 0.0), "allocations": []}

    if basis == "equal":
        weights = [1.0 for _ in rows]
    else:
        field = _BASIS_FIELDS[basis]
        weights = [max(0.0, _num(s.get(field))) for s in rows]
        if sum(weights) <= 0:
            # Driver is all-zero (e.g. no shortfall anywhere) — fall back to equal.
            weights = [1.0 for _ in rows]
            basis = f"{basis} (no signal — split equally)"

    target = float(target or 0.0)
    wsum = sum(weights)
    fractional = [target * w / wsum for w in weights]
    int_alloc = _largest_remainder(weights, int(round(target))) if integer else None

    allocations = []
    for i, s in enumerate(rows):
        alloc = int_alloc[i] if integer else round(fractional[i], 2)
        allocations.append(
            {
                "scope": str(s.get("scope")),
                "ba": s.get("ba"),
                "sba": s.get("sba"),
                "ch": s.get("ch"),
                "site": s.get("site"),
                "weight": round(weights[i], 3),
                "share_pct": round(weights[i] / wsum * 100.0, 1) if wsum > 0 else 0.0,
                "allocation": alloc,
                "required_fte_est": round(_num(s.get("required_fte_est")), 2),
                "supply_fte_est": round(_num(s.get("supply_fte_est")), 2),
            }
        )
    allocations.sort(key=lambda a: a["allocation"], reverse=True)
    allocated_total = sum(a["allocation"] for a in allocations)
    return {
        "status": "ok",
        "basis": basis,
        "target": round(target, 2),
        "integer": integer,
        "allocated_total": round(allocated_total, 2),
        "scope_count": len(allocations),
        "allocations": allocations,
    }
