"""Risk-based staffing: required FTE at demand percentiles (P50/P75/P90).

A plan forecasts a single (P50) demand, so staffing only to that point
under-serves volatile weeks. This derives a demand band from the forecast's
historical error — a coefficient of variation (CV) — and reports the required FTE
if you staffed to a higher percentile of demand. The required FTE per percentile
is computed by re-running the existing capacity engine with volume scaled by the
percentile's demand multiplier (so all the Erlang/shrinkage/rollup logic is reused
unchanged); this module only provides the statistics and the result shaping.
"""
from __future__ import annotations

from typing import Any, Optional

# One-sided normal z-scores for upper demand percentiles.
Z_SCORES = {"p50": 0.0, "p75": 0.674, "p80": 0.842, "p90": 1.282, "p95": 1.645}
DEFAULT_PERCENTILES = ["p50", "p75", "p90"]
# Used when no forecast-error history is available.
DEFAULT_CV = 0.10


def cv_from_mape(mape_pct: Optional[float]) -> Optional[float]:
    """Use realized MAPE (a %) as a proxy for the demand coefficient of variation."""
    try:
        v = float(mape_pct)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v / 100.0


def demand_multiplier(cv: float, pct: str) -> float:
    """Volume scale factor for a percentile: 1 + z·CV (clamped non-negative)."""
    z = Z_SCORES.get(pct, 0.0)
    return max(0.0, 1.0 + z * max(0.0, float(cv or 0.0)))


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "-", "—"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _find_row(upper_rows: list[dict], *needles: str) -> Optional[dict]:
    wanted = [n.lower() for n in needles]
    for row in upper_rows or []:
        if isinstance(row, dict) and all(n in str(row.get("metric") or "").lower() for n in wanted):
            return row
    return None


def required_from_upper(upper_rows: list[dict]) -> dict:
    """Derive per-week required FTE from the upper rows.

    over_under = supply − required  =>  required = supply − over_under.
    """
    ou = _find_row(upper_rows, "over", "mtp") or _find_row(upper_rows, "over/under")
    sup = _find_row(upper_rows, "projected supply")
    if not ou:
        return {"weeks": [], "required": []}
    weeks = sorted([k for k in ou.keys() if k != "metric"], key=lambda k: str(k))
    required = []
    for w in weeks:
        ouv = _num(ou.get(w)) or 0.0
        supv = _num(sup.get(w)) if sup else None
        if supv is None:
            required.append(None)
        else:
            required.append(max(0.0, supv - ouv))
    return {"weeks": weeks, "required": required}


def summarize(required: list) -> dict:
    vals = [r for r in required if isinstance(r, (int, float))]
    if not vals:
        return {"avg": None, "peak": None}
    return {"avg": round(sum(vals) / len(vals), 1), "peak": round(max(vals), 1)}
