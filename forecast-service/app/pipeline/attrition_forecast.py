"""Predictive attrition — forward projection of the weekly attrition series.

Attrition was historical-input-only. This projects the weekly attrition rate
forward with Holt's linear (double exponential) smoothing — level + trend, pure
Python, no heavy dependencies — and an empirical prediction band from the
in-sample one-step residuals (widened with the horizon). Very short series fall
back to a recent-average flat line.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# z for a ~P10/P90 band.
_Z = 1.282


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if f == f else None  # drop NaN
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "-", "—"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def holt_forecast(values: list[float], horizon: int, alpha: float = 0.5, beta: float = 0.3) -> tuple[list[float], float]:
    """Holt's linear forecast. Returns (point_forecasts, residual_sigma)."""
    n = len(values)
    if n == 0:
        return [0.0] * horizon, 0.0
    if n == 1:
        return [max(0.0, values[0])] * horizon, 0.0
    level = values[0]
    trend = values[1] - values[0]
    onestep: list[tuple[float, float]] = []
    for t in range(1, n):
        f = level + trend  # one-step-ahead forecast for period t
        onestep.append((values[t], f))
        prev_level = level
        level = alpha * values[t] + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
    resid = [a - f for a, f in onestep]
    if len(resid) >= 2:
        mean_r = sum(resid) / len(resid)
        sigma = (sum((r - mean_r) ** 2 for r in resid) / (len(resid) - 1)) ** 0.5
    else:
        sigma = 0.0
    fc = [max(0.0, level + (h + 1) * trend) for h in range(horizon)]
    return fc, sigma


def _weekly_overall(rows: list[dict]) -> list[tuple[str, float]]:
    """Collapse per-program weekly rows into one overall attrition% per week:
    sum(leavers) / sum(active) when available, else the mean of attrition_pct."""
    by_week: dict[str, dict[str, float]] = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        wk = str(r.get("week") or "").strip()
        if not wk:
            continue
        leavers = _num(r.get("leavers_fte"))
        active = _num(r.get("avg_active_fte"))
        pct = _num(r.get("attrition_pct"))
        b = by_week.setdefault(wk, {"leavers": 0.0, "active": 0.0, "pct_sum": 0.0, "pct_n": 0.0, "has_la": 0.0})
        if leavers is not None and active is not None and active > 0:
            b["leavers"] += leavers
            b["active"] += active
            b["has_la"] = 1.0
        if pct is not None:
            b["pct_sum"] += pct
            b["pct_n"] += 1.0
    series = []
    for wk in sorted(by_week.keys(), key=lambda k: str(k)):
        b = by_week[wk]
        if b["has_la"] and b["active"] > 0:
            val = b["leavers"] / b["active"] * 100.0
        elif b["pct_n"] > 0:
            val = b["pct_sum"] / b["pct_n"]
        else:
            continue
        series.append((wk, round(val, 3)))
    return series


def _next_mondays(last_week: str, horizon: int) -> list[str]:
    try:
        d = pd.Timestamp(str(last_week)[:10])
    except Exception:
        return [f"+{h+1}w" for h in range(horizon)]
    return [(d + pd.Timedelta(weeks=h + 1)).date().isoformat() for h in range(horizon)]


def project_attrition(rows: list[dict], horizon: int = 12) -> dict:
    horizon = max(1, min(int(horizon or 12), 52))
    series = _weekly_overall(rows)
    n = len(series)
    history = [{"week": w, "value": v} for w, v in series]
    if n == 0:
        return {"status": "no_data", "history": [], "forecast": [], "method": "none"}

    values = [v for _, v in series]
    last_week = series[-1][0]
    future_weeks = _next_mondays(last_week, horizon)

    recent_n = min(8, n)
    recent_avg = round(sum(values[-recent_n:]) / recent_n, 3)

    if n < 3:
        # Too short to fit a trend — flat-line the recent average.
        fc = [recent_avg] * horizon
        sigma = 0.0
        method = "recent-average (insufficient history)"
        trend_per_week = 0.0
    else:
        fc, sigma = holt_forecast(values, horizon)
        method = "Holt linear (level + trend)"
        trend_per_week = round((fc[-1] - fc[0]) / max(1, horizon - 1), 4) if horizon > 1 else 0.0

    forecast = []
    for h, (wk, point) in enumerate(zip(future_weeks, fc)):
        band = _Z * sigma * ((h + 1) ** 0.5)
        forecast.append(
            {
                "week": wk,
                "value": round(point, 3),
                "low": round(max(0.0, point - band), 3),
                "high": round(point + band, 3),
            }
        )

    projected_avg = round(sum(p["value"] for p in forecast) / len(forecast), 3)
    return {
        "status": "ok",
        "method": method,
        "horizon": horizon,
        "history": history,
        "forecast": forecast,
        "summary": {
            "recent_avg_weekly_pct": recent_avg,
            "projected_avg_weekly_pct": projected_avg,
            "recent_avg_annualized_pct": round(recent_avg * 52, 1),
            "projected_avg_annualized_pct": round(projected_avg * 52, 1),
            "trend_per_week_pct": trend_per_week,
            "direction": "rising" if projected_avg > recent_avg + 0.01 else ("falling" if projected_avg < recent_avg - 0.01 else "flat"),
        },
    }
