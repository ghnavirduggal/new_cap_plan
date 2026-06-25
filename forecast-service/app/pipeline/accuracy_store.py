"""Forecast-accuracy tracking.

The forecasting workspace already computes per-model accuracy (Phase 1), but only
the latest table was ever written (overwritten each run), so there was no way to
see which model wins or how accuracy trends over time. This store keeps an
append-only, per-scope history of accuracy snapshots and derives a leaderboard.

A snapshot is recorded each time a forecast result is saved; each snapshot holds
one normalized row per model with whatever accuracy metrics were available
(accuracy bands, MAPE, WAPE, bias) plus a derived rank and the winning model.

Persistence is a JSON file in the exports dir, mirroring how the rest of the
forecast outputs are stored, so it works without a database.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# How many snapshots to keep per scope (newest kept), so the file stays bounded.
_MAX_SNAPSHOTS = 200
_HISTORY_FILE = "forecast_accuracy_history.json"


def _exports_dir() -> Path:
    # forecast-service/app/pipeline/accuracy_store.py -> repo/exports
    outdir = Path(__file__).resolve().parent.parent.parent / "exports"
    outdir.mkdir(exist_ok=True, parents=True)
    return outdir


def _history_path() -> Path:
    return _exports_dir() / _HISTORY_FILE


def _scope_key(scope: Any) -> str:
    return (str(scope or "").strip() or "global").lower()


def _num(value: Any) -> Optional[float]:
    """Coerce a cell to float, tolerating '%' suffixes and blanks."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if pd.notna(value) else None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-", "—"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


# Map possible source column names -> canonical metric keys.
_METRIC_ALIASES = {
    "acc5": ("accuracy(+−5%)", "accuracy(+-5%)", "accuracy(+/-5%)", "accuracy(±5%)", "acc5", "accuracy"),
    "acc7": ("accuracy(+−7%)", "accuracy(+-7%)", "accuracy(+/-7%)", "accuracy(±7%)", "acc7"),
    "acc10": ("accuracy(+−10%)", "accuracy(+-10%)", "accuracy(+/-10%)", "accuracy(±10%)", "acc10"),
    "mape": ("mape%", "mape", "mape_pct"),
    "wape": ("wape%", "wape", "wape_pct"),
    "bias": ("bias%", "bias", "bias_pct"),
}
# Lower-is-better metrics (error); the rest are higher-is-better (accuracy bands).
_LOWER_IS_BETTER = {"mape", "wape"}


def _model_name(row: dict) -> str:
    for key in ("Model", "model", "name", "Name"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def _extract_metrics(row: dict) -> dict:
    lowered = { str(k).strip().lower(): v for k, v in row.items() }
    out: dict[str, float] = {}
    for canon, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                val = _num(lowered[alias])
                if val is not None:
                    out[canon] = val
                    break
    return out


def _primary(metrics: dict) -> tuple[Optional[str], Optional[float], bool]:
    """Pick the metric to rank by. Prefer a tight accuracy band, else MAPE/WAPE."""
    for key in ("acc5", "acc7", "acc10"):
        if key in metrics:
            return key, metrics[key], False  # higher better
    for key in ("mape", "wape"):
        if key in metrics:
            return key, metrics[key], True  # lower better
    return None, None, False


def normalize_models(rows: list[dict]) -> list[dict]:
    """Turn an accuracy table (list of per-model dicts) into ranked model rows."""
    models = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = _model_name(row)
        if not name or name.lower() in {"final_smoothed_values", "final smoothed values"}:
            continue
        metrics = _extract_metrics(row)
        if not metrics:
            continue
        pkey, pval, lower_better = _primary(metrics)
        if pval is None:
            continue
        models.append(
            {
                "model": name,
                "metrics": metrics,
                "primary_metric": pkey,
                "primary_value": pval,
                "lower_is_better": lower_better,
            }
        )
    if not models:
        return []
    # All rows share the same primary metric ordering; rank accordingly.
    lower_better = models[0]["lower_is_better"]
    models.sort(key=lambda m: m["primary_value"], reverse=not lower_better)
    for idx, m in enumerate(models):
        m["rank"] = idx + 1
        m["is_best"] = idx == 0
    return models


def _load_all() -> dict:
    path = _history_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_all(data: dict) -> None:
    try:
        _history_path().write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass


def record_snapshot(
    scope: str,
    rows: list[dict],
    *,
    run_label: str = "",
    actor: str = "",
    ts: Optional[str] = None,
) -> Optional[dict]:
    """Append a normalized accuracy snapshot for a scope. Returns it, or None if
    the table had no usable per-model metrics."""
    models = normalize_models(rows)
    if not models:
        return None
    snapshot = {
        "ts": ts or pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "run_label": str(run_label or ""),
        "actor": str(actor or ""),
        "best_model": models[0]["model"],
        "primary_metric": models[0]["primary_metric"],
        "models": models,
    }
    data = _load_all()
    key = _scope_key(scope)
    history = data.get(key) or []
    history.append(snapshot)
    # Keep newest _MAX_SNAPSHOTS.
    data[key] = history[-_MAX_SNAPSHOTS:]
    _save_all(data)
    return snapshot


def list_scopes() -> list[str]:
    return sorted(_load_all().keys())


def load_history(scope: str) -> list[dict]:
    raw = _load_all().get(_scope_key(scope))
    if not isinstance(raw, list):
        return []
    # Only keep well-formed snapshot dicts so downstream consumers can't choke
    # on a legacy/corrupt entry.
    return [s for s in raw if isinstance(s, dict)]


def leaderboard(scope: str) -> dict:
    """Latest ranked models for a scope, plus a per-model trend of the primary
    metric across snapshots so the UI can show whether accuracy is improving."""
    history = load_history(scope)
    if not history:
        return {"scope": _scope_key(scope), "latest": None, "models": [], "trend": []}
    latest = history[-1]
    # Per-model primary-metric series across snapshots.
    trend = []
    for snap in history:
        point = {"ts": snap.get("ts"), "run_label": snap.get("run_label")}
        for m in snap.get("models") or []:
            if not isinstance(m, dict):
                continue
            name = m.get("model")
            if not name:
                continue
            point[str(name)] = m.get("primary_value")
        trend.append(point)
    return {
        "scope": _scope_key(scope),
        "latest": latest,
        "models": latest.get("models") or [],
        "best_model": latest.get("best_model"),
        "primary_metric": latest.get("primary_metric"),
        "trend": trend,
    }
