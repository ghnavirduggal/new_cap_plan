from __future__ import annotations

import json
from math import isfinite
from typing import Any, Iterable

import numpy as np
import pandas as pd


def df_from_payload(payload: Any) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, (list, tuple)):
        return pd.DataFrame(list(payload))
    if isinstance(payload, dict):
        if "data" in payload and "columns" in payload:
            return pd.DataFrame(payload.get("data", []), columns=payload.get("columns", []))
        if "records" in payload and isinstance(payload.get("records"), Iterable):
            return pd.DataFrame(list(payload.get("records") or []))
        return pd.DataFrame([payload])
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return pd.DataFrame()
        try:
            return pd.read_json(text, orient="split")
        except Exception:
            try:
                return pd.read_json(text, orient="records")
            except Exception:
                try:
                    return pd.read_json(text)
                except Exception:
                    try:
                        return pd.DataFrame(json.loads(text))
                    except Exception:
                        return pd.DataFrame()
    return pd.DataFrame()


def df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    cleaned = df.replace([np.inf, -np.inf], pd.NA)
    cleaned = cleaned.where(pd.notna(cleaned), None)
    return cleaned.to_dict("records")


def df_to_json(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    return df.to_json(date_format="iso", orient="split")


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(val) for val in value]
    if value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, float):
        return value if isfinite(value) else None
    return value
