from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from forecasting.process_and_IQ_data import (
    add_editable_base_volume,
    clean_and_convert_percentage,
    plot_contact_ratio_seasonality,
)
from .utils import df_from_payload, df_to_records


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _clean_contact_ratio_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Year" not in df.columns:
        return pd.DataFrame()
    cols = ["Year"] + [m for m in _MONTHS if m in df.columns]
    out = df[cols].copy()
    for col in out.columns:
        if col != "Year":
            out[col] = clean_and_convert_percentage(out[col])
    return out


def _apply_caps(df: pd.DataFrame, lower: Optional[float], upper: Optional[float]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    capped = df.copy()
    month_cols = [c for c in capped.columns if c not in ["Year", "Avg"]]
    for col in month_cols:
        capped[col] = pd.to_numeric(capped[col], errors="coerce")
    if lower is not None:
        capped[month_cols] = capped[month_cols].clip(lower=float(lower))
    if upper is not None:
        capped[month_cols] = capped[month_cols].clip(upper=float(upper))
    return capped


def _recalculate_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    month_cols = [c for c in out.columns if c not in ["Year", "Avg"]]
    for col in month_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    row_means = out[month_cols].mean(axis=1).replace(0, np.nan)
    out[month_cols] = out[month_cols].div(row_means, axis=0).round(1)
    out[month_cols] = out[month_cols].fillna(0)
    out["Avg"] = out[month_cols].mean(axis=1).round(1)
    return out


def _normalized_ratio_table(df: pd.DataFrame, base_volume: Optional[float]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    base = float(base_volume or 0.0)
    out = df.copy()
    month_cols = [c for c in out.columns if c not in ["Year", "Avg"]]
    out[month_cols] = (out[month_cols].astype(float) * base).round(1)
    out["Avg"] = out[month_cols].mean(axis=1).round(1)
    return out


def _ratio_chart_data(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "Year" not in df.columns:
        return {}
    month_cols = [m for m in _MONTHS if m in df.columns]
    if "Avg" in df.columns:
        month_cols = month_cols + ["Avg"]
    series = []
    for _, row in df.iterrows():
        points = []
        for month in month_cols:
            val = row.get(month)
            num = pd.to_numeric(val, errors="coerce")
            points.append({"x": month, "y": float(num) if pd.notna(num) else None})
        series.append({"name": str(row.get("Year", "")), "points": points})
    return {"x": month_cols, "series": series}


def build_seasonality(payload: Any) -> dict:
    ratio_df = df_from_payload(payload)
    if ratio_df.empty:
        return {"status": "No ratio data supplied.", "results": {}}

    cleaned_ratio = _clean_contact_ratio_table(ratio_df)
    if cleaned_ratio.empty:
        return {"status": "No ratio data supplied.", "results": {}}

    _, capped_df, ratio_seasonality_df = plot_contact_ratio_seasonality(cleaned_ratio)
    ratio_seasonality_df = ratio_seasonality_df.round(1)
    capped_df = capped_df.round(1)

    _, base_volume = add_editable_base_volume(ratio_df)
    recalc_df = _recalculate_seasonality(capped_df)
    normalized_df = _normalized_ratio_table(recalc_df, base_volume)

    return {
        "status": "Seasonality loaded.",
        "results": {
            "ratio": df_to_records(ratio_seasonality_df),
            "capped": df_to_records(capped_df),
            "recalc": df_to_records(recalc_df),
            "normalized": df_to_records(normalized_df),
            "base_volume": float(base_volume) if base_volume is not None else None,
            "ratio_chart": _ratio_chart_data(ratio_seasonality_df),
            "capped_chart": _ratio_chart_data(recalc_df),
        },
    }


def apply_seasonality_changes(
    capped_payload: Any,
    lower_cap: Optional[float],
    upper_cap: Optional[float],
    base_volume: Optional[float],
) -> dict:
    capped_df = df_from_payload(capped_payload)
    if capped_df.empty:
        return {"status": "No seasonality data to apply.", "results": {}}

    capped_df = _apply_caps(capped_df, lower_cap, upper_cap)
    recalc_df = _recalculate_seasonality(capped_df)
    normalized_df = _normalized_ratio_table(recalc_df, base_volume)

    return {
        "status": "Seasonality updated.",
        "results": {
            "capped": df_to_records(capped_df),
            "recalc": df_to_records(recalc_df),
            "normalized": df_to_records(normalized_df),
            "base_volume": float(base_volume) if base_volume is not None else None,
            "capped_chart": _ratio_chart_data(recalc_df),
        },
    }
