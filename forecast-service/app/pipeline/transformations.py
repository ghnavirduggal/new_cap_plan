from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import df_from_payload, df_to_records


TRANSFORMATION_COLS = [
    "Transformation 1",
    "Remarks_Tr 1",
    "Transformation 2",
    "Remarks_Tr 2",
    "Transformation 3",
    "Remarks_Tr 3",
    "IA 1",
    "Remarks_IA 1",
    "IA 2",
    "Remarks_IA 2",
    "IA 3",
    "Remarks_IA 3",
    "Marketing Campaign 1",
    "Remarks_Mkt 1",
    "Marketing Campaign 2",
    "Remarks_Mkt 2",
    "Marketing Campaign 3",
    "Remarks_Mkt 3",
]


def _coerce_num(val: Any):
    if val is None:
        return None
    if isinstance(val, str):
        cleaned = val.replace(",", "").replace("%", "").strip()
        if cleaned == "":
            return None
        lowered = cleaned.lower()
        if lowered in {"nan", "na", "none"}:
            return None
        multiplier = 1.0
        if lowered.endswith("k"):
            multiplier = 1000.0
            lowered = lowered[:-1]
        elif lowered.endswith("m"):
            multiplier = 1_000_000.0
            lowered = lowered[:-1]
        elif lowered.endswith("b"):
            multiplier = 1_000_000_000.0
            lowered = lowered[:-1]
        try:
            return float(lowered) * multiplier
        except Exception:
            return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    try:
        return float(val)
    except Exception:
        return None


def _apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
    def calculate_sequential(row: pd.Series, field: str, base_col: str):
        adj_percent = _coerce_num(row.get(field, None))
        base_val = _coerce_num(row.get(base_col, None))
        if base_val is None:
            return np.nan
        if adj_percent is None:
            return round(base_val, 0)
        return round(base_val * (1 + adj_percent / 100), 0)

    def apply_sequential_adjustments(df_in: pd.DataFrame) -> pd.DataFrame:
        df_copy = df_in.copy()
        adjustment_fields = [
            "Transformation 1",
            "Transformation 2",
            "Transformation 3",
            "IA 1",
            "IA 2",
            "IA 3",
            "Marketing Campaign 1",
            "Marketing Campaign 2",
            "Marketing Campaign 3",
        ]
        prev_col = "Base_Forecast_for_Forecast_Group"
        if prev_col in df_copy.columns:
            df_copy[prev_col] = df_copy[prev_col].apply(_coerce_num)
        for field in adjustment_fields:
            if field in df_copy.columns:
                df_copy[field] = df_copy[field].apply(_coerce_num)
        for field in adjustment_fields:
            new_col = f"Forecast_{field}"
            df_copy[new_col] = df_copy.apply(lambda row: calculate_sequential(row, field, prev_col), axis=1)
            prev_col = new_col
        return df_copy

    processed = apply_sequential_adjustments(df)
    if "Forecast_Marketing Campaign 3" in processed.columns:
        processed["Final_Forecast_Post_Transformations"] = processed["Forecast_Marketing Campaign 3"]
        processed["Final_Forecast"] = processed["Forecast_Marketing Campaign 3"]
    return processed


def _sort_month_year_columns(cols: list[str]) -> list[str]:
    def _sort_key(val: str) -> tuple:
        try:
            dt = pd.to_datetime(val, format="%b-%y", errors="coerce")
            if pd.isna(dt):
                dt = pd.to_datetime(val, errors="coerce")
            if pd.isna(dt):
                return (9999, 99)
            return (dt.year, dt.month)
        except Exception:
            return (9999, 99)
    return sorted(cols, key=_sort_key)


def _sort_year_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Year" in df.columns and "Month" in df.columns:
        month_num = df["Month"].astype(str).str[:3].str.lower().map({
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        })
        df = df.assign(_sort_year=df["Year"], _sort_month=month_num).sort_values(["_sort_year", "_sort_month"])
        df = df.drop(columns=["_sort_year", "_sort_month"], errors="ignore")
    return df


def apply_transformations(payload: Any) -> dict:
    df = df_from_payload(payload)
    if df.empty:
        return {"status": "No data to transform.", "results": {}}

    if "Base_Forecast_for_Forecast_Group" not in df.columns:
        for cand in ("Base_Forecast_Category", "Final_Forecast", "Forecast"):
            if cand in df.columns:
                df["Base_Forecast_for_Forecast_Group"] = df[cand]
                break

    processed = _apply_transformations(df)
    final_value_col = None
    for cand in (
        "Final_Forecast_Post_Transformations",
        "Final_Forecast",
        "Forecast_Marketing Campaign 3",
        "Forecast_Marketing Campaign 2",
        "Forecast_Marketing Campaign 1",
    ):
        if cand in processed.columns:
            final_value_col = cand
            break

    transpose_cols = [
        "Month_Year",
        "Base_Forecast_for_Forecast_Group",
        "Transformation 1",
        "Remarks_Tr 1",
        "Forecast_Transformation 1",
        "Transformation 2",
        "Remarks_Tr 2",
        "Forecast_Transformation 2",
        "Transformation 3",
        "Remarks_Tr 3",
        "Forecast_Transformation 3",
        "IA 1",
        "Remarks_IA 1",
        "Forecast_IA 1",
        "IA 2",
        "Remarks_IA 2",
        "Forecast_IA 2",
        "IA 3",
        "Remarks_IA 3",
        "Forecast_IA 3",
        "Marketing Campaign 1",
        "Remarks_Mkt 1",
        "Forecast_Marketing Campaign 1",
        "Marketing Campaign 2",
        "Remarks_Mkt 2",
        "Forecast_Marketing Campaign 2",
        "Marketing Campaign 3",
        "Remarks_Mkt 3",
        "Forecast_Marketing Campaign 3",
    ]
    available_cols = [c for c in transpose_cols if c in processed.columns]
    transposed = pd.DataFrame()
    if available_cols:
        transposed_source = processed.copy()
        if "Month_Year" in transposed_source.columns:
            month_dt = pd.to_datetime(transposed_source["Month_Year"], format="%b-%y", errors="coerce")
            if month_dt.isna().all():
                month_dt = pd.to_datetime(transposed_source["Month_Year"], errors="coerce")
            if month_dt.notna().any():
                transposed_source = transposed_source.assign(_month_year_dt=month_dt).sort_values("_month_year_dt")
            elif "Year" in transposed_source.columns and "Month" in transposed_source.columns:
                transposed_source = _sort_year_month(transposed_source)
            transposed_source = transposed_source.drop(columns=["_month_year_dt"], errors="ignore")
        elif "Year" in transposed_source.columns and "Month" in transposed_source.columns:
            transposed_source = _sort_year_month(transposed_source)
        transposed = transposed_source[available_cols].copy()
        if "Month_Year" in transposed.columns:
            t = transposed.set_index("Month_Year").transpose().reset_index()
            t.rename(columns={"index": "Category"}, inplace=True)
            if final_value_col:
                final_forecast_values = processed.set_index("Month_Year")[final_value_col]
                t.loc[len(t)] = ["Final Forecast"] + final_forecast_values.tolist()
            transposed = t
    if not transposed.empty and "Category" in transposed.columns:
        month_cols = [c for c in transposed.columns if c != "Category"]
        ordered_months = _sort_month_year_columns(month_cols)
        ordered = ["Category"] + ordered_months + [c for c in month_cols if c not in ordered_months]
        transposed = transposed[ordered]

    final_cols = ["Month_Year"]
    if final_value_col:
        final_cols.append(final_value_col)
    if "forecast_group" in processed.columns:
        final_cols.insert(0, "forecast_group")
    if "Model" in processed.columns:
        final_cols.insert(-1, "Model")
    if "Year" in processed.columns:
        final_cols.insert(-1, "Year")
    final_cols = [c for c in final_cols if c in processed.columns]
    final_tbl = processed[final_cols].copy()
    if final_value_col and final_value_col != "Final_Forecast" and final_value_col in final_tbl.columns:
        final_tbl = final_tbl.rename(columns={final_value_col: "Final_Forecast"})
    if "Month_Year" in final_tbl.columns:
        month_dt = pd.to_datetime(final_tbl["Month_Year"], format="%b-%y", errors="coerce")
        if month_dt.isna().all():
            month_dt = pd.to_datetime(final_tbl["Month_Year"], errors="coerce")
        if month_dt.notna().any():
            final_tbl = final_tbl.assign(_month_year_dt=month_dt).sort_values("_month_year_dt")
            final_tbl = final_tbl.drop(columns=["_month_year_dt"], errors="ignore")

    summary_rows = []
    try:
        summary = {
            "Forecast Group": processed["forecast_group"].iloc[0] if "forecast_group" in processed.columns else None,
            "Model": processed["Model"].iloc[0] if "Model" in processed.columns else None,
            "Selected Year": processed["Year"].min() if "Year" in processed.columns else None,
            "Years Included": sorted(processed["Year"].unique().tolist()) if "Year" in processed.columns else [],
            "Total Rows": len(processed),
            "Base Forecast Total": float(processed["Base_Forecast_for_Forecast_Group"].sum()) if "Base_Forecast_for_Forecast_Group" in processed.columns else None,
            "Final Forecast Total": float(processed["Final_Forecast"].sum()) if "Final_Forecast" in processed.columns else None,
        }
    except Exception:
        summary = {}

    for key, val in (summary or {}).items():
        if isinstance(val, (list, tuple)):
            val = ", ".join(str(v) for v in val)
        summary_rows.append({"Metric": key, "Value": val})
    summary_df = pd.DataFrame(summary_rows)

    return {
        "status": "Transformations applied.",
        "results": {
            "processed": df_to_records(processed),
            "transposed": df_to_records(transposed),
            "final": df_to_records(final_tbl),
            "summary": df_to_records(summary_df),
        },
    }
