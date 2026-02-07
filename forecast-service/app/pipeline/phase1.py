from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd

from forecasting.contact_ratio_dash import run_contact_ratio_forecast, train_and_evaluate_func, iterative_tuning
from forecasting.process_and_IQ_data import (
    accuracy_phase1,
    create_download_csv_with_metadata,
    fill_final_smoothed_row,
    process_forecast_results,
)
import config_manager
from .utils import df_from_payload, df_to_records


def _month_name_to_num(value: Any) -> Optional[int]:
    if value is None:
        return None
    name = str(value).strip().lower()[:3]
    month_map = {
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
    }
    return month_map.get(name)


def _normalize_smoothed_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()

    if "ds" in d.columns:
        d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    elif "Month_Year" in d.columns:
        d["ds"] = pd.to_datetime(d["Month_Year"], format="%b-%Y", errors="coerce")
        if d["ds"].isna().all():
            d["ds"] = pd.to_datetime(d["Month_Year"], errors="coerce")
    elif "Year" in d.columns and "Month" in d.columns:
        month_num = d["Month"].apply(_month_name_to_num)
        date_str = (
            d["Year"].astype(str)
            + "-"
            + month_num.astype("Int64").astype(str).str.zfill(2)
            + "-01"
        )
        d["ds"] = pd.to_datetime(date_str, errors="coerce")

    if "Final_Smoothed_Value" in d.columns:
        d["Final_Smoothed_Value"] = pd.to_numeric(d["Final_Smoothed_Value"], errors="coerce")
    elif "y" in d.columns:
        d["Final_Smoothed_Value"] = pd.to_numeric(d["y"], errors="coerce")

    d = d.dropna(subset=["ds", "Final_Smoothed_Value"]).copy()
    if d.empty:
        return d

    d["ds"] = d["ds"].dt.to_period("M").dt.to_timestamp()
    if d["ds"].duplicated().any():
        numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
        agg = {col: "mean" for col in numeric_cols}
        for col in d.columns:
            if col not in numeric_cols and col != "ds":
                agg[col] = "first"
        d = d.groupby("ds", as_index=False).agg(agg)
    d = d.sort_values("ds").reset_index(drop=True)
    d["y"] = d["Final_Smoothed_Value"] / 100
    return d


def _config_table(cfg: dict) -> pd.DataFrame:
    rows = []
    for model_name, params in (cfg or {}).items():
        if not isinstance(params, dict):
            continue
        for key, val in params.items():
            if isinstance(val, (dict, list, tuple)):
                val = json.dumps(val)
            rows.append({"Model": model_name, "Parameter": key, "Value": val})
    return pd.DataFrame(rows)


def run_phase1(payload: Any, config: Optional[dict] = None, holidays: Optional[dict] = None) -> dict:
    df = df_from_payload(payload)
    df = _normalize_smoothed_df(df)
    if df.empty:
        return {"status": "Prophet data missing.", "results": {}}

    cfg = config_manager.load_config()
    if config:
        cfg = config_manager._ensure_config(config)

    holiday_df = None
    if holidays:
        try:
            holiday_df = pd.DataFrame(
                {"ds": pd.to_datetime(list(holidays.keys()), errors="coerce"), "holiday": list(holidays.values())}
            )
            holiday_df = holiday_df.dropna(subset=["ds"])
        except Exception:
            holiday_df = None

    forecast_res = run_contact_ratio_forecast(
        df,
        12,
        holiday_mapping=None,
        holiday_original_date=holiday_df,
        config=cfg,
    )
    train = forecast_res.get("train", pd.DataFrame())
    test = forecast_res.get("test", pd.DataFrame())

    results = forecast_res.get("forecast_results", {})
    combined = pd.DataFrame()
    wide = pd.DataFrame()
    pivot_smoothed = pd.DataFrame()
    accuracy_tbl = pd.DataFrame()

    if results:
        results_with_smoothed = dict(results)
        smoothed_vals = forecast_res.get("final_smoothed_values")
        if smoothed_vals is not None:
            results_with_smoothed["final_smoothed_values"] = smoothed_vals
        combined, wide, pivot_smoothed = process_forecast_results(results_with_smoothed)
        if not wide.empty and not pivot_smoothed.empty:
            wide = fill_final_smoothed_row(wide.copy(), pivot_smoothed)
        if not wide.empty and not pivot_smoothed.empty:
            try:
                accuracy_tbl = accuracy_phase1(wide, pivot_smoothed)
            except Exception:
                accuracy_tbl = pd.DataFrame()

    forecast_horizon = len(test) if not test.empty else 0
    tuning_tbl = pd.DataFrame()
    final_accuracy_tbl = pd.DataFrame()
    tuned_accuracy_tbl = pd.DataFrame()
    tuned_config = cfg
    download_csv = None

    if not train.empty and forecast_horizon > 0:
        try:
            actual_data = df[["ds", "y"]].dropna()
            initial_accuracy_df, _, _ = train_and_evaluate_func(
                cfg,
                train,
                actual_data,
                forecast_horizon,
                show_details=False,
            )
            tuned_config, _acc_before, tuned_accuracy_df = iterative_tuning(
                cfg,
                initial_accuracy_df,
                train_and_evaluate_func,
                train,
                forecast_horizon,
                actual_data,
            )
            tuning_tbl = _config_table(tuned_config)
            tuned_accuracy_tbl = tuned_accuracy_df.copy() if tuned_accuracy_df is not None else pd.DataFrame()
            final_accuracy_tbl = tuned_accuracy_tbl.copy()

            tuned_forecast = run_contact_ratio_forecast(
                df,
                forecast_horizon,
                holiday_mapping=None,
                holiday_original_date=holiday_df,
                config=tuned_config,
            )
            tuned_results = tuned_forecast.get("forecast_results", {})
            if tuned_results:
                tuned_with_smoothed = dict(tuned_results)
                tuned_smoothed_vals = tuned_forecast.get("final_smoothed_values")
                if tuned_smoothed_vals is not None:
                    tuned_with_smoothed["final_smoothed_values"] = tuned_smoothed_vals
                combined_tuned, wide_tuned, pivot_tuned = process_forecast_results(tuned_with_smoothed)
                if not wide_tuned.empty and not pivot_tuned.empty:
                    wide_tuned = fill_final_smoothed_row(wide_tuned.copy(), pivot_tuned)
                if not wide_tuned.empty and not pivot_tuned.empty:
                    # Use the same schema as Phase 1 Accuracy (Before Iterations) for apples-to-apples comparison.
                    final_accuracy_tbl = accuracy_phase1(wide_tuned, pivot_tuned)
                if not wide_tuned.empty:
                    download_csv = create_download_csv_with_metadata(wide_tuned, tuned_config)
        except Exception:
            pass

    if final_accuracy_tbl.empty:
        final_accuracy_tbl = accuracy_tbl.copy()

    if download_csv is None and not wide.empty:
        try:
            download_csv = create_download_csv_with_metadata(wide, tuned_config)
        except Exception:
            download_csv = None

    status_lines = [
        "Starting Phase 1 Processing",
        f"Total data points: {len(df)} months",
    ]
    if not train.empty:
        status_lines.append(
            f"Train Period: {train['ds'].min():%b %Y} to {train['ds'].max():%b %Y} ({len(train)} months)"
        )
    if not test.empty:
        status_lines.append(
            f"Test Period: {test['ds'].min():%b %Y} to {test['ds'].max():%b %Y} ({len(test)} months)"
        )

    return {
        "status": " | ".join(status_lines),
        "results": {
            "combined": df_to_records(combined),
            "wide": df_to_records(wide),
            "pivot_smoothed": df_to_records(pivot_smoothed),
            "accuracy": df_to_records(accuracy_tbl),
            "tuning": df_to_records(tuning_tbl),
            "final_accuracy": df_to_records(final_accuracy_tbl),
        },
        "config": tuned_config,
        "download_csv": download_csv or "",
    }
