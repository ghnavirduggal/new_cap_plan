from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from forecasting.contact_ratio_dash import run_phase2_forecast
from forecasting.process_and_IQ_data import (
    fmt_millions_1,
    fmt_percent1,
    forecast_group_pivot_and_long_style,
    map_normalized_volume_to_forecast,
    map_original_volume_to_phase2_forecast,
    process_forecast_results,
    unpivot_iq_summary,
)
import config_manager
from .utils import df_from_payload, df_to_records
from .volume_summary import fallback_pivots, normalize_volume_df


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

    d["Final_Smoothed_Value"] = pd.to_numeric(d.get("Final_Smoothed_Value"), errors="coerce")
    d = d.dropna(subset=["ds", "Final_Smoothed_Value"]).copy()
    if d.empty:
        return d

    d["ds"] = d["ds"].dt.to_period("M").dt.to_timestamp()
    return d


def _sort_year_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Year" in df.columns and "Month" in df.columns:
        month_num = df["Month"].apply(_month_name_to_num)
        df = df.assign(_sort_year=df["Year"], _sort_month=month_num).sort_values(["_sort_year", "_sort_month"])
        df = df.drop(columns=["_sort_year", "_sort_month"], errors="ignore")
    return df


def run_phase2(
    payload: Any,
    start_date: str,
    end_date: str,
    config: Optional[dict] = None,
    iq_summary: Optional[Any] = None,
    volume_summary: Optional[Any] = None,
    basis: str = "iq",
    volume_data: Optional[Any] = None,
    category: Optional[str] = None,
) -> dict:
    basis_norm = str(basis or "iq").strip().lower()
    if basis_norm not in {"iq", "volume"}:
        basis_norm = "iq"

    df_smooth = pd.DataFrame()
    if basis_norm == "volume":
        df_norm = normalize_volume_df(df_from_payload(volume_data))
        if not df_norm.empty and category and "category" in df_norm.columns:
            df_norm = df_norm[df_norm["category"] == category]
        if not df_norm.empty and {"date", "volume"}.issubset(df_norm.columns):
            df_norm["date"] = pd.to_datetime(df_norm["date"], errors="coerce")
            df_norm["volume"] = pd.to_numeric(df_norm["volume"], errors="coerce")
            df_norm = df_norm.dropna(subset=["date", "volume"])
            df_norm["ds"] = df_norm["date"].dt.to_period("M").dt.to_timestamp()
            monthly = df_norm.groupby("ds", as_index=False)["volume"].sum()
            monthly["Final_Smoothed_Value"] = (
                pd.to_numeric(monthly["volume"], errors="coerce") / 1_000_000 * 100.0
            )
            df_smooth = monthly[["ds", "Final_Smoothed_Value"]].copy()
            df_smooth["IQ_value"] = 1.0
        if df_smooth.empty and volume_summary is not None:
            vol_df = df_from_payload(volume_summary)
            if not vol_df.empty and "Year" in vol_df.columns:
                month_cols = [c for c in vol_df.columns if c != "Year"]
                volume_long = vol_df.melt(
                    id_vars="Year", value_vars=month_cols, var_name="Month", value_name="volume"
                )
                volume_long["Year"] = pd.to_numeric(volume_long["Year"], errors="coerce")
                volume_long["Month"] = volume_long["Month"].astype(str).str.strip()
                volume_long["volume"] = (
                    volume_long["volume"].astype(str).str.replace(",", "", regex=False).str.strip()
                )

                def _parse_vol(x):
                    if x is None:
                        return None
                    s = str(x).lower()
                    try:
                        if s.endswith("k"):
                            return float(s[:-1]) * 1000
                        if s.endswith("m"):
                            return float(s[:-1]) * 1_000_000
                        return float(s)
                    except Exception:
                        return None

                volume_long["volume"] = volume_long["volume"].apply(_parse_vol)
                volume_long = volume_long.dropna(subset=["Year", "Month", "volume"])
                month_num = volume_long["Month"].apply(_month_name_to_num)
                volume_long["ds"] = pd.to_datetime(
                    volume_long["Year"].astype(int).astype(str)
                    + "-"
                    + month_num.astype("Int64").astype(str).str.zfill(2)
                    + "-01",
                    errors="coerce",
                )
                monthly = (
                    volume_long.dropna(subset=["ds"])
                    .groupby("ds", as_index=False)["volume"]
                    .sum()
                    .sort_values("ds")
                )
                if not monthly.empty:
                    monthly["Final_Smoothed_Value"] = (
                        pd.to_numeric(monthly["volume"], errors="coerce") / 1_000_000 * 100.0
                    )
                    df_smooth = monthly[["ds", "Final_Smoothed_Value"]].copy()
                    df_smooth["IQ_value"] = 1.0
        if df_smooth.empty:
            return {"status": "No volume history found for volume basis.", "results": {}}
    else:
        df_smooth = _normalize_smoothed_df(df_from_payload(payload))
        if df_smooth.empty:
            return {"status": "Prophet smoothing data is empty.", "results": {}}

    try:
        start = pd.to_datetime(start_date).to_period("M").to_timestamp()
        end = pd.to_datetime(end_date).to_period("M").to_timestamp()
    except Exception:
        return {"status": "Invalid dates for Phase 2.", "results": {}}

    if end < start:
        return {"status": "End date must be after start date.", "results": {}}
    forecast_months = (end.year - start.year) * 12 + (end.month - start.month) + 1

    iq_long = pd.DataFrame()
    if basis_norm == "iq" and iq_summary is not None:
        iq_df = df_from_payload(iq_summary)
        if not iq_df.empty:
            iq_long = unpivot_iq_summary(iq_df.copy())
            if not iq_long.empty:
                iq_long["Year"] = iq_long["Year"].astype(int)
                iq_long["Month"] = iq_long["Month"].astype(str).str.strip()

    df_smooth = df_smooth.copy()
    if not iq_long.empty:
        df_smooth["Year"] = df_smooth["ds"].dt.year
        df_smooth["Month"] = df_smooth["ds"].dt.strftime("%b")
        df_smooth = df_smooth.merge(iq_long, on=["Year", "Month"], how="left")

    if "IQ_value" not in df_smooth.columns or df_smooth["IQ_value"].isna().all():
        if "Normalized_Volume" in df_smooth.columns and "Normalized_Ratio_Post_Prophet" in df_smooth.columns:
            denom = pd.to_numeric(df_smooth["Normalized_Ratio_Post_Prophet"], errors="coerce").replace(0, np.nan)
            df_smooth["IQ_value"] = pd.to_numeric(df_smooth["Normalized_Volume"], errors="coerce") / denom
        else:
            df_smooth["IQ_value"] = 1.0

    cfg = config_manager.load_config()
    if config:
        cfg = config_manager._ensure_config(config)
    if basis_norm == "volume":
        for section in ("prophet", "random_forest", "xgboost", "var", "sarimax"):
            if isinstance(cfg, dict) and isinstance(cfg.get(section), dict):
                cfg[section]["use_iq_value_scaled"] = False

    phase2_res = run_phase2_forecast(df_smooth, forecast_months, config=cfg)
    forecast_results = phase2_res.get("forecast_results", {})
    if not forecast_results:
        return {"status": "Phase 2 forecast failed to produce results.", "results": {}}

    combined, wide, pivot_smoothed = process_forecast_results(forecast_results)
    if combined.empty:
        return {"status": "Phase 2 forecast returned no rows.", "results": {}}

    combined = combined.copy()
    combined["ds"] = pd.to_datetime(combined["Month_Year"], format="%b-%y", errors="coerce")
    combined["Year"] = combined["ds"].dt.year
    combined["Month"] = combined["ds"].dt.strftime("%b")

    history_df = df_smooth.copy()
    history_df["ds"] = pd.to_datetime(history_df["ds"], errors="coerce")
    history_df = history_df.dropna(subset=["ds"])
    if not history_df.empty:
        history_df["Year"] = history_df["ds"].dt.year
        history_df["Month"] = history_df["ds"].dt.strftime("%b")
        history_df["Month_Year"] = history_df["ds"].dt.strftime("%b-%y")
        ratio_vals = pd.to_numeric(history_df.get("Final_Smoothed_Value"), errors="coerce")
        if ratio_vals.notna().any() and ratio_vals.dropna().median() > 1:
            ratio_vals = ratio_vals / 100.0
        history_df["Forecast"] = ratio_vals
        history_df["Model"] = "Final_smoothed"
        keep_cols = ["ds", "Year", "Month", "Month_Year", "Model", "Forecast"]
        for col in ("IQ_value", "Normalized_Volume"):
            if col in history_df.columns:
                keep_cols.append(col)
        history_df = history_df[keep_cols]

    merged_df = pd.concat([history_df, combined], ignore_index=True) if not history_df.empty else combined.copy()
    if not iq_long.empty:
        merged_df = merged_df.merge(iq_long, on=["Year", "Month"], how="left", suffixes=("", "_iq"))
        if "IQ_value_iq" in merged_df.columns:
            merged_df["IQ_value"] = merged_df["IQ_value"].fillna(merged_df["IQ_value_iq"])
            merged_df = merged_df.drop(columns=["IQ_value_iq"])
    merged_df["ds"] = pd.to_datetime(merged_df.get("ds"), errors="coerce")
    merged_df = merged_df.sort_values("ds").reset_index(drop=True)
    merged_df["IQ_value"] = pd.to_numeric(merged_df.get("IQ_value"), errors="coerce")
    if merged_df["IQ_value"].notna().any():
        merged_df["IQ_value"] = merged_df.groupby(["Year", "Month"])["IQ_value"].transform(
            lambda s: s.fillna(s.dropna().iloc[0]) if s.dropna().any() else s
        )
        merged_df["IQ_value"] = merged_df["IQ_value"].ffill()

    merged_df["Forecast"] = pd.to_numeric(merged_df.get("Forecast"), errors="coerce")
    merged_df["Base_Forecast_Category"] = (
        merged_df["Forecast"] * merged_df["IQ_value"] * 1_000_000
    ).round().astype("Int64")
    if basis_norm == "iq":
        merged_df["Contact_Ratio_Forecast_Category"] = merged_df["Forecast"].apply(fmt_percent1)
        merged_df["IQ_value_Category"] = merged_df["IQ_value"].apply(fmt_millions_1)
    else:
        merged_df["Contact_Ratio_Forecast_Category"] = None
        merged_df["IQ_value_Category"] = None

    volume_long = pd.DataFrame()
    if volume_summary is not None:
        vol_df = df_from_payload(volume_summary)
        if not vol_df.empty and "Year" in vol_df.columns:
            month_cols = [c for c in vol_df.columns if c != "Year"]
            volume_long = vol_df.melt(id_vars="Year", value_vars=month_cols, var_name="Month", value_name="volume")
            volume_long["Year"] = pd.to_numeric(volume_long["Year"], errors="coerce")
            volume_long["Month"] = volume_long["Month"].astype(str).str.strip()
            volume_long["volume"] = volume_long["volume"].astype(str).str.replace(",", "", regex=False).str.strip()

            def _parse_vol(x):
                if x is None:
                    return None
                s = str(x).lower()
                try:
                    if s.endswith("k"):
                        return float(s[:-1]) * 1000
                    if s.endswith("m"):
                        return float(s[:-1]) * 1_000_000
                    return float(s)
                except Exception:
                    return None

            volume_long["volume"] = volume_long["volume"].apply(_parse_vol)
            volume_long = volume_long.dropna(subset=["Year", "Month", "volume"])
            volume_long = volume_long.rename(columns={"Year": "year", "Month": "month"})

    base_df = merged_df.copy()
    if not volume_long.empty:
        base_df = map_original_volume_to_phase2_forecast(base_df, volume_long)
    smoothing_norm = pd.DataFrame()
    if "Normalized_Volume" in df_smooth.columns:
        smoothing_norm = df_smooth.copy()
        smoothing_norm["Year"] = smoothing_norm["ds"].dt.year
        smoothing_norm["Month"] = smoothing_norm["ds"].dt.strftime("%b")
        smoothing_norm = smoothing_norm[["Year", "Month", "Normalized_Volume"]]
    if not smoothing_norm.empty:
        base_df = map_normalized_volume_to_forecast(base_df, smoothing_norm)

    if "volume" in base_df.columns:
        base_df = base_df.rename(columns={"volume": "Original_volume"})
    base_df["Original_volume_Category"] = pd.to_numeric(base_df.get("Original_volume"), errors="coerce")
    base_df["Normalized_Volume_Category"] = pd.to_numeric(base_df.get("Normalized_Volume"), errors="coerce")

    base_cols = [
        "Year",
        "Month",
        "Model",
        "Contact_Ratio_Forecast_Category",
        "IQ_value_Category",
        "Base_Forecast_Category",
        "Original_volume_Category",
        "Normalized_Volume_Category",
    ]
    base_display = base_df[[c for c in base_cols if c in base_df.columns]].copy()
    base_display = _sort_year_month(base_display)

    fg_summary = pd.DataFrame()
    fg_split = pd.DataFrame()
    fg_monthly = pd.DataFrame()
    volume_split_edit = pd.DataFrame()
    volume_split_info = ""
    if volume_data is not None and category:
        df_norm = normalize_volume_df(df_from_payload(volume_data))
        if not df_norm.empty and "category" in df_norm.columns:
            df_norm = df_norm[df_norm["category"] == category]
        if not df_norm.empty:
            try:
                fg_summary, fg_split, _, _, _ = forecast_group_pivot_and_long_style(df_norm, category)
            except Exception:
                fg_summary, fg_split = fallback_pivots(df_norm, category)
            if {"date", "volume", "forecast_group"}.issubset(df_norm.columns):
                df_norm["ds"] = pd.to_datetime(df_norm["date"], errors="coerce")
                df_norm["Year"] = df_norm["ds"].dt.year
                df_norm["Month"] = df_norm["ds"].dt.strftime("%b")
                df_norm["volume"] = pd.to_numeric(df_norm["volume"], errors="coerce")
                fg_monthly = df_norm.dropna(subset=["Year", "Month", "forecast_group", "volume"])
                if not fg_monthly.empty:
                    fg_monthly = fg_monthly.groupby(
                        ["Year", "Month", "forecast_group"], as_index=False
                    )["volume"].sum()

    if fg_split is not None and not fg_split.empty:
        split_clean = fg_split.copy()
        if "Year" in split_clean.columns:
            split_clean = split_clean[split_clean["Year"] != "--------"].copy()
        split_clean["Year_Numeric"] = pd.to_numeric(split_clean.get("Year"), errors="coerce")

        def _pick_latest(group: pd.DataFrame) -> pd.Series:
            year_numeric = group["Year_Numeric"]
            if year_numeric.notna().any():
                row = group.loc[year_numeric.idxmax()].copy()
            else:
                row = group.iloc[-1].copy()
            if "forecast_group" not in row.index:
                row["forecast_group"] = group.name
            return row

        latest_data = split_clean.groupby("forecast_group").apply(_pick_latest)
        if isinstance(latest_data.index, pd.MultiIndex):
            latest_data = latest_data.reset_index(drop=True)
        if "Vol_Split_Last_3M" in latest_data.columns:
            latest_data["Vol_Split_Last_3M_Numeric"] = (
                latest_data["Vol_Split_Last_3M"].astype(str).str.replace("%", "", regex=False)
            )
            latest_data["Vol_Split_Last_3M_Numeric"] = pd.to_numeric(
                latest_data["Vol_Split_Last_3M_Numeric"], errors="coerce"
            )
        else:
            latest_data["Vol_Split_Last_3M_Numeric"] = 0.0
        total_original = latest_data["Vol_Split_Last_3M_Numeric"].sum()
        if total_original > 0:
            latest_data["Vol_Split_Normalized"] = (
                latest_data["Vol_Split_Last_3M_Numeric"] / total_original * 100
            ).round(1)
        else:
            latest_data["Vol_Split_Normalized"] = 0.0
        volume_split_edit = latest_data[
            ["forecast_group", "Year", "Vol_Split_Last_3M_Numeric", "Vol_Split_Normalized"]
        ].copy()
        volume_split_info = (
            f"Volume Split% last 3 Months total: {total_original:.1f}% | "
            f"Final normalized: {volume_split_edit['Vol_Split_Normalized'].sum():.1f}%"
        )

    return {
        "status": f"Phase 2 forecast ready ({forecast_months} months, basis: {'IQ' if basis_norm == 'iq' else 'Volume'}).",
        "results": {
            "combined": df_to_records(combined),
            "wide": df_to_records(wide),
            "pivot_smoothed": df_to_records(pivot_smoothed),
            "base": df_to_records(base_display),
            "base_raw": df_to_records(base_df),
            "forecast_group_summary": df_to_records(fg_summary),
            "forecast_group_split": df_to_records(fg_split),
            "volume_split_edit": df_to_records(volume_split_edit),
            "volume_split_info": volume_split_info,
            "phase2_store": {
                "base_df": df_to_records(base_df),
                "forecast_group_monthly": df_to_records(fg_monthly),
            },
        },
        "config": cfg,
    }


def apply_volume_split(
    base_payload: Any,
    split_payload: Any,
    forecast_group_monthly_payload: Any = None,
) -> dict:
    base_df = df_from_payload(base_payload)
    if base_df.empty:
        return {"status": "Phase 2 base forecast missing.", "results": {}}

    split_df = df_from_payload(split_payload)
    if split_df.empty or "forecast_group" not in split_df.columns:
        return {"status": "Volume split data missing.", "results": {}}

    split_df["Vol_Split_Normalized"] = pd.to_numeric(split_df["Vol_Split_Normalized"], errors="coerce").fillna(0.0)
    total_norm = split_df["Vol_Split_Normalized"].sum()
    if total_norm > 0:
        split_df["Vol_Split_Final"] = (split_df["Vol_Split_Normalized"] / total_norm * 100).round(1)
    else:
        split_df["Vol_Split_Final"] = 0.0

    mapping = dict(zip(split_df["forecast_group"], split_df["Vol_Split_Final"] / 100.0))
    adjusted_results = []
    for fg, split_pct in mapping.items():
        fg_forecast = base_df.copy()
        fg_forecast["forecast_group"] = fg
        fg_forecast["Volume_Split_%Fg"] = split_pct * 100
        fg_forecast["Base_Forecast_for_Forecast_Group"] = (
            pd.to_numeric(fg_forecast["Base_Forecast_Category"], errors="coerce") * split_pct
        ).round().astype("Int64")
        adjusted_results.append(fg_forecast)

    adjusted_df = pd.concat(adjusted_results, ignore_index=True) if adjusted_results else pd.DataFrame()
    if adjusted_df.empty:
        return {"status": "No adjusted forecast generated.", "results": {}}

    if "Contact_Ratio_Forecast_Category" in adjusted_df.columns:
        adjusted_df["Contact_Ratio_Forecast_Group"] = adjusted_df["Contact_Ratio_Forecast_Category"]
    if "IQ_value_Category" in adjusted_df.columns:
        adjusted_df["IQ_Value_Category"] = adjusted_df["IQ_value_Category"]
    adjusted_df["Volume_Split%_Forecast_Group"] = pd.to_numeric(
        adjusted_df.get("Volume_Split_%Fg"), errors="coerce"
    )

    fg_monthly = df_from_payload(forecast_group_monthly_payload)
    if not fg_monthly.empty:
        fg_monthly["Year"] = pd.to_numeric(fg_monthly["Year"], errors="coerce")
        fg_monthly["Month"] = fg_monthly["Month"].astype(str).str.strip()
        fg_monthly["volume"] = pd.to_numeric(fg_monthly.get("volume"), errors="coerce")
        fg_monthly = fg_monthly.dropna(subset=["Year", "Month", "forecast_group", "volume"])
        fg_monthly = fg_monthly.rename(columns={"volume": "Actual_Forecast_Group_Original_Volume"})
        adjusted_df = adjusted_df.merge(
            fg_monthly[["Year", "Month", "forecast_group", "Actual_Forecast_Group_Original_Volume"]],
            on=["Year", "Month", "forecast_group"],
            how="left",
        )

    verify_df = adjusted_df.groupby(["Year", "Month", "Model"], as_index=False).agg(
        Base_Forecast_Category=("Base_Forecast_Category", "first"),
        Base_Forecast_for_Forecast_Group=("Base_Forecast_for_Forecast_Group", "sum"),
    )
    verify_df["Difference"] = (
        pd.to_numeric(verify_df["Base_Forecast_for_Forecast_Group"], errors="coerce")
        - pd.to_numeric(verify_df["Base_Forecast_Category"], errors="coerce")
    )

    display_cols = [
        "Year",
        "Month",
        "Model",
        "forecast_group",
        "Volume_Split%_Forecast_Group",
        "Contact_Ratio_Forecast_Group",
        "IQ_Value_Category",
        "Base_Forecast_Category",
        "Base_Forecast_for_Forecast_Group",
        "Actual_Forecast_Group_Original_Volume",
    ]
    adjusted_display = adjusted_df[[c for c in display_cols if c in adjusted_df.columns]].copy()
    adjusted_display = adjusted_display.rename(columns={"forecast_group": "Forecast_Group"})
    required_cols = [
        "Year",
        "Month",
        "Model",
        "Forecast_Group",
        "Volume_Split%_Forecast_Group",
        "Contact_Ratio_Forecast_Group",
        "IQ_Value_Category",
        "Base_Forecast_Category",
        "Base_Forecast_for_Forecast_Group",
        "Actual_Forecast_Group_Original_Volume",
    ]
    for col in required_cols:
        if col not in adjusted_display.columns:
            adjusted_display[col] = None
    adjusted_display = adjusted_display[required_cols]
    adjusted_display = _sort_year_month(adjusted_display)
    verify_df = _sort_year_month(verify_df)

    return {
        "status": "Volume Split Applied successfully to base forecast.",
        "results": {
            "adjusted": df_to_records(adjusted_display),
            "verify": df_to_records(verify_df),
            "adjusted_raw": df_to_records(adjusted_df),
        },
    }
