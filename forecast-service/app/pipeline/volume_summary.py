from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from forecasting.process_and_IQ_data import IQ_data, forecast_group_pivot_and_long_style, plot_contact_ratio_seasonality
from .utils import df_to_records


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    lookup: dict[str, str] = {}
    for c in df.columns:
        base = str(c).strip().lower()
        variants = {
            base,
            base.replace(" ", "_"),
            base.replace(" ", ""),
            base.replace("-", "_"),
            base.replace("-", ""),
            base.replace("_", ""),
            re.sub(r"[^a-z0-9]", "", base),
        }
        for v in variants:
            lookup.setdefault(v, c)

    for nm in candidates:
        key = str(nm).strip().lower()
        candidates_norm = {
            key,
            key.replace(" ", "_"),
            key.replace(" ", ""),
            key.replace("-", "_"),
            key.replace("-", ""),
            key.replace("_", ""),
            re.sub(r"[^a-z0-9]", "", key),
        }
        for cand in candidates_norm:
            col = lookup.get(cand)
            if col:
                return col
    return None


def parse_upload(filename: str, content: bytes) -> tuple[pd.DataFrame, str]:
    if not filename:
        return pd.DataFrame(), "No filename supplied."
    try:
        lower = filename.lower()
        sheet_note = ""
        if lower.endswith(".csv"):
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        else:
            xl = pd.ExcelFile(io.BytesIO(content))
            sheet_map = {
                re.sub(r"[^a-z0-9]", "", str(name).lower()): name
                for name in xl.sheet_names
            }
            volume_sheet = sheet_map.get("volume")
            if volume_sheet:
                df = xl.parse(volume_sheet)
                sheet_note = f" (sheet '{volume_sheet}')"
            else:
                first_sheet = xl.sheet_names[0] if xl.sheet_names else None
                df = xl.parse(first_sheet) if first_sheet else pd.DataFrame()
                if first_sheet:
                    sheet_note = f" (sheet '{first_sheet}')"
        msg = f"Loaded {len(df):,} rows from {filename}{sheet_note}."
        return df, msg
    except Exception as exc:
        return pd.DataFrame(), f"Failed to read {filename}: {exc}"


def normalize_volume_df(df: pd.DataFrame, cat_hint: Optional[str] = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]

    cat_col = None
    if cat_hint:
        cat_hint_norm = str(cat_hint).strip().lower()
        if cat_hint_norm in d.columns:
            cat_col = cat_hint_norm
    if not cat_col:
        cat_col = _pick_col(d, ("category", "forecast_group", "queue_name"))
    if cat_col and cat_col != "category":
        d = d.rename(columns={cat_col: "category"})
    if "category" not in d.columns:
        d["category"] = "All"
    fg_col = _pick_col(d, ("forecast_group", "forecast group", "queue_name"))
    sub_col = _pick_col(
        d,
        (
            "category_sub_service",
            "category sub service",
            "category-sub-service",
            "sub_service",
            "sub service",
            "subservice",
        ),
    )
    if sub_col:
        fg_unique = d[fg_col].nunique(dropna=True) if fg_col and fg_col in d.columns else 0
        sub_unique = d[sub_col].nunique(dropna=True)
        if fg_unique <= 1 and sub_unique > 1:
            fg_col = sub_col
    if fg_col and fg_col != "forecast_group":
        d = d.rename(columns={fg_col: "forecast_group"})
    if "forecast_group" not in d.columns:
        d["forecast_group"] = d["category"]
    d["category"] = d["category"].astype(str).str.strip()
    d["forecast_group"] = d["forecast_group"].astype(str).str.strip()
    d.loc[d["category"].str.lower() == "nan", "category"] = None
    d.loc[d["forecast_group"].str.lower() == "nan", "forecast_group"] = None

    ba_col = _pick_col(d, ("business_area", "business area", "businessarea"))
    if ba_col and ba_col != "business_area":
        d = d.rename(columns={ba_col: "business_area"})
    if "business_area" in d.columns:
        d["business_area"] = d["business_area"].astype(str).str.strip()
        d.loc[d["business_area"].str.lower() == "nan", "business_area"] = None

    date_col = _pick_col(d, ("date", "ds", "datetime", "timestamp", "month_start"))
    if date_col and date_col != "date":
        d = d.rename(columns={date_col: "date"})
    vol_col = _pick_col(d, ("volume", "items", "calls", "count", "value", "y"))
    if vol_col and vol_col != "volume":
        d = d.rename(columns={vol_col: "volume"})

    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if "volume" in d.columns:
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce")

    if "date" in d.columns and "volume" in d.columns:
        d = d.dropna(subset=["date", "volume"])

    return d


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    d = df.copy()
    date_col = _pick_col(d, ("date", "ds", "datetime", "timestamp", "month_start"))
    val_col = _pick_col(d, ("volume", "items", "calls", "count", "value"))

    if date_col and val_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col])
        d["_month"] = d[date_col].dt.to_period("M").dt.to_timestamp()
        d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
        grouped = (
            d.groupby("_month", as_index=False)[val_col]
            .sum()
            .rename(columns={val_col: "Total"})
        )
        grouped["Month"] = grouped["_month"].dt.strftime("%b-%y")
        grouped = grouped[["Month", "Total"]]
        return grouped
    return d.describe(include="all").reset_index().rename(columns={"index": "metric"})


def category_month_pivot(df: pd.DataFrame) -> pd.DataFrame:
    norm = normalize_volume_df(df)
    if norm.empty or "date" not in norm.columns or "volume" not in norm.columns:
        return pd.DataFrame()

    norm = norm.dropna(subset=["date", "volume"])
    if norm.empty:
        return pd.DataFrame()

    norm["__month_period"] = norm["date"].dt.to_period("M")
    grouped = norm.groupby(["category", "__month_period"], as_index=False)["volume"].sum()
    if grouped.empty:
        return pd.DataFrame()

    ordered_months = sorted(grouped["__month_period"].unique())
    month_labels = {p: p.strftime("%b-%y") for p in ordered_months}

    pivot = grouped.pivot_table(
        index="category",
        columns="__month_period",
        values="volume",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.rename(columns=month_labels).reset_index()

    ordered_cols = ["category"] + [month_labels[p] for p in ordered_months]
    pivot = pivot.loc[:, ordered_cols]
    pivot = pivot.rename(columns={"category": "Category"})

    month_cols = [c for c in pivot.columns if c != "Category"]
    for col in month_cols:
        pivot[col] = pd.to_numeric(pivot[col], errors="coerce").round(0)

    return pivot.sort_values("Category").reset_index(drop=True)


def fallback_pivots(df: pd.DataFrame, cat: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = df.copy()
    if "category" not in d.columns:
        return pd.DataFrame(), pd.DataFrame()
    d = d[d["category"] == cat]
    if d.empty or "date" not in d.columns or "volume" not in d.columns:
        return pd.DataFrame(), pd.DataFrame()

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d = d.dropna(subset=["date", "volume"])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    d["Year"] = d["date"].dt.year
    d["Month_Num"] = d["date"].dt.month
    d["Month"] = d["date"].dt.strftime("%b")
    if "forecast_group" not in d.columns:
        d["forecast_group"] = d["category"]

    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_raw = (
        d.pivot_table(index=["Year", "forecast_group"], columns="Month", values="volume", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    month_cols = [m for m in months_order if m in pivot_raw.columns]

    display = pivot_raw.copy()
    display["Yearly_Avg"] = display[month_cols].mean(axis=1) if month_cols else 0
    display["Growth_%"] = None
    for fg in display["forecast_group"].unique():
        idx = display["forecast_group"] == fg
        growth = display[idx].sort_values("Year")["Yearly_Avg"].pct_change().values
        display.loc[idx, "Growth_%"] = growth
    for col in month_cols + ["Yearly_Avg"]:
        display[col] = display[col].apply(lambda x: f"{float(x) / 1000:,.1f}k" if pd.notna(x) else "")
    display["Growth_%"] = display["Growth_%"].apply(
        lambda x: f"{float(x) * 100:.1f}%" if pd.notna(x) else ""
    )
    display = display[["Year", "forecast_group"] + month_cols + ["Yearly_Avg", "Growth_%"]]

    split = pivot_raw.copy()
    row_totals = split[month_cols].sum(axis=1) if month_cols else pd.Series(dtype=float)
    for col in month_cols:
        split[col] = split[col].where(row_totals == 0, split[col] / row_totals * 100).round(1)

    def _row_avg(row):
        vals = [row[m] for m in month_cols if pd.notna(row[m]) and row[m] > 0]
        return round(sum(vals) / len(vals), 1) if vals else pd.NA

    def _last3(row):
        vals = [row[m] for m in month_cols if pd.notna(row[m]) and row[m] > 0]
        if len(vals) >= 3:
            return round(sum(vals[-3:]) / 3, 1)
        return round(sum(vals) / len(vals), 1) if vals else pd.NA

    split["Avg"] = split.apply(_row_avg, axis=1) if month_cols else pd.NA
    split["Vol_Split_Last_3M"] = split.apply(_last3, axis=1) if month_cols else pd.NA
    for col in month_cols + ["Avg", "Vol_Split_Last_3M"]:
        split[col] = split[col].apply(lambda x: f"{float(x):.1f}%" if pd.notna(x) else "")
    split = split[["Year", "forecast_group"] + month_cols + ["Avg", "Vol_Split_Last_3M"]]
    return display, split


def smoothing_core(df: pd.DataFrame, window: int, threshold: float, prophet_order: Optional[int] = None) -> dict:
    date_col = _pick_col(df, ("date", "ds", "timestamp"))
    val_col = _pick_col(df, ("final_smoothed_value", "volume", "value", "items", "calls", "count", "y"))
    if not date_col or not val_col:
        raise ValueError("Expected columns for date and volume.")

    d = df.copy()
    d["ds"] = pd.to_datetime(d[date_col], errors="coerce")
    d["y"] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna(subset=["ds", "y"]).sort_values("ds")

    if prophet_order:
        from prophet import Prophet

        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.add_seasonality(name="monthly", period=30.5, fourier_order=int(prophet_order or 5))
        m.fit(d[["ds", "y"]])
        preds = m.predict(d[["ds"]])
        d["smoothed"] = preds["yhat"]
    else:
        span = max(int(window or 6), 1)
        d["smoothed"] = d["y"].ewm(span=span, adjust=False).mean()

    resid = d["y"] - d["smoothed"]
    std = resid.std() or 1e-9
    d["zscore"] = (resid - resid.mean()) / std
    d["is_anomaly"] = d["zscore"].abs() > float(threshold or 3.0)

    d["Year"] = d["ds"].dt.year
    d["Month"] = d["ds"].dt.strftime("%b")
    pivot = d.pivot_table(index="Year", columns="Month", values="smoothed", aggfunc="mean").reset_index()
    pivot = pivot.fillna(0)

    try:
        _, capped, ratio = plot_contact_ratio_seasonality(pivot)
    except Exception:
        capped = pd.DataFrame()
        ratio = pd.DataFrame()

    ratio_disp = ratio.copy()
    capped_disp = capped.copy()
    for col in ratio_disp.columns:
        if col != "Year":
            ratio_disp[col] = pd.to_numeric(ratio_disp[col], errors="coerce").round(1)
    for col in capped_disp.columns:
        if col != "Year":
            capped_disp[col] = pd.to_numeric(capped_disp[col], errors="coerce").round(1)

    anomalies = d[d["is_anomaly"]][["ds", "y", "smoothed", "zscore"]]
    smoothed_tbl = d[["ds", "y", "smoothed", "zscore", "is_anomaly"]]

    return {
        "ratio": ratio_disp,
        "capped": capped_disp,
        "smoothed": smoothed_tbl,
        "anomalies": anomalies,
        "pivot": pivot,
    }


def _score_smoothing_candidate(smoothed_tbl: pd.DataFrame) -> tuple[float, dict]:
    if smoothed_tbl is None or smoothed_tbl.empty:
        return float("inf"), {}
    y = pd.to_numeric(smoothed_tbl.get("y"), errors="coerce")
    s = pd.to_numeric(smoothed_tbl.get("smoothed"), errors="coerce")
    if y.isna().all() or s.isna().all():
        return float("inf"), {}

    resid = y - s
    rmse = float(np.sqrt(np.nanmean(resid**2))) if np.isfinite(resid).any() else float("inf")
    y_std = float(y.std()) or 1e-9
    rmse_norm = rmse / y_std

    y_diff = y.diff().abs().mean()
    s_diff = s.diff().abs().mean()
    diff_base = float(y_diff) if pd.notna(y_diff) else 0.0
    smooth_ratio = float(s_diff) / (diff_base + 1e-9) if diff_base else 0.0

    anomaly_rate = 0.0
    if "is_anomaly" in smoothed_tbl.columns:
        anomaly_rate = float(smoothed_tbl["is_anomaly"].fillna(False).mean())

    target_smooth = 0.6
    target_anomaly = 0.03
    smooth_penalty = abs(smooth_ratio - target_smooth)
    anomaly_penalty = abs(anomaly_rate - target_anomaly)

    flat_penalty = 0.0
    if diff_base and smooth_ratio < 0.1:
        flat_penalty = 0.3
    if anomaly_rate > 0.2:
        anomaly_penalty += 0.3

    score = rmse_norm + 0.8 * smooth_penalty + 1.2 * anomaly_penalty + flat_penalty
    metrics = {
        "rmse_norm": rmse_norm,
        "smooth_ratio": smooth_ratio,
        "anomaly_rate": anomaly_rate,
    }
    return score, metrics


def auto_smoothing_sweep(
    df: pd.DataFrame,
    windows: Optional[list[int]] = None,
    thresholds: Optional[list[float]] = None,
) -> tuple[Optional[dict], dict]:
    if df is None or df.empty:
        return None, {"error": "no-data"}

    default_windows = [3, 4, 5, 6, 8, 10, 12]
    default_thresholds = [2.0, 2.5, 3.0, 3.5]
    windows = windows or default_windows
    thresholds = thresholds or default_thresholds

    data_len = len(df)
    max_win = max(2, min(12, data_len))
    windows = sorted({int(w) for w in windows if int(w) > 1 and int(w) <= max_win})
    if not windows:
        windows = [max(2, min(6, data_len))]
    thresholds = sorted({float(t) for t in thresholds if float(t) > 0})
    if not thresholds:
        thresholds = [2.5]

    best = None
    candidates: list[dict[str, Any]] = []
    for window in windows:
        for threshold in thresholds:
            try:
                res = smoothing_core(df, window, threshold, None)
            except Exception:
                continue
            score, metrics = _score_smoothing_candidate(res.get("smoothed"))
            if not np.isfinite(score):
                continue
            meta = {
                "method": "ewma",
                "window": int(window),
                "threshold": float(threshold),
                "score": float(score),
                "metrics": metrics,
            }
            candidates.append(meta)
            if best is None or score < best["meta"]["score"]:
                best = {"res": res, "meta": meta}

    if best is None:
        return None, {"error": "no-valid-candidate"}

    payload = {
        "smoothed": df_to_records(best["res"]["smoothed"]),
        "anomalies": df_to_records(best["res"]["anomalies"]),
        "ratio": df_to_records(best["res"]["ratio"]),
        "capped": df_to_records(best["res"]["capped"]),
        "pivot": df_to_records(best["res"]["pivot"]),
        "meta": best["meta"],
        "candidates": candidates,
    }
    return payload, best["meta"]


def run_volume_summary(filename: str, content: bytes) -> dict:
    df, msg = parse_upload(filename, content)
    norm = normalize_volume_df(df)
    summary = summarize(df)
    if not norm.empty:
        try:
            exports_dir = Path(__file__).resolve().parent.parent.parent / "exports"
            exports_dir.mkdir(exist_ok=True)
            norm.to_csv(exports_dir / "original_data.csv", index=False)
        except Exception:
            pass

    categories = sorted(norm["category"].dropna().astype(str).unique().tolist()) if not norm.empty else []
    chosen = categories[0] if categories else None

    pivots = pd.DataFrame()
    split = pd.DataFrame()
    if chosen:
        try:
            pivots, split, _, _, _ = forecast_group_pivot_and_long_style(norm, chosen)
            if pivots is None or split is None or pivots.empty or split.empty:
                raise ValueError("Primary pivot empty")
        except Exception:
            pivots, split = fallback_pivots(norm, chosen)

    iq_summary = {}
    holiday_store = None
    if filename.lower().endswith((".xlsx", ".xls", ".xlsm")):
        try:
            raw_iq = IQ_data(io.BytesIO(content))
            iq_summary = {
                key: {sub_key: df_to_records(sub_df) for sub_key, sub_df in value.items()}
                for key, value in raw_iq.items()
            }
        except Exception:
            iq_summary = {}
        try:
            xl = pd.ExcelFile(io.BytesIO(content))
            holiday_sheet = next(
                (s for s in xl.sheet_names if "holiday" in str(s).lower()),
                None,
            )
            if holiday_sheet:
                df_holidays = xl.parse(holiday_sheet)
                df_holidays.columns = [
                    str(col).strip().lower().replace(" ", "_") for col in df_holidays.columns
                ]
                date_col = _pick_col(df_holidays, ("date", "holiday_date"))
                name_col = _pick_col(df_holidays, ("holidays", "holiday", "name"))
                if date_col and name_col:
                    df_holidays[date_col] = pd.to_datetime(df_holidays[date_col], errors="coerce")
                    df_holidays = df_holidays.dropna(subset=[date_col, name_col])
                    holiday_store = {
                        "mapping": {
                            str(k): v for k, v in zip(df_holidays[date_col], df_holidays[name_col])
                        }
                    }
        except Exception:
            holiday_store = None

    auto_payload, auto_meta = (None, None)
    if not norm.empty and {"date", "volume"}.issubset(norm.columns):
        auto_payload, auto_meta = auto_smoothing_sweep(norm)

    return {
        "message": msg,
        "summary": df_to_records(summary),
        "normalized": df_to_records(norm),
        "categories": categories,
        "pivot": df_to_records(pivots),
        "volume_split": df_to_records(split),
        "iq_summary": iq_summary,
        "auto_smoothing": auto_payload,
        "auto_meta": auto_meta,
        "holidays": holiday_store,
    }
