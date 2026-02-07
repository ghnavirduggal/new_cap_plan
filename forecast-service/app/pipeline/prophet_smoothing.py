from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

from forecasting.process_and_IQ_data import clean_and_convert_millions
from .utils import df_from_payload, df_to_records


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _month_name_to_num(name: str) -> Optional[int]:
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
    if not name:
        return None
    return month_map.get(str(name).strip().lower()[:3])


def _table_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if df is None or df.empty or "Year" not in df.columns:
        return pd.DataFrame()
    month_cols = [c for c in df.columns if c != "Year"]
    if not month_cols:
        return pd.DataFrame()
    long_df = df.melt(id_vars=["Year"], value_vars=month_cols, var_name="Month", value_name=value_name)
    long_df["Month_num"] = long_df["Month"].apply(_month_name_to_num)
    long_df = long_df.dropna(subset=["Month_num"])
    long_df["ds"] = pd.to_datetime(
        long_df["Year"].astype(str)
        + "-"
        + long_df["Month_num"].astype(int).astype(str).str.zfill(2)
        + "-01",
        errors="coerce",
    )
    return long_df.dropna(subset=["ds"])


def _iq_table_to_long(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Year" not in df.columns:
        return pd.DataFrame()
    month_cols = [c for c in df.columns if c != "Year"]
    clean_df = df.copy()
    for col in month_cols:
        clean_df[col] = clean_and_convert_millions(clean_df[col])
    long_df = clean_df.melt(id_vars=["Year"], value_vars=month_cols, var_name="Month", value_name="IQ_value")
    long_df["Month_num"] = long_df["Month"].apply(_month_name_to_num)
    long_df = long_df.dropna(subset=["Month_num"])
    long_df["ds"] = pd.to_datetime(
        long_df["Year"].astype(str)
        + "-"
        + long_df["Month_num"].astype(int).astype(str).str.zfill(2)
        + "-01",
        errors="coerce",
    )
    return long_df.dropna(subset=["ds"])


def _prophet_param_grid(include_holidays: bool) -> list[dict]:
    cps = [0.003, 0.01, 0.03, 0.1, 0.3]
    crange = [0.8, 0.9, 0.95]
    sps = [0.1, 0.3, 1, 3, 10]
    smode = ["additive", "multiplicative"]
    n_chg = [3, 5, 8, 10]
    y_fourier = [3, 4, 5, 6, 8]
    hps = [0.1, 0.3, 1, 3] if include_holidays else [None]

    combos = []
    for cp in cps:
        for cr in crange:
            for sp in sps:
                for mode in smode:
                    for ncp in n_chg:
                        for yf in y_fourier:
                            for hp in hps:
                                combos.append(
                                    {
                                        "changepoint_prior_scale": cp,
                                        "changepoint_range": cr,
                                        "seasonality_prior_scale": sp,
                                        "seasonality_mode": mode,
                                        "n_changepoints": ncp,
                                        "yearly_fourier_order": yf,
                                        "holidays_prior_scale": hp,
                                    }
                                )
    max_candidates = 10
    if len(combos) > max_candidates:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(combos), size=max_candidates, replace=False)
        combos = [combos[i] for i in idx]
    return combos


def _prophet_cv_splits(n: int) -> list[tuple[int, int]]:
    splits = []
    for train_len in [15, 18, 21]:
        if n >= train_len + 3:
            splits.append((train_len, 3))
    if not splits and n >= 6:
        splits.append((n - 3, 3))
    return splits


def _residual_anomaly_rate(residuals: np.ndarray, z_thresh: float = 3.5) -> float:
    if residuals.size == 0:
        return 0.0
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    if mad == 0:
        std = np.std(residuals)
        if std == 0:
            return 0.0
        zscores = np.abs((residuals - np.mean(residuals)) / std)
    else:
        zscores = 0.6745 * np.abs(residuals - median) / mad
    return float(np.mean(zscores > z_thresh))


def _prophet_score_candidate(
    df: pd.DataFrame,
    params: dict,
    splits: list[tuple[int, int]],
    regressors: list[str],
    holiday_df: Optional[pd.DataFrame],
) -> float:
    scores = []
    anomaly_threshold = 0.5
    for train_len, horizon in splits:
        train = df.iloc[:train_len].copy()
        val = df.iloc[train_len : train_len + horizon].copy()
        if train.empty or val.empty:
            continue

        m = Prophet(
            changepoint_prior_scale=params["changepoint_prior_scale"],
            changepoint_range=params["changepoint_range"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            seasonality_mode=params["seasonality_mode"],
            n_changepoints=params["n_changepoints"],
            yearly_seasonality=int(params["yearly_fourier_order"]),
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holiday_df if params.get("holidays_prior_scale") is not None else None,
            holidays_prior_scale=params.get("holidays_prior_scale") or 0.1,
        )
        reg_cols = []
        for reg in regressors:
            if reg in train.columns:
                m.add_regressor(reg)
                reg_cols.append(reg)
        fit_cols = ["ds", "y"] + reg_cols
        m.fit(train[fit_cols])

        pred = m.predict(val[["ds"] + reg_cols])
        yhat = pred["yhat"].values
        y = val["y"].values
        residuals = y - yhat
        anomaly_rate = _residual_anomaly_rate(residuals)
        if anomaly_rate > anomaly_threshold:
            return float("inf")
        denom = np.sum(np.abs(y)) or 1e-9
        wape = np.sum(np.abs(y - yhat)) / denom
        bias_pct = np.sum(yhat - y) / denom
        scores.append(wape + 0.5 * abs(bias_pct))
    if not scores:
        return float("inf")
    return float(np.mean(scores))


def _prophet_cv_best(
    df: pd.DataFrame,
    holiday_df: Optional[pd.DataFrame],
    regressors: list[str],
) -> tuple[dict, float]:
    candidates = _prophet_param_grid(include_holidays=holiday_df is not None)
    splits = _prophet_cv_splits(len(df))
    best_score = float("inf")
    best_params = candidates[0] if candidates else {}
    for params in candidates:
        score = _prophet_score_candidate(df, params, splits, regressors, holiday_df)
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, best_score


def _prophet_fit_full(
    df: pd.DataFrame,
    params: dict,
    regressors: list[str],
    holiday_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    model = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        changepoint_range=params["changepoint_range"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        n_changepoints=params["n_changepoints"],
        yearly_seasonality=int(params["yearly_fourier_order"]),
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=holiday_df if params.get("holidays_prior_scale") is not None else None,
        holidays_prior_scale=params.get("holidays_prior_scale") or 0.1,
    )
    reg_cols = []
    for reg in regressors:
        if reg in df.columns:
            model.add_regressor(reg)
            reg_cols.append(reg)
    fit_cols = ["ds", "y"] + reg_cols
    model.fit(df[fit_cols])
    pred = model.predict(df[["ds"] + reg_cols])
    return pred


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


def _line_chart_data(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "ds" not in df.columns:
        return {}
    labels = pd.to_datetime(df["ds"], errors="coerce").dt.strftime("%b-%Y").tolist()
    return {
        "x": labels,
        "series": [
            {"name": "Normalized Ratio 1", "points": [{"x": labels[i], "y": float(v) if pd.notna(v) else None} for i, v in enumerate(df["Normalized_Ratio_1"])]},
            {"name": "Final Smoothed Value", "points": [{"x": labels[i], "y": float(v) if pd.notna(v) else None} for i, v in enumerate(df["Final_Smoothed_Value"])]},
        ],
    }


def _ensure_ds(df: pd.DataFrame) -> pd.DataFrame:
    df_work = df.copy()
    if "ds" in df_work.columns and df_work["ds"].notna().any():
        df_work["ds"] = pd.to_datetime(df_work["ds"], errors="coerce")
        return df_work
    if "Month_Year" in df_work.columns:
        df_work["ds"] = pd.to_datetime(df_work["Month_Year"], format="%b-%Y", errors="coerce")
        if df_work["ds"].isna().all():
            df_work["ds"] = pd.to_datetime(df_work["Month_Year"], errors="coerce")
        return df_work
    if "Year" in df_work.columns and "Month" in df_work.columns:
        month_num = df_work["Month"].apply(_month_name_to_num)
        date_str = (
            df_work["Year"].astype(str)
            + "-"
            + month_num.astype("Int64").astype(str).str.zfill(2)
            + "-01"
        )
        df_work["ds"] = pd.to_datetime(date_str, errors="coerce")
    return df_work


def save_prophet_changes(edited_payload: Any, original_payload: Any = None) -> dict:
    new_df = df_from_payload(edited_payload)
    if new_df.empty:
        return {"status": "No data to save.", "results": {}}

    orig_df = df_from_payload(original_payload) if original_payload is not None else pd.DataFrame()

    changed = True
    if not orig_df.empty and "Final_Smoothed_Value" in new_df.columns and "Final_Smoothed_Value" in orig_df.columns:
        try:
            changed = not np.allclose(
                pd.to_numeric(new_df["Final_Smoothed_Value"], errors="coerce"),
                pd.to_numeric(orig_df["Final_Smoothed_Value"], errors="coerce"),
                equal_nan=True,
            )
        except Exception:
            changed = True

    new_df = _ensure_ds(new_df)
    if "ds" in new_df.columns:
        new_df["ds"] = pd.to_datetime(new_df["ds"], errors="coerce")
    new_df["Final_Smoothed_Value"] = pd.to_numeric(new_df.get("Final_Smoothed_Value"), errors="coerce")
    new_df = new_df.dropna(subset=["ds", "Final_Smoothed_Value"]).copy()
    if new_df.empty:
        return {"status": "No valid smoothed rows found.", "results": {}}

    new_df["ds"] = new_df["ds"].dt.to_period("M").dt.to_timestamp()
    if new_df["ds"].duplicated().any():
        numeric_cols = new_df.select_dtypes(include=[np.number]).columns.tolist()
        agg = {col: "mean" for col in numeric_cols}
        for col in new_df.columns:
            if col not in numeric_cols and col != "ds":
                agg[col] = "first"
        new_df = new_df.groupby("ds", as_index=False).agg(agg)
    new_df = new_df.sort_values("ds").reset_index(drop=True)

    new_df["Year"] = new_df["ds"].dt.year
    new_df["Month"] = new_df["ds"].dt.strftime("%b")

    norm2_pivot = (
        new_df.pivot_table(index="Year", columns="Month", values="Final_Smoothed_Value")
        .reset_index()
    )
    month_cols = [m for m in _MONTHS if m in norm2_pivot.columns]
    norm2_pivot["Avg"] = norm2_pivot[month_cols].mean(axis=1).round(1)
    cols = ["Year"] + month_cols + ["Avg"]
    norm2_pivot = norm2_pivot[cols]

    status = (
        "No changes detected - Phase 1 is ready immediately! You can run Phase 1 below (no need to Save Changes if there aren't any!)."
        if not changed
        else "Changes saved. Phase 1 is ready."
    )

    line_df = new_df.copy()
    if "Normalized Ratio 1" in line_df.columns and "Normalized_Ratio_1" not in line_df.columns:
        line_df = line_df.rename(columns={"Normalized Ratio 1": "Normalized_Ratio_1"})

    return {
        "status": status,
        "results": {
            "prophet_table": df_to_records(new_df),
            "norm2": df_to_records(norm2_pivot),
            "changed": changed,
            "ready": True,
            "norm2_chart": _ratio_chart_data(norm2_pivot),
            "line_chart": _line_chart_data(line_df),
        },
    }


def run_prophet_smoothing(
    normalized_payload: Any,
    ratio_payload: Any,
    iq_payload: Any,
    holiday_payload: Optional[dict] = None,
) -> dict:
    normalized_df = df_from_payload(normalized_payload)
    if normalized_df.empty:
        return {"status": "Normalized ratio not found.", "results": {}}
    normalized_long = _table_to_long(normalized_df, "Normalized_Ratio_1")
    if normalized_long.empty:
        return {"status": "Normalized ratio is empty.", "results": {}}

    ratio_long = pd.DataFrame()
    ratio_df = df_from_payload(ratio_payload)
    if not ratio_df.empty:
        ratio_long = _table_to_long(ratio_df, "Original_Contact_Ratio")

    iq_df = df_from_payload(iq_payload)
    iq_long = _iq_table_to_long(iq_df)
    if iq_long.empty:
        return {"status": "IQ data missing.", "results": {}}

    df_long = normalized_long.merge(iq_long[["ds", "IQ_value"]], on="ds", how="left")
    if not ratio_long.empty:
        df_long = df_long.merge(ratio_long[["ds", "Original_Contact_Ratio"]], on="ds", how="left")

    df_long["IQ_value"] = pd.to_numeric(df_long["IQ_value"], errors="coerce").fillna(1.0)
    df_long["ds"] = pd.to_datetime(df_long["ds"], errors="coerce")
    df_long = df_long.dropna(subset=["ds"]).copy()
    df_long["ds"] = df_long["ds"].dt.to_period("M").dt.to_timestamp()
    if df_long["ds"].duplicated().any():
        numeric_cols = df_long.select_dtypes(include=[np.number]).columns.tolist()
        agg = {col: "mean" for col in numeric_cols}
        for col in df_long.columns:
            if col not in numeric_cols and col != "ds":
                agg[col] = "first"
        df_long = df_long.groupby("ds", as_index=False).agg(agg)
    df_long = df_long.sort_values("ds").reset_index(drop=True)

    scaler = MinMaxScaler()
    df_long["IQ_value_scaled"] = scaler.fit_transform(df_long[["IQ_value"]]).round(4)
    df_long["y"] = df_long["Normalized_Ratio_1"]

    holiday_df = None
    mapping = holiday_payload.get("mapping", {}) if isinstance(holiday_payload, dict) else {}
    if mapping:
        holiday_df = pd.DataFrame(
            {"ds": pd.to_datetime(list(mapping.keys()), errors="coerce"), "holiday": list(mapping.values())}
        ).dropna(subset=["ds"])

    regressors = ["IQ_value_scaled"]
    best_params: dict[str, Any] = {}
    best_score: Optional[float] = None
    warning = None
    use_prophet = True
    try:
        best_params, best_score = _prophet_cv_best(df_long, holiday_df, regressors)
        pred = _prophet_fit_full(df_long, best_params, regressors, holiday_df)
        df_long["Normalized_Ratio_Post_Prophet"] = pred["yhat"].values.round(4)
    except Exception as exc:
        use_prophet = False
        warning = f"Prophet smoothing failed; using normalized ratio. {exc}"
        df_long["Normalized_Ratio_Post_Prophet"] = df_long["Normalized_Ratio_1"]
    df_long["Normalized_Volume"] = (
        pd.to_numeric(df_long["Normalized_Ratio_Post_Prophet"], errors="coerce")
        * pd.to_numeric(df_long["IQ_value"], errors="coerce")
    ).round(4)
    df_long["Final_Smoothed_Value"] = df_long["Normalized_Ratio_Post_Prophet"].round(4)
    df_long["Year"] = df_long["ds"].dt.year
    df_long["Month"] = df_long["ds"].dt.strftime("%b")
    df_long["Month_Year"] = df_long["ds"].dt.strftime("%b-%Y")
    if "Original_Contact_Ratio" in df_long.columns:
        df_long["Contact_Ratio"] = df_long["Original_Contact_Ratio"]
    else:
        df_long["Contact_Ratio"] = np.nan
    df_long["IQ_Value_Scaled"] = df_long["IQ_value_scaled"]

    holiday_name_map = {}
    if holiday_df is not None and not holiday_df.empty and "ds" in holiday_df.columns:
        holiday_df = holiday_df.copy()
        holiday_df["Month_Year"] = pd.to_datetime(holiday_df["ds"], errors="coerce").dt.strftime("%b-%Y")
        name_col = "holiday" if "holiday" in holiday_df.columns else None
        if name_col:
            holiday_name_map = (
                holiday_df.groupby("Month_Year")[name_col]
                .apply(lambda x: ", ".join(sorted({str(v) for v in x if str(v).strip()})))
                .to_dict()
            )
    df_long["Holiday_Name"] = df_long["Month_Year"].map(holiday_name_map).fillna("")

    norm2_pivot = (
        df_long.pivot_table(index="Year", columns="Month", values="Final_Smoothed_Value")
        .reset_index()
    )
    month_cols = [m for m in _MONTHS if m in norm2_pivot.columns]
    norm2_pivot["Avg"] = norm2_pivot[month_cols].mean(axis=1).round(1)
    cols = ["Year"] + month_cols + ["Avg"]
    norm2_pivot = norm2_pivot[cols]

    table_cols = [
        "Year",
        "Month",
        "Contact_Ratio",
        "Month_Year",
        "Holiday_Name",
        "Normalized_Ratio_1",
        "IQ_Value_Scaled",
        "Normalized_Ratio_Post_Prophet",
        "Normalized_Volume",
        "Final_Smoothed_Value",
        "ds",
    ]
    table_df = df_long[table_cols].copy()
    table_df = table_df.rename(columns={"Normalized_Ratio_1": "Normalized Ratio 1"})

    status_msg = (
        f"Prophet smoothing complete. Best score (WAPE + 0.5*bias): {best_score:.4f}"
        if use_prophet and best_score is not None
        else "Prophet smoothing fallback applied."
    )

    return {
        "status": status_msg,
        "warning": warning,
        "results": {
            "prophet_table": df_to_records(table_df),
            "norm2": df_to_records(norm2_pivot),
            "params": best_params,
            "score": best_score,
            "norm2_chart": _ratio_chart_data(norm2_pivot),
            "line_chart": _line_chart_data(table_df.rename(columns={"Normalized Ratio 1": "Normalized_Ratio_1"})),
        },
    }
