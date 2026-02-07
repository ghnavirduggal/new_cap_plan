from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _exports_dir() -> Path:
    outdir = _repo_root() / "exports"
    outdir.mkdir(exist_ok=True)
    return outdir


def _read_base_dir(filename: str) -> Optional[Path]:
    path = _repo_root() / filename
    if not path.exists():
        return None
    try:
        txt = path.read_text().strip()
    except Exception:
        return None
    if not txt:
        return None
    try:
        base_dir = Path(txt)
        base_dir.mkdir(exist_ok=True, parents=True)
        return base_dir
    except Exception:
        return None


def _safe_user() -> str:
    user = os.getenv("USERNAME") or os.getenv("USER") or "user"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(user)).strip("_")
    return cleaned or "user"


def save_smoothing(payload: dict) -> dict:
    outdir = _exports_dir()
    smoothed = pd.DataFrame(payload.get("smoothed", []))
    seasonality = pd.DataFrame(payload.get("capped", []))
    smoothed_path = outdir / "smoothing_smoothed.csv"
    seasonality_path = outdir / "smoothing_seasonality.csv"
    smoothed.to_csv(smoothed_path, index=False)
    seasonality.to_csv(seasonality_path, index=False)
    return {
        "status": "saved",
        "paths": [str(smoothed_path), str(seasonality_path)],
    }


def save_forecast_results(payload: dict) -> dict:
    outdir = _exports_dir()
    combined = pd.DataFrame(payload.get("combined", []))
    accuracy = pd.DataFrame(payload.get("accuracy", []))
    combined_path = outdir / "forecast_results.csv"
    accuracy_path = outdir / "forecast_accuracy.csv"
    combined.to_csv(combined_path, index=False)
    accuracy.to_csv(accuracy_path, index=False)
    return {
        "status": "saved",
        "paths": [str(combined_path), str(accuracy_path)],
    }


def save_adjusted_forecast(df: pd.DataFrame, group_name: Optional[str] = None) -> dict:
    base_dir = _read_base_dir("latest_forecast_base_dir.txt") or _exports_dir()
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    user = _safe_user()
    group = re.sub(r"[^A-Za-z0-9_-]+", "_", str(group_name or "forecast_group")).strip("_") or "forecast_group"
    filename = f"Monthly_Forecast_{group}_{ts}_{user}.csv"
    fpath = base_dir / filename
    df.to_csv(fpath, index=False)
    return {"status": "saved", "paths": [str(fpath)]}


def save_transformations(df: pd.DataFrame) -> dict:
    base_dir = _read_base_dir("latest_forecast_base_dir.txt") or _exports_dir()
    if "Forecast_Marketing Campaign 3" in df.columns and "Final_Forecast_Post_Transformations" not in df.columns:
        df = df.copy()
        df["Final_Forecast_Post_Transformations"] = df["Forecast_Marketing Campaign 3"]

    transformation_columns = [
        "Month_Year",
        "Base_Forecast_for_Forecast_Group",
        "Final_Forecast_Post_Transformations",
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
    if "forecast_group" in df.columns:
        transformation_columns.insert(0, "forecast_group")
    available_cols = [col for col in transformation_columns if col in df.columns]
    export_df = df[available_cols].copy()

    ts = pd.Timestamp.now()
    date_str = ts.strftime("%d_%b_%Y")
    time_str = ts.strftime("%H-%M-%S")
    owner = _safe_user()
    fname = f"Monthly_Forecast_with_Adjustments_{date_str}_{time_str}_{owner}.csv"
    fpath = base_dir / fname
    export_df.to_csv(fpath, index=False)
    return {"status": "saved", "paths": [str(fpath)]}


def save_daily_interval(payload: dict) -> dict:
    daily = pd.DataFrame(payload.get("daily", []))
    interval = pd.DataFrame(payload.get("interval", []))
    meta = payload.get("meta", {}) or {}
    month_label = meta.get("month") or ""
    month_folder = None
    if month_label:
        try:
            month_dt = pd.to_datetime(month_label, errors="coerce")
        except Exception:
            month_dt = pd.NaT
        if pd.notna(month_dt):
            month_folder = month_dt.strftime("%b_%Y")

    base_dir = _read_base_dir("saved_folder_path.txt") or _read_base_dir("latest_forecast_base_dir.txt") or _exports_dir()
    output_dir = base_dir / month_folder if month_folder else base_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    ts = pd.Timestamp.now()
    daily_path = output_dir / f"Final_Daily_Forecast_{ts:%Y%m%d_%H%M%S}.csv"
    interval_path = output_dir / f"Interval_Daily_Forecast_{ts:%Y%m%d_%H%M%S}.csv"
    daily.to_csv(daily_path, index=False)
    interval.to_csv(interval_path, index=False)
    try:
        (output_dir / "latest_forecast_path.txt").write_text(str(daily_path))
    except Exception:
        pass
    return {"status": "saved", "paths": [str(daily_path), str(interval_path)]}


def _candidate_base_dirs() -> list[Path]:
    candidates = []
    for filename in ("latest_forecast_base_dir.txt", "saved_folder_path.txt"):
        base = _read_base_dir(filename)
        if base is not None:
            candidates.append(base)
    candidates.append(_exports_dir())
    unique = []
    seen = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def list_saved_forecasts() -> list[dict]:
    patterns = (
        "Monthly_Forecast_*.csv",
        "Monthly_Forecast_with_Adjustments_*.csv",
    )
    runs: dict[str, dict] = {}
    for base_dir in _candidate_base_dirs():
        for pattern in patterns:
            for path in base_dir.glob(pattern):
                try:
                    stat = path.stat()
                except Exception:
                    continue
                name = path.name
                if name in runs:
                    continue
                updated = pd.to_datetime(stat.st_mtime, unit="s", errors="coerce")
                label_ts = updated.strftime("%d %b %Y %H:%M") if pd.notna(updated) else "unknown time"
                runs[name] = {
                    "name": name,
                    "label": f"{name} (saved {label_ts})",
                    "updated_at": label_ts,
                    "mtime": float(stat.st_mtime or 0.0),
                }
    return sorted(
        runs.values(),
        key=lambda item: item.get("mtime", 0.0),
        reverse=True,
    )


def load_saved_forecast(filename: str) -> dict:
    if not filename:
        return {"status": "missing", "rows": []}
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".csv") or not safe_name.lower().startswith("monthly_forecast"):
        return {"status": "invalid", "rows": []}
    for base_dir in _candidate_base_dirs():
        path = base_dir / safe_name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame()
        return {"status": "ok", "rows": df.to_dict("records"), "name": safe_name}
    return {"status": "not_found", "rows": [], "name": safe_name}


def load_original_data() -> dict:
    path = _exports_dir() / "original_data.csv"
    if not path.exists():
        return {"status": "not_found", "rows": []}
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    return {"status": "ok", "rows": df.to_dict("records")}
