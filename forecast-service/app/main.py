from __future__ import annotations

import json
import io
import pandas as pd
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import config_manager
import os
from app.pipeline.headcount import (
    business_areas,
    channels_for_scope,
    headcount_template,
    locations,
    preview_headcount,
    save_headcount,
    sites,
    sub_business_areas,
)
from app.pipeline.budget_store import load_budget_rows, upsert_budget_rows
from app.pipeline.dataset_dashboard import dataset_snapshot
from app.pipeline.ops_dashboard import ops_options, refresh_ops
from app.pipeline.newhire_store import ingest_new_hires, load_new_hires, new_hire_template_df, save_new_hires
from app.pipeline.postgres import (
    ensure_budget_schema,
    ensure_newhire_schema,
    ensure_ops_schema,
    ensure_planning_schema,
    ensure_roster_schema,
    ensure_shrinkage_schema,
)
from app.pipeline.roster_store import (
    build_roster_template_wide,
    load_roster_long,
    load_roster_wide,
    normalize_roster_wide,
    save_roster,
)
from app.pipeline.settings_store import load_holidays, load_settings, save_holidays, save_settings
from app.pipeline.shrinkage_store import (
    attrition_weekly_from_raw,
    is_voice_shrinkage_like,
    load_attrition_weekly,
    load_shrinkage_weekly,
    merge_shrink_weekly,
    normalize_shrink_weekly,
    normalize_shrinkage_bo,
    normalize_shrinkage_voice,
    save_attrition_raw,
    save_attrition_weekly,
    save_shrinkage_raw,
    save_shrinkage_weekly,
    shrinkage_bo_raw_template_df,
    shrinkage_voice_raw_template_df,
    summarize_shrinkage_bo,
    summarize_shrinkage_voice,
    weekly_shrinkage_from_bo_summary,
    weekly_shrinkage_from_voice_summary,
)
from app.pipeline.timeseries_store import save_timeseries
from app.pipeline.daily_interval import run_daily_interval
from app.pipeline.ingest import parse_upload_any
from app.pipeline.phase1 import run_phase1
from app.pipeline.phase2 import apply_volume_split, run_phase2
from app.pipeline.prophet_smoothing import run_prophet_smoothing
from app.pipeline.seasonality import apply_seasonality_changes, build_seasonality
from app.pipeline.planning_calc import get_cached_consolidated_calcs, serialize_bundle
from app.pipeline.planning_calc_engine import ensure_plan_calc
from app.pipeline.plan_detail_engine import compute_plan_detail_tables, persist_plan_detail_tables, extract_upper_rows
from app.pipeline.ba_rollup_plan import compute_ba_rollup_tables, month_cols_for_ba, week_cols_for_ba
from app.pipeline.plan_detail._common import (
    clone_plan,
    current_user_fallback,
    extend_plan_weeks,
    get_class_level_options,
    get_class_type_options,
    load_nh_classes,
    next_class_reference,
    save_nh_classes,
    save_plan_meta,
)
from app.pipeline.planning_store import (
    delete_plan,
    list_business_areas,
    list_plans,
    load_plan,
    load_plan_table,
    save_plan_table,
    upsert_plan,
)
from app.pipeline.save import (
    list_saved_forecasts,
    load_saved_forecast,
    load_original_data,
    save_adjusted_forecast,
    save_daily_interval,
    save_forecast_results,
    save_smoothing,
    save_transformations,
)
from app.pipeline.transformations import apply_transformations
from app.pipeline.volume_summary import auto_smoothing_sweep, run_volume_summary, smoothing_core
from app.pipeline.utils import df_from_payload, df_to_records, sanitize_for_json
from app.cap_db import load_df, save_df

app = FastAPI(title="CAP Connect Forecast Service")

_COMBINED_TIMESERIES_KINDS = {
    "voice_forecast",
    "voice_actual",
    "voice_tactical",
    "bo_forecast",
    "bo_actual",
    "bo_tactical",
    "chat_forecast",
    "chat_actual",
    "chat_tactical",
    "ob_forecast",
    "ob_actual",
    "ob_tactical",
}


def _norm_cols(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def _pick_col(cols: dict[str, str], names: list[str]) -> Optional[str]:
    for name in names:
        key = str(name).strip().lower()
        if key in cols:
            return cols[key]
    return None


def _normalize_timeseries(kind: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df, {}
    if kind not in _COMBINED_TIMESERIES_KINDS:
        return df, {}

    prefix, stage = kind.split("_", 1)
    stage = stage.strip().lower()
    cols = _norm_cols(df)
    date_col = _pick_col(cols, ["date", "day", "week"])
    interval_col = _pick_col(cols, ["interval", "time", "interval_start", "start_time", "interval start", "start time"])

    def _build_base(volume_label: str, aht_label: str, item_col: str = "volume"):
        out = pd.DataFrame()
        if date_col:
            out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        if interval_col:
            out["interval"] = df[interval_col].astype(str)
        if volume_label:
            out[item_col] = pd.to_numeric(df[volume_label], errors="coerce")
        if aht_label:
            out["aht_sec"] = pd.to_numeric(df[aht_label], errors="coerce")
        if "date" in out.columns:
            out = out.dropna(subset=["date"])
        return out

    extras: dict[str, pd.DataFrame] = {}
    if prefix == "voice":
        volume_col = _pick_col(
            cols,
            [
                f"{stage} volume",
                f"{stage}_volume",
                "forecast volume",
                "actual volume",
                "tactical volume",
                "volume",
                "vol",
                "calls",
                "offered",
                "items",
                "count",
            ],
        )
        aht_col = _pick_col(
            cols,
            [
                f"{stage} aht",
                f"{stage}_aht",
                "forecast aht",
                "actual aht",
                "tactical aht",
                "aht_sec",
                "aht",
                "avg_aht",
                "aht (sec)",
                "aht_seconds",
                "talk_sec",
            ],
        )
        base = _build_base(volume_col, aht_col, "volume")
        if "volume" in base.columns:
            extras[f"voice_{stage}_volume"] = base[["date"] + (["interval"] if "interval" in base.columns else []) + ["volume"]]
        if "aht_sec" in base.columns:
            extras[f"voice_{stage}_aht"] = base[["date", "aht_sec"]]
        return base if not base.empty else df, extras

    if prefix == "chat":
        items_col = _pick_col(
            cols,
            [
                f"{stage} volume",
                f"{stage}_volume",
                "forecast volume",
                "actual volume",
                "tactical volume",
                "items",
                "chats",
                "volume",
                "count",
            ],
        )
        aht_col = _pick_col(
            cols,
            [
                f"{stage} aht",
                f"{stage}_aht",
                "forecast aht",
                "actual aht",
                "tactical aht",
                "aht_sec",
                "aht",
                "avg_aht",
            ],
        )
        base = _build_base(items_col, aht_col, "items")
        if "items" in base.columns:
            extras[f"chat_{stage}_volume"] = base[["date"] + (["interval"] if "interval" in base.columns else []) + ["items"]]
        if "aht_sec" in base.columns:
            extras[f"chat_{stage}_aht"] = base[["date", "aht_sec"]]
        return base if not base.empty else df, extras

    if prefix == "bo":
        items_col = _pick_col(
            cols,
            [
                f"{stage} volume",
                f"{stage}_volume",
                "forecast volume",
                "actual volume",
                "tactical volume",
                "items",
                "txns",
                "transactions",
                "volume",
            ],
        )
        sut_col = _pick_col(
            cols,
            [
                f"{stage} sut",
                f"{stage}_sut",
                "forecast sut",
                "actual sut",
                "tactical sut",
                "sut_sec",
                "sut",
                "aht_sec",
                "aht",
                "avg_sut",
            ],
        )
        base = _build_base(items_col, sut_col, "items")
        if "items" in base.columns:
            extras[f"bo_{stage}_volume"] = base[["date", "items"]]
        if "aht_sec" in base.columns:
            extras[f"bo_{stage}_sut"] = base[["date", "aht_sec"]].rename(columns={"aht_sec": "sut_sec"})
        return base if not base.empty else df, extras

    if prefix == "ob":
        opc_col = _pick_col(cols, ["opc", "dials", "calls", "attempts", "volume"])
        conn_col = _pick_col(cols, ["connect_rate", "connect%", "connect pct", "connect"])
        rpc_col = _pick_col(cols, ["rpc", "rpc_count"])
        rpc_rate_col = _pick_col(cols, ["rpc_rate", "rpc%", "rpc pct"])
        aht_col = _pick_col(cols, ["aht_sec", "aht", "avg_aht", "talk_sec"])
        base = pd.DataFrame()
        if date_col:
            base["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        if interval_col:
            base["interval"] = df[interval_col].astype(str)
        if opc_col:
            base["opc"] = pd.to_numeric(df[opc_col], errors="coerce")
        if conn_col:
            base["connect_rate"] = pd.to_numeric(df[conn_col], errors="coerce")
        if rpc_col:
            base["rpc"] = pd.to_numeric(df[rpc_col], errors="coerce")
        if rpc_rate_col:
            base["rpc_rate"] = pd.to_numeric(df[rpc_rate_col], errors="coerce")
        if aht_col:
            base["aht_sec"] = pd.to_numeric(df[aht_col], errors="coerce")
        if "date" in base.columns:
            base = base.dropna(subset=["date"])
        if "opc" in base.columns:
            extras[f"ob_{stage}_opc"] = base[["date", "opc"]]
        if "connect_rate" in base.columns:
            extras[f"ob_{stage}_connect_rate"] = base[["date", "connect_rate"]]
        if "rpc" in base.columns:
            extras[f"ob_{stage}_rpc"] = base[["date", "rpc"]]
        if "rpc_rate" in base.columns:
            extras[f"ob_{stage}_rpc_rate"] = base[["date", "rpc_rate"]]
        if "aht_sec" in base.columns:
            extras[f"ob_{stage}_aht"] = base[["date", "aht_sec"]]
        return base if not base.empty else df, extras

    return df, {}

def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS") or os.getenv("CORS_ORIGIN") or ""
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def _startup():
    ensure_budget_schema()
    ensure_ops_schema()
    ensure_newhire_schema()
    ensure_roster_schema()
    ensure_shrinkage_schema()
    ensure_planning_schema()


@app.get("/api/forecast/config")
def get_config():
    return config_manager.load_config()


@app.post("/api/forecast/config")
def save_config(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Config payload must be a JSON object.")
    config_manager.save_config(payload)
    return {"status": "saved", "config": config_manager.load_config()}


@app.get("/api/forecast/settings")
def get_settings(
    scope_type: str = Query("global"),
    location: Optional[str] = None,
    ba: Optional[str] = None,
    sba: Optional[str] = None,
    channel: Optional[str] = None,
    site: Optional[str] = None,
):
    settings = load_settings(scope_type, location, ba, sba, channel, site)
    return {"status": "ok", "settings": settings}


@app.post("/api/forecast/settings")
def save_settings_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Settings payload must be a JSON object.")
    scope = payload.get("scope") or {}
    settings = payload.get("settings") or {}
    if not isinstance(scope, dict) or not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="Scope and settings must be objects.")
    saved = save_settings(
        scope.get("scope_type") or "global",
        scope.get("location"),
        scope.get("business_area"),
        scope.get("sub_business_area"),
        scope.get("channel"),
        scope.get("site"),
        settings,
    )
    return {"status": "saved", "settings": saved}


@app.get("/api/forecast/holidays")
def get_holidays(
    scope_type: str = Query("global"),
    location: Optional[str] = None,
    ba: Optional[str] = None,
    sba: Optional[str] = None,
    channel: Optional[str] = None,
    site: Optional[str] = None,
):
    df = load_holidays(scope_type, location, ba, sba, channel, site)
    return {"status": "ok", "rows": df_to_records(df)}


@app.post("/api/forecast/holidays")
def save_holidays_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Holiday payload must be a JSON object.")
    scope = payload.get("scope") or {}
    rows = payload.get("rows") or []
    df = df_from_payload(rows)
    if not df.empty:
        cols = {str(c).strip().lower(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("holiday_date")
        name_col = cols.get("name") or cols.get("holiday") or cols.get("holiday_name")
        if date_col and date_col != "date":
            df = df.rename(columns={date_col: "date"})
        if name_col and name_col != "name":
            df = df.rename(columns={name_col: "name"})
        if "name" not in df.columns:
            df["name"] = ""
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            df = df.dropna(subset=["date"])
        df = df[["date", "name"]]
    saved = save_holidays(
        scope.get("scope_type") or "global",
        scope.get("location"),
        scope.get("business_area"),
        scope.get("sub_business_area"),
        scope.get("channel"),
        scope.get("site"),
        df,
    )
    return {"status": "saved", "rows": df_to_records(saved)}


@app.post("/api/uploads/timeseries")
def save_timeseries_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Timeseries payload must be a JSON object.")
    kind = payload.get("kind")
    scope_key = payload.get("scope_key") or "global"
    mode = (payload.get("mode") or "append").lower()
    rows = payload.get("rows") or []
    df = df_from_payload(rows)
    base_df, extras = _normalize_timeseries(str(kind or ""), df)
    result = save_timeseries(kind, scope_key, base_df, mode=mode)
    extra_results: dict[str, int] = {}
    for extra_kind, extra_df in extras.items():
        if isinstance(extra_df, pd.DataFrame) and not extra_df.empty:
            save_timeseries(extra_kind, scope_key, extra_df, mode=mode)
            extra_results[extra_kind] = len(extra_df.index)
    payload_out = {"status": result.get("status"), "rows": result.get("rows", 0), "path": result.get("path")}
    if extra_results:
        payload_out["extras"] = extra_results
    return payload_out


@app.get("/api/forecast/budget")
def get_budget(
    ba: Optional[str] = None,
    subba: Optional[str] = None,
    channel: Optional[str] = None,
    site: Optional[str] = None,
):
    if not (ba and subba and channel and site):
        return {"rows": []}
    df = load_budget_rows(ba, subba, channel, site)
    return {"rows": df_to_records(df)}


@app.post("/api/forecast/budget")
def save_budget(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Budget payload must be a JSON object.")
    scope = payload.get("scope") or {}
    rows = payload.get("rows") or []
    if not isinstance(scope, dict):
        raise HTTPException(status_code=400, detail="Scope must be an object.")
    count = upsert_budget_rows(
        scope.get("business_area") or "",
        scope.get("sub_business_area") or "",
        scope.get("channel") or "",
        scope.get("site") or "",
        rows,
    )
    return {"status": "saved", "rows": count}


@app.get("/api/forecast/headcount")
def get_headcount_preview(rows: int = Query(50, ge=1, le=500)):
    return {"status": "ok", "rows": preview_headcount(rows)}


@app.post("/api/forecast/headcount")
def upload_headcount(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Headcount payload must be a JSON object.")
    rows = payload.get("rows") or []
    df = df_from_payload(rows)
    result = save_headcount(df)
    preview = df.head(50).to_dict("records") if not df.empty else []
    return {"status": result.get("status"), "rows": preview, "saved": result}


@app.get("/api/forecast/headcount/options")
def headcount_options(
    ba: Optional[str] = None,
    sba: Optional[str] = None,
    location: Optional[str] = None,
):
    return {
        "business_areas": business_areas(),
        "sub_business_areas": sub_business_areas(ba) if ba else [],
        "locations": locations(ba),
        "sites": sites(ba, location) if ba else [],
        "channels": channels_for_scope(ba, sba),
    }


@app.post("/api/forecast/dataset")
def dataset_snapshot_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Dataset payload must be a JSON object.")
    return dataset_snapshot(
        payload.get("start_date"),
        payload.get("end_date"),
        payload.get("series") or "auto",
        payload.get("ba") or [],
        payload.get("sba") or [],
        payload.get("ch") or [],
        payload.get("loc") or [],
        payload.get("site") or [],
    )


@app.get("/api/forecast/headcount/template")
def headcount_template_endpoint(rows: int = Query(5, ge=1, le=50)):
    df = headcount_template(rows)
    return {"rows": df_to_records(df)}


@app.get("/api/forecast/roster")
def get_roster():
    wide = load_roster_wide()
    long = load_roster_long()
    return {"wide": df_to_records(wide), "long": df_to_records(long)}


@app.post("/api/forecast/roster")
def save_roster_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Roster payload must be a JSON object.")
    wide_rows = payload.get("wide") or []
    long_rows = payload.get("long") or []
    result = save_roster(wide_rows, long_rows)
    return {"status": result.get("status"), "counts": result}


@app.post("/api/forecast/roster/normalize")
def normalize_roster_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Roster payload must be a JSON object.")
    rows = payload.get("rows") or []
    df = df_from_payload(rows)
    norm = normalize_roster_wide(df)
    return {"rows": df_to_records(norm)}


@app.get("/api/forecast/roster/template")
def roster_template_endpoint(
    start: str = Query(...),
    end: str = Query(...),
    sample: bool = Query(False),
):
    df = build_roster_template_wide(start, end, include_sample=sample)
    return {"rows": df_to_records(df)}


def _parse_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


@app.get("/api/forecast/ops/options")
def get_ops_options(
    ba: Optional[str] = None,
    sba: Optional[str] = None,
    ch: Optional[str] = None,
    loc: Optional[str] = None,
):
    return ops_options(_parse_list(ba), _parse_list(sba), _parse_list(ch), _parse_list(loc))


@app.post("/api/forecast/ops/summary")
def get_ops_summary(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Ops payload must be a JSON object.")
    result = refresh_ops(
        payload.get("start_date"),
        payload.get("end_date"),
        payload.get("grain") or "D",
        payload.get("ba") or [],
        payload.get("sba") or [],
        payload.get("ch") or [],
        payload.get("site") or [],
        payload.get("loc") or [],
    )
    return sanitize_for_json(result)


@app.post("/api/planning/calc/consolidated")
def planning_consolidated(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Planning calc payload must be a JSON object.")

    scope = payload.get("scope") or {}
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else None
    plan_key = payload.get("plan_key")
    if not plan_key:
        plan_key = "|".join(
            str(scope.get(k) or "").strip()
            for k in ("business_area", "sub_business_area", "channel", "site")
        ).strip("|")
    plan_key = plan_key or "global"

    plan_date = payload.get("plan_date")
    try:
        plan_date = pd.to_datetime(plan_date, errors="coerce").date() if plan_date else None
    except Exception:
        plan_date = None

    version_token = payload.get("version_token")
    extra = {"scope": scope}

    def _builder():
        return get_cached_consolidated_calcs(scope, settings=settings, plan_date=plan_date, version_token=version_token)

    result, status, meta = ensure_plan_calc(
        plan_key,
        grain="consolidated",
        fw_cols=None,
        whatif=None,
        interval_date=None,
        plan_type=None,
        version_token=version_token,
        builder=_builder,
        extra=extra,
    )

    if status != "ready":
        return {"status": status, "job": meta}
    return {"status": "ready", "job": meta, "data": serialize_bundle(result)}


@app.get("/api/planning/plan")
def get_plans(
    plan_id: Optional[int] = Query(None),
    ba: Optional[str] = Query(None),
    sba: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    site: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    if plan_id:
        return {"plan": load_plan(int(plan_id))}
    return {"plans": list_plans(ba, sba, channel, location, site, status_filter=status)}


@app.get("/api/planning/business-areas")
def planning_business_areas(status: Optional[str] = Query("current")):
    return {"business_areas": list_business_areas(status)}


@app.post("/api/planning/plan")
def save_plan(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Plan payload must be a JSON object.")
    result = upsert_plan(payload)
    if result.get("status") == "missing_dsn":
        raise HTTPException(status_code=503, detail="Planning database is not configured.")
    if result.get("status") == "duplicate":
        raise HTTPException(status_code=409, detail="Plan already exists for the same scope.")
    if not result.get("id"):
        raise HTTPException(status_code=500, detail="Plan was not created.")
    return result


@app.get("/api/planning/plan/scope-options")
def plan_scope_options(plan_id: int = Query(...)):
    plan = load_plan(int(plan_id))
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found.")
    options = []
    plans = list_plans(
        business_area=plan.get("business_area"),
        sub_business_area=plan.get("sub_business_area"),
        channel=plan.get("channel"),
        location=plan.get("location"),
        site=plan.get("site"),
        status_filter=None,
        include_deleted=False,
        limit=200,
    )
    for row in plans:
        if int(row.get("id") or 0) == int(plan_id):
            continue
        status = row.get("status") or ("current" if row.get("is_current") else "history")
        name = row.get("plan_name") or f"Plan {row.get('id')}"
        label = f"{name} (id {row.get('id')}, status {status})"
        options.append(
            {
                "id": row.get("id"),
                "plan_name": name,
                "status": status,
                "is_current": bool(row.get("is_current")),
                "label": label,
            }
        )
    return {"options": options}


@app.post("/api/planning/plan/delete")
def delete_plan_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Delete payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    return delete_plan(int(plan_id))


@app.post("/api/planning/plan/save-as")
def save_plan_as(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Save-as payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    name = (payload.get("name") or "").strip()
    if not plan_id or not name:
        raise HTTPException(status_code=400, detail="plan_id and name are required.")
    try:
        new_id = clone_plan(int(plan_id), name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "created", "id": new_id}


@app.post("/api/planning/plan/extend")
def extend_plan_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Extend payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    weeks = payload.get("weeks")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    if not weeks:
        raise HTTPException(status_code=400, detail="weeks is required.")
    extend_plan_weeks(int(plan_id), int(weeks))
    return {"status": "extended"}


@app.post("/api/planning/plan/export")
def export_plan(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Export payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    buf = io.BytesIO()
    keys = ["fw", "hc", "attr", "shr", "train", "ratio", "seat", "bva", "nh", "emp", "bulk_files", "notes"]
    try:
        data, status, _meta = compute_plan_detail_tables(int(plan_id), grain="week")
        upper_rows = data.get("upper") if status == "ready" and data else []
    except Exception:
        upper_rows = []
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        if upper_rows:
            pd.DataFrame(upper_rows).to_excel(xw, sheet_name="upper", index=False)
        for key in keys:
            rows = load_plan_table(int(plan_id), key)
            df = pd.DataFrame(rows or [])
            df.to_excel(xw, sheet_name=key[:31], index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=plan_{plan_id}_export.xlsx"},
    )


@app.get("/api/planning/plan/whatif")
def get_plan_whatif(plan_id: int = Query(...)):
    df = load_df(f"plan_{plan_id}_whatif")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"overrides": {}}
    row = df.iloc[-1].to_dict()
    overrides = row.get("overrides") or {}
    if isinstance(overrides, str):
        try:
            overrides = json.loads(overrides)
        except Exception:
            overrides = {}
    return {"overrides": overrides}


@app.post("/api/planning/plan/whatif")
def save_plan_whatif(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="What-if payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    action = (payload.get("action") or "apply").lower()
    overrides = payload.get("overrides") if isinstance(payload.get("overrides"), dict) else {}
    if action == "clear":
        overrides = {}
    rec = {
        "ts": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "start_week": "",
        "end_week": "",
        "overrides": overrides,
    }
    save_df(f"plan_{plan_id}_whatif", pd.DataFrame([rec]))
    save_plan_meta(int(plan_id), {"last_updated_on": rec["ts"], "last_updated_by": current_user_fallback()})
    return {"status": "saved", "overrides": overrides}


@app.get("/api/planning/plan/new-hire/classes")
def get_plan_new_hire_classes(plan_id: int = Query(...)):
    df = load_nh_classes(int(plan_id))
    return {
        "rows": df_to_records(df),
        "class_types": get_class_type_options(),
        "class_levels": get_class_level_options(),
    }


@app.post("/api/planning/plan/new-hire/classes")
def save_plan_new_hire_classes(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="New-hire payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    rows = payload.get("rows") or []
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    df = df_from_payload(rows)
    save_nh_classes(int(plan_id), df)
    return {"status": "saved", "rows": df_to_records(load_nh_classes(int(plan_id)))}


@app.post("/api/planning/plan/new-hire/class")
def add_plan_new_hire_class(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="New-hire payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    data = payload.get("data") or {}
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="data must be a JSON object.")
    df = load_nh_classes(int(plan_id))
    now = pd.Timestamp.utcnow().isoformat(timespec="seconds")
    class_ref = next_class_reference(str(plan_id), df)
    row = {
        "class_reference": class_ref,
        "source_system_id": data.get("source_system_id") or class_ref,
        "emp_type": data.get("emp_type") or "full-time",
        "status": data.get("status") or "tentative",
        "class_type": data.get("class_type") or "ramp-up",
        "class_level": data.get("class_level") or "new-agent",
        "grads_needed": data.get("grads_needed") or 0,
        "billable_hc": data.get("billable_hc") or 0,
        "training_weeks": data.get("training_weeks") or 0,
        "nesting_weeks": data.get("nesting_weeks") or 0,
        "induction_start": data.get("induction_start") or "",
        "training_start": data.get("training_start") or "",
        "training_end": data.get("training_end") or "",
        "nesting_start": data.get("nesting_start") or "",
        "nesting_end": data.get("nesting_end") or "",
        "production_start": data.get("production_start") or "",
        "created_by": current_user_fallback(),
        "created_ts": now,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_nh_classes(int(plan_id), df)
    return {"status": "saved", "rows": df_to_records(df)}


@app.post("/api/planning/plan/compare")
def compare_plans(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Compare payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    compare_id = payload.get("compare_id")
    if not plan_id or not compare_id:
        raise HTTPException(status_code=400, detail="plan_id and compare_id are required.")

    def _upper_df(pid: int) -> pd.DataFrame:
        data, status, _meta = compute_plan_detail_tables(int(pid), grain="week")
        if status != "ready" or not data:
            return pd.DataFrame()
        rows = data.get("upper") or []
        df = pd.DataFrame(rows)
        if "metric" in df.columns:
            df = df.set_index("metric")
        return df

    curr_df = _upper_df(int(plan_id))
    comp_df = _upper_df(int(compare_id))
    if curr_df.empty or comp_df.empty:
        raise HTTPException(status_code=404, detail="Upper summary data not available.")

    merged_cols = sorted(set(curr_df.columns) | set(comp_df.columns))
    curr_df = curr_df.reindex(columns=merged_cols).fillna("")
    comp_df = comp_df.reindex(index=curr_df.index, columns=merged_cols).fillna("")
    diff_df = pd.DataFrame(index=curr_df.index)
    for col in merged_cols:
        curr_vals = pd.to_numeric(curr_df[col], errors="coerce")
        comp_vals = pd.to_numeric(comp_df[col], errors="coerce")
        diff = curr_vals - comp_vals
        diff_df[col] = diff.where(curr_vals.notna() & comp_vals.notna(), "")

    def _records(df: pd.DataFrame) -> list[dict]:
        if df.empty:
            return []
        return df.reset_index().to_dict("records")

    return {"current": _records(curr_df), "compare": _records(comp_df), "delta": _records(diff_df)}


@app.get("/api/planning/plan/table")
def get_plan_table(plan_id: int = Query(...), name: str = Query(...)):
    return {"rows": load_plan_table(int(plan_id), name)}


@app.post("/api/planning/plan/table")
def save_plan_table_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Plan table payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    name = payload.get("name")
    rows = payload.get("rows") or []
    if not plan_id or not name:
        raise HTTPException(status_code=400, detail="plan_id and name are required.")
    return save_plan_table(int(plan_id), str(name), rows)


@app.post("/api/planning/plan/detail/compute")
def compute_plan_detail_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Plan detail payload must be a JSON object.")
    plan_id = payload.get("plan_id") or payload.get("id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    grain = payload.get("grain") or "week"
    whatif = payload.get("whatif") if isinstance(payload.get("whatif"), dict) else None
    interval_date = payload.get("interval_date")
    version_token = payload.get("version_token")
    persist = bool(payload.get("persist", True))

    data, status, meta = compute_plan_detail_tables(
        int(plan_id),
        grain=grain,
        whatif=whatif,
        interval_date=interval_date,
        version_token=version_token,
    )
    if status != "ready":
        return sanitize_for_json({"status": status, "job": meta})
    if persist:
        persist_plan_detail_tables(int(plan_id), data or {}, grain=grain, interval_date=interval_date)
    return sanitize_for_json({"status": "ready", "job": meta, "data": data})


@app.post("/api/planning/rollup")
def planning_rollup_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Rollup payload must be a JSON object.")
    ba = payload.get("business_area") or payload.get("ba")
    if not ba:
        raise HTTPException(status_code=400, detail="business_area is required.")
    grain = str(payload.get("grain") or "week").lower()
    whatif = payload.get("whatif") if isinstance(payload.get("whatif"), dict) else None
    status_filter = payload.get("status_filter") or "current"

    if grain == "month":
        fw_cols, _ = month_cols_for_ba(str(ba), status_filter=status_filter)
    else:
        fw_cols, _ = week_cols_for_ba(str(ba), status_filter=status_filter)

    results = compute_ba_rollup_tables(str(ba), fw_cols, whatif=whatif, status_filter=status_filter, grain=grain)
    upper, fw, hc, attr, shr, train, ratio, seat, bva, nh, emp, bulk_files, notes = results
    data = {
        "upper": extract_upper_rows(upper),
        "tables": {
            "fw": fw or [],
            "hc": hc or [],
            "attr": attr or [],
            "shr": shr or [],
            "train": train or [],
            "ratio": ratio or [],
            "seat": seat or [],
            "bva": bva or [],
            "nh": nh or [],
            "emp": emp or [],
            "bulk_files": bulk_files or [],
            "notes": notes or [],
        },
        "grain": grain,
    }
    return {"status": "ready", "data": data}


@app.get("/api/forecast/new-hire")
def get_new_hires():
    df = load_new_hires()
    return {"rows": df_to_records(df)}


@app.post("/api/forecast/new-hire")
def save_new_hires_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="New hire payload must be a JSON object.")
    rows = payload.get("rows") or []
    count = save_new_hires(rows)
    return {"status": "saved", "rows": count}


@app.post("/api/forecast/new-hire/ingest")
def ingest_new_hires_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="New hire payload must be a JSON object.")
    rows = payload.get("rows") or []
    source_id = payload.get("source_id")
    default_ba = payload.get("default_ba")
    df = ingest_new_hires(rows, source_id=source_id, default_ba=default_ba)
    return {"rows": df_to_records(df)}


@app.get("/api/forecast/new-hire/template")
def get_new_hire_template():
    df = new_hire_template_df()
    return {"rows": df_to_records(df)}


@app.get("/api/forecast/shrinkage")
def get_shrinkage_weekly():
    df = load_shrinkage_weekly()
    return {"rows": df_to_records(df)}


@app.post("/api/forecast/shrinkage")
def save_shrinkage_weekly_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Shrinkage payload must be a JSON object.")
    rows = payload.get("rows") or []
    df = normalize_shrink_weekly(rows)
    count = save_shrinkage_weekly(df)
    return {"status": "saved", "rows": df_to_records(df), "count": count}


@app.get("/api/forecast/shrinkage/template")
def shrinkage_template_endpoint(kind: str = Query("voice")):
    kind_norm = str(kind or "").strip().lower()
    if kind_norm in {"bo", "back office", "backoffice"}:
        df = shrinkage_bo_raw_template_df()
    else:
        df = shrinkage_voice_raw_template_df()
    return {"rows": df_to_records(df)}


@app.post("/api/forecast/shrinkage/raw")
def shrinkage_raw_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Shrinkage payload must be a JSON object.")
    kind = str(payload.get("kind") or "").strip().lower()
    rows = payload.get("rows") or []
    save_flag = bool(payload.get("save"))
    df = df_from_payload(rows)

    def _apply_voice_channel(frame: pd.DataFrame, channel: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        if "Channel" in out.columns:
            out["Channel"] = out["Channel"].replace("", pd.NA).fillna(channel)
        else:
            out["Channel"] = channel
        return out

    def _apply_bo_channel(frame: pd.DataFrame, channel: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        if "channel" in out.columns:
            out["channel"] = out["channel"].replace("", pd.NA).fillna(channel)
        else:
            out["channel"] = channel
        return out

    if kind in {"bo", "back office", "backoffice"}:
        if is_voice_shrinkage_like(df):
            norm = _apply_voice_channel(normalize_shrinkage_voice(df), "Back Office")
            daily = summarize_shrinkage_voice(norm)
            weekly = weekly_shrinkage_from_voice_summary(daily)
        else:
            norm = _apply_bo_channel(normalize_shrinkage_bo(df), "Back Office")
            daily = summarize_shrinkage_bo(norm)
            weekly = weekly_shrinkage_from_bo_summary(daily)
        raw_kind = "backoffice"
    elif kind in {"chat"}:
        if is_voice_shrinkage_like(df):
            norm = _apply_voice_channel(normalize_shrinkage_voice(df), "Chat")
            daily = summarize_shrinkage_voice(norm)
            weekly = weekly_shrinkage_from_voice_summary(daily)
        else:
            norm = _apply_bo_channel(normalize_shrinkage_bo(df), "Chat")
            daily = summarize_shrinkage_bo(norm)
            weekly = weekly_shrinkage_from_bo_summary(daily)
        raw_kind = "chat"
    elif kind in {"ob", "outbound"}:
        if is_voice_shrinkage_like(df):
            norm = _apply_voice_channel(normalize_shrinkage_voice(df), "Outbound")
            daily = summarize_shrinkage_voice(norm)
            weekly = weekly_shrinkage_from_voice_summary(daily)
        else:
            norm = _apply_bo_channel(normalize_shrinkage_bo(df), "Outbound")
            daily = summarize_shrinkage_bo(norm)
            weekly = weekly_shrinkage_from_bo_summary(daily)
        raw_kind = "outbound"
    else:
        norm = normalize_shrinkage_voice(df)
        daily = summarize_shrinkage_voice(norm)
        weekly = weekly_shrinkage_from_voice_summary(daily)
        raw_kind = "voice"

    combined = merge_shrink_weekly(load_shrinkage_weekly(), weekly)
    if save_flag:
        save_shrinkage_raw(raw_kind, norm)
        save_shrinkage_weekly(combined)

    return {
        "raw": df_to_records(norm),
        "daily": df_to_records(daily),
        "weekly": df_to_records(weekly),
        "combined": df_to_records(combined),
    }


@app.get("/api/forecast/attrition")
def get_attrition_weekly():
    df = load_attrition_weekly()
    return {"rows": df_to_records(df)}


@app.post("/api/forecast/attrition")
def save_attrition_weekly_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Attrition payload must be a JSON object.")
    rows = payload.get("rows") or []
    df = pd.DataFrame(rows or [])
    count = save_attrition_weekly(df)
    return {"status": "saved", "rows": df_to_records(df), "count": count}


@app.post("/api/forecast/attrition/raw")
def attrition_raw_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Attrition payload must be a JSON object.")
    rows = payload.get("rows") or []
    save_flag = bool(payload.get("save"))
    raw_df = df_from_payload(rows)
    weekly = attrition_weekly_from_raw(raw_df)
    if save_flag:
        save_attrition_raw(raw_df)
        save_attrition_weekly(weekly)
    return {"weekly": df_to_records(weekly)}


@app.post("/api/forecast/volume-summary")
async def volume_summary(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    result = run_volume_summary(file.filename or "upload", content)
    return sanitize_for_json(result)


@app.post("/api/forecast/ingest/original")
async def ingest_original(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    df, msg = parse_upload_any(
        file.filename or "upload",
        content,
        preferred_sheets=("Volume", "Original", "Data"),
    )
    return sanitize_for_json({"message": msg, "rows": df_to_records(df)})


@app.post("/api/forecast/ingest/interval")
async def ingest_interval(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    df, msg = parse_upload_any(
        file.filename or "upload",
        content,
        preferred_sheets=("Interval", "Intervals", "Timeslot", "Time", "Volume"),
    )
    return sanitize_for_json({"message": msg, "rows": df_to_records(df)})


@app.get("/api/forecast/saved-runs")
def list_saved_runs():
    return sanitize_for_json({"runs": list_saved_forecasts()})


@app.get("/api/forecast/saved-runs/{filename}")
def load_saved_run(filename: str):
    result = load_saved_forecast(filename)
    if not result.get("rows"):
        raise HTTPException(status_code=404, detail="Saved forecast not found.")
    return sanitize_for_json(result)


@app.get("/api/forecast/original-data")
def get_original_data():
    result = load_original_data()
    if not result.get("rows"):
        raise HTTPException(status_code=404, detail="Original data not found.")
    return sanitize_for_json(result)


@app.post("/api/forecast/smoothing")
async def smoothing(payload: dict):
    data = payload.get("data")
    window = payload.get("window", 6)
    threshold = payload.get("threshold", 2.5)
    prophet_order = payload.get("prophet_order")

    df = df_from_payload(data)
    if df.empty:
        raise HTTPException(status_code=400, detail="Smoothing input data is empty.")

    warning = None
    try:
        result = smoothing_core(df, int(window), float(threshold), prophet_order)
    except Exception as exc:
        if prophet_order:
            try:
                result = smoothing_core(df, int(window), float(threshold), None)
                warning = f"Prophet failed; used EWMA instead. {exc}"
            except Exception as exc2:
                raise HTTPException(status_code=400, detail=str(exc2))
        else:
            raise HTTPException(status_code=400, detail=str(exc))

    return sanitize_for_json({
        "status": "smoothing_complete",
        "results": {
            "smoothed": result["smoothed"].to_dict("records"),
            "anomalies": result["anomalies"].to_dict("records"),
            "ratio": result["ratio"].to_dict("records"),
            "capped": result["capped"].to_dict("records"),
            "pivot": result["pivot"].to_dict("records"),
        },
        "warning": warning,
    })


@app.post("/api/forecast/smoothing/auto")
async def smoothing_auto(payload: dict):
    data = payload.get("data")
    windows = payload.get("windows")
    thresholds = payload.get("thresholds")

    df = df_from_payload(data)
    if df.empty:
        raise HTTPException(status_code=400, detail="Smoothing input data is empty.")

    auto_payload, meta = auto_smoothing_sweep(df, windows=windows, thresholds=thresholds)
    if not auto_payload:
        raise HTTPException(status_code=400, detail="Auto smoothing failed.")
    return sanitize_for_json({"status": "auto_smoothing_complete", "results": auto_payload, "meta": meta})


@app.post("/api/forecast/phase1")
async def phase1(payload: dict):
    data = payload.get("data")
    config = payload.get("config")
    holidays = payload.get("holidays")

    result = run_phase1(data, config=config, holidays=holidays)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Phase 1 failed."))
    return sanitize_for_json(result)


@app.post("/api/forecast/phase2")
async def phase2(payload: dict):
    data = payload.get("data")
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    config = payload.get("config")
    iq_summary = payload.get("iq_summary")
    volume_summary = payload.get("volume_summary")
    basis = payload.get("basis", "iq")
    volume_data = payload.get("volume_data")
    category = payload.get("category")

    if not start_date or not end_date:
        raise HTTPException(status_code=400, detail="start_date and end_date are required.")

    result = run_phase2(
        data,
        start_date=start_date,
        end_date=end_date,
        config=config,
        iq_summary=iq_summary,
        volume_summary=volume_summary,
        basis=basis,
        volume_data=volume_data,
        category=category,
    )
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Phase 2 failed."))
    return sanitize_for_json(result)


@app.post("/api/forecast/transformations/apply")
async def transformations_apply(payload: dict):
    data = payload.get("data")
    result = apply_transformations(data)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Transformations failed."))
    return sanitize_for_json(result)


@app.post("/api/forecast/seasonality/build")
async def build_seasonality_endpoint(payload: dict):
    ratio_table = payload.get("ratio_table")
    result = build_seasonality(ratio_table)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Seasonality failed."))
    return sanitize_for_json(result)


@app.post("/api/forecast/seasonality/apply")
async def apply_seasonality_endpoint(payload: dict):
    capped_rows = payload.get("capped_rows")
    lower_cap = payload.get("lower_cap")
    upper_cap = payload.get("upper_cap")
    base_volume = payload.get("base_volume")
    result = apply_seasonality_changes(capped_rows, lower_cap, upper_cap, base_volume)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Seasonality failed."))
    return sanitize_for_json(result)


@app.post("/api/forecast/volume-summary/prophet-smoothing")
async def volume_summary_prophet(payload: dict):
    normalized_ratio = payload.get("normalized_ratio")
    ratio_table = payload.get("ratio_table")
    iq_table = payload.get("iq_table")
    holidays = payload.get("holidays")
    result = run_prophet_smoothing(normalized_ratio, ratio_table, iq_table, holidays)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Prophet smoothing failed."))
    return sanitize_for_json(result)


@app.post("/api/forecast/phase2/volume-split")
async def phase2_volume_split(payload: dict):
    base_df = payload.get("base_df")
    split_rows = payload.get("split_rows")
    fg_monthly = payload.get("forecast_group_monthly")
    result = apply_volume_split(base_df, split_rows, fg_monthly)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Volume split failed."))
    return sanitize_for_json(result)


@app.post("/api/forecast/daily-interval")
async def daily_interval(payload: dict):
    transform_df = payload.get("transform_df")
    interval_df = payload.get("interval_df")
    forecast_month = payload.get("forecast_month")
    group_value = payload.get("group_value")
    model_value = payload.get("model_value")
    distribution_override = payload.get("distribution_override")
    original_data = payload.get("original_data")
    holidays = payload.get("holidays")
    group_level = payload.get("group_level", "forecast_group")

    if not forecast_month:
        raise HTTPException(status_code=400, detail="forecast_month is required.")

    result = run_daily_interval(
        transform_payload=transform_df,
        interval_payload=interval_df,
        forecast_month=forecast_month,
        group_value=group_value,
        model_value=model_value,
        distribution_override=distribution_override,
        original_data=original_data,
        holidays=holidays,
        group_level=group_level,
    )
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Daily/interval failed."))
    return sanitize_for_json(result)


@app.post("/api/forecast/save/smoothing")
async def save_smoothing_results(payload: dict):
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, dict):
        payload = results
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid smoothing payload.")
    if not payload.get("smoothed") and not payload.get("capped"):
        raise HTTPException(status_code=400, detail="No smoothing data to save.")
    return sanitize_for_json(save_smoothing(payload))


@app.post("/api/forecast/save/forecast-results")
async def save_forecast(payload: dict):
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, dict):
        payload = results
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid forecast payload.")
    if not payload.get("combined") and not payload.get("accuracy"):
        raise HTTPException(status_code=400, detail="No forecast results to save.")
    return sanitize_for_json(save_forecast_results(payload))


@app.post("/api/forecast/save/adjusted-forecast")
async def save_adjusted(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid adjusted forecast payload.")
    rows = (
        payload.get("data")
        or payload.get("rows")
        or payload.get("base")
        or payload.get("results")
        or payload.get("adjusted")
    )
    df = df_from_payload(rows)
    if df.empty:
        raise HTTPException(status_code=400, detail="No adjusted forecast rows to save.")
    return sanitize_for_json(save_adjusted_forecast(df, group_name=payload.get("group_name")))


@app.post("/api/forecast/save/transformations")
async def save_transformations_results(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid transformations payload.")
    rows = payload.get("data") or payload.get("rows") or payload.get("final")
    df = df_from_payload(rows)
    if df.empty:
        raise HTTPException(status_code=400, detail="No transformed rows to save.")
    return sanitize_for_json(save_transformations(df))


@app.post("/api/forecast/save/daily-interval")
async def save_daily_interval_results(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid daily/interval payload.")
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    daily = payload.get("daily") or results.get("daily")
    interval = payload.get("interval") or results.get("interval")
    meta = payload.get("meta") or results.get("meta") or {}
    if not daily and not interval:
        raise HTTPException(status_code=400, detail="No daily/interval rows to save.")
    return sanitize_for_json(save_daily_interval({"daily": daily or [], "interval": interval or [], "meta": meta}))
