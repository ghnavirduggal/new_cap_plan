from __future__ import annotations

import json
import io
import re
import numpy as np
import pandas as pd
import logging
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import config_manager
from app import cap_store, plan_store, security
from app.cap_store import resolve_settings as resolve_cap_settings
from app.pipeline import settings_store
from app.pipeline import user_store
import os
import time
from app.pipeline.headcount import (
    business_areas,
    channels_for_scope,
    critical_team_options,
    headcount_template,
    locations,
    preview_headcount,
    save_headcount,
    sites,
    sub_business_areas,
)
from app.pipeline.budget_store import load_budget_rows, upsert_budget_rows
from app.pipeline.dataset_dashboard import dataset_snapshot
from app.pipeline.ops_dashboard import ops_options, refresh_ops_async, refresh_ops_part, workforce_preview
from app.pipeline.ops_store import (
    get_latest_timeseries_hash,
    load_timeseries_any,
    normalize_scope_key,
    record_timeseries_upload_hash,
    timeseries_dataset_hash,
    preview_timeseries_rows,
)
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
    _parse_date_series,
    attrition_weekly_from_raw,
    is_voice_shrinkage_like,
    load_attrition_raw,
    load_attrition_weekly,
    load_shrinkage_weekly,
    merge_shrink_weekly,
    normalize_attrition_raw_upload,
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
from app.pipeline.prophet_smoothing import run_prophet_smoothing, save_prophet_changes
from app.pipeline.seasonality import apply_seasonality_changes, build_seasonality
from app.pipeline.planning_calc import get_cached_consolidated_calcs, serialize_bundle
from app.pipeline.capacity_core import required_fte_daily, voice_requirements_interval, voice_rollups
from app.pipeline.planning_calc_engine import ensure_plan_calc
from app.pipeline.plan_detail_engine import (
    compute_plan_detail_tables,
    load_plan_detail_tables,
    persist_plan_detail_tables,
    extract_upper_rows,
    prefetch_plan_detail_grains,
)
from app.pipeline.plan_detail.calc_engine import mark_plan_dirty as mark_plan_detail_dirty, mark_plan_dirty_deps as mark_plan_detail_dirty_deps, dep_snapshot_all
from app.pipeline import scenario_store
from app.pipeline import approval as approval_mod
from app.pipeline import monte_carlo as monte_carlo_mod
from app.pipeline import risk_band as risk_band_mod
from app.pipeline import accuracy_store
from app.pipeline import hiring_solver
from app.pipeline.ba_rollup_plan import compute_ba_rollup_tables, invalidate_rollup_cache, month_cols_for_ba, week_cols_for_ba
from app.pipeline.plan_detail._common import (
    clone_plan,
    current_user_fallback,
    get_class_level_options,
    get_class_type_options,
    load_nh_classes,
    next_class_reference,
    save_nh_classes,
    save_plan_meta,
    _assemble_voice,
    _weekly_voice,
)
from app.pipeline.planning_store import (
    delete_plan,
    list_business_areas,
    list_activity,
    list_plans,
    load_plan,
    load_plan_table,
    record_activity,
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
from app.pipeline.forecast_run_store import save_forecast_run
from app.pipeline.transformations import apply_transformations
from app.pipeline.volume_summary import auto_smoothing_sweep, run_volume_summary, smoothing_core
from app.pipeline.utils import df_from_payload, df_to_records, sanitize_for_json
from app.cap_db import load_df, save_df

app = FastAPI(title="CAP Connect Forecast Service")
logger = logging.getLogger("forecast-service")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# Diagnostic/debug endpoints leak filesystem paths and config; keep them off
# unless explicitly enabled.
_DEBUG_ENDPOINTS_ENABLED = os.getenv("ENABLE_DEBUG_ENDPOINTS", "0").strip().lower() in {"1", "true", "yes"}


def _require_debug_enabled() -> None:
    if not _DEBUG_ENDPOINTS_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")


def _max_request_bytes() -> int:
    try:
        return int(os.getenv("MAX_REQUEST_BYTES", str(50 * 1024 * 1024)))
    except (TypeError, ValueError):
        return 50 * 1024 * 1024


def _max_zip_uncompressed_bytes() -> int:
    try:
        return int(os.getenv("MAX_ZIP_UNCOMPRESSED_BYTES", str(300 * 1024 * 1024)))
    except (TypeError, ValueError):
        return 300 * 1024 * 1024


async def _read_upload_capped(file: "UploadFile") -> bytes:
    """Read an uploaded file in chunks, aborting past the request-size cap.

    The Content-Length middleware can be bypassed with Transfer-Encoding: chunked,
    so we also enforce the byte budget here while streaming, before the whole body
    is buffered in memory.
    """
    cap = _max_request_bytes()
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            raise HTTPException(status_code=413, detail="Request body too large.")
        chunks.append(chunk)
    return b"".join(chunks)


def _guard_zip_bomb(content: bytes) -> None:
    """Reject decompression bombs: .xlsx is a ZIP, and a small file can inflate to
    many GB. Sum the declared uncompressed sizes and abort past the cap before any
    parser (pandas/openpyxl) expands the archive."""
    if len(content) < 4 or content[:2] != b"PK":
        return  # not a zip (e.g. CSV) — nothing to expand
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            total = sum(max(0, info.file_size) for info in zf.infolist())
    except zipfile.BadZipFile:
        return  # let the downstream parser surface a clean error
    if total > _max_zip_uncompressed_bytes():
        raise HTTPException(status_code=413, detail="Spreadsheet expands too large to process.")


@app.middleware("http")
async def _limit_request_size(request, call_next):
    # Reject oversized bodies up front (by Content-Length) so huge uploads can't
    # exhaust memory before they're even parsed. Chunked-encoding requests omit
    # Content-Length and are additionally capped while streaming in
    # _read_upload_capped at the upload handlers.
    from fastapi.responses import JSONResponse

    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > _max_request_bytes():
        return JSONResponse(status_code=413, content={"detail": "Request body too large."})
    return await call_next(request)


@app.middleware("http")
async def _authenticate(request, call_next):
    # Attach the authenticated principal (if any) to request.state, and when
    # AUTH_ENABLED reject unauthenticated requests to protected routes. All of
    # this is a no-op unless the AUTH_* env flags are set.
    from fastapi.responses import JSONResponse

    request.state.principal = security.principal_from_request(request)
    if (
        security.auth_enabled()
        and request.state.principal is None
        and not security.is_exempt_path(request.url.path)
        and request.method != "OPTIONS"
    ):
        return JSONResponse(status_code=401, content={"detail": "Authentication required."})
    return await call_next(request)


@app.post("/api/auth/token")
def issue_token(request: Request):
    """Mint a short-lived session token from trusted proxy identity.

    Only works when AUTH_JWT_SECRET is configured and the upstream proxy is
    trusted (TRUST_PROXY_AUTH=1).

    When auth is disabled entirely (no AUTH_JWT_SECRET — the default/dev case)
    we return 200 with a null token instead of 404: there is nothing to mint
    and nothing to protect, so the frontend's one-time token probe gets a clean
    "no token needed" answer rather than a 404 logged on every page load. When
    a secret IS configured but the proxy isn't trusted, we still 404 so the
    endpoint can't be used as an open token oracle.
    """
    secret = os.getenv("AUTH_JWT_SECRET", "")
    if not secret:
        return {"token": None, "auth": "disabled"}
    if not security.trust_proxy_auth() or not security.proxy_request_verified(request):
        raise HTTPException(status_code=404, detail="Not found.")
    email = (
        request.headers.get("x-forwarded-email")
        or request.headers.get("x-email")
        or request.headers.get("x-user-email")
        or ""
    ).strip()
    if not email:
        raise HTTPException(status_code=401, detail="No upstream identity.")
    token = security.mint_jwt(email, secret, ttl_seconds=int(os.getenv("AUTH_TOKEN_TTL", "3600") or 3600))
    return {"token": token, "token_type": "bearer", "email": email}


def _user_identity(request: Request) -> tuple[str, str, str]:
    """Resolve (key, email, name) for the current user. Prefers a verified
    principal (JWT), then trusted proxy headers, then the process fallback."""
    email = ""
    name = ""
    try:
        principal = security.principal_from_request(request)
        email = (getattr(principal, "email", "") or "").strip()
    except Exception:
        principal = None
    if not email:
        email = (
            request.headers.get("x-forwarded-email")
            or request.headers.get("x-email")
            or request.headers.get("x-user-email")
            or ""
        ).strip()
    name = (
        request.headers.get("x-forwarded-preferred-username")
        or request.headers.get("x-forwarded-user")
        or ""
    ).strip()
    key = (email or name or "").strip().lower()
    if not key:
        try:
            key = (current_user_fallback() or "system").strip()
        except Exception:
            key = "system"
        name = name or key
    return key, email, name


@app.get("/api/user")
def get_user(request: Request):
    key, email, name = _user_identity(request)
    try:
        return sanitize_for_json(user_store.get_user_profile(key, email=email, name=name))
    except Exception:
        logger.exception("get_user failed for key=%s", key)
        return {"key": key, "email": email, "name": name or email or key, "photo_url": ""}


def _safe_photo_url(value) -> str | None:
    """Allowlist photo URL schemes. The value is later rendered as <img src=...>
    in the browser, so block javascript:/vbscript:/arbitrary data: payloads and
    only permit https:, root-relative paths, and image data URLs."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return ""  # explicit clear
    lowered = candidate.lower()
    if lowered.startswith("https://") or candidate.startswith("/"):
        return candidate
    if lowered.startswith("data:image/png") or lowered.startswith("data:image/jpeg") or lowered.startswith("data:image/webp"):
        return candidate
    raise HTTPException(status_code=400, detail="Unsupported photo URL (use https:// or an image data URL).")


@app.patch("/api/user")
def patch_user(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Profile payload must be a JSON object.")
    key, email, name = _user_identity(request)
    display_name = payload.get("name")
    photo_url = _safe_photo_url(payload.get("photo_url")) if "photo_url" in payload else None
    # Data-URL photos can balloon; cap here (the body-size middleware also guards).
    if isinstance(photo_url, str) and len(photo_url) > 3_000_000:
        raise HTTPException(status_code=413, detail="Profile photo too large (max ~2MB).")
    try:
        user_store.update_user_profile(
            key,
            display_name=display_name if isinstance(display_name, str) else None,
            photo_url=photo_url if isinstance(photo_url, str) else None,
        )
        return sanitize_for_json(user_store.get_user_profile(key, email=email, name=name))
    except HTTPException:
        raise
    except Exception:
        logger.exception("patch_user failed for key=%s", key)
        raise HTTPException(status_code=500, detail="Could not update profile.")


@app.post("/api/users/display")
def users_display(payload: dict, request: Request):
    """Resolve a batch of user identifiers (BRID/email/name keys) to display
    name + photo so owners/actors render as names instead of opaque IDs."""
    security.require_user(request)  # no-op unless AUTH_ENABLED
    keys = payload.get("keys") if isinstance(payload, dict) else None
    if not isinstance(keys, list):
        return {}
    try:
        return sanitize_for_json(user_store.get_user_display_map([str(k) for k in keys]))
    except Exception:
        logger.exception("users_display failed")
        return {}


def _authorize_plan_access(plan_id, request: Request) -> dict:
    """Load a plan and enforce per-plan authorization (no-op unless AUTHZ_ENABLED)."""
    plan = load_plan(int(plan_id))
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found.")
    security.authorize_plan(plan, security.require_user(request))
    return plan

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


def _norm_str(value: object) -> str:
    return str(value or "").strip()


def _scope_key_from_payload(payload: dict) -> str:
    if not isinstance(payload, dict):
        return "global"
    ba = _norm_str(payload.get("business_area") or payload.get("ba") or payload.get("vertical"))
    sba = _norm_str(payload.get("sub_business_area") or payload.get("sub_ba") or payload.get("sba"))
    channel = _norm_str(payload.get("channel") or payload.get("lob"))
    site = _norm_str(payload.get("site") or payload.get("location"))
    if ba or sba or channel or site:
        return f"{ba}|{sba}|{channel}|{site}".strip("|")
    group_val = _norm_str(payload.get("group_value") or payload.get("group") or payload.get("forecast_group"))
    if group_val:
        return f"forecast_group|{group_val}"
    return "global"


def _scope_from_key(scope_key: str) -> dict:
    raw = _norm_str(scope_key)
    if not raw:
        return {"scope_type": "global"}
    if raw.lower().startswith("location|"):
        _, loc = raw.split("|", 1)
        return {"scope_type": "location", "location": loc.strip()}
    if "|" in raw:
        parts = raw.split("|")
        while len(parts) < 4:
            parts.append("")
        ba, sba, ch, site = parts[:4]
        return {
            "scope_type": "hier",
            "business_area": ba.strip(),
            "sub_business_area": sba.strip(),
            "channel": ch.strip(),
            "site": site.strip(),
        }
    return {"scope_type": "global"}


def _list_all_plans(limit: int = 1000) -> list[dict]:
    try:
        return list_plans(include_deleted=False, limit=limit) or []
    except Exception:
        return []


def _plans_matching_scope(scope: dict) -> list[dict]:
    def _norm_part(val: object) -> str:
        s = _norm_str(val).lower()
        s = s.replace(",", " ")
        return " ".join(s.split())

    scope_type = _norm_str(scope.get("scope_type") or "global").lower()
    if scope_type == "location":
        location = _norm_part(scope.get("location"))
        if not location:
            return _list_all_plans()
        return list_plans(location=location, include_deleted=False, limit=1000) or []
    if scope_type == "hier":
        ba_raw = _norm_str(scope.get("business_area"))
        sba_raw = _norm_str(scope.get("sub_business_area"))
        if not ba_raw or not sba_raw:
            return _list_all_plans()
        candidates = list_plans(
            business_area=ba_raw,
            sub_business_area=sba_raw,
            include_deleted=False,
            limit=1000,
        ) or []
        if not candidates:
            candidates = _list_all_plans()
        channel = _norm_part(scope.get("channel"))
        site = _norm_part(scope.get("site"))
        if channel:
            candidates = [
                plan for plan in candidates
                if _norm_part(plan.get("channel")) == channel
            ]
        if site:
            candidates = [
                plan for plan in candidates
                if _norm_part(plan.get("site")) == site
            ]
        return candidates
    return _list_all_plans()


_PRECOMPUTE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="plan-precompute")


def _precompute_plan_detail(pid: int) -> None:
    try:
        data, status, _meta = compute_plan_detail_tables(
            int(pid),
            grain="week",
            whatif=None,
            interval_date=None,
            version_token=None,
            persist=True,
        )
        if status == "ready" and data:
            persist_plan_detail_tables(int(pid), data, grain="week")
            for g in ("month", "day"):
                data_g, status_g, _ = compute_plan_detail_tables(
                    int(pid),
                    grain=g,
                    whatif=None,
                    interval_date=None,
                    version_token=None,
                    persist=True,
                )
                if status_g == "ready" and data_g:
                    persist_plan_detail_tables(int(pid), data_g, grain=g)
    except Exception:
        return


def _invalidate_plan_detail_for_scope(scope: dict, dep: str | None = None) -> None:
    for plan in _plans_matching_scope(scope):
        pid = plan.get("id")
        if pid:
            try:
                if dep:
                    from app.pipeline.plan_detail.calc_engine import mark_plan_dirty_deps
                    mark_plan_dirty_deps(int(pid), dep)
                else:
                    mark_plan_detail_dirty(int(pid))
            except Exception:
                continue
            try:
                _PRECOMPUTE_EXECUTOR.submit(_precompute_plan_detail, int(pid))
            except Exception:
                continue


def _norm_brid(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.match(r"^\d+\.0+$", text):
        text = text.split(".", 1)[0]
    return text.lower()


def _canon_channel_name(value: object) -> str:
    key = " ".join(str(value or "").strip().lower().replace(",", " ").split())
    if key in {"back office", "backoffice", "bo"}:
        return "back office"
    if key in {"outbound", "out bound", "ob"}:
        return "outbound"
    if key in {"chat", "message us", "messageus", "messaging"}:
        return "chat"
    if key in {"voice", "inbound", "call", "telephony"}:
        return "voice"
    return key


def _attrition_termination_map(df_raw: pd.DataFrame, scope: Optional[dict]) -> dict[str, str]:
    if df_raw is None or df_raw.empty:
        return {}
    try:
        norm = normalize_attrition_raw_upload(df_raw, scope=scope)
    except Exception:
        return {}
    if norm is None or norm.empty or "BRID" not in norm.columns or "Termination Date" not in norm.columns:
        return {}
    brid_norm = norm["BRID"].map(_norm_brid)
    term_series = _parse_date_series(norm["Termination Date"])
    work = pd.DataFrame({"brid_norm": brid_norm, "term": term_series})
    work["term"] = pd.to_datetime(work["term"], errors="coerce").dt.date
    work = work[
        (work["brid_norm"].str.len() > 0)
        & work["term"].notna()
    ].copy()
    if work.empty:
        return {}
    work = work.sort_values("term").drop_duplicates(subset=["brid_norm"], keep="last")
    return {
        str(row["brid_norm"]): row["term"].isoformat()
        for _, row in work.iterrows()
    }


def _sync_attrition_to_plan_roster(
    df_raw: pd.DataFrame,
    scope: Optional[dict],
    target_plan_ids: Optional[set[int] | list[int]] = None,
) -> dict:
    term_map = _attrition_termination_map(df_raw, scope)
    if not term_map:
        return {"updated_plans": 0, "updated_rows": 0}

    business_area = _norm_str((scope or {}).get("business_area") or (scope or {}).get("ba"))
    sub_business_area = _norm_str((scope or {}).get("sub_business_area") or (scope or {}).get("sba"))
    if not business_area or not sub_business_area:
        return {"updated_plans": 0, "updated_rows": 0}

    scope_obj = {
        "scope_type": "hier",
        "business_area": business_area,
        "sub_business_area": sub_business_area,
        "channel": _norm_str((scope or {}).get("channel")),
        "site": _norm_str((scope or {}).get("site")),
    }

    def _norm(v: object) -> str:
        return " ".join(str(v or "").strip().lower().split())

    plans = _plans_matching_scope(scope_obj)
    if not plans:
        try:
            fallback = list_plans(
                business_area=business_area,
                sub_business_area=sub_business_area,
                site=scope_obj.get("site") or None,
                status_filter=None,
                include_deleted=False,
                limit=1000,
            ) or []
        except Exception:
            fallback = []
        target_channel = _canon_channel_name(scope_obj.get("channel"))
        if target_channel:
            narrowed: list[dict] = []
            for plan in fallback:
                chan_raw = str(plan.get("channel") or "")
                chan_parts = [_canon_channel_name(part) for part in chan_raw.split(",") if _canon_channel_name(part)]
                if target_channel in chan_parts:
                    narrowed.append(plan)
            plans = narrowed
        else:
            plans = fallback
    if not plans:
        return {"updated_plans": 0, "updated_rows": 0}

    plan_id_filter: set[int] = set()
    if target_plan_ids:
        for pid in target_plan_ids:
            try:
                plan_id_filter.add(int(pid))
            except Exception:
                continue
    if plan_id_filter:
        plans = [p for p in plans if int(p.get("id") or 0) in plan_id_filter]
        if not plans:
            return {"updated_plans": 0, "updated_rows": 0}

    updated_plans = 0
    updated_rows = 0
    for plan in plans:
        pid = int(plan.get("id") or 0)
        if not pid:
            continue
        rows = load_plan_table(pid, "emp")
        if not isinstance(rows, list) or not rows:
            continue
        changed = False
        row_updates = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            brid = _norm_brid(
                row.get("brid")
                or row.get("BRID")
                or row.get("employee_id")
                or row.get("Employee ID")
                or row.get("Employee Id")
            )
            if not brid:
                continue
            term_iso = term_map.get(brid)
            if not term_iso:
                continue
            current_term = _norm_str(
                row.get("terminate_date")
                or row.get("Terminate Date")
                or row.get("termination_date")
                or row.get("Termination Date")
            )
            current_term = current_term[:10] if len(current_term) >= 10 else current_term
            if current_term != term_iso:
                row["terminate_date"] = term_iso
                if "Terminate Date" in row:
                    row["Terminate Date"] = term_iso
                changed = True
            current_status = _norm(row.get("current_status") or row.get("Current Status"))
            if current_status != "terminated":
                row["current_status"] = "Terminated"
                if "Current Status" in row:
                    row["Current Status"] = "Terminated"
                changed = True
            row_updates += 1
        if not changed:
            continue
        try:
            result = save_plan_table(pid, "emp", rows)
        except Exception:
            continue
        if str(result.get("status") or "").lower() == "saved":
            updated_plans += 1
            updated_rows += row_updates
    return {"updated_plans": updated_plans, "updated_rows": updated_rows}


def _plan_scope_for_attrition(plan: Optional[dict]) -> Optional[dict]:
    plan_obj = plan if isinstance(plan, dict) else {}
    ba = _norm_str(plan_obj.get("vertical") or plan_obj.get("business_area"))
    sba = _norm_str(plan_obj.get("sub_ba") or plan_obj.get("sub_business_area"))
    if not ba or not sba:
        return None
    channel_raw = _norm_str(plan_obj.get("channel") or plan_obj.get("lob"))
    channel = channel_raw.split(",")[0].strip() if channel_raw else ""
    site = _norm_str(plan_obj.get("site") or plan_obj.get("location") or plan_obj.get("country"))
    return {
        "business_area": ba,
        "sub_business_area": sba,
        "channel": channel,
        "site": site,
    }


def _resync_saved_attrition_to_roster(
    scope: Optional[dict],
    *,
    target_plan_ids: Optional[set[int] | list[int]] = None,
) -> dict:
    if not isinstance(scope, dict):
        return {"updated_plans": 0, "updated_rows": 0}
    try:
        raw_df = load_attrition_raw(scope=scope)
    except Exception:
        return {"updated_plans": 0, "updated_rows": 0}
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        return {"updated_plans": 0, "updated_rows": 0}
    try:
        return _sync_attrition_to_plan_roster(raw_df, scope, target_plan_ids=target_plan_ids)
    except Exception:
        return {"updated_plans": 0, "updated_rows": 0}


def _settings_path(scope_type: str, location: Optional[str], ba: Optional[str], sba: Optional[str], channel: Optional[str], site: Optional[str]) -> str:
    key = settings_store._scope_key(scope_type, location, ba, sba, channel, site)
    return str(settings_store._exports_dir() / f"settings_{key}.json")


def _settings_debug(ba: str, sba: str, channel: str, site: str) -> dict:
    site_path = _settings_path("hier", None, ba, sba, channel, site)
    site_path_lower = _settings_path(
        "hier",
        None,
        (ba or "").lower() if ba else None,
        (sba or "").lower() if sba else None,
        (channel or "").lower() if channel else None,
        (site or "").lower() if site else None,
    )
    nosite_path = _settings_path("hier", None, ba, sba, channel, None)
    nosite_path_lower = _settings_path(
        "hier",
        None,
        (ba or "").lower() if ba else None,
        (sba or "").lower() if sba else None,
        (channel or "").lower() if channel else None,
        None,
    )
    global_path = _settings_path("global", None, None, None, None, None)
    resolved = resolve_cap_settings(ba=ba, subba=sba, lob=channel, site=site) or {}
    return {
        "files": {
            "site": site_path,
            "site_exists": os.path.exists(site_path),
            "site_lower": site_path_lower,
            "site_lower_exists": os.path.exists(site_path_lower),
            "nosite": nosite_path,
            "nosite_exists": os.path.exists(nosite_path),
            "nosite_lower": nosite_path_lower,
            "nosite_lower_exists": os.path.exists(nosite_path_lower),
            "global": global_path,
            "global_exists": os.path.exists(global_path),
        },
        "resolved": {
            "interval_minutes": resolved.get("interval_minutes"),
            "hours_per_fte": resolved.get("hours_per_fte"),
            "shrinkage_pct": resolved.get("shrinkage_pct"),
            "voice_shrinkage_pct": resolved.get("voice_shrinkage_pct"),
            "bo_shrinkage_pct": resolved.get("bo_shrinkage_pct"),
            "chat_shrinkage_pct": resolved.get("chat_shrinkage_pct"),
            "ob_shrinkage_pct": resolved.get("ob_shrinkage_pct"),
            "occupancy_cap_voice": resolved.get("occupancy_cap_voice"),
            "util_bo": resolved.get("util_bo"),
            "util_ob": resolved.get("util_ob"),
            "util_chat": resolved.get("util_chat"),
        },
    }


def _ts_stats(kind: str, scope_key: str) -> dict:
    df = load_timeseries_any(kind, [scope_key])
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"rows": 0}
    out: dict[str, object] = {"rows": int(len(df.index))}
    if "date" in df.columns:
        try:
            dates = pd.to_datetime(df["date"], errors="coerce")
            out["date_min"] = str(dates.min().date()) if dates.notna().any() else None
            out["date_max"] = str(dates.max().date()) if dates.notna().any() else None
        except Exception:
            out["date_min"] = None
            out["date_max"] = None
    if "interval" in df.columns:
        try:
            s = df["interval"].astype(str).replace("nan", "").replace("NaT", "").str.strip()
            out["has_interval"] = bool((s != "").any())
        except Exception:
            out["has_interval"] = False
    return out


def _norm_cols(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def _pick_col(cols_or_df: object, names: list[str] | tuple[str, ...]) -> Optional[str]:
    """
    Flexible column picker. Accepts a DataFrame or a pre-built name map.
    Performs fuzzy matching over common separators/casing.
    """
    lookup: dict[str, str] = {}
    if isinstance(cols_or_df, pd.DataFrame):
        for c in cols_or_df.columns:
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
                lookup.setdefault(v, str(c))
    elif isinstance(cols_or_df, dict):
        for k, v in cols_or_df.items():
            base = str(k).strip().lower()
            variants = {
                base,
                base.replace(" ", "_"),
                base.replace(" ", ""),
                base.replace("-", "_"),
                base.replace("-", ""),
                base.replace("_", ""),
                re.sub(r"[^a-z0-9]", "", base),
            }
            for v_key in variants:
                lookup.setdefault(v_key, v)
    else:
        return None

    for name in names:
        key = str(name).strip().lower()
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


def _parse_date_series(series: pd.Series) -> pd.Series:
    s = pd.Series(series)
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            return pd.to_datetime(s, errors="coerce").dt.date
    except Exception:
        pass

    sample = s.dropna().astype(str).str.strip()
    if sample.empty:
        return pd.to_datetime(s, errors="coerce").dt.date

    iso_mask = sample.str.match(r"^\\d{4}-\\d{1,2}-\\d{1,2}$")
    if iso_mask.any() and iso_mask.mean() > 0.5:
        return pd.to_datetime(s, errors="coerce", format="%Y-%m-%d").dt.date

    dash_mask = sample.str.match(r"^\\d{1,2}-\\d{1,2}-\\d{2,4}$")
    if dash_mask.any() and dash_mask.mean() > 0.5:
        parts = sample[dash_mask].str.extract(r"^(\\d{1,2})-(\\d{1,2})-(\\d{2,4})$")
        first = pd.to_numeric(parts[0], errors="coerce")
        fmt = "%d-%m-%Y" if (first > 12).any() else "%m-%d-%Y"
        return pd.to_datetime(s, errors="coerce", format=fmt).dt.date

    slash_mask = sample.str.match(r"^\\d{1,2}/\\d{1,2}/\\d{2,4}$")
    if slash_mask.any() and slash_mask.mean() > 0.5:
        parts = sample[slash_mask].str.extract(r"^(\\d{1,2})/(\\d{1,2})/(\\d{2,4})$")
        first = pd.to_numeric(parts[0], errors="coerce")
        fmt = "%d/%m/%Y" if (first > 12).any() else "%m/%d/%Y"
        return pd.to_datetime(s, errors="coerce", format=fmt).dt.date

    return pd.to_datetime(s, errors="coerce").dt.date


def _to_seconds_value(value):
    """Convert common duration formats to seconds for AHT/SUT fields."""
    try:
        if value is None:
            return np.nan
        if value is pd.NA:
            return np.nan
        try:
            if pd.isna(value):
                return np.nan
        except Exception:
            pass
        if isinstance(value, pd.Timedelta):
            return float(value.total_seconds())
        if isinstance(value, (np.integer, int)):
            return float(value)
        if isinstance(value, (np.floating, float)):
            v = float(value)
            if not np.isfinite(v):
                return np.nan
            # Excel often stores time-of-day / durations as day fractions.
            if 0.0 < abs(v) < 1.0:
                return float(v * 86400.0)
            return v
        if hasattr(value, "hour") and hasattr(value, "minute") and hasattr(value, "second"):
            return float(int(value.hour) * 3600 + int(value.minute) * 60 + int(value.second))
        s = str(value).strip()
        if not s:
            return np.nan
        if ":" in s:
            parts = [p.strip() for p in s.split(":")]
            if len(parts) == 3:
                h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
                return (h * 3600.0) + (m * 60.0) + sec
            if len(parts) == 2:
                m = float(parts[0]); sec = float(parts[1])
                return (m * 60.0) + sec
        v = float(s.replace(",", ""))
        if 0.0 < abs(v) < 1.0:
            return float(v * 86400.0)
        return v
    except Exception:
        return np.nan


def _to_seconds_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(pd.Series(series).map(_to_seconds_value), errors="coerce")


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
            out["date"] = _parse_date_series(df[date_col])
        if interval_col:
            out["interval"] = df[interval_col].astype(str)
        if volume_label:
            out[item_col] = pd.to_numeric(df[volume_label], errors="coerce")
        if aht_label:
            out["aht_sec"] = _to_seconds_series(df[aht_label])
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
                "actual volume/forecast volume",
                "forecast volume/actual volume",
                "actual/forecast volume",
                "forecast/actual volume",
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
                f"{stage} aht (sec)",
                f"{stage}_aht_sec",
                f"{stage} aht/sut",
                f"{stage}_aht_sut",
                "actual aht/forecast aht",
                "forecast aht/actual aht",
                "actual/forecast aht",
                "forecast/actual aht",
                "forecast aht",
                "actual aht",
                "tactical aht",
                "forecast aht/sut",
                "actual aht/sut",
                "tactical aht/sut",
                "aht_sec",
                "aht",
                "aht/sut",
                "aht_sut",
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
            extras[f"voice_{stage}_aht"] = base[["date"] + (["interval"] if "interval" in base.columns else []) + ["aht_sec"]]
        return base if not base.empty else df, extras

    if prefix == "chat":
        items_col = _pick_col(
            cols,
            [
                f"{stage} volume",
                f"{stage}_volume",
                "actual volume/forecast volume",
                "forecast volume/actual volume",
                "actual/forecast volume",
                "forecast/actual volume",
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
                f"{stage} aht (sec)",
                f"{stage}_aht_sec",
                f"{stage} aht/sut",
                f"{stage}_aht_sut",
                "actual aht/forecast aht",
                "forecast aht/actual aht",
                "actual/forecast aht",
                "forecast/actual aht",
                "forecast aht",
                "actual aht",
                "tactical aht",
                "forecast aht/sut",
                "actual aht/sut",
                "tactical aht/sut",
                "aht_sec",
                "aht",
                "aht/sut",
                "aht_sut",
                "avg_aht",
                "sut_sec",
            ],
        )
        base = _build_base(items_col, aht_col, "items")
        if "items" in base.columns:
            extras[f"chat_{stage}_volume"] = base[["date"] + (["interval"] if "interval" in base.columns else []) + ["items"]]
        if "aht_sec" in base.columns:
            extras[f"chat_{stage}_aht"] = base[["date"] + (["interval"] if "interval" in base.columns else []) + ["aht_sec"]]
        return base if not base.empty else df, extras

    if prefix == "bo":
        items_col = _pick_col(
            cols,
            [
                f"{stage} volume",
                f"{stage}_volume",
                "actual volume/forecast volume",
                "forecast volume/actual volume",
                "actual/forecast volume",
                "forecast/actual volume",
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
                f"{stage} sut (sec)",
                f"{stage}_sut_sec",
                f"{stage} aht/sut",
                f"{stage}_aht_sut",
                "actual aht/forecast aht",
                "forecast aht/actual aht",
                "actual/forecast aht",
                "forecast/actual aht",
                "actual sut/forecast sut",
                "forecast sut/actual sut",
                "actual/forecast sut",
                "forecast/actual sut",
                "forecast sut",
                "actual sut",
                "tactical sut",
                "forecast aht/sut",
                "actual aht/sut",
                "tactical aht/sut",
                "sut_sec",
                "sut",
                "aht/sut",
                "aht_sut",
                "aht_sec",
                "aht",
                "avg_sut",
            ],
        )
        base = _build_base(items_col, sut_col, "items")
        if "items" in base.columns:
            extras[f"bo_{stage}_volume"] = base[["date", "items"]]
        if "aht_sec" in base.columns:
            extras[f"bo_{stage}_sut"] = base[["date"] + (["interval"] if "interval" in base.columns else []) + ["aht_sec"]].rename(columns={"aht_sec": "sut_sec"})
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
            base["aht_sec"] = _to_seconds_series(df[aht_col])
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
    effective_date: Optional[str] = Query(None),
):
    settings = load_settings(scope_type, location, ba, sba, channel, site, effective_date)

    def _scope_candidates() -> list[str]:
        if not (ba and sba and channel):
            return []
        parts = [str(ba).strip(), str(sba).strip(), str(channel).strip()]
        scopes = ["|".join(parts)]
        if site:
            scopes.insert(0, "|".join(parts + [str(site).strip()]))
        return [s for s in scopes if s and s != "||"]

    def _last_actual_value(df: pd.DataFrame, value_col: str, weight_col: Optional[str] = None) -> Optional[float]:
        if not isinstance(df, pd.DataFrame) or df.empty or value_col not in df.columns:
            return None
        x = df.copy()
        x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
        if "date" in x.columns:
            x["_date"] = pd.to_datetime(x["date"], errors="coerce")
            x = x.dropna(subset=["_date"])
            if x.empty:
                return None
            last_date = x["_date"].max()
            x = x[x["_date"] == last_date]
        if x.empty:
            return None
        if weight_col and weight_col in x.columns:
            weights = pd.to_numeric(x[weight_col], errors="coerce").fillna(0.0)
            values = x[value_col]
            mask = values.notna() & (weights > 0)
            if mask.any():
                denom = float(weights[mask].sum())
                if denom > 0:
                    return float((values[mask] * weights[mask]).sum() / denom)
        values = x[value_col].dropna()
        return float(values.mean()) if not values.empty else None

    def _last_budget_value(df: pd.DataFrame, value_col: str) -> Optional[float]:
        if not isinstance(df, pd.DataFrame) or df.empty or value_col not in df.columns:
            return None
        x = df.copy()
        x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
        x = x.dropna(subset=[value_col])
        if x.empty:
            return None
        if "week" in x.columns:
            x["_week"] = pd.to_datetime(x["week"], errors="coerce")
            x = x.sort_values(["_week", "week"], na_position="last")
        return float(x.iloc[-1][value_col])

    scopes = _scope_candidates()
    if scopes:
        raw_channel = str(channel or "").strip().lower()
        kind = ""
        value_col = "aht_sec"
        weight_col = None
        if raw_channel in ("voice", "call", "telephony"):
            kind = "voice_actual"
            weight_col = "volume"
        elif raw_channel in ("back office", "backoffice", "bo"):
            kind = "bo_actual"
            value_col = "sut_sec"
            weight_col = "items"
        elif raw_channel in ("chat", "messaging", "messageus", "message us"):
            kind = "chat_actual"
            weight_col = "items"
        elif raw_channel in ("outbound", "ob", "out bound"):
            kind = "ob_actual"
        if kind:
            # Multi-scope load: use the batched single-query path instead of one
            # DB round-trip per scope.
            ts_df = load_timeseries_any(kind, scopes, batch=True)
            actual_val = _last_actual_value(ts_df, value_col, weight_col=weight_col)
            if actual_val is None and value_col == "sut_sec" and "aht_sec" in ts_df.columns:
                actual_val = _last_actual_value(ts_df, "aht_sec", weight_col=weight_col)
            if value_col == "sut_sec":
                settings["last_actual_sut_sec"] = actual_val
            else:
                settings["last_actual_aht_sec"] = actual_val

        if all([ba, sba, channel, site]):
            budget_df = load_budget_rows(str(ba), str(sba), str(channel), str(site))
            budget_col = "budget_sut_sec" if raw_channel in ("back office", "backoffice", "bo") else "budget_aht_sec"
            budget_val = _last_budget_value(budget_df, budget_col)
            if budget_col == "budget_sut_sec":
                settings["last_budget_sut_sec"] = budget_val
            else:
                settings["last_budget_aht_sec"] = budget_val
    return {"status": "ok", "settings": settings}


@app.post("/api/forecast/settings")
def save_settings_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Settings payload must be a JSON object.")
    scope = payload.get("scope") or {}
    settings = payload.get("settings") or {}
    effective_date = payload.get("effective_date")
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
        effective_date=effective_date,
    )
    try:
        record_activity(
            action="updated settings",
            actor=current_user_fallback(),
            entity_type="settings",
            payload={"scope": scope, "effective_date": effective_date},
        )
    except Exception:
        pass
    try:
        _invalidate_plan_detail_for_scope(scope, "settings")
    except Exception:
        pass
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
    try:
        _invalidate_plan_detail_for_scope(scope, "settings")
    except Exception:
        pass
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
    # Normalize scope key: drop trailing empty site segment (e.g., "BA|SBA|Channel|")
    try:
        raw = str(scope_key or "").strip()
        if raw and raw.lower() != "global":
            parts = raw.split("|")
            if len(parts) >= 4 and not parts[-1].strip():
                scope_key = "|".join(p.strip() for p in parts[:3])
    except Exception:
        pass
    try:
        logger.info(
            "timeseries upload: kind=%s scope_key=%s mode=%s rows_in=%s cols=%s",
            kind,
            scope_key,
            mode,
            len(df.index) if isinstance(df, pd.DataFrame) else 0,
            list(df.columns) if isinstance(df, pd.DataFrame) else [],
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            logger.info("timeseries sample rows: %s", df.head(3).to_dict("records"))
    except Exception:
        pass
    base_df, extras = _normalize_timeseries(str(kind or ""), df)
    try:
        if isinstance(base_df, pd.DataFrame):
            logger.info(
                "timeseries normalized: kind=%s rows=%s cols=%s date_min=%s date_max=%s",
                kind,
                len(base_df.index),
                list(base_df.columns),
                base_df["date"].min() if "date" in base_df.columns else None,
                base_df["date"].max() if "date" in base_df.columns else None,
            )
            if "interval" in base_df.columns and not base_df.empty:
                logger.info("timeseries interval sample: %s", base_df["interval"].astype(str).head(5).tolist())
        if extras:
            logger.info("timeseries extras: %s", {k: len(v.index) for k, v in extras.items() if isinstance(v, pd.DataFrame)})
    except Exception:
        pass
    scope_norm = normalize_scope_key(scope_key)
    data_hash = timeseries_dataset_hash(str(kind or ""), base_df)
    if data_hash:
        latest = get_latest_timeseries_hash(str(kind or ""), scope_norm)
        if latest and latest == data_hash:
            return {
                "status": "unchanged",
                "rows": len(base_df.index) if isinstance(base_df, pd.DataFrame) else 0,
                "hash": data_hash,
                "unchanged": True,
            }
    result = save_timeseries(kind, scope_key, base_df, mode=mode)
    extra_results: dict[str, int] = {}
    for extra_kind, extra_df in extras.items():
        if isinstance(extra_df, pd.DataFrame) and not extra_df.empty:
            save_timeseries(extra_kind, scope_key, extra_df, mode=mode)
            extra_results[extra_kind] = len(extra_df.index)
    if data_hash:
        try:
            record_timeseries_upload_hash(
                str(kind or ""),
                scope_norm,
                data_hash,
                len(base_df.index) if isinstance(base_df, pd.DataFrame) else 0,
            )
        except Exception:
            pass
    try:
        try:
            scope_dbg = _scope_from_key(scope_key)
            plans = _plans_matching_scope(scope_dbg)
            logger.info(
                "timeseries scope match: scope=%s plans=%s",
                scope_dbg,
                [p.get("id") for p in plans][:10],
            )
        except Exception:
            pass
        record_activity(
            action=f"uploaded timeseries {str(kind or '').strip() or 'data'}",
            actor=current_user_fallback(),
            entity_type="timeseries",
            entity_id=str(kind or ""),
            payload={"scope_key": scope_key, "mode": mode},
        )
    except Exception:
        pass
    try:
        _invalidate_plan_detail_for_scope(_scope_from_key(scope_key), "timeseries")
    except Exception:
        pass
    payload_out = {"status": result.get("status"), "rows": result.get("rows", 0), "path": result.get("path")}
    if extra_results:
        payload_out["extras"] = extra_results
    return payload_out


# --- Programmatic REST ingest API (v1) ------------------------------------
# A stable, documented, machine-facing contract for pushing volume/AHT actuals
# and forecasts, so refreshes can be automated instead of manual file uploads.
# It reuses the same normalization + persistence as the interactive upload path.

_INGEST_CHANNEL_TYPES = {"voice", "bo", "chat", "ob"}
_INGEST_METRICS = {"forecast", "actual", "tactical"}
_INGEST_MAX_ROWS = 200_000


def _require_ingest_auth(request: Request) -> None:
    """Auth for the ingest API: an INGEST_API_KEY (X-API-Key header) when one is
    configured, otherwise the normal interactive principal."""
    if security.ingest_api_key_configured():
        if not security.ingest_key_ok(request):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")
    else:
        security.require_user(request)


def _ingest_timeseries_core(kind: str, scope_key: str, rows: list, mode: str, actor: str) -> dict:
    """Normalize + persist a timeseries batch, reusing the interactive path's
    helpers. Returns a clean integrator-facing summary."""
    df = df_from_payload(rows)
    base_df, extras = _normalize_timeseries(str(kind or ""), df)
    scope_norm = normalize_scope_key(scope_key)
    data_hash = timeseries_dataset_hash(str(kind or ""), base_df)
    if data_hash:
        latest = get_latest_timeseries_hash(str(kind or ""), scope_norm)
        if latest and latest == data_hash:
            return {"status": "unchanged", "rows_ingested": 0, "unchanged": True, "kind": kind, "scope_key": scope_key}
    result = save_timeseries(kind, scope_key, base_df, mode=mode)
    extra_results: dict[str, int] = {}
    for extra_kind, extra_df in extras.items():
        if isinstance(extra_df, pd.DataFrame) and not extra_df.empty:
            save_timeseries(extra_kind, scope_key, extra_df, mode=mode)
            extra_results[extra_kind] = len(extra_df.index)
    if data_hash:
        try:
            record_timeseries_upload_hash(str(kind or ""), scope_norm, data_hash, len(base_df.index))
        except Exception:
            pass
    try:
        record_activity(
            action=f"ingested timeseries {str(kind or '').strip() or 'data'} (API)",
            actor=actor,
            entity_type="timeseries",
            entity_id=str(kind or ""),
            payload={"scope_key": scope_key, "mode": mode, "source": "ingest_api"},
        )
    except Exception:
        pass
    try:
        _invalidate_plan_detail_for_scope(_scope_from_key(scope_key), "timeseries")
    except Exception:
        pass
    date_range = None
    try:
        if isinstance(base_df, pd.DataFrame) and "date" in base_df.columns and not base_df.empty:
            date_range = {"from": str(base_df["date"].min()), "to": str(base_df["date"].max())}
    except Exception:
        date_range = None
    out = {
        "status": result.get("status"),
        "kind": kind,
        "scope_key": scope_key,
        "mode": mode,
        "rows_ingested": result.get("rows", 0),
        "date_range": date_range,
    }
    if extra_results:
        out["derived"] = extra_results
    return out


@app.get("/api/ingest/v1/schema")
def ingest_schema(request: Request):
    """Self-describing contract for the ingest API so integrators can self-serve."""
    _require_ingest_auth(request)
    return {
        "version": "1",
        "endpoint": "POST /api/ingest/v1/timeseries",
        "auth": "X-API-Key header (set INGEST_API_KEY), or a normal session token.",
        "channel_types": sorted(_INGEST_CHANNEL_TYPES),
        "metrics": sorted(_INGEST_METRICS),
        "kinds": sorted(_COMBINED_TIMESERIES_KINDS),
        "scope_fields": ["business_area", "sub_business_area", "channel", "site"],
        "modes": ["append", "replace"],
        "max_rows_per_request": _INGEST_MAX_ROWS,
        "row_fields": {
            "date": "YYYY-MM-DD (required)",
            "interval": "HH:MM (optional; for interval-grain voice/chat data)",
            "volume": "contacts/calls/items handled (voice/chat/ob: volume; bo: items)",
            "aht": "average handle time in seconds (bo: use 'sut')",
        },
        "example": {
            "business_area": "Cards",
            "sub_business_area": "Servicing",
            "channel": "Voice",
            "channel_type": "voice",
            "metric": "actual",
            "mode": "append",
            "rows": [{"date": "2026-06-01", "interval": "09:00", "volume": 120, "aht": 280}],
        },
    }


@app.post("/api/ingest/v1/timeseries")
def ingest_timeseries(payload: dict, request: Request):
    """Push volume/AHT actuals or forecasts programmatically.

    Identify the series either with an explicit `kind` (e.g. voice_actual) or with
    `channel_type` (voice|bo|chat|ob) + `metric` (forecast|actual|tactical).
    Identify the scope with business_area/sub_business_area/channel/site (or a raw
    `scope_key`). Send `rows` as a list of {date, interval?, volume|items, aht|sut}.
    """
    _require_ingest_auth(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Ingest payload must be a JSON object.")

    kind = str(payload.get("kind") or "").strip().lower()
    if not kind:
        channel_type = str(payload.get("channel_type") or "").strip().lower()
        metric = str(payload.get("metric") or "").strip().lower()
        if channel_type not in _INGEST_CHANNEL_TYPES:
            raise HTTPException(status_code=400, detail=f"channel_type must be one of {sorted(_INGEST_CHANNEL_TYPES)}.")
        if metric not in _INGEST_METRICS:
            raise HTTPException(status_code=400, detail=f"metric must be one of {sorted(_INGEST_METRICS)}.")
        kind = f"{channel_type}_{metric}"
    if kind not in _COMBINED_TIMESERIES_KINDS:
        raise HTTPException(status_code=400, detail=f"Unsupported kind '{kind}'. Valid kinds: {sorted(_COMBINED_TIMESERIES_KINDS)}.")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="A non-empty 'rows' list is required.")
    if len(rows) > _INGEST_MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"Too many rows ({len(rows)}); max {_INGEST_MAX_ROWS} per request.")

    scope_key = str(payload.get("scope_key") or "").strip() or _scope_key_from_payload(payload)
    mode = (payload.get("mode") or "append").lower()
    if mode not in {"append", "replace"}:
        raise HTTPException(status_code=400, detail="mode must be 'append' or 'replace'.")

    actor = ""
    try:
        actor = (getattr(security.principal_from_request(request), "email", "") or "").strip()
    except Exception:
        actor = ""
    actor = actor or "ingest-api"

    return sanitize_for_json(_ingest_timeseries_core(kind, scope_key, rows, mode, actor))


@app.post("/api/uploads/timeseries/preview")
def preview_timeseries_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Timeseries payload must be a JSON object.")
    kind = payload.get("kind")
    scope_key = payload.get("scope_key") or "global"
    rows = payload.get("rows") or []
    if not kind:
        raise HTTPException(status_code=400, detail="kind is required.")
    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="rows must be a list.")
    preview = preview_timeseries_rows(str(kind), str(scope_key), rows)
    preview["kind"] = str(kind)
    preview["scope_key"] = str(scope_key)
    return sanitize_for_json(preview)


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
    try:
        _invalidate_plan_detail_for_scope(
            {
                "scope_type": "hier",
                "business_area": scope.get("business_area"),
                "sub_business_area": scope.get("sub_business_area"),
                "channel": scope.get("channel"),
                "site": scope.get("site"),
            }
        )
    except Exception:
        pass
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
    preview = df_to_records(df.head(50)) if not df.empty else []
    return sanitize_for_json({"status": result.get("status"), "rows": preview, "saved": result})


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


@app.get("/api/forecast/settings/critical-team-options")
def settings_critical_team_options(
    ba: Optional[str] = None,
    sba: Optional[str] = None,
    ch: Optional[str] = None,
    location: Optional[str] = None,
    site: Optional[str] = None,
):
    return {
        "options": critical_team_options(
            ba=ba,
            sba=sba,
            ch=ch,
            location=location,
            site=site,
        )
    }


@app.post("/api/forecast/dataset")
def dataset_snapshot_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Dataset payload must be a JSON object.")
    return sanitize_for_json(
        dataset_snapshot(
            payload.get("start_date"),
            payload.get("end_date"),
            payload.get("series") or "auto",
            payload.get("ba") or [],
            payload.get("sba") or [],
            payload.get("ch") or [],
            payload.get("loc") or [],
            payload.get("site") or [],
        )
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
    debug: bool = Query(False),
):
    return ops_options(_parse_list(ba), _parse_list(sba), _parse_list(ch), _parse_list(loc), debug=debug)


@app.post("/api/forecast/ops/summary")
def get_ops_summary(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Ops payload must be a JSON object.")
    result = refresh_ops_async(
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


@app.post("/api/forecast/ops/summary/part")
def get_ops_summary_part(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Ops payload must be a JSON object.")
    part = payload.get("part")
    result = refresh_ops_part(
        part,
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


@app.post("/api/planning/plan/rebalancing")
def planning_rebalancing_preview(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Rebalancing payload must be a JSON object.")

    plan_id = payload.get("plan_id")
    rollup_ba = str(payload.get("rollup_ba") or payload.get("business_area") or "").strip()
    grain = str(payload.get("grain") or "D")
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")

    ba: list[str] = []
    sba: list[str] = []
    ch: list[str] = []
    site: list[str] = []
    loc: list[str] = []

    if plan_id:
        plan = _authorize_plan_access(plan_id, request)
        ba_val = str(plan.get("business_area") or plan.get("vertical") or "").strip()
        if ba_val:
            ba = [ba_val]
        if not start_date:
            start_date = plan.get("start_week")
        if not end_date:
            end_date = plan.get("end_week")
    elif rollup_ba:
        ba = [rollup_ba]

    overrides = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    policy: dict[str, Any] = {}

    def _copy_ratio(dst: str, *keys: str):
        for key in keys:
            val = overrides.get(key)
            if val is None:
                continue
            policy[dst] = val
            return

    _copy_ratio("cross_skill_efficiency_pct", "cross_skill_efficiency_pct", "xskill_efficiency_pct")
    _copy_ratio("max_lend_pct", "max_lend_pct", "xskill_max_lend_pct")
    if isinstance(overrides.get("lock_critical_teams"), list):
        policy["lock_critical_teams"] = overrides.get("lock_critical_teams")

    workforce = workforce_preview(
        start_date,
        end_date,
        grain,
        ba,
        sba,
        ch,
        site,
        loc,
        policy_overrides=policy,
    )
    return sanitize_for_json({"status": "ready", "workforce": workforce})


# --- Plan approval workflow ------------------------------------------------
# Draft → Submitted → Approved | Rejected (reopen back to Draft). Transitions are
# attributed and logged to the planning activity feed (audit trail).

def _plan_hier(plan: dict) -> dict:
    """Non-column plan metadata is stored in the hierarchy_json blob."""
    raw = plan.get("hierarchy_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            val = json.loads(raw)
            return val if isinstance(val, dict) else {}
        except Exception:
            return {}
    return {}


def _approval_state(plan: dict) -> dict:
    hier = _plan_hier(plan)
    status = hier.get("approval_status")
    return {
        "status": approval_mod.normalize_state(status),
        "actor": hier.get("approval_actor") or "",
        "ts": hier.get("approval_ts") or "",
        "note": hier.get("approval_note") or "",
        "available_actions": approval_mod.available_actions(status),
    }


@app.get("/api/planning/plan/approval")
def get_plan_approval(request: Request, plan_id: int = Query(...)):
    plan = _authorize_plan_access(plan_id, request)
    state = _approval_state(plan)
    try:
        history = [a for a in list_activity(limit=50, plan_id=int(plan_id)) if (a.get("entity_type") == "approval")]
    except Exception:
        history = []
    return sanitize_for_json({**state, "history": history})


@app.post("/api/planning/plan/approval")
def post_plan_approval(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Approval payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    action = str(payload.get("action") or "").strip().lower()
    note = str(payload.get("note") or "").strip()
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    plan = _authorize_plan_access(plan_id, request)
    current = approval_mod.normalize_state(_plan_hier(plan).get("approval_status"))
    new_state = approval_mod.transition(current, action)
    if new_state is None:
        raise HTTPException(status_code=409, detail=f"Cannot '{action}' a plan that is '{current}'.")
    # Approve/reject are decisions — restrict to admins when authz is enabled.
    if action in approval_mod.DECISION_ACTIONS and security.authz_enabled():
        principal = security.require_user(request)
        if not getattr(principal, "is_admin", False):
            raise HTTPException(status_code=403, detail="Only an admin can approve or reject a plan.")
    actor = current_user_fallback()
    now = pd.Timestamp.utcnow().isoformat(timespec="seconds")
    save_plan_meta(
        int(plan_id),
        {"approval_status": new_state, "approval_actor": actor, "approval_ts": now, "approval_note": note},
    )
    try:
        record_activity(
            plan_id=int(plan_id),
            action=f"approval: {action} ({current} → {new_state})",
            actor=actor,
            entity_type="approval",
            payload={"from": current, "to": new_state, "note": note},
        )
    except Exception:
        pass
    return sanitize_for_json(
        {"status": new_state, "actor": actor, "ts": now, "note": note,
         "available_actions": approval_mod.available_actions(new_state)}
    )


@app.get("/api/planning/plan")
def get_plans(
    request: Request,
    plan_id: Optional[int] = Query(None),
    ba: Optional[str] = Query(None),
    sba: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    site: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    if plan_id:
        plan = _authorize_plan_access(plan_id, request)
        return {"plan": plan}
    return {"plans": list_plans(ba, sba, channel, location, site, status_filter=status)}


@app.get("/api/planning/plan/debug")
def plan_debug(plan_id: int = Query(...)):
    _require_debug_enabled()
    plan = load_plan(int(plan_id))
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found.")

    ba = (plan.get("business_area") or plan.get("vertical") or "").strip()
    sba = (plan.get("sub_business_area") or plan.get("sub_ba") or "").strip()
    ch = (plan.get("channel") or plan.get("lob") or "").split(",")[0].strip()
    site = (plan.get("site") or plan.get("location") or plan.get("country") or "").strip()

    scope_key_4 = f"{ba}|{sba}|{ch}|{site}"
    scope_key_3 = f"{ba}|{sba}|{ch}"
    scope_norm_4 = normalize_scope_key(scope_key_4)
    scope_norm_3 = normalize_scope_key(scope_key_3)

    kinds = [
        "voice_forecast_volume",
        "voice_forecast_aht",
        "voice_actual_volume",
        "voice_actual_aht",
        "voice_tactical_volume",
        "voice_tactical_aht",
        "voice_forecast",
        "voice_actual",
        "voice_tactical",
    ]
    ts = {}
    for kind in kinds:
        ts[kind] = {
            "scope_4": _ts_stats(kind, scope_key_4),
            "scope_3": _ts_stats(kind, scope_key_3),
        }

    debug = {
        "plan": {
            "id": plan.get("id"),
            "plan_name": plan.get("plan_name"),
            "business_area": ba,
            "sub_business_area": sba,
            "channel": plan.get("channel"),
            "channel_first": ch,
            "site": plan.get("site"),
            "location": plan.get("location"),
            "start_week": str(plan.get("start_week") or ""),
            "end_week": str(plan.get("end_week") or ""),
            "plan_key": plan.get("plan_key"),
        },
        "scope": {
            "scope_key_4": scope_key_4,
            "scope_key_3": scope_key_3,
            "scope_norm_4": scope_norm_4,
            "scope_norm_3": scope_norm_3,
        },
        "settings": _settings_debug(ba, sba, ch, site),
        "timeseries": ts,
        "dep_tokens": dep_snapshot_all(int(plan_id)),
        "has_dsn": bool(os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")),
    }
    return debug


@app.get("/api/planning/plan/debug/voice-weekly")
def plan_debug_voice_weekly(
    plan_id: int = Query(...),
    week: Optional[str] = Query(None),
):
    _require_debug_enabled()
    plan = load_plan(int(plan_id))
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found.")
    ba = (plan.get("business_area") or plan.get("vertical") or "").strip()
    sba = (plan.get("sub_business_area") or plan.get("sub_ba") or "").strip()
    ch = (plan.get("channel") or plan.get("lob") or "").split(",")[0].strip()
    site = (plan.get("site") or plan.get("location") or plan.get("country") or "").strip()
    scope_key = f"{ba}|{sba}|{ch}|{site}"

    def _canon_ivl(x: str) -> str:
        s = str(x or "").strip()
        if not s or s.lower() in ("nan", "nat"):
            return ""
        try:
            t = pd.to_datetime(s, errors="coerce")
            if pd.notna(t):
                return t.strftime("%H:%M")
        except Exception:
            pass
        m = re.match(r"^(\\d{1,2}):(\\d{2})(?::\\d{2})?(?:\\s*[APap][Mm])?$", s)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2))
            hh = max(0, min(23, hh))
            mm = max(0, min(59, mm))
            return f"{hh:02d}:{mm:02d}"
        return s

    def _summarize(which: str):
        df = _assemble_voice(scope_key, which)
        if df is None or df.empty:
            return {"rows": 0}
        x = df.copy()
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        x = x.dropna(subset=["date"])
        if "interval" in x.columns:
            x["_interval_norm"] = x["interval"].map(_canon_ivl)
        else:
            x["_interval_norm"] = ""
        x["week"] = (x["date"] - pd.to_timedelta(x["date"].dt.weekday, unit="D")).dt.date.astype(str)
        if week:
            x = x[x["week"] == str(week)]
        daily = (
            x.groupby("date", as_index=False)
            .agg(
                rows=("date", "count"),
                uniq_intervals=("_interval_norm", "nunique"),
                vol_sum=("volume", "sum"),
            )
        )
        weekly = _weekly_voice(df)
        wk_val = None
        if isinstance(weekly, pd.DataFrame) and not weekly.empty:
            wk = weekly.set_index("week")
            if week and str(week) in wk.index:
                wk_val = float(wk.loc[str(week), "vol"])
        return {
            "rows": int(len(x.index)),
            "weekly_vol": wk_val,
            "daily": sanitize_for_json(daily.assign(date=daily["date"].dt.date.astype(str)).to_dict("records")),
        }

    return {
        "scope_key": scope_key,
        "week": week,
        "forecast": _summarize("forecast"),
        "actual": _summarize("actual"),
        "tactical": _summarize("tactical"),
    }


@app.get("/api/planning/plan/debug/upper")
def plan_debug_upper(plan_id: int = Query(...)):
    _require_debug_enabled()
    plan = load_plan(int(plan_id))
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found.")
    ba = (plan.get("business_area") or plan.get("vertical") or "").strip()
    sba = (plan.get("sub_business_area") or plan.get("sub_ba") or "").strip()
    ch = (plan.get("channel") or plan.get("lob") or "").split(",")[0].strip()
    site = (plan.get("site") or plan.get("location") or plan.get("country") or "").strip()
    scope_key = f"{ba}|{sba}|{ch}|{site}"
    settings = resolve_cap_settings(ba=ba, subba=sba, lob=ch, site=site) or {}

    vF = _assemble_voice(scope_key, "forecast")
    req_daily_error = None
    try:
        req_daily = required_fte_daily(vF, pd.DataFrame(), pd.DataFrame(), settings)
    except Exception as exc:
        req_daily = pd.DataFrame()
        req_daily_error = f"{type(exc).__name__}: {exc}"
    ivl_error = None
    try:
        voice_ivl = voice_requirements_interval(vF, settings)
    except Exception as exc:
        voice_ivl = pd.DataFrame()
        ivl_error = f"{type(exc).__name__}: {exc}"
    rollup_error = None
    try:
        voice_roll = voice_rollups(voice_ivl, settings)
    except Exception as exc:
        voice_roll = {"daily": pd.DataFrame(), "weekly": pd.DataFrame(), "monthly": pd.DataFrame()}
        rollup_error = f"{type(exc).__name__}: {exc}"

    def _df_summary(df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {"rows": 0}
        out = {"rows": int(len(df.index)), "cols": list(df.columns)}
        if "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce").dropna()
            if not d.empty:
                out["date_min"] = d.min().date().isoformat()
                out["date_max"] = d.max().date().isoformat()
        if "interval" in df.columns:
            out["interval_nulls"] = int(df["interval"].isna().sum())
            out["interval_sample"] = list(df["interval"].astype(str).head(5))
        for c in ("volume", "aht_sec"):
            if c in df.columns:
                out[f"{c}_nulls"] = int(pd.to_numeric(df[c], errors="coerce").isna().sum())
        out["sample"] = sanitize_for_json(df.head(3).to_dict("records"))
        return out

    return {
        "scope_key": scope_key,
        "settings": {
            "interval_minutes": settings.get("interval_minutes"),
            "target_sl": settings.get("target_sl"),
            "sl_seconds": settings.get("sl_seconds"),
            "occupancy_cap_voice": settings.get("occupancy_cap_voice"),
            "shrinkage_pct": settings.get("shrinkage_pct"),
            "hours_per_fte": settings.get("hours_per_fte"),
        },
        "voice_forecast": _df_summary(vF),
        "req_daily": _df_summary(req_daily),
        "req_daily_error": req_daily_error,
        "voice_ivl": _df_summary(voice_ivl),
        "voice_ivl_error": ivl_error,
        "voice_roll_daily": _df_summary(voice_roll.get("daily")),
        "voice_roll_weekly": _df_summary(voice_roll.get("weekly")),
        "voice_roll_error": rollup_error,
    }


@app.get("/api/planning/activity")
def get_planning_activity(request: Request, limit: int = Query(10), plan_id: Optional[int] = Query(None)):
    if plan_id is not None:
        _authorize_plan_access(plan_id, request)
    return {"rows": list_activity(limit=limit, plan_id=plan_id)}


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
def plan_scope_options(request: Request, plan_id: int = Query(...)):
    plan = _authorize_plan_access(plan_id, request)
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
def delete_plan_endpoint(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Delete payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    _authorize_plan_access(plan_id, request)
    invalidate_rollup_cache()
    return delete_plan(int(plan_id))


@app.post("/api/planning/plan/save-as")
def save_plan_as(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Save-as payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    name = (payload.get("name") or "").strip()
    if not plan_id or not name:
        raise HTTPException(status_code=400, detail="plan_id and name are required.")
    _authorize_plan_access(plan_id, request)
    try:
        new_id = clone_plan(int(plan_id), name)
    except Exception as exc:
        logger.exception("clone_plan failed: plan_id=%s", plan_id)
        raise HTTPException(status_code=500, detail="Failed to save plan.") from exc
    return {"status": "created", "id": new_id}


@app.post("/api/planning/plan/extend")
def extend_plan_endpoint(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Extend payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    weeks = payload.get("weeks")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    if not weeks:
        raise HTTPException(status_code=400, detail="weeks is required.")
    _authorize_plan_access(plan_id, request)
    try:
        plan_store.extend_plan_weeks(int(plan_id), int(weeks))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "extended"}


@app.post("/api/planning/plan/export")
def export_plan(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Export payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    _authorize_plan_access(plan_id, request)
    buf = io.BytesIO()
    keys = ["fw", "hc", "attr", "shr", "train", "ratio", "seat", "bva", "nh", "emp", "bulk_files", "notes"]
    try:
        data, status, _meta = compute_plan_detail_tables(int(plan_id), grain="week")
        upper_rows = data.get("upper") if status == "ready" and data else []
    except Exception:
        upper_rows = []
    from app.pipeline.save import sanitize_export

    # Prefer xlsxwriter; fall back to openpyxl so the export still works if the
    # primary engine is missing in a given deployment (both are declared deps).
    try:
        _xl_engine = "xlsxwriter"
        import xlsxwriter  # noqa: F401
    except Exception:
        _xl_engine = "openpyxl"

    with pd.ExcelWriter(buf, engine=_xl_engine) as xw:
        if upper_rows:
            sanitize_export(pd.DataFrame(upper_rows)).to_excel(xw, sheet_name="upper", index=False)
        for key in keys:
            rows = load_plan_table(int(plan_id), key)
            df = pd.DataFrame(rows or [])
            sanitize_export(df).to_excel(xw, sheet_name=key[:31], index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=plan_{plan_id}_export.xlsx"},
    )


@app.get("/api/planning/plan/whatif")
def get_plan_whatif(request: Request, plan_id: int = Query(...)):
    _authorize_plan_access(plan_id, request)
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
def save_plan_whatif(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="What-if payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    _authorize_plan_access(plan_id, request)
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
    try:
        mark_plan_detail_dirty_deps(int(plan_id), "whatif")
        _PRECOMPUTE_EXECUTOR.submit(_precompute_plan_detail, int(plan_id))
    except Exception:
        pass
    try:
        record_activity(
            plan_id=int(plan_id),
            action="updated what-if",
            actor=current_user_fallback(),
            entity_type="whatif",
        )
    except Exception:
        pass
    return {"status": "saved", "overrides": overrides}


# --- Saved scenarios -------------------------------------------------------
# A scenario is a named, reusable snapshot of the what-if dials. Unlike the live
# What-If (a single transient record), scenarios are durable cases a planner can
# save, re-apply, and compare side by side.

def _scenario_upper_rows(plan_id: int, overrides: dict) -> list[dict]:
    """Compute the weekly upper-summary rows for a plan under an arbitrary
    override set, ignoring whatever What-If is currently persisted."""
    wf = dict(overrides or {})
    wf["__replace_persisted__"] = True
    data, status, _meta = compute_plan_detail_tables(int(plan_id), grain="week", whatif=wf)
    rows: list[dict] = []
    if status == "ready" and data:
        rows = data.get("upper") or []
    if not rows:
        cached = load_plan_detail_tables(int(plan_id), grain="week")
        rows = cached.get("upper") or []
    return rows


@app.get("/api/planning/plan/scenarios")
def get_plan_scenarios(plan_id: int = Query(...), request: Request = None):
    _authorize_plan_access(plan_id, request)
    return {"scenarios": scenario_store.list_scenarios(int(plan_id))}


@app.post("/api/planning/plan/scenarios")
def save_plan_scenario(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Scenario payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    _authorize_plan_access(plan_id, request)
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Scenario name is required.")
    overrides = payload.get("overrides") if isinstance(payload.get("overrides"), dict) else {}
    saved = scenario_store.save_scenario(
        int(plan_id),
        scenario_id=payload.get("scenario_id"),
        name=name,
        note=str(payload.get("note") or ""),
        overrides=overrides,
        start_week=str(payload.get("start_week") or ""),
        end_week=str(payload.get("end_week") or ""),
        actor=current_user_fallback(),
    )
    try:
        record_activity(
            plan_id=int(plan_id),
            action=f"saved scenario '{name}'",
            actor=current_user_fallback(),
            entity_type="scenario",
            entity_id=str(saved.get("scenario_id") or ""),
        )
    except Exception:
        pass
    return {"status": "saved", "scenario": saved}


@app.post("/api/planning/plan/scenarios/delete")
def delete_plan_scenario(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Scenario payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    scenario_id = payload.get("scenario_id")
    if not plan_id or not scenario_id:
        raise HTTPException(status_code=400, detail="plan_id and scenario_id are required.")
    _authorize_plan_access(plan_id, request)
    removed = scenario_store.delete_scenario(int(plan_id), str(scenario_id))
    return {"status": "deleted" if removed else "not_found"}


@app.post("/api/planning/plan/scenarios/apply")
def apply_plan_scenario(payload: dict, request: Request):
    """Make a saved scenario the live What-If: writes its overrides + window into
    the plan's What-If record and triggers a recompute."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Scenario payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    scenario_id = payload.get("scenario_id")
    if not plan_id or not scenario_id:
        raise HTTPException(status_code=400, detail="plan_id and scenario_id are required.")
    _authorize_plan_access(plan_id, request)
    scenario = scenario_store.get_scenario(int(plan_id), str(scenario_id))
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found.")
    rec = {
        "ts": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "start_week": scenario.get("start_week") or "",
        "end_week": scenario.get("end_week") or "",
        "overrides": scenario.get("overrides") or {},
    }
    save_df(f"plan_{int(plan_id)}_whatif", pd.DataFrame([rec]))
    save_plan_meta(int(plan_id), {"last_updated_on": rec["ts"], "last_updated_by": current_user_fallback()})
    try:
        mark_plan_detail_dirty_deps(int(plan_id), "whatif")
        _PRECOMPUTE_EXECUTOR.submit(_precompute_plan_detail, int(plan_id))
    except Exception:
        pass
    try:
        record_activity(
            plan_id=int(plan_id),
            action=f"applied scenario '{scenario.get('name')}'",
            actor=current_user_fallback(),
            entity_type="scenario",
            entity_id=str(scenario_id),
        )
    except Exception:
        pass
    return {"status": "applied", "overrides": rec["overrides"], "start_week": rec["start_week"], "end_week": rec["end_week"]}


@app.post("/api/planning/plan/scenarios/compare")
def compare_plan_scenarios(payload: dict, request: Request):
    """Compute the weekly upper summary for a baseline (no dials) plus each
    requested scenario, side by side, without disturbing the live plan."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Scenario payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    _authorize_plan_access(plan_id, request)
    ids = payload.get("scenario_ids")
    ids = [str(s) for s in ids] if isinstance(ids, list) else None
    all_scenarios = scenario_store.list_scenarios(int(plan_id))
    if ids:
        chosen = [s for s in all_scenarios if s["scenario_id"] in ids]
    else:
        chosen = all_scenarios
    columns = []
    include_baseline = payload.get("include_baseline", True)
    if include_baseline:
        columns.append({"scenario_id": "__baseline__", "name": "Baseline (no dials)", "overrides": {}})
    for s in chosen:
        columns.append({"scenario_id": s["scenario_id"], "name": s["name"], "overrides": s.get("overrides") or {}})
    out = []
    for col in columns:
        rows = _scenario_upper_rows(int(plan_id), col["overrides"])
        out.append({
            "scenario_id": col["scenario_id"],
            "name": col["name"],
            "overrides": col["overrides"],
            "upper": sanitize_for_json(rows),
        })
    return {"columns": out}


@app.get("/api/planning/plan/new-hire/classes")
def get_plan_new_hire_classes(request: Request, plan_id: int = Query(...)):
    _authorize_plan_access(plan_id, request)
    df = load_nh_classes(int(plan_id))
    return {
        "rows": df_to_records(df),
        "class_types": get_class_type_options(),
        "class_levels": get_class_level_options(),
    }


@app.post("/api/planning/plan/new-hire/classes")
def save_plan_new_hire_classes(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="New-hire payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    rows = payload.get("rows") or []
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    _authorize_plan_access(plan_id, request)
    df = df_from_payload(rows)
    save_nh_classes(int(plan_id), df)
    try:
        mark_plan_detail_dirty_deps(int(plan_id), "newhire")
        _PRECOMPUTE_EXECUTOR.submit(_precompute_plan_detail, int(plan_id))
    except Exception:
        pass
    try:
        record_activity(
            plan_id=int(plan_id),
            action="updated new hire classes",
            actor=current_user_fallback(),
            entity_type="newhire",
        )
    except Exception:
        pass
    return {"status": "saved", "rows": df_to_records(load_nh_classes(int(plan_id)))}


@app.post("/api/planning/plan/new-hire/class")
def add_plan_new_hire_class(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="New-hire payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    data = payload.get("data") or {}
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    _authorize_plan_access(plan_id, request)
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
    try:
        mark_plan_detail_dirty_deps(int(plan_id), "newhire")
        _PRECOMPUTE_EXECUTOR.submit(_precompute_plan_detail, int(plan_id))
    except Exception:
        pass
    try:
        record_activity(
            plan_id=int(plan_id),
            action="added new hire class",
            actor=current_user_fallback(),
            entity_type="newhire",
            entity_id=str(class_ref),
        )
    except Exception:
        pass
    return {"status": "saved", "rows": df_to_records(df)}


@app.post("/api/planning/plan/new-hire/confirm")
def confirm_plan_new_hire_classes(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="New-hire payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    class_refs = payload.get("class_refs") or []
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    _authorize_plan_access(plan_id, request)
    if not isinstance(class_refs, list):
        raise HTTPException(status_code=400, detail="class_refs must be a list.")
    df = load_nh_classes(int(plan_id))
    if not isinstance(df, pd.DataFrame) or df.empty or not class_refs:
        return {"status": "saved", "rows": df_to_records(df if isinstance(df, pd.DataFrame) else pd.DataFrame())}
    key_set = {str(v).strip().lower() for v in class_refs if str(v).strip()}
    if not key_set:
        return {"status": "saved", "rows": df_to_records(df)}
    if "class_reference" in df.columns:
        mask = df["class_reference"].astype(str).str.strip().str.lower().isin(key_set)
        if mask.any():
            df.loc[mask, "status"] = "confirmed"
            save_nh_classes(int(plan_id), df)
            try:
                mark_plan_detail_dirty_deps(int(plan_id), "newhire")
                _PRECOMPUTE_EXECUTOR.submit(_precompute_plan_detail, int(plan_id))
            except Exception:
                pass
            try:
                record_activity(
                    plan_id=int(plan_id),
                    action="confirmed new hire class",
                    actor=current_user_fallback(),
                    entity_type="newhire",
                )
            except Exception:
                pass
    return {"status": "saved", "rows": df_to_records(df)}


# --- Risk-based staffing (demand percentiles) ------------------------------
# Required FTE at P50/P75/P90 demand. The demand band comes from the forecast's
# historical error (a coefficient of variation); each percentile re-runs the
# capacity engine with volume scaled by 1 + z·CV (reusing the what-if path).

@app.post("/api/planning/plan/risk-band")
def plan_risk_band(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Risk-band payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    plan = _authorize_plan_access(plan_id, request)

    # Resolve CV: explicit param > estimate from this scope's forecast MAPE > default.
    cv = None
    cv_source = "default"
    try:
        cv = float(payload.get("cv")) if payload.get("cv") is not None else None
    except (TypeError, ValueError):
        cv = None
    if cv is not None:
        cv_source = "manual"
    else:
        scope = str(plan.get("business_area") or plan.get("vertical") or "global")
        for sc in (scope, "global"):
            try:
                lb = accuracy_store.leaderboard(sc)
            except Exception:
                lb = {}
            models = lb.get("models") or []
            if models:
                top = models[0]
                mape = (top.get("metrics") or {}).get("mape")
                if mape is None and top.get("primary_metric") == "mape":
                    mape = top.get("primary_value")
                est = risk_band_mod.cv_from_mape(mape)
                if est is not None:
                    cv = est
                    cv_source = f"forecast accuracy MAPE (scope={sc})"
                    break
    if cv is None:
        cv = risk_band_mod.DEFAULT_CV

    requested = payload.get("percentiles")
    pcts = [str(p).lower() for p in requested] if isinstance(requested, list) else list(risk_band_mod.DEFAULT_PERCENTILES)
    pcts = [p for p in pcts if p in risk_band_mod.Z_SCORES]
    if not pcts:
        pcts = list(risk_band_mod.DEFAULT_PERCENTILES)
    if "p50" not in pcts:
        pcts = ["p50"] + pcts

    out = []
    for pct in pcts:
        mult = risk_band_mod.demand_multiplier(cv, pct)
        vol_delta = (mult - 1.0) * 100.0
        upper = _scenario_upper_rows(int(plan_id), {"vol_delta": vol_delta})
        req = risk_band_mod.required_from_upper(upper)
        summ = risk_band_mod.summarize(req["required"])
        out.append(
            {
                "percentile": pct,
                "demand_multiplier": round(mult, 4),
                "vol_delta_pct": round(vol_delta, 2),
                "weeks": req["weeks"],
                "required": [None if r is None else round(r, 1) for r in req["required"]],
                "avg": summ["avg"],
                "peak": summ["peak"],
            }
        )
    base = next((p for p in out if p["percentile"] == "p50"), out[0] if out else None)
    return sanitize_for_json(
        {
            "cv": round(float(cv), 4),
            "cv_source": cv_source,
            "percentiles": out,
            "baseline_avg": base["avg"] if base else None,
            "baseline_peak": base["peak"] if base else None,
        }
    )


# --- Hiring-plan solver ----------------------------------------------------
# Turns new-hire classes from an input into a recommended output: given the
# weekly required-vs-supply curve and the training+nesting lead time, suggest
# class start weeks and sizes that close the projected shortfall.

def _plan_upper_weekly(plan_id: int) -> list[dict]:
    data, status, _meta = compute_plan_detail_tables(int(plan_id), grain="week")
    if status == "ready" and data and data.get("upper"):
        return data.get("upper") or []
    cached = load_plan_detail_tables(int(plan_id), grain="week")
    return cached.get("upper") or []


@app.post("/api/planning/plan/hiring-plan")
def recommend_hiring_plan(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Hiring-plan payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    _authorize_plan_access(plan_id, request)
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    # Default the ramp lead time to training_weeks + nesting_weeks when not given.
    if "ramp_weeks" not in params:
        tw = int(params.get("training_weeks", 4) or 0)
        nw = int(params.get("nesting_weeks", 2) or 0)
        params = {**params, "ramp_weeks": tw + nw}
    upper = _plan_upper_weekly(int(plan_id))
    result = hiring_solver.solve_hiring_plan(upper, params)
    return sanitize_for_json(result)


@app.post("/api/planning/plan/hiring-plan/apply")
def apply_hiring_plan(payload: dict, request: Request):
    """Write recommended classes into the plan's new-hire classes so the
    recommendation becomes an actual, editable hiring plan."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Hiring-plan payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    classes = payload.get("classes")
    if not plan_id or not isinstance(classes, list) or not classes:
        raise HTTPException(status_code=400, detail="plan_id and a non-empty 'classes' list are required.")
    _authorize_plan_access(plan_id, request)
    training_weeks = int(payload.get("training_weeks", 4) or 0)
    nesting_weeks = int(payload.get("nesting_weeks", 2) or 0)

    def _add_weeks(iso: str, weeks: int) -> str:
        try:
            d = pd.Timestamp(str(iso)[:10])
            return (d + pd.Timedelta(weeks=weeks)).date().isoformat()
        except Exception:
            return str(iso or "")

    df = load_nh_classes(int(plan_id))
    now = pd.Timestamp.utcnow().isoformat(timespec="seconds")
    new_rows = []
    for c in classes:
        if not isinstance(c, dict):
            continue
        try:
            grads = int(round(float(c.get("grads_needed") or 0)))
        except Exception:
            grads = 0
        if grads <= 0:
            continue
        start = str(c.get("start_week") or "")
        training_end = _add_weeks(start, training_weeks)
        nesting_end = _add_weeks(training_end, nesting_weeks)
        production = str(c.get("production_week") or _add_weeks(start, training_weeks + nesting_weeks))
        class_ref = next_class_reference(str(plan_id), df)
        row = {
            "class_reference": class_ref,
            "source_system_id": class_ref,
            "emp_type": "full-time",
            "status": "tentative",
            "class_type": "ramp-up",
            "class_level": "new-agent",
            "grads_needed": grads,
            "billable_hc": 0,
            "training_weeks": training_weeks,
            "nesting_weeks": nesting_weeks,
            "induction_start": start,
            "training_start": start,
            "training_end": training_end,
            "nesting_start": training_end,
            "nesting_end": nesting_end,
            "production_start": production,
            "created_by": current_user_fallback(),
            "created_ts": now,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        new_rows.append(class_ref)
    if not new_rows:
        raise HTTPException(status_code=400, detail="No valid classes to apply.")
    save_nh_classes(int(plan_id), df)
    try:
        mark_plan_detail_dirty_deps(int(plan_id), "newhire")
        _PRECOMPUTE_EXECUTOR.submit(_precompute_plan_detail, int(plan_id))
    except Exception:
        pass
    try:
        record_activity(
            plan_id=int(plan_id),
            action=f"applied hiring plan ({len(new_rows)} classes)",
            actor=current_user_fallback(),
            entity_type="newhire",
        )
    except Exception:
        pass
    return {"status": "applied", "added": len(new_rows), "rows": df_to_records(load_nh_classes(int(plan_id)))}


# --- Monte Carlo demand simulation -----------------------------------------
# Distribution of required FTE (and coverage probability) under uncertain demand.
# A small required-FTE response curve over volume multipliers is computed via the
# capacity engine, then demand draws are simulated and interpolated off the curve.

_MC_MULTIPLIERS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]


def _avg_required_and_supply(upper_rows: list[dict]) -> tuple[Optional[float], Optional[float]]:
    """Mean weekly required FTE (= supply − over/under) and mean supply."""
    def _row(*needles):
        for r in upper_rows or []:
            if isinstance(r, dict) and all(n in str(r.get("metric") or "").lower() for n in needles):
                return r
        return None

    ou = _row("over", "mtp") or _row("over/under")
    sup = _row("projected supply")
    if not ou:
        return None, None
    weeks = [k for k in ou.keys() if k != "metric"]

    def _n(v):
        try:
            return float(str(v).replace(",", "").rstrip("%"))
        except (TypeError, ValueError):
            return None

    reqs, sups = [], []
    for w in weeks:
        ouv = _n(ou.get(w))
        supv = _n(sup.get(w)) if sup else None
        if supv is not None:
            sups.append(supv)
            if ouv is not None:
                reqs.append(max(0.0, supv - ouv))
    avg_req = sum(reqs) / len(reqs) if reqs else None
    avg_sup = sum(sups) / len(sups) if sups else None
    return avg_req, avg_sup


@app.post("/api/planning/plan/monte-carlo")
def plan_monte_carlo(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Monte Carlo payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    plan = _authorize_plan_access(plan_id, request)

    # Resolve demand CV: param > this scope's tracked forecast MAPE > default 12%.
    cv = None
    cv_source = "default"
    try:
        cv = float(payload.get("cv")) if payload.get("cv") is not None else None
    except (TypeError, ValueError):
        cv = None
    if cv is not None:
        cv_source = "manual"
    else:
        scope = str(plan.get("business_area") or plan.get("vertical") or "global")
        for sc in (scope, "global"):
            try:
                lb = accuracy_store.leaderboard(sc)
            except Exception:
                lb = {}
            models = lb.get("models") or []
            if models:
                mape = (models[0].get("metrics") or {}).get("mape")
                if mape is None and models[0].get("primary_metric") == "mape":
                    mape = models[0].get("primary_value")
                try:
                    if mape is not None and float(mape) > 0:
                        cv = float(mape) / 100.0
                        cv_source = f"forecast accuracy MAPE (scope={sc})"
                        break
                except (TypeError, ValueError):
                    pass
    if cv is None:
        cv = 0.12

    # Build the required-FTE response curve and the baseline supply.
    curve: list[tuple[float, float]] = []
    supply_level: Optional[float] = None
    for m in _MC_MULTIPLIERS:
        upper = _scenario_upper_rows(int(plan_id), {"vol_delta": (m - 1.0) * 100.0})
        req, sup = _avg_required_and_supply(upper)
        if req is not None:
            curve.append((m, req))
        if abs(m - 1.0) < 1e-9 and sup is not None:
            supply_level = sup
    if not curve:
        return {"status": "no_data", "reason": "No required-FTE data for this plan."}

    draws = payload.get("draws", 2000)
    result = monte_carlo_mod.simulate(curve, supply_level, cv, draws=draws)
    result["status"] = "ok"
    result["cv_source"] = cv_source
    result["response_curve"] = [{"multiplier": m, "required": round(r, 1)} for m, r in curve]
    return sanitize_for_json(result)


@app.post("/api/planning/plan/compare")
def compare_plans(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Compare payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    compare_id = payload.get("compare_id")
    if not plan_id or not compare_id:
        raise HTTPException(status_code=400, detail="plan_id and compare_id are required.")
    _authorize_plan_access(plan_id, request)
    _authorize_plan_access(compare_id, request)

    def _upper_df(pid: int) -> pd.DataFrame:
        data, status, _meta = compute_plan_detail_tables(int(pid), grain="week")
        rows = []
        if status == "ready" and data:
            rows = data.get("upper") or []
        if not rows:
            cached = load_plan_detail_tables(int(pid), grain="week")
            rows = cached.get("upper") or []
        if not rows:
            return pd.DataFrame()
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
        return sanitize_for_json(df.reset_index().to_dict("records"))

    return {"current": _records(curr_df), "compare": _records(comp_df), "delta": _records(diff_df)}


@app.get("/api/planning/plan/table")
def get_plan_table(request: Request, plan_id: int = Query(...), name: str = Query(...)):
    _authorize_plan_access(plan_id, request)
    pid = int(plan_id)
    table_name = str(name or "")
    if table_name.strip().lower() == "emp":
        try:
            plan = load_plan(pid) or {}
        except Exception:
            plan = {}
        scope = _plan_scope_for_attrition(plan)
        if scope:
            _resync_saved_attrition_to_roster(scope, target_plan_ids={pid})
    return {"rows": load_plan_table(pid, name)}


@app.post("/api/planning/plan/table")
def save_plan_table_endpoint(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Plan table payload must be a JSON object.")
    plan_id = payload.get("plan_id")
    name = payload.get("name")
    rows = payload.get("rows") or []
    if not plan_id or not name:
        raise HTTPException(status_code=400, detail="plan_id and name are required.")
    _authorize_plan_access(plan_id, request)
    result = save_plan_table(int(plan_id), str(name), rows)
    if result.get("status") == "locked":
        raise HTTPException(status_code=409, detail="Plan is locked (history).")
    # A table edit changes the plan's contribution to any BA rollup it belongs to.
    invalidate_rollup_cache()
    try:
        base = str(name or "").split("_")[0].lower()
        if base in {"notes", "bulk_files"}:
            return result
        _PRECOMPUTE_EXECUTOR.submit(_precompute_plan_detail, int(plan_id))
    except Exception:
        pass
    return result


@app.post("/api/planning/plan/detail/compute")
def compute_plan_detail_endpoint(payload: dict, request: Request):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Plan detail payload must be a JSON object.")
    plan_id = payload.get("plan_id") or payload.get("id")
    if not plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required.")
    _authorize_plan_access(plan_id, request)
    try:
        plan = load_plan(int(plan_id))
        if plan and str(plan.get("status") or "").lower() == "history":
            cached = load_plan_detail_tables(int(plan_id), grain=payload.get("grain"), interval_date=payload.get("interval_date"))
            has_cached = False
            try:
                has_cached = any(cached.get("upper") or []) or any(len(v or []) for v in (cached.get("tables") or {}).values())
            except Exception:
                has_cached = False
            if has_cached:
                return sanitize_for_json({"status": "ready", "job": {"locked": True, "cached": True}, "data": cached})
            raise HTTPException(status_code=409, detail="Plan is locked (history) and no snapshot is available.")
    except HTTPException:
        raise
    except Exception:
        pass
    grain = payload.get("grain") or "week"
    whatif = payload.get("whatif") if isinstance(payload.get("whatif"), dict) else None
    interval_date = payload.get("interval_date")
    version_token = payload.get("version_token")
    force_recompute = bool(payload.get("force_recompute"))
    allow_cached = bool(payload.get("allow_cached", True))
    persist = bool(payload.get("persist", True))
    if str(grain).lower() in {"interval", "day", "week", "month"}:
        allow_cached = False

    if force_recompute and version_token is None:
        # Force a fresh compute by bumping the version token.
        version_token = f"force:{time.time()}"

    try:
        data, status, meta = compute_plan_detail_tables(
            int(plan_id),
            grain=grain,
            whatif=whatif,
            interval_date=interval_date,
            version_token=version_token,
            persist=persist,
        )
    except Exception:
        logger.exception("plan detail compute failed: plan_id=%s grain=%s interval_date=%s", plan_id, grain, interval_date)
        raise
    if status != "ready":
        if force_recompute:
            return sanitize_for_json({"status": status, "job": meta})
        cached = load_plan_detail_tables(int(plan_id), grain=grain, interval_date=interval_date)
        has_cached = False
        try:
            has_cached = any(cached.get("upper") or []) or any(len(v or []) for v in (cached.get("tables") or {}).values())
        except Exception:
            has_cached = False
        dep_changed = []
        if isinstance(meta, dict):
            dep_changed = meta.get("dep_changed") or []
        allow_cached = True
        if dep_changed:
            light_only = set(dep_changed).issubset({"plan_tables:notes", "plan_tables:bulk_files"})
            allow_cached = light_only
        if has_cached and allow_cached:
            meta = dict(meta or {})
            meta["cached"] = True
            return sanitize_for_json({"status": "ready", "job": meta, "data": cached})
        return sanitize_for_json({"status": status, "job": meta})
    if payload.get("prefetch", True) and not whatif:
        g = str(grain or "week").strip().lower()
        prefetch_grains = [item for item in ["week", "month", "day"] if item != g]
        prefetch_plan_detail_grains(
            int(plan_id),
            grains=prefetch_grains,
            whatif=None,
            version_token=version_token,
        )
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
    try:
        record_activity(
            action="uploaded shrinkage",
            actor=current_user_fallback(),
            entity_type="shrinkage",
            payload={"rows": int(count or 0)},
        )
    except Exception:
        pass
    try:
        _invalidate_plan_detail_for_scope({"scope_type": "global"}, "shrinkage")
    except Exception:
        pass
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
    mode = str(payload.get("mode") or "replace").strip().lower()
    if mode not in {"replace", "append"}:
        mode = "replace"
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

    existing = load_shrinkage_weekly()
    if mode == "append":
        combined = merge_shrink_weekly(existing, weekly)
    else:
        if existing is None or existing.empty:
            combined = weekly
        else:
            combined = pd.concat([existing, weekly], ignore_index=True)
            combined = normalize_shrink_weekly(combined)
            combined = combined.drop_duplicates(subset=["week", "program"], keep="last")
            combined = combined.sort_values(["week", "program"]).reset_index(drop=True)
    if save_flag:
        save_shrinkage_raw(raw_kind, norm)
        save_shrinkage_weekly(combined)
        try:
            _invalidate_plan_detail_for_scope({"scope_type": "global"}, "shrinkage")
        except Exception:
            pass

    return {
        "raw": df_to_records(norm),
        "daily": df_to_records(daily),
        "weekly": df_to_records(weekly),
        "combined": df_to_records(combined),
    }


@app.get("/api/forecast/attrition")
def get_attrition_weekly(
    ba: Optional[str] = None,
    sba: Optional[str] = None,
    channel: Optional[str] = None,
    site: Optional[str] = None,
):
    scope = None
    if any(str(v or "").strip() for v in (ba, sba, channel, site)):
        scope = {
            "business_area": ba,
            "sub_business_area": sba,
            "channel": channel,
            "site": site,
        }
        _resync_saved_attrition_to_roster(scope)
    df = load_attrition_weekly(scope=scope)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.replace([np.inf, -np.inf], np.nan)
    return {"rows": sanitize_for_json(df_to_records(df))}


@app.get("/api/forecast/attrition/forecast")
def get_attrition_forecast(
    ba: Optional[str] = None,
    sba: Optional[str] = None,
    channel: Optional[str] = None,
    site: Optional[str] = None,
    horizon: int = 12,
):
    """Project the weekly attrition rate forward (level+trend) with a P10/P90 band."""
    from app.pipeline import attrition_forecast

    scope = None
    if any(str(v or "").strip() for v in (ba, sba, channel, site)):
        scope = {"business_area": ba, "sub_business_area": sba, "channel": channel, "site": site}
    df = load_attrition_weekly(scope=scope)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.replace([np.inf, -np.inf], np.nan)
    rows = df_to_records(df) if isinstance(df, pd.DataFrame) else []
    return sanitize_for_json(attrition_forecast.project_attrition(rows, horizon=horizon))


@app.post("/api/forecast/attrition")
def save_attrition_weekly_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Attrition payload must be a JSON object.")
    rows = payload.get("rows") or []
    mode = str(payload.get("mode") or "replace").strip().lower()
    if mode not in {"replace", "append"}:
        mode = "replace"
    scope_payload = payload.get("scope") or {}
    scope = None
    if isinstance(scope_payload, dict) and any(
        str(scope_payload.get(k) or "").strip()
        for k in ("business_area", "ba", "sub_business_area", "sba", "channel", "site")
    ):
        scope = {
            "business_area": scope_payload.get("business_area") or scope_payload.get("ba"),
            "sub_business_area": scope_payload.get("sub_business_area") or scope_payload.get("sba"),
            "channel": scope_payload.get("channel"),
            "site": scope_payload.get("site"),
        }
    df = pd.DataFrame(rows or [])
    count = save_attrition_weekly(df, mode=mode, scope=scope)
    combined = load_attrition_weekly(scope=scope)
    try:
        record_activity(
            action="uploaded attrition",
            actor=current_user_fallback(),
            entity_type="attrition",
            payload={"rows": int(count or 0), "mode": mode, "scope": scope or {"scope_type": "global"}},
        )
    except Exception:
        pass
    try:
        _invalidate_plan_detail_for_scope(
            {
                "scope_type": "hier" if scope else "global",
                "business_area": (scope or {}).get("business_area"),
                "sub_business_area": (scope or {}).get("sub_business_area"),
                "channel": (scope or {}).get("channel"),
                "site": (scope or {}).get("site"),
            },
            "attrition",
        )
    except Exception:
        pass
    return {"status": "saved", "rows": df_to_records(combined), "count": count, "mode": mode}


@app.post("/api/forecast/attrition/raw")
def attrition_raw_endpoint(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Attrition payload must be a JSON object.")
    rows = payload.get("rows") or []
    save_flag = bool(payload.get("save"))
    mode = str(payload.get("mode") or "replace").strip().lower()
    if mode not in {"replace", "append"}:
        mode = "replace"
    scope_payload = payload.get("scope") or {}
    scope = None
    if isinstance(scope_payload, dict) and any(
        str(scope_payload.get(k) or "").strip()
        for k in ("business_area", "ba", "sub_business_area", "sba", "channel", "site")
    ):
        scope = {
            "business_area": scope_payload.get("business_area") or scope_payload.get("ba"),
            "sub_business_area": scope_payload.get("sub_business_area") or scope_payload.get("sba"),
            "channel": scope_payload.get("channel"),
            "site": scope_payload.get("site"),
        }
    raw_df = df_from_payload(rows)
    weekly = attrition_weekly_from_raw(raw_df, scope=scope)
    combined = weekly
    roster_sync = {"updated_plans": 0, "updated_rows": 0}
    if save_flag:
        save_attrition_raw(raw_df, scope=scope)
        save_attrition_weekly(weekly, mode=mode, scope=scope)
        combined = load_attrition_weekly(scope=scope)
        try:
            roster_sync = _sync_attrition_to_plan_roster(raw_df, scope)
        except Exception:
            roster_sync = {"updated_plans": 0, "updated_rows": 0}
        try:
            _invalidate_plan_detail_for_scope(
                {
                    "scope_type": "hier" if scope else "global",
                    "business_area": (scope or {}).get("business_area"),
                    "sub_business_area": (scope or {}).get("sub_business_area"),
                    "channel": (scope or {}).get("channel"),
                    "site": (scope or {}).get("site"),
                },
                "attrition",
            )
        except Exception:
            pass
    return {
        "weekly": df_to_records(weekly),
        "combined": df_to_records(combined),
        "mode": mode,
        "roster_sync": roster_sync,
    }


@app.post("/api/forecast/volume-summary")
async def volume_summary(file: UploadFile = File(...)):
    content = await _read_upload_capped(file)
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    _guard_zip_bomb(content)
    result = run_volume_summary(file.filename or "upload", content)
    try:
        save_forecast_run(
            "volume_summary",
            result,
            scope_key="global",
            meta={"filename": file.filename or "upload"},
            created_by=current_user_fallback(),
        )
    except Exception:
        pass
    return sanitize_for_json(result)


@app.post("/api/forecast/ingest/original")
async def ingest_original(file: UploadFile = File(...)):
    content = await _read_upload_capped(file)
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    _guard_zip_bomb(content)
    df, msg = parse_upload_any(
        file.filename or "upload",
        content,
        preferred_sheets=("Volume", "Original", "Data"),
    )
    return sanitize_for_json({"message": msg, "rows": df_to_records(df)})


@app.post("/api/forecast/ingest/interval")
async def ingest_interval(file: UploadFile = File(...)):
    content = await _read_upload_capped(file)
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    _guard_zip_bomb(content)
    df, msg = parse_upload_any(
        file.filename or "upload",
        content,
        preferred_sheets=("Interval", "Intervals", "Timeslot", "Time", "Volume"),
    )
    return sanitize_for_json({"message": msg, "rows": df_to_records(df)})


@app.get("/api/forecast/saved-runs")
def list_saved_runs():
    return sanitize_for_json({"runs": list_saved_forecasts()})


# --- Forecast accuracy tracking -------------------------------------------
# Phase 1 computes per-model accuracy; these endpoints expose the persisted
# history + a "which model is winning" leaderboard for a scope.

@app.get("/api/forecast/accuracy")
def get_forecast_accuracy(scope: str = Query("global")):
    from app.pipeline import accuracy_store

    return sanitize_for_json(
        {
            "scopes": accuracy_store.list_scopes(),
            "leaderboard": accuracy_store.leaderboard(scope),
            "history": accuracy_store.load_history(scope),
        }
    )


@app.post("/api/forecast/accuracy/record")
def record_forecast_accuracy(payload: dict, request: Request):
    """Record an accuracy snapshot from a Phase-1 accuracy table. Auth is a no-op
    unless AUTH_ENABLED; the forecast result-save path also records automatically."""
    security.require_user(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Accuracy payload must be a JSON object.")
    from app.pipeline import accuracy_store

    rows = payload.get("accuracy") if isinstance(payload.get("accuracy"), list) else payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="An 'accuracy' table (list of per-model rows) is required.")
    scope = payload.get("scope") or payload.get("business_area") or payload.get("forecast_group") or "global"
    snap = accuracy_store.record_snapshot(
        str(scope),
        rows,
        run_label=str(payload.get("run_label") or ""),
        actor=current_user_fallback(),
    )
    if not snap:
        raise HTTPException(status_code=422, detail="No usable per-model accuracy metrics found in the table.")
    return sanitize_for_json({"status": "recorded", "snapshot": snap})


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
def smoothing(payload: dict):
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
def smoothing_auto(payload: dict):
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
def phase1(payload: dict):
    data = payload.get("data")
    config = payload.get("config")
    holidays = payload.get("holidays")

    result = run_phase1(data, config=config, holidays=holidays)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Phase 1 failed."))
    try:
        save_forecast_run(
            "phase1",
            result,
            scope_key=_scope_key_from_payload(payload),
            meta={"holidays": bool(holidays), "config": bool(config)},
            created_by=current_user_fallback(),
        )
    except Exception:
        pass
    return sanitize_for_json(result)


@app.post("/api/forecast/phase2")
def phase2(payload: dict):
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
    try:
        save_forecast_run(
            "phase2",
            result,
            scope_key=_scope_key_from_payload(payload),
            meta={"basis": basis, "start_date": start_date, "end_date": end_date},
            created_by=current_user_fallback(),
        )
    except Exception:
        pass
    return sanitize_for_json(result)


@app.post("/api/forecast/transformations/apply")
def transformations_apply(payload: dict):
    data = payload.get("data")
    result = apply_transformations(data)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Transformations failed."))
    try:
        save_forecast_run(
            "transformations",
            result,
            scope_key=_scope_key_from_payload(payload),
            meta={"action": "apply"},
            created_by=current_user_fallback(),
        )
    except Exception:
        pass
    return sanitize_for_json(result)


@app.post("/api/forecast/seasonality/build")
def build_seasonality_endpoint(payload: dict):
    ratio_table = payload.get("ratio_table")
    result = build_seasonality(ratio_table)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Seasonality failed."))
    try:
        save_forecast_run(
            "seasonality_build",
            result,
            scope_key=_scope_key_from_payload(payload),
            created_by=current_user_fallback(),
        )
    except Exception:
        pass
    return sanitize_for_json(result)


@app.post("/api/forecast/seasonality/apply")
def apply_seasonality_endpoint(payload: dict):
    capped_rows = payload.get("capped_rows")
    lower_cap = payload.get("lower_cap")
    upper_cap = payload.get("upper_cap")
    base_volume = payload.get("base_volume")
    result = apply_seasonality_changes(capped_rows, lower_cap, upper_cap, base_volume)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Seasonality failed."))
    try:
        save_forecast_run(
            "seasonality_apply",
            result,
            scope_key=_scope_key_from_payload(payload),
            created_by=current_user_fallback(),
        )
    except Exception:
        pass
    return sanitize_for_json(result)


@app.post("/api/forecast/volume-summary/prophet-smoothing")
def volume_summary_prophet(payload: dict):
    normalized_ratio = payload.get("normalized_ratio")
    ratio_table = payload.get("ratio_table")
    iq_table = payload.get("iq_table")
    holidays = payload.get("holidays")
    result = run_prophet_smoothing(normalized_ratio, ratio_table, iq_table, holidays)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Prophet smoothing failed."))
    try:
        save_forecast_run(
            "prophet_smoothing",
            result,
            scope_key=_scope_key_from_payload(payload),
            created_by=current_user_fallback(),
        )
    except Exception:
        pass
    return sanitize_for_json(result)


@app.post("/api/forecast/volume-summary/prophet-save")
def volume_summary_prophet_save(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Prophet save payload must be a JSON object.")
    edited = payload.get("edited")
    original = payload.get("original")
    result = save_prophet_changes(edited, original)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Prophet save failed."))
    return sanitize_for_json(result)


@app.post("/api/forecast/phase2/volume-split")
def phase2_volume_split(payload: dict):
    base_df = payload.get("base_df")
    split_rows = payload.get("split_rows")
    fg_monthly = payload.get("forecast_group_monthly")
    result = apply_volume_split(base_df, split_rows, fg_monthly)
    if not result.get("results"):
        raise HTTPException(status_code=400, detail=result.get("status", "Volume split failed."))
    return sanitize_for_json(result)


@app.post("/api/forecast/daily-interval")
def daily_interval(payload: dict):
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
    try:
        scope_key = _scope_key_from_payload({"group_value": group_value}) if not payload.get("scope_key") else str(payload.get("scope_key"))
        meta = {
            "forecast_month": forecast_month,
            "group_value": group_value,
            "model_value": model_value,
            "group_level": group_level,
        }
        save_forecast_run(
            "daily_interval",
            result,
            scope_key=scope_key,
            meta=meta,
            created_by=current_user_fallback(),
        )
    except Exception:
        pass
    return sanitize_for_json(result)


@app.post("/api/forecast/push-to-plan")
def push_forecast_to_plan(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Push payload must be a JSON object.")

    ba = (payload.get("business_area") or payload.get("ba") or payload.get("vertical") or "").strip()
    sba = (payload.get("sub_business_area") or payload.get("sub_ba") or payload.get("sba") or "").strip()
    channel = (payload.get("channel") or payload.get("lob") or "").strip()
    site = (payload.get("site") or payload.get("location") or "").strip()
    if not (ba and sba and channel):
        return {"ok": False, "message": "Select Business Area, Sub BA, and Channel before pushing."}

    results = payload.get("results")
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except Exception:
            results = None
    if not isinstance(results, dict):
        results = {}

    mode = str(payload.get("mode") or payload.get("on_duplicate") or "append").strip().lower()
    if mode not in ("append", "override", "replace"):
        mode = "append"

    daily_rows = payload.get("daily") or results.get("daily")
    interval_rows = payload.get("interval") or results.get("interval")

    def _build_df(rows, prefer_cols, *, interval_required: bool = False, prefer_sut: bool = False):
        df = pd.DataFrame(rows or [])
        if df.empty:
            return pd.DataFrame()
        date_col = _pick_col(df, ("date", "Date"))
        vol_col = _pick_col(df, prefer_cols)
        if not date_col or not vol_col:
            return pd.DataFrame()
        out = df[[date_col, vol_col]].copy()
        interval_col = _pick_col(
            df,
            (
                "interval",
                "Interval",
                "interval_start",
                "interval start",
                "interval_start_time",
                "interval start time",
                "intervalstart",
                "start_time",
                "start time",
                "starttime",
                "time",
                "Time",
                "timeslot",
                "time_slot",
                "time slot",
                "slot",
            ),
        )
        if not interval_col and interval_required:
            def _looks_like_interval(series: pd.Series) -> float:
                s = series.astype(str).str.strip()
                if s.empty:
                    return 0.0
                pat = r"^\\d{1,2}:\\d{2}(?::\\d{2})?(?:\\s*-\\s*\\d{1,2}:\\d{2}(?::\\d{2})?)?$"
                matches = s.str.match(pat, na=False)
                return float(matches.mean()) if len(matches) else 0.0

            best_col = None
            best_score = 0.0
            for col in df.columns:
                if col in (date_col, vol_col):
                    continue
                score = _looks_like_interval(df[col])
                if score > best_score:
                    best_score = score
                    best_col = col
            if best_score >= 0.6:
                interval_col = best_col
        if interval_col:
            out["interval"] = df[interval_col]
        if interval_required and "interval" not in out.columns:
            return pd.DataFrame()
        dur_col = _pick_col(
            df,
            (
                "aht_sec",
                "AHT (sec)",
                "aht",
                "aht/sut",
                "aht sut",
                "aht_sut",
                "forecast aht",
                "forecast_aht",
                "forecast aht (sec)",
                "forecast_aht_sec",
                "actual aht",
                "actual_aht",
                "actual aht (sec)",
                "actual_aht_sec",
                "tactical aht",
                "tactical_aht",
                "tactical aht (sec)",
                "tactical_aht_sec",
                "forecast sut",
                "forecast_sut",
                "actual sut",
                "actual_sut",
                "tactical sut",
                "tactical_sut",
                "avg_handle_time",
                "avg handle time",
                "avg_handle_time_sec",
                "avg handle time sec",
                "avg_talk_sec",
                "talk_sec",
                "sut_sec",
                "sut",
                "SUT (sec)",
            ),
        )
        if dur_col:
            dur_low = str(dur_col).strip().lower()
            is_sut = prefer_sut or ("sut" in dur_low)
            tgt = "sut_sec" if is_sut else "aht_sec"
            out[tgt] = pd.to_numeric(df[dur_col], errors="coerce")
        out = out.rename(columns={date_col: "date", vol_col: "volume"})
        # Robust numeric parse (strip commas/percents)
        out["volume"] = (
            out["volume"]
            .astype(str)
            .str.replace(r"[%,$]", "", regex=True)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if out["date"].isna().all():
            out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
        out = out.dropna(subset=["date", "volume"])
        return out

    channel_key = (channel or "").strip().lower()
    ch_low = channel_key
    is_voice = ch_low in ("voice", "call", "telephony")
    is_bo = ch_low in ("back office", "backoffice", "bo")
    is_chat = ch_low in ("chat", "messaging", "messageus", "message us")
    is_ob = ch_low in ("outbound", "ob", "out bound")

    sel = str(payload.get("granularity") or payload.get("push_granularity") or "auto").strip().lower()
    if sel not in ("auto", "interval", "daily"):
        sel = "auto"
    use_daily = bool(is_bo) if sel == "auto" else sel == "daily"

    if is_voice and use_daily:
        return {"ok": False, "message": "Voice forecasts must be pushed at interval granularity."}
    if is_bo and not use_daily:
        return {"ok": False, "message": "Back Office forecasts must be pushed at daily granularity."}

    if use_daily:
        forecast_df = _build_df(
            daily_rows,
            ("daily_forecast", "forecast", "volume"),
            interval_required=False,
            prefer_sut=is_bo,
        )
    else:
        interval_required = bool(is_voice or is_chat or is_ob)
        forecast_df = _build_df(
            interval_rows,
            ("interval_forecast", "forecast", "volume"),
            interval_required=interval_required,
            prefer_sut=False,
        )

    if forecast_df.empty:
        if not use_daily and (is_voice or is_chat or is_ob):
            debug = bool(payload.get("debug"))
            cols = []
            date_col = None
            vol_col = None
            interval_col = None
            sample = None
            try:
                raw_df = pd.DataFrame(interval_rows or [])
                cols = list(raw_df.columns)
                if not raw_df.empty:
                    sample_rows = sanitize_for_json(raw_df.head(1).to_dict("records"))
                    sample = sample_rows[0] if sample_rows else None
                date_col = _pick_col(raw_df, ("date", "Date"))
                vol_col = _pick_col(raw_df, ("interval_forecast", "forecast", "volume"))
                interval_col = _pick_col(
                    raw_df,
                    (
                        "interval",
                        "Interval",
                        "interval_start",
                        "interval start",
                        "interval_start_time",
                        "interval start time",
                        "intervalstart",
                        "start_time",
                        "start time",
                        "starttime",
                        "time",
                        "Time",
                        "timeslot",
                        "time_slot",
                        "time slot",
                        "slot",
                    ),
                )
            except Exception:
                pass
            suffix = f" Columns detected: {cols}" if cols else ""
            msg = "No interval forecast rows found (missing interval/time column). Re-run the forecast to generate interval output, then push again." + suffix
            if debug:
                return {
                    "ok": False,
                    "message": msg,
                    "debug": {
                        "date_col": date_col,
                        "volume_col": vol_col,
                        "interval_col": interval_col,
                        "columns": cols,
                        "sample_row": sample,
                    },
                }
            return {"ok": False, "message": msg}
        return {"ok": False, "message": "No forecast rows found for the selected channel and granularity."}

    aht_override = payload.get("aht_sec") or payload.get("aht") or payload.get("sut_sec")
    aht_value = None
    if aht_override is not None and aht_override != "":
        try:
            aht_value = float(aht_override)
        except Exception:
            aht_value = None

    scope_key = f"{ba}|{sba}|{channel_key}|{site}"
    metadata = {"source": "daily-interval", "granularity": "daily" if use_daily else "interval"}
    _ = metadata  # reserved for future audit use
    ok, msg, _run_id = cap_store.push_forecast_to_planning(
        scope_key,
        channel_key or channel,
        forecast_df,
        default_aht_sec=aht_value,
        mode=mode,
    )
    if ok:
        try:
            _invalidate_plan_detail_for_scope(_scope_from_key(scope_key), "timeseries")
        except Exception:
            pass
    return {"ok": ok, "message": msg}


@app.post("/api/forecast/save/smoothing")
def save_smoothing_results(payload: dict):
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, dict):
        payload = results
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid smoothing payload.")
    if not payload.get("smoothed") and not payload.get("capped"):
        raise HTTPException(status_code=400, detail="No smoothing data to save.")
    return sanitize_for_json(save_smoothing(payload))


@app.post("/api/forecast/save/forecast-results")
def save_forecast(payload: dict):
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, dict):
        payload = results
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid forecast payload.")
    if not payload.get("combined") and not payload.get("accuracy"):
        raise HTTPException(status_code=400, detail="No forecast results to save.")
    out = save_forecast_results(payload)
    try:
        save_forecast_run(
            "forecast_results",
            payload,
            scope_key=_scope_key_from_payload(payload),
            meta={"saved": True},
            created_by=current_user_fallback(),
        )
    except Exception:
        pass
    return sanitize_for_json(out)


@app.post("/api/forecast/save/adjusted-forecast")
def save_adjusted(payload: dict):
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
def save_transformations_results(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid transformations payload.")
    rows = payload.get("data") or payload.get("rows") or payload.get("final")
    df = df_from_payload(rows)
    if df.empty:
        raise HTTPException(status_code=400, detail="No transformed rows to save.")
    out = save_transformations(df)
    try:
        save_forecast_run(
            "transformations_save",
            {"rows": df_to_records(df)},
            scope_key=_scope_key_from_payload(payload),
            meta={"saved": True},
            created_by=current_user_fallback(),
        )
    except Exception:
        pass
    return sanitize_for_json(out)


@app.post("/api/forecast/save/daily-interval")
def save_daily_interval_results(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid daily/interval payload.")
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    daily = payload.get("daily") or results.get("daily")
    interval = payload.get("interval") or results.get("interval")
    meta = payload.get("meta") or results.get("meta") or {}
    if not daily and not interval:
        raise HTTPException(status_code=400, detail="No daily/interval rows to save.")
    out = save_daily_interval({"daily": daily or [], "interval": interval or [], "meta": meta})
    try:
        save_forecast_run(
            "daily_interval_save",
            {"daily": daily or [], "interval": interval or [], "meta": meta},
            scope_key=_scope_key_from_payload(payload),
            meta={"saved": True},
            created_by=current_user_fallback(),
        )
    except Exception:
        pass
    return sanitize_for_json(out)
