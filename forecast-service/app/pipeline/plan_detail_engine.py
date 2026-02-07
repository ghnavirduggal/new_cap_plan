from __future__ import annotations

import datetime as dt
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from app.pipeline.plan_detail._calc import _fill_tables_fixed
from app.pipeline.plan_detail._common import _month_cols, _week_cols, _week_span, resolve_settings
from app.pipeline.plan_detail._fill_tables_fixed_daily import _fill_tables_fixed_daily
from app.pipeline.plan_detail._fill_tables_fixed_interval import _fill_tables_fixed_interval
from app.pipeline.plan_detail._fill_tables_fixed_monthly import _fill_tables_fixed_monthly
from app.pipeline.plan_detail._grain_cols import day_cols_for_weeks, interval_cols_for_day
from app.pipeline.plan_detail.calc_engine import ensure_plan_calc, dep_snapshot, dep_snapshot_all


_LAST_DEP_SNAPSHOT: dict[tuple[int, str, str], dict] = {}
CALC_VERSION = 2
from app.pipeline.planning_store import load_plan, load_plan_table, save_plan_table


_TABLE_KEYS = [
    "fw",
    "hc",
    "attr",
    "shr",
    "train",
    "ratio",
    "seat",
    "bva",
    "nh",
    "emp",
    "bulk_files",
    "notes",
]

_PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="plan-prefetch")


def _normalize_grain(grain: Optional[str]) -> str:
    g = str(grain or "week").strip().lower()
    return g if g in {"week", "month", "day", "interval"} else "week"


def _parse_date(value: Optional[str]) -> dt.date:
    if not value:
        return dt.date.today()
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return dt.date.today()
    if pd.isna(parsed):
        return dt.date.today()
    return parsed.date()


def _resolve_ivl_min(plan: dict) -> int:
    ch_first = str(plan.get("channel") or plan.get("lob") or "").split(",")[0].strip()
    settings = resolve_settings(
        ba=plan.get("business_area") or plan.get("vertical"),
        subba=plan.get("sub_business_area") or plan.get("sub_ba"),
        lob=ch_first,
        site=(plan.get("site") or plan.get("location") or plan.get("country") or "").strip(),
    )
    try:
        return int(float(settings.get("interval_minutes", 30) or 30))
    except Exception:
        return 30


def _build_fw_cols(plan: dict, grain: str, interval_date: Optional[str]) -> list[dict]:
    weeks = _week_span(plan.get("start_week"), plan.get("end_week"))
    if grain == "month":
        cols, _ = _month_cols(weeks)
        return cols
    if grain == "day":
        cols, _ = day_cols_for_weeks(weeks)
        return cols
    if grain == "interval":
        day = _parse_date(interval_date)
        ivl_min = _resolve_ivl_min(plan)
        cols, _ = interval_cols_for_day(day, ivl_min=ivl_min)
        return cols
    cols, _ = _week_cols(weeks)
    return cols


def extract_upper_rows(component: Any) -> list[dict]:
    if component is None:
        return []
    if hasattr(component, "to_plotly_json"):
        try:
            payload = component.to_plotly_json() or {}
            props = payload.get("props") or {}
            data = props.get("data")
            if isinstance(data, list):
                return data
        except Exception:
            pass
    if isinstance(component, dict):
        data = component.get("data")
        if isinstance(data, list):
            return data
        props = component.get("props")
        if isinstance(props, dict) and isinstance(props.get("data"), list):
            return props.get("data")
    data_attr = getattr(component, "data", None)
    return data_attr if isinstance(data_attr, list) else []


def _normalize_rows(rows: Any) -> list[dict]:
    return rows if isinstance(rows, list) else []


def _table_suffix(grain: str, interval_date: Optional[str]) -> str:
    if grain == "week":
        return ""
    if grain == "interval":
        stamp = _parse_date(interval_date).isoformat()
        return f"_{grain}_{stamp}"
    return f"_{grain}"


def compute_plan_detail_tables(
    plan_id: int,
    *,
    grain: Optional[str] = None,
    whatif: Optional[dict] = None,
    interval_date: Optional[str] = None,
    version_token: Any = None,
    persist: bool = False,
) -> tuple[Optional[dict], str, dict]:
    plan = load_plan(int(plan_id))
    if not plan:
        return None, "missing", {"reason": "plan not found"}

    g = _normalize_grain(grain)
    fw_cols = _build_fw_cols(plan, g, interval_date)
    plan_type = plan.get("plan_type") or "Volume Based"
    ivl_min = _resolve_ivl_min(plan)
    dep_tokens = dep_snapshot(
        int(plan_id),
        ["timeseries", "settings", "plan_meta", "roster", "newhire", "shrinkage", "attrition", "plan_tables", "whatif"],
    )
    dep_all = dep_snapshot_all(int(plan_id))
    dep_key = (int(plan_id), _normalize_grain(grain), str(interval_date or ""))
    last_dep = _LAST_DEP_SNAPSHOT.get(dep_key)
    changed_deps = set()
    if isinstance(last_dep, dict):
        for k, v in dep_all.items():
            if last_dep.get(k) != v:
                changed_deps.add(k)
    light_only = bool(changed_deps) and changed_deps.issubset({"plan_tables:notes", "plan_tables:bulk_files"})
    if light_only:
        cached = load_plan_detail_tables(int(plan_id), grain=_normalize_grain(grain), interval_date=interval_date)
        if cached:
            meta = {"cached": True, "light_skip": True, "dep_changed": sorted(changed_deps), "dep_tokens": dep_tokens}
            _LAST_DEP_SNAPSHOT[dep_key] = dep_all
            return cached, "ready", meta
    if version_token is not None:
        tick = version_token
    else:
        tick = hash((dep_tokens.get("timeseries", 0), dep_tokens.get("settings", 0), dep_tokens.get("plan_meta", 0), CALC_VERSION))

    def _build_payload(results: Any) -> Optional[dict]:
        if not results:
            return None
        upper, *rows = results
        normalized = [_normalize_rows(r) for r in rows]
        tables = dict(zip(_TABLE_KEYS, normalized))
        return {
            "upper": extract_upper_rows(upper),
            "tables": tables,
            "grain": g,
        }

    def _builder():
        if g == "month":
            result = _fill_tables_fixed_monthly(plan_type, plan_id, fw_cols, tick, whatif)
        elif g == "day":
            result = _fill_tables_fixed_daily(plan_type, plan_id, fw_cols, tick, whatif)
        elif g == "interval":
            result = _fill_tables_fixed_interval(
                plan_type, plan_id, fw_cols, tick, whatif, ivl_min=ivl_min, sel_date=interval_date 
            )
        else:
            result = _fill_tables_fixed(plan_type, plan_id, fw_cols, tick, whatif, grain="week")

        if persist and not whatif and result:
            payload = _build_payload(result)
            if payload:
                persist_plan_detail_tables(int(plan_id), payload, grain=g, interval_date=interval_date)
        return result

    results, status, meta = ensure_plan_calc(
        plan_id,
        grain=g,
        fw_cols=fw_cols,
        whatif=whatif,
        interval_date=interval_date,
        plan_type=plan_type,
        version_token=version_token,
        builder=_builder,
        extra={"plan_type": plan_type, "dep_tokens": dep_tokens},
    )
    if isinstance(meta, dict):
        meta["dep_tokens"] = dep_tokens
        if changed_deps:
            meta["dep_changed"] = sorted(changed_deps)
    if status != "ready" or not results:
        return None, status, meta

    payload = _build_payload(results)
    if not payload:
        return None, "failed", meta
    _LAST_DEP_SNAPSHOT[dep_key] = dep_all
    return payload, status, meta


def prefetch_plan_detail_grains(
    plan_id: int,
    *,
    grains: list[str],
    whatif: Optional[dict] = None,
    version_token: Any = None,
) -> None:
    if not grains:
        return
    if whatif:
        return

    def _run(target_grain: str):
        try:
            compute_plan_detail_tables(
                int(plan_id),
                grain=target_grain,
                whatif=None,
                interval_date=None,
                version_token=version_token,
            )
        except Exception:
            return

    for grain in grains:
        g = _normalize_grain(grain)
        if g == "interval":
            continue
        _PREFETCH_EXECUTOR.submit(_run, g)


def persist_plan_detail_tables(
    plan_id: int,
    payload: dict,
    *,
    grain: Optional[str] = None,
    interval_date: Optional[str] = None,
) -> None:
    if not payload or "tables" not in payload:
        return
    g = _normalize_grain(grain)
    suffix = _table_suffix(g, interval_date)
    tables = payload.get("tables") or {}
    for key in _TABLE_KEYS:
        rows = tables.get(key, [])
        save_plan_table(int(plan_id), f"{key}{suffix}", rows)
    upper_rows = payload.get("upper")
    if isinstance(upper_rows, list):
        save_plan_table(int(plan_id), f"upper{suffix}", upper_rows)


def load_plan_detail_tables(
    plan_id: int,
    *,
    grain: Optional[str] = None,
    interval_date: Optional[str] = None,
) -> dict:
    g = _normalize_grain(grain)
    suffix = _table_suffix(g, interval_date)
    tables: dict[str, list[dict]] = {}
    for key in _TABLE_KEYS:
        tables[key] = load_plan_table(int(plan_id), f"{key}{suffix}")
    upper_rows = load_plan_table(int(plan_id), f"upper{suffix}")
    return {"tables": tables, "upper": upper_rows, "grain": g}
