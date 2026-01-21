from __future__ import annotations

import datetime as dt
from typing import Any, Optional

import pandas as pd

from app.pipeline.plan_detail._calc import _fill_tables_fixed
from app.pipeline.plan_detail._common import _month_cols, _week_cols, _week_span, resolve_settings
from app.pipeline.plan_detail._fill_tables_fixed_daily import _fill_tables_fixed_daily
from app.pipeline.plan_detail._fill_tables_fixed_interval import _fill_tables_fixed_interval
from app.pipeline.plan_detail._fill_tables_fixed_monthly import _fill_tables_fixed_monthly
from app.pipeline.plan_detail._grain_cols import day_cols_for_weeks, interval_cols_for_day
from app.pipeline.plan_detail.calc_engine import ensure_plan_calc
from app.pipeline.planning_store import load_plan, save_plan_table


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
) -> tuple[Optional[dict], str, dict]:
    plan = load_plan(int(plan_id))
    if not plan:
        return None, "missing", {"reason": "plan not found"}

    g = _normalize_grain(grain)
    fw_cols = _build_fw_cols(plan, g, interval_date)
    plan_type = plan.get("plan_type") or "Volume Based"
    ivl_min = _resolve_ivl_min(plan)
    tick = version_token

    def _builder():
        if g == "month":
            return _fill_tables_fixed_monthly(plan_type, plan_id, fw_cols, tick, whatif)
        if g == "day":
            return _fill_tables_fixed_daily(plan_type, plan_id, fw_cols, tick, whatif)
        if g == "interval":
            return _fill_tables_fixed_interval(
                plan_type, plan_id, fw_cols, tick, whatif, ivl_min=ivl_min, sel_date=interval_date
            )
        return _fill_tables_fixed(plan_type, plan_id, fw_cols, tick, whatif, grain="week")

    results, status, meta = ensure_plan_calc(
        plan_id,
        grain=g,
        fw_cols=fw_cols,
        whatif=whatif,
        interval_date=interval_date,
        plan_type=plan_type,
        version_token=version_token,
        builder=_builder,
        extra={"plan_type": plan_type},
    )
    if status != "ready" or not results:
        return None, status, meta

    upper, *rows = results
    normalized = [_normalize_rows(r) for r in rows]
    tables = dict(zip(_TABLE_KEYS, normalized))
    payload = {
        "upper": extract_upper_rows(upper),
        "tables": tables,
        "grain": g,
    }
    return payload, status, meta


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
