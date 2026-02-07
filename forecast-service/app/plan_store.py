from __future__ import annotations

from typing import Any, Optional

from app.pipeline.planning_store import (
    delete_plan as _delete_plan,
    list_business_areas as _list_business_areas,
    list_plans as _list_plans,
    load_plan as _load_plan,
    upsert_plan as _upsert_plan,
)
from app.pipeline.postgres import db_conn, ensure_planning_schema, has_dsn


def _map_payload(payload: dict) -> dict:
    data = dict(payload or {})
    if "vertical" in data and "business_area" not in data:
        data["business_area"] = data.get("vertical")
    if "sub_ba" in data and "sub_business_area" not in data:
        data["sub_business_area"] = data.get("sub_ba")
    return data


def create_plan(payload: dict) -> int:
    result = _upsert_plan(_map_payload(payload))
    status = result.get("status")
    if status == "duplicate":
        raise ValueError("Duplicate: that plan already exists for this scope.")
    return int(result.get("id") or 0)


def get_plan(plan_id: int) -> Optional[dict]:
    plan = _load_plan(int(plan_id)) if plan_id else {}
    if not plan:
        return None
    merged = dict(plan)
    if "business_area" in plan:
        merged.setdefault("vertical", plan.get("business_area"))
    if "sub_business_area" in plan:
        merged.setdefault("sub_ba", plan.get("sub_business_area"))
    return merged


def list_plans(
    vertical: Optional[str] = None,
    status_filter: Optional[str] = None,
    include_deleted: bool = False,
) -> list[dict]:
    rows = _list_plans(
        business_area=vertical,
        status_filter=status_filter,
        include_deleted=include_deleted,
        limit=500,
    )
    out = []
    for row in rows:
        merged = dict(row)
        if "business_area" in row:
            merged.setdefault("vertical", row.get("business_area"))
        if "sub_business_area" in row:
            merged.setdefault("sub_ba", row.get("sub_business_area"))
        out.append(merged)
    return out


def list_business_areas(status_filter: Optional[str] = "current") -> list[str]:
    return _list_business_areas(status_filter)


def delete_plan(plan_id: int, hard_if_missing: bool = True) -> None:
    _delete_plan(int(plan_id), hard_if_missing=hard_if_missing)


def extend_plan_weeks(plan_id: int, add_weeks: int) -> None:
    try:
        pid = int(plan_id)
        delta = int(add_weeks)
    except Exception as exc:
        raise ValueError("Invalid plan id or weeks") from exc
    if delta <= 0:
        return
    plan = None
    try:
        plan = _load_plan(pid)
    except Exception:
        plan = None
    if plan and str(plan.get("status") or "").lower() == "history":
        raise ValueError("Plan is locked (history).")
    if has_dsn():
        ensure_planning_schema()
        with db_conn() as conn:
            conn.execute(
                """
                UPDATE planning_plans
                SET end_week = COALESCE(end_week, CURRENT_DATE) + (%s * INTERVAL '7 days'),
                    updated_at = NOW()
                WHERE id = %s
                """,
                (delta, pid),
            )
    try:
        from app.pipeline.plan_detail._common import extend_plan_weeks as _extend_meta

        _extend_meta(pid, delta)
    except Exception:
        pass
