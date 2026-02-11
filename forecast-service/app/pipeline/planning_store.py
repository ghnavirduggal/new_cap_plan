from __future__ import annotations

import datetime as dt
import getpass
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Any
import math

import pandas as pd
from psycopg2.extras import Json

from app.pipeline.planning_calc_engine import mark_plan_dirty
from app.pipeline.postgres import db_conn, ensure_planning_schema, has_dsn
from app.pipeline.utils import sanitize_for_json


def _norm_channel_csv(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip().lower() for v in value if str(v).strip()]
    else:
        parts = [p.strip().lower() for p in str(value).split(",") if p.strip()]
    parts = sorted(set(parts))
    return ", ".join(parts)


def _plan_key(
    business_area: Optional[str],
    sub_business_area: Optional[str],
    channel: Optional[str],
    site: Optional[str],
    location: Optional[str],
) -> str:
    channel_norm = _norm_channel_csv(channel)
    parts = [
        str(business_area or "").strip(),
        str(sub_business_area or "").strip(),
        channel_norm.strip(),
        str(site or location or "").strip(),
    ]
    key = "|".join([p for p in parts if p])
    return key.lower() if key else "global"


def _parse_date(value: Optional[object]) -> Optional[dt.date]:
    if value is None or value == "":
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.normalize().date()

def _user() -> str:
    return os.environ.get("HOSTNAME") or os.environ.get("USERNAME") or getpass.getuser() or "system"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _safe_component(value: Optional[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "")).strip("_")
    return cleaned or "all"


def _plan_table_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_dir = str(os.getenv("CAP_EXPORTS_DIR") or "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(_repo_root() / "exports")
    candidates.append(Path(tempfile.gettempdir()) / "cap_exports")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _plan_table_dir() -> Path:
    for base in _plan_table_candidates():
        outdir = base / "plan_tables"
        try:
            outdir.mkdir(exist_ok=True, parents=True)
            probe = outdir / ".perm_probe"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            return outdir
        except Exception:
            continue
    outdir = _plan_table_candidates()[-1] / "plan_tables"
    outdir.mkdir(exist_ok=True, parents=True)
    return outdir


def _plan_table_path(plan_id: int, table_name: str) -> Path:
    return _plan_table_dir() / f"plan_{int(plan_id)}_{_safe_component(table_name)}.json"


def _load_plan_table_local(plan_id: int, table_name: str) -> list[dict]:
    path = _plan_table_path(plan_id, table_name)
    if not path.exists():
        # Case-insensitive fallback for local files.
        try:
            target = path.name.lower()
            for cand in _plan_table_dir().glob(f"plan_{int(plan_id)}_*.json"):
                if cand.name.lower() == target:
                    path = cand
                    break
        except Exception:
            pass
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _save_plan_table_local(plan_id: int, table_name: str, rows: list[dict]) -> dict:
    path = _plan_table_path(plan_id, table_name)
    payload = _sanitize_json(rows or [])
    try:
        path.write_text(json.dumps(payload))
    except Exception:
        return {"status": "invalid"}
    return {"status": "saved", "rows": len(rows or []), "path": str(path), "storage": "local"}


def _to_json(value: object) -> Optional[Json]:
    if value in (None, ""):
        return None
    if isinstance(value, Json):
        return value
    if isinstance(value, (dict, list, tuple)):
        return Json(value)
    if isinstance(value, str):
        try:
            return Json(json.loads(value))
        except Exception:
            return Json(value)
    return Json(value)

def upsert_plan(payload: dict) -> dict:
    if not has_dsn():
        return {"status": "missing_dsn"}
    ensure_planning_schema()
    data = dict(payload or {})
    plan_id = data.get("id")
    plan_key = data.get("plan_key") or _plan_key(
        data.get("business_area"),
        data.get("sub_business_area"),
        data.get("channel"),
        data.get("site"),
        data.get("location"),
    )
    data["plan_key"] = plan_key
    data["start_week"] = _parse_date(data.get("start_week"))
    data["end_week"] = _parse_date(data.get("end_week"))
    data["channel"] = _norm_channel_csv(data.get("channel"))

    is_current = bool(data.get("is_current") or str(data.get("status") or "").lower() == "current")
    status = data.get("status") or ("current" if is_current else "draft")
    data["is_current"] = is_current
    data["status"] = status

    user = _user()
    data.setdefault("created_by", user)
    data.setdefault("updated_by", user)
    data.setdefault("owner", user)

    tags_json = _to_json(data.get("tags"))
    hier_json = _to_json(data.get("hierarchy_json"))

    if plan_id:
        try:
            existing = load_plan(int(plan_id))
            if existing and str(existing.get("status") or "").lower() == "history":
                return {"status": "locked", "id": int(plan_id)}
        except Exception:
            pass
        if is_current:
            _demote_current_plans(
                data.get("business_area"),
                data.get("sub_business_area"),
                data.get("channel"),
                data.get("location"),
                data.get("site"),
                exclude_id=int(plan_id),
            )
        with db_conn() as conn:
            conn.execute(
                """
                UPDATE planning_plans
                SET plan_key = %s,
                    org = %s,
                    business_entity = %s,
                    business_area = %s,
                    sub_business_area = %s,
                    channel = %s,
                    location = %s,
                    site = %s,
                    plan_name = %s,
                    plan_type = %s,
                    start_week = %s,
                    end_week = %s,
                    ft_weekly_hours = %s,
                    pt_weekly_hours = %s,
                    tags = %s,
                    is_current = %s,
                    status = %s,
                    hierarchy_json = %s,
                    owner = %s,
                    updated_by = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (
                    plan_key,
                    data.get("org"),
                    data.get("business_entity"),
                    data.get("business_area"),
                    data.get("sub_business_area"),
                    data.get("channel"),
                    data.get("location"),
                    data.get("site"),
                    data.get("plan_name"),
                    data.get("plan_type"),
                    data.get("start_week"),
                    data.get("end_week"),
                    data.get("ft_weekly_hours"),
                    data.get("pt_weekly_hours"),
                    tags_json,
                    is_current,
                    status,
                    hier_json,
                    data.get("owner"),
                    data.get("updated_by") or user,
                    int(plan_id),
                ),
            )
        mark_plan_dirty(plan_key)
        return {"status": "updated", "id": int(plan_id), "plan_key": plan_key}

    # Duplicate guard: BA + SBA + Plan Name + Location + Site + channel set.
    if _has_duplicate(
        data.get("business_area"),
        data.get("sub_business_area"),
        data.get("plan_name"),
        data.get("location"),
        data.get("site"),
        data.get("channel"),
    ):
        return {"status": "duplicate"}

    if is_current:
        _demote_current_plans(
            data.get("business_area"),
            data.get("sub_business_area"),
            data.get("channel"),
            data.get("location"),
            data.get("site"),
            exclude_id=None,
        )
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO planning_plans (
                plan_key,
                org,
                business_entity,
                business_area,
                sub_business_area,
                channel,
                location,
                site,
                plan_name,
                plan_type,
                start_week,
                end_week,
                ft_weekly_hours,
                pt_weekly_hours,
                tags,
                is_current,
                status,
                hierarchy_json,
                owner,
                created_by,
                updated_by
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                plan_key,
                data.get("org"),
                data.get("business_entity"),
                data.get("business_area"),
                data.get("sub_business_area"),
                data.get("channel"),
                data.get("location"),
                data.get("site"),
                data.get("plan_name"),
                data.get("plan_type"),
                data.get("start_week"),
                data.get("end_week"),
                data.get("ft_weekly_hours"),
                data.get("pt_weekly_hours"),
                tags_json,
                is_current,
                status,
                hier_json,
                data.get("owner"),
                data.get("created_by") or user,
                data.get("updated_by") or user,
            ),
        )
        row = cur.fetchone()
    plan_id = int(row[0]) if row else None
    mark_plan_dirty(plan_key)
    return {"status": "created", "id": plan_id, "plan_key": plan_key}


def _demote_current_plans(
    business_area: Optional[str],
    sub_business_area: Optional[str],
    channel: Optional[str],
    location: Optional[str],
    site: Optional[str],
    exclude_id: Optional[int],
) -> None:
    if not has_dsn():
        return
    ensure_planning_schema()
    chan_norm = _norm_channel_csv(channel)
    filters = [
        "LOWER(COALESCE(TRIM(business_area),'')) = LOWER(COALESCE(TRIM(%s),''))",
        "LOWER(COALESCE(TRIM(sub_business_area),'')) = LOWER(COALESCE(TRIM(%s),''))",
        "LOWER(COALESCE(TRIM(location),'')) = LOWER(COALESCE(TRIM(%s),''))",
        "LOWER(COALESCE(TRIM(site),'')) = LOWER(COALESCE(TRIM(%s),''))",
        "COALESCE(is_deleted,false) = false",
        "(COALESCE(is_current,false) = true OR status = 'current')",
    ]
    params = [business_area, sub_business_area, location, site]
    if exclude_id is not None:
        filters.append("id <> %s")
        params.append(int(exclude_id))

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, channel
            FROM planning_plans
            WHERE {' AND '.join(filters)}
            """,
            tuple(params),
        )
        rows = cur.fetchall()
        to_demote = [row[0] for row in rows if _norm_channel_csv(row[1]) == chan_norm]
        if not to_demote:
            return
        cur.execute(
            """
            UPDATE planning_plans
            SET is_current = FALSE, status = 'history', updated_at = NOW()
            WHERE id = ANY(%s)
            """,
            (to_demote,),
        )


def _has_duplicate(
    business_area: Optional[str],
    sub_business_area: Optional[str],
    plan_name: Optional[str],
    location: Optional[str],
    site: Optional[str],
    channel: Optional[str],
) -> bool:
    if not has_dsn():
        return False
    ensure_planning_schema()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, channel
            FROM planning_plans
            WHERE LOWER(COALESCE(TRIM(business_area),'')) = LOWER(COALESCE(TRIM(%s),'')) 
              AND LOWER(COALESCE(TRIM(sub_business_area),'')) = LOWER(COALESCE(TRIM(%s),'')) 
              AND LOWER(COALESCE(TRIM(plan_name),'')) = LOWER(COALESCE(TRIM(%s),'')) 
              AND LOWER(COALESCE(TRIM(location),'')) = LOWER(COALESCE(TRIM(%s),'')) 
              AND LOWER(COALESCE(TRIM(site),'')) = LOWER(COALESCE(TRIM(%s),'')) 
              AND COALESCE(is_deleted,false) = false
            """,
            (
                business_area or "",
                sub_business_area or "",
                plan_name or "",
                location or "",
                site or "",
            ),
        )
        rows = cur.fetchall()
    chan_norm = _norm_channel_csv(channel)
    return any(_norm_channel_csv(row[1]) == chan_norm for row in rows)


def load_plan(plan_id: int) -> dict:
    if not has_dsn():
        return {}
    ensure_planning_schema()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, plan_key, org, business_entity, business_area, sub_business_area, channel,
                   location, site, plan_name, plan_type, start_week, end_week,
                   ft_weekly_hours, pt_weekly_hours, tags, is_current, status,
                   hierarchy_json, owner, created_by, updated_by, is_deleted, deleted_at,
                   created_at, updated_at
            FROM planning_plans
            WHERE id = %s
            """,
            (int(plan_id),),
        )
        row = cur.fetchone()
        if not row:
            return {}
        cols = [desc[0] for desc in cur.description]
    return dict(zip(cols, row))


def _is_plan_locked(plan_id: int) -> bool:
    if not plan_id:
        return False
    try:
        plan = load_plan(int(plan_id))
    except Exception:
        return False
    if not plan:
        return False
    if bool(plan.get("is_deleted")):
        return True
    return str(plan.get("status") or "").lower() == "history"


def list_plans(
    business_area: Optional[str] = None,
    sub_business_area: Optional[str] = None,
    channel: Optional[str] = None,
    location: Optional[str] = None,
    site: Optional[str] = None,
    status_filter: Optional[str] = None,
    include_deleted: bool = False,
    limit: int = 50,
) -> list[dict]:
    if not has_dsn():
        return []
    ensure_planning_schema()
    filters = []
    params = []
    if business_area:
        filters.append("business_area = %s")
        params.append(business_area)
    if sub_business_area:
        filters.append("sub_business_area = %s")
        params.append(sub_business_area)
    if channel:
        filters.append("channel = %s")
        params.append(channel)
    if location:
        filters.append("location = %s")
        params.append(location)
    if site:
        filters.append("site = %s")
        params.append(site)
    if status_filter == "current":
        filters.append("(COALESCE(is_current,false) = true OR status = 'current')")
    elif status_filter == "history":
        filters.append("(COALESCE(is_current,false) = false OR COALESCE(status,'') IN ('history','draft'))")
    elif status_filter:
        filters.append("status = %s")
        params.append(status_filter)
    if not include_deleted:
        filters.append("COALESCE(is_deleted,false) = false")

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, plan_key, org, business_entity, business_area, sub_business_area, channel,
                   location, site, plan_name, plan_type, start_week, end_week,
                   ft_weekly_hours, pt_weekly_hours, tags, is_current, status,
                   hierarchy_json, owner, created_by, updated_by, is_deleted, deleted_at,
                   created_at, updated_at
            FROM planning_plans
            {where_sql}
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (*params, int(limit)),
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return [dict(zip(cols, row)) for row in rows]


def list_business_areas(status_filter: Optional[str] = "current") -> list[str]:
    if not has_dsn():
        return []
    ensure_planning_schema()
    filters = ["business_area IS NOT NULL", "COALESCE(TRIM(business_area),'') <> ''", "COALESCE(is_deleted,false) = false"]
    params: list[object] = []
    if status_filter == "current":
        filters.append("(COALESCE(is_current,false) = true OR status = 'current')")
    elif status_filter == "history":
        filters.append("(COALESCE(is_current,false) = false OR COALESCE(status,'') IN ('history','draft'))")
    elif status_filter:
        filters.append("status = %s")
        params.append(status_filter)
    where_sql = f"WHERE {' AND '.join(filters)}"
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT DISTINCT business_area, LOWER(business_area) AS business_area_sort
            FROM planning_plans
            {where_sql}
            ORDER BY business_area_sort
            """,
            tuple(params),
        )
        rows = cur.fetchall()
    return [row[0] for row in rows if row and row[0]]


def delete_plan(plan_id: int, hard_if_missing: bool = True) -> dict:
    if not has_dsn():
        return {"status": "missing_dsn"}
    ensure_planning_schema()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM planning_plans WHERE id = %s", (int(plan_id),))
        row = cur.fetchone()
        if not row:
            return {"status": "missing" if not hard_if_missing else "missing"}
        cur.execute(
            """
            UPDATE planning_plans
            SET is_deleted = TRUE, deleted_at = NOW(), updated_at = NOW()
            WHERE id = %s
            """,
            (int(plan_id),),
        )
    return {"status": "deleted", "id": int(plan_id)}


def save_plan_table(plan_id: int, table_name: str, rows: list[dict]) -> dict:
    if not plan_id or not table_name:
        return {"status": "invalid"}
    if not has_dsn():
        return _save_plan_table_local(int(plan_id), str(table_name), rows)
    ensure_planning_schema()
    if _is_plan_locked(int(plan_id)):
        return {"status": "locked", "rows": 0}
    payload = Json(_sanitize_json(rows or []))
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO planning_plan_tables (plan_id, table_name, payload)
            VALUES (%s, %s, %s)
            ON CONFLICT (plan_id, table_name)
            DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
            """,
            (int(plan_id), table_name, payload),
        )
        cur.execute(
            """
            INSERT INTO planning_plan_table_history (plan_id, table_name, payload, created_by)
            VALUES (%s, %s, %s, %s)
            """,
            (int(plan_id), table_name, payload, _user()),
        )
        cur.execute("SELECT plan_key FROM planning_plans WHERE id = %s", (int(plan_id),))
        row = cur.fetchone()
    if row:
        mark_plan_dirty(row[0])
    try:
        _record_plan_activity(int(plan_id), table_name)
    except Exception:
        pass
    try:
        from app.pipeline.plan_detail.calc_engine import mark_plan_dirty_deps
        base = str(table_name or "").split("_")[0].lower()
        if base == "emp":
            dep = "roster"
        elif base == "nh":
            dep = "newhire"
        elif base == "shr":
            dep = "shrinkage"
        elif base == "attr":
            dep = "attrition"
        else:
            dep = f"plan_tables:{base or 'unknown'}"
        mark_plan_dirty_deps(int(plan_id), dep)
    except Exception:
        pass
    return {"status": "saved", "rows": len(rows or [])}


def _record_plan_activity(plan_id: int, table_name: str) -> None:
    if not has_dsn():
        return
    base = str(table_name or "").split("_")[0].lower()
    action_map = {
        "fw": "updated forecast & workload",
        "hc": "updated headcount",
        "shr": "updated shrinkage",
        "attr": "updated attrition",
        "train": "updated training lifecycle",
        "ratio": "updated ratios",
        "seat": "updated seat utilization",
        "bva": "updated budget vs actual",
        "nh": "updated new hire plan",
        "emp": "updated employee roster",
        "bulk_files": "uploaded roster file",
        "notes": "added note",
    }
    action = action_map.get(base, "updated plan data")
    record_activity(
        plan_id=plan_id,
        action=action,
        actor=_user(),
        entity_type="plan_table",
        entity_id=table_name,
    )


def record_activity(
    *,
    plan_id: Optional[int] = None,
    action: str,
    actor: Optional[str] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    payload: Any = None,
) -> None:
    if not has_dsn():
        return
    ensure_planning_schema()
    actor_name = actor or _user()
    payload_json = _to_json(payload)
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO planning_activity (plan_id, actor, action, entity_type, entity_id, payload)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                int(plan_id) if plan_id else None,
                actor_name,
                str(action or ""),
                str(entity_type or "") if entity_type else None,
                str(entity_id or "") if entity_id else None,
                payload_json,
            ),
        )


def list_activity(limit: int = 20, plan_id: Optional[int] = None) -> list[dict]:
    if not has_dsn():
        return []
    ensure_planning_schema()
    limit = max(1, min(int(limit or 20), 200))
    where_sql = ""
    params: list[Any] = []
    if plan_id:
        where_sql = "WHERE a.plan_id = %s"
        params.append(int(plan_id))
    params.append(limit)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT a.id, a.plan_id, a.actor, a.action, a.entity_type, a.entity_id, a.payload, a.created_at,
                   p.plan_name, p.business_area, p.sub_business_area, p.channel, p.site, p.start_week, p.end_week
              FROM planning_activity a
              LEFT JOIN planning_plans p ON p.id = a.plan_id
              {where_sql}
             ORDER BY a.created_at DESC
             LIMIT %s
            """,
            tuple(params),
        )
        rows = cur.fetchall()
    out: list[dict] = []
    for row in rows:
        (
            act_id,
            pid,
            actor,
            action,
            entity_type,
            entity_id,
            payload,
            created_at,
            plan_name,
            business_area,
            sub_business_area,
            channel,
            site,
            start_week,
            end_week,
        ) = row
        out.append(
            {
                "id": act_id,
                "plan_id": pid,
                "actor": actor,
                "action": action,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "payload": payload,
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                "plan_name": plan_name,
                "business_area": business_area,
                "sub_business_area": sub_business_area,
                "channel": channel,
                "site": site,
                "start_week": str(start_week or ""),
                "end_week": str(end_week or ""),
            }
        )
    return out


def _sanitize_json(value: Any) -> Any:
    return sanitize_for_json(value)


def load_plan_table(plan_id: int, table_name: str) -> list[dict]:
    if not plan_id or not table_name:
        return []
    if not has_dsn():
        return _load_plan_table_local(int(plan_id), str(table_name))
    ensure_planning_schema()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT payload
            FROM planning_plan_tables
            WHERE plan_id = %s AND table_name = %s
            """,
            (int(plan_id), table_name),
        )
        row = cur.fetchone()
    if not row:
        return []
    return row[0] or []
