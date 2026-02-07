from __future__ import annotations

import hashlib
from typing import Any, Optional

import pandas as pd
from psycopg.types.json import Json

from app.pipeline.postgres import db_conn, has_dsn, ensure_forecast_schema
from app.pipeline.ops_store import normalize_scope_key


def _safe_json(value: Any) -> Json:
    if isinstance(value, Json):
        return value
    return Json(value)


def _dataset_hash(rows: list[dict]) -> str:
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    df = df.copy()
    df = df.fillna("")
    df = df.reindex(sorted(df.columns), axis=1)
    hashed = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.sha1(hashed.tobytes()).hexdigest()


def create_forecast_run(
    run_type: str,
    *,
    scope_key: Optional[str] = None,
    meta: Optional[dict] = None,
    created_by: Optional[str] = None,
) -> Optional[int]:
    if not has_dsn():
        return None
    ensure_forecast_schema()
    scope_norm = normalize_scope_key(scope_key or "global")
    meta_json = _safe_json(meta or {})
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO forecast_runs (run_type, scope_key, scope_key_norm, meta, created_by)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (str(run_type or "").strip(), scope_key or "global", scope_norm, meta_json, created_by),
        )
        row = cur.fetchone()
    return int(row[0]) if row else None


def save_forecast_step(run_id: int, step: str, payload: Any) -> None:
    if not has_dsn() or not run_id:
        return
    ensure_forecast_schema()
    rows = payload if isinstance(payload, list) else None
    row_count = len(rows) if rows else None
    dataset_hash = _dataset_hash(rows) if rows else None
    payload_json = _safe_json(payload)
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO forecast_run_steps (run_id, step, payload, row_count, dataset_hash)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (int(run_id), str(step or "result"), payload_json, row_count, dataset_hash),
        )


def save_forecast_run(
    run_type: str,
    payload: dict,
    *,
    scope_key: Optional[str] = None,
    meta: Optional[dict] = None,
    created_by: Optional[str] = None,
) -> Optional[int]:
    run_id = create_forecast_run(run_type, scope_key=scope_key, meta=meta, created_by=created_by)
    if not run_id:
        return None
    # Store the entire payload as a single step for audit/history.
    save_forecast_step(run_id, "result", payload)
    return run_id
