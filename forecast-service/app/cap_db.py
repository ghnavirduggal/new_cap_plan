from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

import pandas as pd

from app.pipeline.planning_store import load_plan_table, save_plan_table
from app.pipeline.shrinkage_store import (
    load_attrition_raw,
    load_attrition_weekly,
    load_shrinkage_raw,
    load_shrinkage_weekly,
)
from app.pipeline.utils import df_to_records

_PLAN_RE = re.compile(r"^plan_(\d+)_([a-zA-Z0-9_]+)$")
_LOAD_DF_CACHE: ContextVar[dict[str, pd.DataFrame] | None] = ContextVar("load_df_cache", default=None)


@contextmanager
def load_df_cache():
    """Cache DataFrame loads for one calculation request."""
    token = _LOAD_DF_CACHE.set({})
    try:
        yield
    finally:
        _LOAD_DF_CACHE.reset(token)


def _parse_key(key: str) -> tuple[Optional[int], str]:
    if not key:
        return None, ""
    match = _PLAN_RE.match(str(key))
    if not match:
        return None, str(key)
    return int(match.group(1)), match.group(2)


def load_df(key: str) -> pd.DataFrame:
    cache = _LOAD_DF_CACHE.get()
    if cache is not None and key in cache:
        return cache[key].copy()

    if key in {"shrinkage_raw_backoffice", "shrinkage_raw_bo"}:
        df = load_shrinkage_raw("backoffice")
    elif key in {"shrinkage_raw_voice"}:
        df = load_shrinkage_raw("voice")
    elif key in {"shrinkage_raw_chat"}:
        df = load_shrinkage_raw("chat")
    elif key in {"shrinkage_raw_outbound", "shrinkage_raw_ob"}:
        df = load_shrinkage_raw("outbound")
    elif key in {"shrinkage_weekly", "shrinkage"}:
        df = load_shrinkage_weekly()
    elif key in {"attrition_weekly", "attrition"}:
        df = load_attrition_weekly()
    elif key in {"attrition_raw"}:
        df = load_attrition_raw()
    else:
        pid, name = _parse_key(key)
        if pid is None or not name:
            df = pd.DataFrame()
        else:
            rows = load_plan_table(pid, name)
            try:
                df = pd.DataFrame(rows or [])
            except Exception:
                df = pd.DataFrame()
    if cache is not None:
        cache[key] = df.copy()
    return df


def save_df(key: str, df: pd.DataFrame) -> None:
    pid, name = _parse_key(key)
    if pid is None or not name:
        return
    rows = df_to_records(df if isinstance(df, pd.DataFrame) else pd.DataFrame())
    save_plan_table(pid, name, rows)


def delete_datasets_by_prefix(prefix: str) -> None:
    # Optional: no-op for now. Plan detail tables are managed via planning_plan_tables.
    return None
