from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from app.pipeline.planning_store import load_plan_table, save_plan_table
from app.pipeline.utils import df_to_records

_PLAN_RE = re.compile(r"^plan_(\d+)_([a-zA-Z0-9_]+)$")


def _parse_key(key: str) -> tuple[Optional[int], str]:
    if not key:
        return None, ""
    match = _PLAN_RE.match(str(key))
    if not match:
        return None, str(key)
    return int(match.group(1)), match.group(2)


def load_df(key: str) -> pd.DataFrame:
    pid, name = _parse_key(key)
    if pid is None or not name:
        return pd.DataFrame()
    rows = load_plan_table(pid, name)
    try:
        df = pd.DataFrame(rows or [])
    except Exception:
        df = pd.DataFrame()
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
