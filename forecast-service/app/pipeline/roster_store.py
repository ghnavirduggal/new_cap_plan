from __future__ import annotations

import datetime as dt
import re
from typing import Optional

import pandas as pd
from psycopg.types.json import Json

from app.pipeline.headcount import load_headcount
from app.pipeline.postgres import db_conn, ensure_roster_schema, has_dsn


def _json_safe_value(value):
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        return pd.to_datetime(value, errors="coerce").date().isoformat()
    return value


def _json_safe_row(row: dict) -> dict:
    return {key: _json_safe_value(val) for key, val in (row or {}).items()}


def build_roster_template_wide(
    start_date: dt.date | str,
    end_date: dt.date | str,
    include_sample: bool = False,
) -> pd.DataFrame:
    base_cols = [
        "BRID",
        "Name",
        "Team Manager",
        "Business Area",
        "Sub Business Area",
        "LOB",
        "Site",
        "Location",
        "Country",
    ]
    if not isinstance(start_date, dt.date):
        start_date = pd.to_datetime(start_date, errors="coerce").date()
    if not isinstance(end_date, dt.date):
        end_date = pd.to_datetime(end_date, errors="coerce").date()
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    date_cols = [
        (start_date + dt.timedelta(days=i)).isoformat()
        for i in range((end_date - start_date).days + 1)
    ]
    cols = base_cols + date_cols
    df = pd.DataFrame(columns=cols)

    if include_sample and date_cols:
        r1 = {c: "" for c in cols}
        r1.update(
            {
                "BRID": "IN0001",
                "Name": "Asha Rao",
                "Team Manager": "Priyanka Menon",
                "Business Area": "Retail",
                "Sub Business Area": "Cards",
                "LOB": "Back Office",
                "Site": "Chennai",
                "Location": "IN-Chennai",
                "Country": "India",
                date_cols[0]: "09:00-17:30",
            }
        )
        r2 = {c: "" for c in cols}
        r2.update(
            {
                "BRID": "UK0002",
                "Name": "Alex Doe",
                "Team Manager": "Chris Lee",
                "Business Area": "Retail",
                "Sub Business Area": "Cards",
                "LOB": "Voice",
                "Site": "Glasgow",
                "Location": "UK-Glasgow",
                "Country": "UK",
                date_cols[0]: "Leave",
            }
        )
        if len(date_cols) > 1:
            r1[date_cols[1]] = "10:00-18:00"
        df = pd.DataFrame([r1, r2])[cols]
    return df


def normalize_roster_wide(df_wide: pd.DataFrame) -> pd.DataFrame:
    if df_wide is None or df_wide.empty:
        return pd.DataFrame(
            columns=[
                "BRID",
                "Name",
                "Team Manager",
                "Business Area",
                "Sub Business Area",
                "LOB",
                "Site",
                "Location",
                "Country",
                "date",
                "entry",
            ]
        )
    id_cols = [
        "BRID",
        "Name",
        "Team Manager",
        "Business Area",
        "Sub Business Area",
        "LOB",
        "Site",
        "Location",
        "Country",
    ]
    id_cols = [c for c in id_cols if c in df_wide.columns]
    date_cols = [c for c in df_wide.columns if c not in id_cols]
    long = df_wide.melt(
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="date",
        value_name="entry",
    )
    long["entry"] = long["entry"].fillna("").astype(str).str.strip()
    long = long[long["entry"] != ""]
    long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.date
    long = long[pd.notna(long["date"])]
    long["is_leave"] = long["entry"].str.lower().isin({"leave", "l", "off", "pto"})
    return long


def _manager_map() -> pd.DataFrame:
    df = load_headcount()
    if df is None or df.empty:
        return pd.DataFrame(columns=["brid", "line_manager_brid", "line_manager_full_name"])
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).strip().lower()): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for name in names:
            key = re.sub(r"[^a-z0-9]", "", str(name).strip().lower())
            col = lookup.get(key)
            if col:
                return col
        return None

    brid_col = pick("brid", "employee id", "employee_id")
    mgr_brid_col = pick("line manager brid", "line_manager_brid", "manager brid", "manager_brid")
    mgr_name_col = pick(
        "line manager full name",
        "line_manager_full_name",
        "manager name",
        "manager_name",
    )
    if not brid_col:
        return pd.DataFrame(columns=["brid", "line_manager_brid", "line_manager_full_name"])

    out = pd.DataFrame()
    out["brid"] = df[brid_col].astype(str).str.strip()
    out["line_manager_brid"] = df[mgr_brid_col].astype(str).str.strip() if mgr_brid_col else ""
    out["line_manager_full_name"] = df[mgr_name_col].astype(str).str.strip() if mgr_name_col else ""
    out = out[out["brid"] != ""]
    return out.drop_duplicates(subset=["brid"], keep="last")


def enrich_with_manager(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mgr = _manager_map()
    if mgr.empty:
        return df

    out = df.copy()
    cols = {str(c).strip().lower(): c for c in out.columns}
    brid_col = cols.get("brid") or cols.get("employee id") or cols.get("employee_id") or (
        "BRID" if "BRID" in out.columns else None
    )
    if not brid_col:
        return out

    map_name = dict(zip(mgr["brid"].astype(str), mgr["line_manager_full_name"].astype(str)))
    map_brid = dict(zip(mgr["brid"].astype(str), mgr["line_manager_brid"].astype(str)))
    brid_series = out[brid_col].astype(str).str.strip()

    if "Team Manager" not in out.columns:
        out["Team Manager"] = brid_series.map(map_name)
    else:
        out["Team Manager"] = out["Team Manager"].fillna(brid_series.map(map_name))
    if "Manager BRID" not in out.columns:
        out["Manager BRID"] = brid_series.map(map_brid)
    return out


def load_roster_wide() -> pd.DataFrame:
    ensure_roster_schema()
    if not has_dsn():
        return pd.DataFrame()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT payload FROM roster_wide_entries ORDER BY id")
        rows = cur.fetchall()
    payloads = [row[0] if isinstance(row, tuple) else row for row in rows]
    return pd.DataFrame(payloads) if payloads else pd.DataFrame()


def load_roster_long() -> pd.DataFrame:
    ensure_roster_schema()
    if not has_dsn():
        return pd.DataFrame()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT payload FROM roster_long_entries ORDER BY date, id")
        rows = cur.fetchall()
    payloads = [row[0] if isinstance(row, tuple) else row for row in rows]
    return pd.DataFrame(payloads) if payloads else pd.DataFrame()


def save_roster(wide_rows: list[dict], long_rows: Optional[list[dict]] = None) -> dict:
    ensure_roster_schema()
    if not has_dsn():
        return {"status": "missing_dsn", "wide": 0, "long": 0}

    df_wide = pd.DataFrame(wide_rows or [])
    df_long = pd.DataFrame(long_rows or [])
    if df_long.empty and not df_wide.empty:
        df_long = normalize_roster_wide(df_wide)

    df_wide = enrich_with_manager(df_wide)
    df_long = enrich_with_manager(df_long)

    if not df_long.empty:
        df_long["date"] = pd.to_datetime(df_long.get("date"), errors="coerce").dt.date
        df_long = df_long.dropna(subset=["date"])
        entry_col = "entry" if "entry" in df_long.columns else None
        if entry_col:
            df_long["is_leave"] = (
                df_long[entry_col].astype(str).str.strip().str.lower().isin({"leave", "l", "off", "pto"})
            )

    with db_conn() as conn:
        conn.execute("DELETE FROM roster_wide_entries")
        conn.execute("DELETE FROM roster_long_entries")

        if not df_wide.empty:
            conn.executemany(
                "INSERT INTO roster_wide_entries (payload) VALUES (%s)",
                [(Json(_json_safe_row(row)),) for row in df_wide.to_dict("records")],
            )

        if not df_long.empty:
            brid_col = None
            for cand in ("BRID", "brid", "employee_id", "employee id"):
                if cand in df_long.columns:
                    brid_col = cand
                    break
            entry_col = "entry" if "entry" in df_long.columns else None
            records = []
            for _, row in df_long.iterrows():
                payload = _json_safe_row(row.to_dict())
                records.append(
                    (
                        str(row.get(brid_col)).strip() if brid_col else None,
                        row.get("date"),
                        str(row.get(entry_col)).strip() if entry_col else None,
                        bool(row.get("is_leave")) if "is_leave" in row else None,
                        Json(payload),
                    )
                )
            conn.executemany(
                """
                INSERT INTO roster_long_entries (brid, date, entry, is_leave, payload)
                VALUES (%s, %s, %s, %s, %s)
                """,
                records,
            )

    return {
        "status": "saved",
        "wide": int(len(df_wide.index)) if not df_wide.empty else 0,
        "long": int(len(df_long.index)) if not df_long.empty else 0,
    }
