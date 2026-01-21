from __future__ import annotations

import re
from typing import Iterable, Optional

import pandas as pd

from app.pipeline.postgres import db_conn, has_dsn, ensure_ops_schema
from app.pipeline.timeseries_store import load_timeseries_csv


def _norm_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _pick_col(columns: Iterable[str], *candidates: str) -> Optional[str]:
    lookup = {_norm_col(col): col for col in columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in lookup:
            return lookup[key]
    return None


def normalize_scope_key(scope_key: str) -> str:
    raw = str(scope_key or "").strip()
    if not raw:
        return "global"
    if raw.lower() == "global":
        return "global"
    parts = [p.strip() for p in raw.split("|") if p is not None]
    if not parts:
        return "global"
    if parts[0].lower() == "location":
        loc = parts[1] if len(parts) > 1 else ""
        return f"location|{loc.strip().lower()}"
    lower = [p.strip().lower() for p in parts if p is not None]
    if len(lower) >= 4:
        return "|".join(lower[:4])
    if len(lower) >= 3:
        return "|".join(lower[:3])
    return lower[0] if lower else "global"


def normalize_timeseries_rows(kind: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    if df.empty:
        return df
    cols = list(df.columns)
    date_col = _pick_col(cols, "date", "ds", "day")
    interval_col = _pick_col(cols, "interval", "interval start", "interval_start", "time")

    volume_col = _pick_col(cols, "volume", "forecast volume", "actual volume", "calls", "items", "count")
    aht_col = _pick_col(cols, "aht_sec", "aht", "forecast aht", "actual aht", "average handle time")
    sut_col = _pick_col(cols, "sut_sec", "sut", "forecast sut", "actual sut")
    items_col = _pick_col(cols, "items", "transactions", "txns")

    out = pd.DataFrame()
    if date_col:
        out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    else:
        out["date"] = pd.NaT
    if interval_col:
        out["interval"] = df[interval_col].astype(str).str.strip()
    else:
        out["interval"] = None

    if volume_col:
        out["volume"] = pd.to_numeric(df[volume_col], errors="coerce")
    else:
        out["volume"] = None
    if aht_col:
        out["aht_sec"] = pd.to_numeric(df[aht_col], errors="coerce")
    else:
        out["aht_sec"] = None
    if sut_col:
        out["sut_sec"] = pd.to_numeric(df[sut_col], errors="coerce")
    else:
        out["sut_sec"] = None

    if items_col:
        out["items"] = pd.to_numeric(df[items_col], errors="coerce")
    else:
        out["items"] = None

    if out["items"].isna().all() and out["volume"].notna().any():
        out["items"] = out["volume"]

    return out.dropna(subset=["date"]).copy()


def save_timeseries_rows(kind: str, scope_key: str, rows: list[dict], mode: str = "append") -> int:
    if not kind:
        return 0
    if not has_dsn():
        return 0
    ensure_ops_schema()
    df = normalize_timeseries_rows(kind, rows)
    if df.empty:
        return 0

    scope_norm = normalize_scope_key(scope_key)
    if mode == "replace":
        with db_conn() as conn:
            conn.execute(
                "DELETE FROM timeseries_entries WHERE kind = %s AND scope_key_norm = %s",
                (kind, scope_norm),
            )

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                kind,
                scope_key or "global",
                scope_norm,
                row.get("date"),
                row.get("interval"),
                row.get("volume"),
                row.get("aht_sec"),
                row.get("sut_sec"),
                row.get("items"),
            )
        )
    with db_conn() as conn:
        conn.executemany(
            """
            INSERT INTO timeseries_entries (
                kind,
                scope_key,
                scope_key_norm,
                date,
                interval,
                volume,
                aht_sec,
                sut_sec,
                items
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            records,
        )
    return len(records)


def load_timeseries_any(kind: str, scopes: list[str]) -> pd.DataFrame:
    if not kind or not scopes:
        return pd.DataFrame()
    if not has_dsn():
        frames = []
        for scope_key in scopes:
            df_raw = load_timeseries_csv(kind, scope_key)
            df = normalize_timeseries_rows(kind, df_raw.to_dict("records")) if not df_raw.empty else pd.DataFrame()
            if not df.empty:
                df["scope_key"] = scope_key
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    frames = []
    for scope in scopes:
        scope_norm = normalize_scope_key(scope)
        if not scope_norm:
            continue
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT scope_key, date, interval, volume, aht_sec, sut_sec, items
                FROM timeseries_entries
                WHERE kind = %s AND scope_key_norm = %s
                """,
                (kind, scope_norm),
            )
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        if not df.empty:
            frames.append(df)
            continue

        parts = scope_norm.split("|")
        if len(parts) >= 3 and not scope_norm.startswith("location|"):
            prefix = "|".join(parts[:3])
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT scope_key, date, interval, volume, aht_sec, sut_sec, items
                    FROM timeseries_entries
                    WHERE kind = %s AND scope_key_norm LIKE %s
                    """,
                    (kind, f"{prefix}|%"),
                )
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
            df_prefix = pd.DataFrame(rows, columns=cols)
            if not df_prefix.empty:
                frames.append(df_prefix)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def list_timeseries_scope_keys(kinds: list[str]) -> list[str]:
    if not kinds:
        return []
    if not has_dsn():
        return []
    placeholders = ", ".join(["%s"] * len(kinds))
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT DISTINCT scope_key FROM timeseries_entries WHERE kind IN ({placeholders})",
            tuple(kinds),
        )
        rows = cur.fetchall()
    return sorted({(row[0] if isinstance(row, tuple) else row.get("scope_key")) for row in rows if row})


def load_roster() -> pd.DataFrame:
    if not has_dsn():
        return pd.DataFrame()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT start_date, end_date, fte, program, status
            FROM roster_entries
            """
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def load_hiring() -> pd.DataFrame:
    if not has_dsn():
        return pd.DataFrame()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT start_week, fte, program
            FROM hiring_entries
            """
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)
