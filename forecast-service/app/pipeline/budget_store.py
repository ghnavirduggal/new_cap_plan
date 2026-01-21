from __future__ import annotations

from typing import Optional

import pandas as pd

from app.pipeline.postgres import db_conn, ensure_budget_schema


def _canon_channel(channel: str) -> str:
    raw = (channel or "").strip().lower()
    if raw in {"voice", "call", "telephony"}:
        return "Voice"
    if raw in {"back office", "backoffice", "bo"}:
        return "Back Office"
    if raw in {"chat", "messaging", "messageus", "message us"}:
        return "Chat"
    if raw in {"outbound", "ob", "out bound"}:
        return "Outbound"
    return channel.strip() if channel else ""


def _week_monday(value: Optional[object]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    ts = ts.normalize()
    return ts - pd.Timedelta(days=int(ts.weekday()))


def _pick_col(columns: list[str], *candidates: str) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def normalize_budget_rows(channel: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    if df.empty:
        return df
    cols = list(df.columns)
    week_col = _pick_col(cols, "week", "start_week", "monday") or cols[0]
    hc_col = _pick_col(cols, "budget_headcount", "headcount")
    aht_col = _pick_col(cols, "budget_aht_sec", "aht_sec", "aht")
    sut_col = _pick_col(cols, "budget_sut_sec", "sut_sec", "sut")

    out = df.rename(columns={week_col: "week"}).copy()
    if hc_col and hc_col != "budget_headcount":
        out = out.rename(columns={hc_col: "budget_headcount"})
    if aht_col and aht_col != "budget_aht_sec":
        out = out.rename(columns={aht_col: "budget_aht_sec"})
    if sut_col and sut_col != "budget_sut_sec":
        out = out.rename(columns={sut_col: "budget_sut_sec"})

    if "budget_headcount" not in out.columns:
        out["budget_headcount"] = None
    if "budget_aht_sec" not in out.columns:
        out["budget_aht_sec"] = None
    if "budget_sut_sec" not in out.columns:
        out["budget_sut_sec"] = None

    out["week"] = out["week"].apply(_week_monday)
    out = out.dropna(subset=["week"])
    out["budget_headcount"] = pd.to_numeric(out["budget_headcount"], errors="coerce")
    out["budget_aht_sec"] = pd.to_numeric(out["budget_aht_sec"], errors="coerce")
    out["budget_sut_sec"] = pd.to_numeric(out["budget_sut_sec"], errors="coerce")
    out = out.drop_duplicates(subset=["week"], keep="last").sort_values("week")

    canon = _canon_channel(channel)
    if canon == "Back Office":
        return out[["week", "budget_headcount", "budget_sut_sec"]]
    if canon in {"Voice", "Chat", "Outbound"}:
        return out[["week", "budget_headcount", "budget_aht_sec"]]
    return out[["week", "budget_headcount", "budget_aht_sec", "budget_sut_sec"]]


def load_budget_rows(
    ba: str,
    subba: str,
    channel: str,
    site: str,
) -> pd.DataFrame:
    ensure_budget_schema()
    canon = _canon_channel(channel)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT week, budget_headcount, budget_aht_sec, budget_sut_sec
            FROM budget_entries
            WHERE business_area = %s
              AND sub_business_area = %s
              AND channel = %s
              AND site = %s
            ORDER BY week
            """,
            (ba, subba, canon, site),
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    if canon == "Back Office":
        return df[["week", "budget_headcount", "budget_sut_sec"]]
    if canon in {"Voice", "Chat", "Outbound"}:
        return df[["week", "budget_headcount", "budget_aht_sec"]]
    return df


def upsert_budget_rows(
    ba: str,
    subba: str,
    channel: str,
    site: str,
    rows: list[dict],
) -> int:
    if not (ba and subba and channel and site):
        return 0
    ensure_budget_schema()
    canon = _canon_channel(channel)
    df = normalize_budget_rows(canon, rows)
    if df.empty:
        return 0
    records = []
    for _, row in df.iterrows():
        records.append(
            (
                ba,
                subba,
                canon,
                site,
                pd.to_datetime(row["week"]).date(),
                row.get("budget_headcount"),
                row.get("budget_aht_sec"),
                row.get("budget_sut_sec"),
            )
        )
    with db_conn() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO budget_entries (
                business_area,
                sub_business_area,
                channel,
                site,
                week,
                budget_headcount,
                budget_aht_sec,
                budget_sut_sec
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (business_area, sub_business_area, channel, site, week)
            DO UPDATE SET
                budget_headcount = EXCLUDED.budget_headcount,
                budget_aht_sec = EXCLUDED.budget_aht_sec,
                budget_sut_sec = EXCLUDED.budget_sut_sec,
                updated_at = NOW()
            """,
            records,
        )
    return len(records)
