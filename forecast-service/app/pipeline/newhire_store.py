from __future__ import annotations

import datetime as dt
import re
from typing import Optional

import pandas as pd

from app.pipeline.postgres import db_conn, ensure_newhire_schema, has_dsn

NH_LABELS = {
    "business_area": "Business Area",
    "class_reference": "Class Reference",
    "source_system_id": "Source System ID",
    "emp_type": "Emp Type",
    "status": "Status",
    "class_type": "Class Type",
    "class_level": "Class Level",
    "grads_needed": "Grads Needed",
    "billable_hc": "Billable HC",
    "training_weeks": "Training Weeks",
    "nesting_weeks": "Nesting Weeks",
    "induction_start": "Induction Start",
    "training_start": "Training Start",
    "training_end": "Training End",
    "nesting_start": "Nesting Start",
    "nesting_end": "Nesting End",
    "production_start": "Production Start",
    "created_by": "Created By",
    "created_ts": "Created On",
}

NH_COLS = [
    "business_area",
    "class_reference",
    "source_system_id",
    "emp_type",
    "status",
    "class_type",
    "class_level",
    "grads_needed",
    "billable_hc",
    "training_weeks",
    "nesting_weeks",
    "induction_start",
    "training_start",
    "training_end",
    "nesting_start",
    "nesting_end",
    "production_start",
    "created_by",
    "created_ts",
]


def current_user_fallback() -> str:
    import os
    import getpass

    return os.environ.get("USERNAME") or os.environ.get("USER") or getpass.getuser() or "system"


def new_hire_template_df() -> pd.DataFrame:
    sample = {
        "business_area": "Example BA",
        "class_reference": "",
        "source_system_id": "",
        "emp_type": "full-time",
        "status": "tentative",
        "class_type": "ramp-up",
        "class_level": "new-agent",
        "grads_needed": 10,
        "billable_hc": 0,
        "training_weeks": 2,
        "nesting_weeks": 1,
        "induction_start": "",
        "training_start": "",
        "training_end": "",
        "nesting_start": "",
        "nesting_end": "",
        "production_start": "",
        "created_by": "",
        "created_ts": "",
    }
    return pd.DataFrame([sample])[NH_COLS]


def _iso_date(value) -> Optional[str]:
    if value in (None, "", "nan"):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def _auto_dates(training_start, training_weeks, nesting_start, nesting_weeks, production_start):
    def to_date(val):
        if not val:
            return None
        if isinstance(val, dt.date):
            return val
        return pd.to_datetime(val, errors="coerce").date()

    ts = to_date(training_start)
    ns = to_date(nesting_start)
    ps = to_date(production_start)
    tw = int(training_weeks or 0)
    nw = int(nesting_weeks or 0)
    te = ts + dt.timedelta(days=7 * tw) - dt.timedelta(days=1) if (ts and tw > 0) else None
    if ns is None and te is not None:
        ns = te + dt.timedelta(days=1)
    ne = ns + dt.timedelta(days=7 * nw) - dt.timedelta(days=1) if (ns and nw > 0) else None
    if ps is None and ne is not None:
        ps = ne + dt.timedelta(days=1)
    return _iso_date(ts), _iso_date(te), _iso_date(ns), _iso_date(ne), _iso_date(ps)


def _ensure_nh_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=NH_COLS)
    out = df.copy()
    for col in NH_COLS:
        if col not in out.columns:
            if col in ("grads_needed", "billable_hc", "training_weeks", "nesting_weeks"):
                out[col] = 0
            elif col == "created_by":
                out[col] = current_user_fallback()
            elif col == "created_ts":
                out[col] = pd.Timestamp.utcnow().isoformat(timespec="seconds")
            else:
                out[col] = None
    return out[NH_COLS + [c for c in out.columns if c not in NH_COLS]]


def _pick_col(columns: list[str], *candidates: str) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def load_new_hires() -> pd.DataFrame:
    ensure_newhire_schema()
    if not has_dsn():
        return pd.DataFrame(columns=NH_COLS)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT business_area, class_reference, source_system_id, emp_type, status, class_type,
                   class_level, grads_needed, billable_hc, training_weeks, nesting_weeks,
                   induction_start, training_start, training_end, nesting_start, nesting_end,
                   production_start, created_by, created_ts
            FROM new_hire_entries
            ORDER BY production_start NULLS LAST, class_reference
            """
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    return _ensure_nh_cols(df)


def save_new_hires(rows: list[dict]) -> int:
    ensure_newhire_schema()
    if not has_dsn():
        return 0
    df = _ensure_nh_cols(pd.DataFrame(rows or []))
    if df.empty:
        return 0

    for col in ("grads_needed", "billable_hc", "training_weeks", "nesting_weeks"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ("induction_start", "training_start", "training_end", "nesting_start", "nesting_end", "production_start"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    df["created_by"] = df["created_by"].fillna(current_user_fallback())
    df["created_ts"] = pd.to_datetime(df["created_ts"], errors="coerce")
    df["created_ts"] = df["created_ts"].fillna(pd.Timestamp.utcnow())

    with db_conn() as conn:
        conn.execute("DELETE FROM new_hire_entries")
        conn.executemany(
            """
            INSERT INTO new_hire_entries (
                business_area,
                class_reference,
                source_system_id,
                emp_type,
                status,
                class_type,
                class_level,
                grads_needed,
                billable_hc,
                training_weeks,
                nesting_weeks,
                induction_start,
                training_start,
                training_end,
                nesting_start,
                nesting_end,
                production_start,
                created_by,
                created_ts
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            df[NH_COLS].itertuples(index=False, name=None),
        )
    return len(df.index)


def next_class_reference(existing_df: Optional[pd.DataFrame] = None, ba: Optional[str] = None) -> str:
    existing = existing_df if isinstance(existing_df, pd.DataFrame) else load_new_hires()
    today = dt.date.today().strftime("%Y%m%d")
    ba_tag = ""
    if ba:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", str(ba))
        ba_tag = f"-{cleaned[:6].upper()}" if cleaned else ""
    head = f"NH-{today}{ba_tag}"
    seq = 1
    if not existing.empty and "class_reference" in existing.columns:
        match = existing["class_reference"].astype(str).str.extract(rf"^{re.escape(head)}-(\d+)$", expand=False).dropna()
        if not match.empty:
            seq = int(match.astype(int).max()) + 1
    return f"{head}-{seq:02d}"


def normalize_nh_upload_master(raw_df: pd.DataFrame, source_id: Optional[str] = None, default_ba: Optional[str] = None) -> pd.DataFrame:
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        return _ensure_nh_cols(raw_df)

    df = raw_df.copy()
    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names):
        for name in names:
            if name in cols:
                return cols[name]
        return None

    col_map = dict(
        business_area=pick("business area", "ba", "journey", "vertical", "business_area"),
        emp_type=pick("emp type", "emptype", "ft/pt", "ftpt", "emp_type"),
        status=pick("status", "class status", "class_status"),
        class_type=pick("class type", "classtype", "class_type"),
        class_level=pick("class level", "classlevel", "class_level"),
        grads_needed=pick("grads needed", "graduates", "grads", "headcount", "grads_needed"),
        billable_hc=pick("billable hc", "billable", "billable_headcount", "billable_hc"),
        training_weeks=pick("training weeks", "train weeks", "training_weeks"),
        nesting_weeks=pick("nesting weeks", "nest weeks", "nesting_weeks"),
        induction_start=pick("induction start", "induction_start"),
        training_start=pick("training start", "training_start", "train start"),
        training_end=pick("training end", "training_end", "train end"),
        nesting_start=pick("nesting start", "nesting_start"),
        nesting_end=pick("nesting end", "nesting_end"),
        production_start=pick("production start", "production_start", "go live", "golive"),
        class_reference=pick("class reference", "class_reference", "class"),
        source_system_id=pick("source system id", "source_system_id", "source id", "sourceid"),
    )

    out = pd.DataFrame()
    for key in [
        "business_area",
        "emp_type",
        "status",
        "class_type",
        "class_level",
        "grads_needed",
        "billable_hc",
        "training_weeks",
        "nesting_weeks",
        "induction_start",
        "training_start",
        "training_end",
        "nesting_start",
        "nesting_end",
        "production_start",
        "class_reference",
        "source_system_id",
    ]:
        src = col_map.get(key)
        out[key] = df[src] if src in df else None

    out["business_area"] = out["business_area"].fillna(default_ba).astype(object)
    out["emp_type"] = out["emp_type"].astype(str).str.lower().replace(
        {"full time": "full-time", "fulltime": "full-time", "ft": "full-time", "pt": "part-time", "nan": None, "none": None, "": None}
    )
    out["status"] = out["status"].astype(str).str.lower().replace({"nan": None, "none": None, "": None})
    out["class_type"] = out["class_type"].astype(str).str.lower().replace(
        {"ramp up": "ramp-up", "rampup": "ramp-up", "nan": None, "none": None, "": None}
    )
    out["class_level"] = out["class_level"].astype(str).str.lower().replace(
        {
            "newagent": "new-agent",
            "new agent": "new-agent",
            "cross skill": "cross-skill",
            "up skill": "up-skill",
            "nan": None,
            "none": None,
            "": None,
        }
    )

    for col in ("grads_needed", "billable_hc", "training_weeks", "nesting_weeks"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    for idx, row in out.iterrows():
        ts, te, ns, ne, ps = _auto_dates(
            row.get("training_start"),
            row.get("training_weeks"),
            row.get("nesting_start"),
            row.get("nesting_weeks"),
            row.get("production_start"),
        )
        out.at[idx, "induction_start"] = _iso_date(row.get("induction_start"))
        out.at[idx, "training_start"] = ts
        out.at[idx, "training_end"] = te
        out.at[idx, "nesting_start"] = ns
        out.at[idx, "nesting_end"] = ne
        out.at[idx, "production_start"] = ps

    batch_id = source_id or f"upload-{current_user_fallback()}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M%S')}"
    out["source_system_id"] = out["source_system_id"].replace({None: batch_id, "": batch_id})

    existing = load_new_hires()
    for ba_val, idxs in out.groupby(out["business_area"].fillna("").astype(str)).groups.items():
        for idx in idxs:
            if not str(out.at[idx, "class_reference"] or "").strip():
                out.at[idx, "class_reference"] = next_class_reference(existing_df=existing, ba=ba_val)
                existing = pd.concat(
                    [existing, out.loc[[idx], existing.columns.intersection(["class_reference"])]],
                    ignore_index=True,
                )

    out["created_by"] = current_user_fallback()
    out["created_ts"] = pd.Timestamp.utcnow().isoformat(timespec="seconds")
    return _ensure_nh_cols(out)


def ingest_new_hires(rows: list[dict], source_id: Optional[str] = None, default_ba: Optional[str] = None) -> pd.DataFrame:
    raw_df = pd.DataFrame(rows or [])
    norm = normalize_nh_upload_master(raw_df, source_id=source_id, default_ba=default_ba)
    current = load_new_hires()
    if not current.empty and "class_reference" in current.columns and "class_reference" in norm.columns:
        keep_old = current[~current["class_reference"].astype(str).isin(norm["class_reference"].astype(str))]
        merged = pd.concat([keep_old, norm], ignore_index=True)
    else:
        merged = norm
    save_new_hires(merged)
    return merged
