from __future__ import annotations

from pathlib import Path
import re
import json
import hashlib
from typing import Any, Optional

import numpy as np
import pandas as pd
from psycopg.types.json import Json

from app.pipeline.postgres import db_conn, has_dsn, ensure_headcount_schema

CHANNEL_LIST = ["Voice", "Back Office", "Outbound", "Blended", "Chat", "MessageUs"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _exports_dir() -> Path:
    outdir = _repo_root() / "exports"
    outdir.mkdir(exist_ok=True)
    return outdir


def _headcount_path() -> Path:
    return _exports_dir() / "headcount.csv"

def _brid_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]", "", str(col).strip().lower())
        if key == "brid":
            return col
    return None

def _normalize_brid_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()

def _normalize_df_for_hash(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    dfc = df.copy()
    dfc = dfc.replace({np.nan: ""})
    dfc = dfc.astype(str)
    dfc = dfc.reindex(sorted(dfc.columns), axis=1)
    try:
        dfc = dfc.sort_values(by=list(dfc.columns))
    except Exception:
        pass
    return dfc.to_csv(index=False)

def _dataset_hash(df: pd.DataFrame) -> str:
    raw = _normalize_df_for_hash(df)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _row_hash(row: dict) -> str:
    raw = json.dumps(row, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_headcount(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"status": "empty", "rows": 0}
    brid_col = _brid_col(df)
    dataset_hash = _dataset_hash(df)
    if has_dsn():
        ensure_headcount_schema()
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM headcount_upload_meta WHERE dataset_hash = %s",
                (dataset_hash,),
            )
            if cur.fetchone():
                return {"status": "duplicate", "rows": 0, "dataset_hash": dataset_hash}
            cur.execute(
                """
                INSERT INTO headcount_upload_meta (dataset_hash, row_count)
                VALUES (%s, %s)
                ON CONFLICT (dataset_hash) DO NOTHING
                """,
                (dataset_hash, int(len(df.index))),
            )
            cols = _hcu_cols(df)
            if brid_col:
                brid_vals = _normalize_brid_series(df[brid_col])
                brid_vals = brid_vals[brid_vals != ""]
                if not brid_vals.empty:
                    cur.execute(
                        "DELETE FROM headcount_entries WHERE brid = ANY(%s)",
                        (brid_vals.drop_duplicates().tolist(),),
                    )
            for _, row in df.iterrows():
                payload = row.where(pd.notna(row), None).to_dict()
                row_hash = _row_hash(payload)
                brid_val = None
                if brid_col:
                    brid_val = str(payload.get(brid_col) or "").strip() or None
                cur.execute(
                    """
                    INSERT INTO headcount_entries (
                        row_hash, business_area, sub_business_area, channel, site, location, brid, payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (row_hash) DO NOTHING
                    """,
                    (
                        row_hash,
                        str(payload.get(cols.get("ba"))).strip() if cols.get("ba") else None,
                        str(payload.get(cols.get("sba"))).strip() if cols.get("sba") else None,
                        str(payload.get(cols.get("lob"))).strip() if cols.get("lob") else None,
                        str(payload.get(cols.get("site"))).strip() if cols.get("site") else None,
                        str(payload.get(cols.get("loc"))).strip() if cols.get("loc") else None,
                        brid_val or str(payload.get("BRID") or payload.get("brid") or "").strip() or None,
                        Json(payload),
                    ),
                )
    # Keep CSV in sync for legacy reads
    path = _headcount_path()
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty:
            existing_brid_col = _brid_col(existing)
            if brid_col and existing_brid_col:
                new_brids = _normalize_brid_series(df[brid_col])
                new_brids = set(new_brids[new_brids != ""])
                if new_brids:
                    existing_brids = _normalize_brid_series(existing[existing_brid_col])
                    keep_mask = ~existing_brids.isin(new_brids)
                    existing = existing.loc[keep_mask].copy()
                combined = pd.concat([existing, df], ignore_index=True)
            else:
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates()
            combined.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)
    return {"status": "saved", "rows": len(df.index), "path": str(path), "dataset_hash": dataset_hash}


def load_headcount() -> pd.DataFrame:
    if has_dsn():
        ensure_headcount_schema()
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT payload FROM headcount_entries ORDER BY id ASC")
                rows = cur.fetchall()
            if rows:
                payloads = [row[0] for row in rows if row and row[0]]
                if payloads:
                    return pd.DataFrame(payloads)
        except Exception:
            pass
    path = _headcount_path()
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _hcu_df() -> pd.DataFrame:
    try:
        df = load_headcount()
    except Exception:
        df = pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _hcu_cols(df: pd.DataFrame) -> dict:
    raw = {str(c).strip().lower(): c for c in df.columns}
    norm = {re.sub(r"[^a-z0-9]", "", key): col for key, col in raw.items()}

    def pick(*names: str) -> Optional[str]:
        for name in names:
            key = str(name).strip().lower()
            col = raw.get(key)
            if col:
                return col
            norm_key = re.sub(r"[^a-z0-9]", "", key)
            col = norm.get(norm_key)
            if col:
                return col
        return None

    ba = pick(
        "journey",
        "business area",
        "vertical",
        "current org unit description",
        "current_org_unit_description",
        "level 0",
        "level_0",
    )
    sba = pick(
        "level 3",
        "level_3",
        "sub business area",
        "sub_business_area",
    )
    loc = pick(
        "position location country",
        "position_location_country",
        "location country",
        "location_country",
        "country",
        "location",
    )
    site = pick(
        "position location building description",
        "position_location_building_description",
        "building description",
        "building",
        "site name",
        "site_name",
        "location site",
        "location_site",
        "location building",
        "location_building",
        "site",
    )
    if not site:
        for key, col in norm.items():
            if any(
                token in key
                for token in (
                    "positionlocationbuildingdescription",
                    "locationbuildingdescription",
                    "buildingdescription",
                    "locationbuilding",
                    "locationsite",
                )
            ):
                site = col
                break
    lob = pick(
        "lob",
        "channel",
        "program",
        "position group",
        "position_group",
    )
    return {"ba": ba, "sba": sba, "loc": loc, "site": site, "lob": lob}


def _clean_values(series: pd.Series) -> list[str]:
    cleaned = (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace({"": np.nan})
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    return cleaned.tolist()


def business_areas() -> list[str]:
    df = _hcu_df()
    if df.empty:
        return []
    C = _hcu_cols(df)
    if not C["ba"]:
        return []
    return _clean_values(df[C["ba"]])


def sub_business_areas(ba: Optional[str] = None) -> list[str]:
    df = _hcu_df()
    if df.empty:
        return []
    C = _hcu_cols(df)
    if not (C["ba"] and C["sba"]):
        return []
    dff = df[[C["ba"], C["sba"]]].dropna()
    if ba:
        mask = dff[C["ba"]].astype(str).str.strip().str.lower() == str(ba).strip().lower()
        return _clean_values(dff.loc[mask, C["sba"]])
    return _clean_values(dff[C["sba"]])


def locations(ba: Optional[str] = None) -> list[str]:
    df = _hcu_df()
    if df.empty:
        return []
    C = _hcu_cols(df)
    if not C["loc"]:
        return []
    if ba and C["ba"]:
        dff = df[[C["ba"], C["loc"]]].dropna()
        mask = dff[C["ba"]].astype(str).str.strip().str.lower() == str(ba).strip().lower()
        return _clean_values(dff.loc[mask, C["loc"]])
    return _clean_values(df[C["loc"]])


def sites(ba: Optional[str] = None, location: Optional[str] = None) -> list[str]:
    df = _hcu_df()
    if df.empty:
        return []
    C = _hcu_cols(df)
    if not C["site"]:
        return []
    dff = df.copy()
    if ba and C["ba"]:
        dff = dff[dff[C["ba"]].astype(str).str.strip().str.lower() == str(ba).strip().lower()]
    if C["loc"] and location:
        dff = dff[dff[C["loc"]].astype(str).str.strip().str.lower() == str(location).strip().lower()]
    return _clean_values(dff[C["site"]])


def channels_for_scope(_ba: Optional[str] = None, _sba: Optional[str] = None) -> list[str]:
    return CHANNEL_LIST[:]


def preview_headcount(rows: Optional[int] = 50) -> list[dict[str, Any]]:
    df = _hcu_df()
    if df.empty:
        return []
    limit = int(rows or 50)
    return df.head(limit).to_dict("records")


def headcount_template(rows: int = 5) -> pd.DataFrame:
    columns = [
        "Level 0",
        "Level 1",
        "Level 2",
        "Level 3",
        "Level 4",
        "Level 5",
        "Level 6",
        "BRID",
        "Full Name",
        "Position Description",
        "Headcount Operational Status Description",
        "Employee Group Description",
        "Corporate Grade Description",
        "Line Manager BRID",
        "Line Manager Full Name",
        "Current Organisation Unit",
        "Current Organisation Unit Description",
        "Position Location Country",
        "Position Location City",
        "Position Location Building Description",
        "CCID",
        "CC Name",
        "Journey",
        "Position Group",
    ]
    sample = [
        [
            "BUK",
            "COO",
            "Business Services",
            "BFA",
            "Refers",
            "",
            "",
            "IN0001",
            "Asha Rao",
            "Agent",
            "Active",
            "FT",
            "BA4",
            "IN9999",
            "Priyanka Menon",
            "Ops|BFA|Refers",
            "Ops BFA Refers",
            "India",
            "Chennai",
            "DLF IT Park",
            "12345",
            "Complaints",
            "Onboarding",
            "Back Office",
        ],
        [
            "BUK",
            "COO",
            "Business Services",
            "BFA",
            "Appeals",
            "",
            "",
            "IN0002",
            "Rahul Jain",
            "Team Leader",
            "Active",
            "FT",
            "BA5",
            "IN8888",
            "Arjun Mehta",
            "Ops|BFA|Appeals",
            "Ops BFA Appeals",
            "India",
            "Pune",
            "EON Cluster C",
            "12345",
            "Complaints",
            "Onboarding",
            "Voice",
        ],
    ]
    df = pd.DataFrame(sample[:rows], columns=columns)
    if rows > len(sample):
        df = pd.concat([df, pd.DataFrame(columns=columns)], ignore_index=True)
    return df
