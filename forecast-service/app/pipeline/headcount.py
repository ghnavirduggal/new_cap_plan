from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

CHANNEL_LIST = ["Voice", "Back Office", "Outbound", "Blended", "Chat", "MessageUs"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _exports_dir() -> Path:
    outdir = _repo_root() / "exports"
    outdir.mkdir(exist_ok=True)
    return outdir


def _headcount_path() -> Path:
    return _exports_dir() / "headcount.csv"


def save_headcount(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"status": "empty", "rows": 0}
    path = _headcount_path()
    df.to_csv(path, index=False)
    return {"status": "saved", "rows": len(df.index), "path": str(path)}


def load_headcount() -> pd.DataFrame:
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
        "site",
    )
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
