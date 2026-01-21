from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from app.pipeline.headcount import load_headcount
from app.pipeline.ops_store import load_hiring, load_roster
from app.pipeline.roster_store import load_roster_long, load_roster_wide
from app.pipeline.settings_store import load_holidays, load_settings
from app.pipeline.timeseries_store import load_timeseries_csv


def load_timeseries(kind: str, scope_key: str) -> pd.DataFrame:
    if not kind or not scope_key:
        return pd.DataFrame()
    return load_timeseries_csv(kind, scope_key)


def resolve_settings(ba: Optional[str] = None, subba: Optional[str] = None, lob: Optional[str] = None, for_date: Optional[str] = None):
    # Match the resolution order used in plan_detail/planning_calc.
    if ba and subba and lob:
        return load_settings("hier", None, ba, subba, lob, None)
    return load_settings("global", None, None, None, None, None)


def resolve_holidays(
    scope_type: str = "global",
    location: Optional[str] = None,
    ba: Optional[str] = None,
    sba: Optional[str] = None,
    subba: Optional[str] = None,
    channel: Optional[str] = None,
    lob: Optional[str] = None,
    site: Optional[str] = None,
    **_kwargs: object,
) -> pd.DataFrame:
    if sba is None and subba is not None:
        sba = subba
    if channel is None and lob is not None:
        channel = lob
    return load_holidays(scope_type, location, ba, sba, channel, site)


def load_defaults() -> dict:
    return {}


def _headcount_cols(df: pd.DataFrame) -> dict:
    raw = {str(c).strip().lower(): c for c in df.columns}
    norm = {"".join(ch for ch in key if ch.isalnum()): col for key, col in raw.items()}

    def pick(*names: str) -> Optional[str]:
        for name in names:
            key = str(name).strip().lower()
            if key in raw:
                return raw[key]
            norm_key = "".join(ch for ch in key if ch.isalnum())
            if norm_key in norm:
                return norm[norm_key]
        return None

    return {
        "ba": pick("journey", "business area", "vertical", "current org unit description", "level 0", "level_0"),
        "sba": pick("level 3", "level_3", "sub business area", "sub_business_area"),
        "lob": pick("lob", "channel", "program", "position group", "position_group"),
        "site": pick("position location building description", "position_location_building_description", "site"),
        "loc": pick("position location country", "position_location_country", "location", "country"),
    }


def get_clients_hierarchy() -> Tuple[dict, list, list]:
    df = load_headcount()
    if df is None or df.empty:
        return {}, [], []
    cols = _headcount_cols(df)
    ba_col = cols.get("ba")
    sba_col = cols.get("sba")
    lob_col = cols.get("lob")
    site_col = cols.get("site")
    loc_col = cols.get("loc")

    hier: dict = {}
    sites: set[str] = set()
    locs: set[str] = set()

    if site_col and site_col in df.columns:
        sites = set(df[site_col].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist())
    if loc_col and loc_col in df.columns:
        locs = set(df[loc_col].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist())

    if ba_col and sba_col and lob_col:
        for _, row in df[[ba_col, sba_col, lob_col]].dropna().iterrows():
            ba = str(row[ba_col]).strip()
            sba = str(row[sba_col]).strip()
            lob = str(row[lob_col]).strip()
            if not ba or not sba or not lob:
                continue
            hier.setdefault(ba, {}).setdefault(sba, set()).add(lob)

    hier_out = {ba: {sba: sorted(list(lobs)) for sba, lobs in sbas.items()} for ba, sbas in hier.items()}
    return hier_out, sorted(sites), sorted(locs)
