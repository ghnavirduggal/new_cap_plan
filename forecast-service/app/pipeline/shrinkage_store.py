from __future__ import annotations

import datetime as dt
import re
from typing import Optional

import numpy as np
import pandas as pd
from psycopg.types.json import Json

from app.pipeline.headcount import load_headcount
from app.pipeline.ops_store import load_roster as load_roster_supply
from app.pipeline.postgres import db_conn, ensure_shrinkage_schema, has_dsn
from app.pipeline.roster_store import load_roster_long


SHRINK_WEEKLY_FIELDS = [
    "week",
    "program",
    "ooo_hours",
    "ino_hours",
    "base_hours",
    "ooo_pct",
    "ino_pct",
    "overall_pct",
]

_SHRINK_COLUMN_ALIASES = {
    "week": "week",
    "startweek": "week",
    "program": "program",
    "businessarea": "program",
    "journey": "program",
    "outofofficehours": "ooo_hours",
    "ooohours": "ooo_hours",
    "ooohrs": "ooo_hours",
    "inofficehours": "ino_hours",
    "inohours": "ino_hours",
    "productivehours": "base_hours",
    "basehours": "base_hours",
    "baseproductivehours": "base_hours",
    "outofofficepct": "ooo_pct",
    "ooopct": "ooo_pct",
    "inofficepct": "ino_pct",
    "inopct": "ino_pct",
    "overallshrinkpct": "overall_pct",
    "overallpct": "overall_pct",
    "overallshrink": "overall_pct",
}


def _json_safe_value(value):
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        return pd.to_datetime(value, errors="coerce").date().isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe_row(row: dict) -> dict:
    return {key: _json_safe_value(val) for key, val in (row or {}).items()}


def is_voice_shrinkage_like(df: pd.DataFrame) -> bool:
    """Detect Voice-style shrinkage uploads (Superstate + Hours)."""
    if df is None or df.empty:
        return False
    cols = {str(c).strip().lower() for c in df.columns}
    return ("superstate" in cols) and ("hours" in cols)


def _shrink_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "", str(name).strip().lower())
    slug = slug.replace("overallshrink", "overallshrink")
    return slug


def _week_floor(value: pd.Timestamp | str | dt.date, week_start: str = "Monday") -> dt.date:
    day = pd.to_datetime(value).date()
    wd = day.weekday()
    if (week_start or "Monday").lower().startswith("sun"):
        return day - dt.timedelta(days=(wd + 1) % 7)
    return day - dt.timedelta(days=wd)


def _hhmm_to_minutes(value) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        parts = text.split(":")
        if len(parts) >= 2:
            try:
                hours = int(parts[0])
                minutes = int(parts[1])
                return float(hours * 60 + minutes)
            except Exception:
                pass
    try:
        return float(text)
    except Exception:
        return 0.0


def _parse_date_series(series: pd.Series) -> pd.Series:
    s = pd.Series(series)
    try:
        if np.issubdtype(s.dtype, np.datetime64):
            return pd.to_datetime(s, errors="coerce").dt.date
    except Exception:
        pass

    sample = s.dropna().astype(str).str.strip()
    if sample.empty:
        return pd.to_datetime(s, errors="coerce").dt.date

    iso_mask = sample.str.match(r"^\\d{4}-\\d{1,2}-\\d{1,2}$")
    if iso_mask.any() and iso_mask.mean() > 0.5:
        return pd.to_datetime(s, errors="coerce", format="%Y-%m-%d").dt.date

    slash_mask = sample.str.match(r"^\\d{1,2}/\\d{1,2}/\\d{2,4}$")
    if slash_mask.any() and slash_mask.mean() > 0.5:
        parts = sample[slash_mask].str.extract(r"^(\\d{1,2})/(\\d{1,2})/(\\d{2,4})$")
        first = pd.to_numeric(parts[0], errors="coerce")
        fmt = "%d/%m/%Y" if (first > 12).any() else "%m/%d/%Y"
        return pd.to_datetime(s, errors="coerce", format=fmt).dt.date

    dash_mask = sample.str.match(r"^\\d{1,2}-\\d{1,2}-\\d{2,4}$")
    if dash_mask.any() and dash_mask.mean() > 0.5:
        parts = sample[dash_mask].str.extract(r"^(\\d{1,2})-(\\d{1,2})-(\\d{2,4})$")
        first = pd.to_numeric(parts[0], errors="coerce")
        fmt = "%d-%m-%Y" if (first > 12).any() else "%m-%d-%Y"
        return pd.to_datetime(s, errors="coerce", format=fmt).dt.date

    return pd.to_datetime(s, errors="coerce").dt.date


def _pick_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).strip().lower()): c for c in df.columns}
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", str(name).strip().lower())
        if key in lookup:
            return lookup[key]
    return None


def _hc_lookup() -> dict:
    df = load_headcount()
    if df is None or df.empty:
        return {}
    brid_col = _pick_col(df, "brid", "employee id", "employee_id")
    if not brid_col:
        return {}
    lm_name_col = _pick_col(df, "line manager full name", "line_manager_full_name", "manager name")
    lm_brid_col = _pick_col(df, "line manager brid", "line_manager_brid", "manager brid")
    site_col = _pick_col(df, "position location building description", "site", "building")
    city_col = _pick_col(df, "position location city", "city")
    country_col = _pick_col(df, "position location country", "country", "location")
    journey_col = _pick_col(df, "journey", "business area", "level 0")
    level3_col = _pick_col(df, "level 3", "sub business area", "sub_business_area")

    out = {}
    for _, row in df.iterrows():
        brid = str(row.get(brid_col, "")).strip()
        if not brid:
            continue
        out[brid] = {
            "lm_name": row.get(lm_name_col) if lm_name_col else None,
            "lm_brid": row.get(lm_brid_col) if lm_brid_col else None,
            "site": row.get(site_col) if site_col else None,
            "city": row.get(city_col) if city_col else None,
            "country": row.get(country_col) if country_col else None,
            "journey": row.get(journey_col) if journey_col else None,
            "level_3": row.get(level3_col) if level3_col else None,
        }
    return out


def shrinkage_bo_raw_template_df(rows: int = 16) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize().date()
    cats = [
        "Staff Complement",
        "Flextime",
        "Borrowed Staff",
        "Lend Staff",
        "Overtime",
        "Core Time",
        "Diverted",
        "Downtime",
        "Time Worked",
        "Work out",
    ]
    demo = []
    for i in range(rows):
        cat = cats[i % len(cats)]
        dur = 1800 if cat in ("Diverted", "Downtime") else (3600 if cat in ("Core Time", "Time Worked") else 1200)
        brid = f"IN{1000 + i}"
        demo.append(
            {
                "Category": "Shrinkage",
                "StartDate": today.isoformat(),
                "EndDate": today.isoformat(),
                "DateId": int(pd.Timestamp(today).strftime("%Y%m%d")),
                "Date": today.isoformat(),
                "GroupId": "BO1",
                "WorkgroupId": "WG1",
                "WorkgroupName": "BO Cases",
                "Activity": cat,
                "SaffMemberId": brid,
                "StaffLastName": "Doe",
                "SatffFirstName": "Alex",
                "StaffReferenceId": brid,
                "TaskId": "T-001",
                "Units": 10 if cat == "Work out" else 0,
                "DurationSeconds": dur,
                "EmploymentType": "FT",
                "AgentID(BRID)": brid,
                "Agent Name": "Alex Doe",
                "TL Name": "",
                "Time": round(dur / 3600, 2),
                "Sub Business Area": "",
            }
        )
    return pd.DataFrame(demo)


def shrinkage_voice_raw_template_df(rows: int = 18) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize().date()
    superstates = [
        "SC_INCLUDED_TIME",
        "SC_ABSENCE_TOTAL",
        "SC_A_Sick_Long_Term",
        "SC_HOLIDAY",
        "SC_TRAINING_TOTAL",
        "SC_BREAKS",
        "SC_SYSTEM_EXCEPTION",
    ]
    demo = []
    for i in range(rows):
        state = superstates[i % len(superstates)]
        hhmm = f"{(i % 3) + 1:02d}:{(i * 10) % 60:02d}"
        brid = f"UK{2000 + i}"
        demo.append(
            {
                "Employee": f"User {i + 1}",
                "BRID": brid,
                "First Name": "Sam",
                "Last Name": "Patel",
                "Superstate": state,
                "Date": today.isoformat(),
                "Day of Week": "Mon",
                "Day": int(pd.Timestamp(today).day),
                "Month": int(pd.Timestamp(today).month),
                "Year": int(pd.Timestamp(today).year),
                "Week Number": int(pd.Timestamp(today).isocalendar().week),
                "Week of": (pd.Timestamp(today) - pd.Timedelta(days=pd.Timestamp(today).weekday())).date().isoformat(),
                "Hours": hhmm,
                "Management_Line": "",
                "Location": "",
                "CSM": "",
                "Monthly": "",
                "Weekly": "",
                "Business Area": "",
                "Sub Business Area": "",
                "Channel": "Voice",
            }
        )
    return pd.DataFrame(demo)


def _bo_bucket(activity: str) -> str:
    try:
        if isinstance(activity, str):
            value = activity
        elif pd.isna(activity):
            value = ""
        else:
            value = str(activity)
    except Exception:
        value = ""
    text = value.strip().lower()
    if "divert" in text:
        return "diverted"
    if "down" in text or text == "downtime":
        return "downtime"
    if "staff complement" in text:
        return "staff_complement"
    if "flex" in text:
        return "flextime"
    if "lend" in text:
        return "lend_staff"
    if "borrow" in text:
        return "borrowed_staff"
    if "overtime" in text or text == "ot":
        return "overtime"
    if "core time" in text or text == "core":
        return "core_time"
    if "time worked" in text:
        return "time_worked"
    if "work out" in text or "workout" in text:
        return "work_out"
    return "other"


def normalize_shrinkage_bo(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    col_act = _pick_col(df, "Activity")
    col_sec = _pick_col(df, "DurationSeconds", "Duration (sec)", "duration_seconds")
    col_date = _pick_col(df, "Date")
    col_units = _pick_col(df, "Units")
    col_brid = _pick_col(df, "AgentID(BRID)", "StaffReferenceId", "SaffMemberId", "StaffMemberId", "BRID")
    col_fname = _pick_col(df, "SatffFirstName", "StaffFirstName", "FirstName")
    col_lname = _pick_col(df, "StaffLastName", "LastName")
    if not (col_act and col_sec and col_date and col_brid):
        return pd.DataFrame()

    out = df.copy()
    out = out.rename(
        columns={
            col_act: "activity",
            col_sec: "duration_seconds",
            col_date: "date",
            col_units if col_units else "units": "units",
            col_brid: "brid",
            col_fname or "": "first_name",
            col_lname or "": "last_name",
        }
    )
    out["date"] = _parse_date_series(out["date"])
    out["duration_seconds"] = pd.to_numeric(out["duration_seconds"], errors="coerce").fillna(0).astype(float)
    out["units"] = pd.to_numeric(out.get("units"), errors="coerce").fillna(0).astype(float)
    out["brid"] = out["brid"].astype(str).str.strip()

    hc = _hc_lookup()
    out["tl_name"] = out["brid"].map(lambda x: (hc.get(x) or {}).get("lm_name"))
    out["journey"] = out["brid"].map(lambda x: (hc.get(x) or {}).get("journey"))
    out["sub_business_area"] = out["brid"].map(lambda x: (hc.get(x) or {}).get("level_3"))
    out["time_hours"] = out["duration_seconds"] / 3600.0
    out["channel"] = "Back Office"
    return out


def summarize_shrinkage_bo(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    dff = df.copy()
    dff["date"] = pd.to_datetime(dff.get("date"), errors="coerce").dt.date
    dff["bucket"] = dff["activity"].map(_bo_bucket) if "activity" in dff.columns else "other"

    keys = ["date", "journey", "sub_business_area", "channel"]
    if "country" in dff.columns:
        keys.append("country")
    if "site" in dff.columns:
        keys.append("site")

    val_col = "time_hours" if "time_hours" in dff.columns else "duration_seconds"
    factor = 1.0 if val_col == "time_hours" else 1.0 / 3600.0

    agg = dff.groupby(keys + ["bucket"], dropna=False)[val_col].sum().reset_index()
    pivot = agg.pivot_table(index=keys, columns="bucket", values=val_col, fill_value=0.0).reset_index()

    def _col(frame: pd.DataFrame, names: list[str]) -> pd.Series:
        for name in names:
            if name in frame.columns:
                return frame[name]
        return pd.Series(0.0, index=frame.index)

    staff = _col(pivot, ["staff_complement"]) * factor
    dwn = _col(pivot, ["downtime"]) * factor
    flx = _col(pivot, ["flextime"]) * factor
    ot = _col(pivot, ["overtime"]) * factor
    bor = _col(pivot, ["borrowed_staff", "borrowed"]) * factor
    lnd = _col(pivot, ["lend_staff", "lend"]) * factor
    div = _col(pivot, ["diverted"]) * factor

    ttw = staff - dwn + flx + ot + bor - lnd

    pivot["OOO Hours"] = dwn
    pivot["In Office Hours"] = div
    pivot["Base Hours"] = staff
    pivot["TTW Hours"] = ttw

    pivot = pivot.rename(
        columns={
            "journey": "Business Area",
            "sub_business_area": "Sub Business Area",
            "channel": "Channel",
            "country": "Country",
            "site": "Site",
        }
    )
    keep_keys = [c for c in ["date", "Business Area", "Sub Business Area", "Channel", "Country", "Site"] if c in pivot.columns]
    keep = keep_keys + ["OOO Hours", "In Office Hours", "Base Hours", "TTW Hours"]
    return pivot[keep].sort_values(keep_keys)


def weekly_shrinkage_from_bo_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame(columns=SHRINK_WEEKLY_FIELDS)
    df = daily.copy()
    df["week"] = pd.to_datetime(df["date"], errors="coerce").dt.date.apply(lambda x: _week_floor(x, "Monday"))
    df["program"] = df.get("Business Area", "All").fillna("All").astype(str)

    grp = df.groupby(["week", "program"], as_index=False)[["OOO Hours", "In Office Hours", "Base Hours", "TTW Hours"]].sum()
    base = grp["Base Hours"].replace({0.0: np.nan})
    ttw = grp["TTW Hours"].replace({0.0: np.nan})
    grp["ooo_pct"] = np.where(base.gt(0), (grp["OOO Hours"] / base) * 100.0, np.nan)
    grp["ino_pct"] = np.where(ttw.gt(0), (grp["In Office Hours"] / ttw) * 100.0, np.nan)
    grp["overall_pct"] = grp["ooo_pct"].fillna(0.0) + grp["ino_pct"].fillna(0.0)
    grp = grp.rename(
        columns={
            "OOO Hours": "ooo_hours",
            "In Office Hours": "ino_hours",
            "Base Hours": "base_hours",
        }
    )
    return grp[SHRINK_WEEKLY_FIELDS]


def normalize_shrinkage_voice(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    col_date = _pick_col(df, "Date")
    col_state = _pick_col(df, "Superstate")
    col_hours = _pick_col(df, "Hours")
    col_brid = _pick_col(df, "BRID", "AgentID(BRID)", "Employee Id", "EmployeeID")
    if not (col_date and col_state and col_hours and col_brid):
        return pd.DataFrame()

    out = df.copy()
    out = out.rename(columns={col_date: "date", col_state: "superstate", col_hours: "hours_raw", col_brid: "brid"})
    out["date"] = _parse_date_series(out["date"])
    out["brid"] = out["brid"].astype(str).str.strip()
    mins = out["hours_raw"].map(_hhmm_to_minutes).fillna(0.0)
    out["hours"] = mins / 60.0

    hc = _hc_lookup()
    out["TL Name"] = out["brid"].map(lambda x: (hc.get(x) or {}).get("lm_name"))
    out["Site"] = out["brid"].map(lambda x: (hc.get(x) or {}).get("site"))
    out["City"] = out["brid"].map(lambda x: (hc.get(x) or {}).get("city"))
    out["Country"] = out["brid"].map(lambda x: (hc.get(x) or {}).get("country"))
    out["Business Area"] = out.get("Business Area", pd.Series(index=out.index)).fillna(
        out["brid"].map(lambda x: (hc.get(x) or {}).get("journey"))
    )
    out["Sub Business Area"] = out.get("Sub Business Area", pd.Series(index=out.index)).fillna(
        out["brid"].map(lambda x: (hc.get(x) or {}).get("level_3"))
    )
    if "Channel" not in out.columns:
        out["Channel"] = "Voice"

    for col, default in [("Business Area", "All"), ("Sub Business Area", "All"), ("Country", "All")]:
        out[col] = out[col].replace("", np.nan).fillna(default)
    out["Channel"] = out["Channel"].replace("", np.nan).fillna("Voice")
    return out


def summarize_shrinkage_voice(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    keys = ["date", "Business Area", "Sub Business Area", "Channel"]
    if "Country" in out.columns and out["Country"].notna().any():
        keys.append("Country")

    pivot = out.pivot_table(index=keys, columns="superstate", values="hours", aggfunc="sum", fill_value=0.0).reset_index()

    def _series(name: str) -> pd.Series:
        return pivot[name] if name in pivot.columns else pd.Series(0.0, index=pivot.index)

    ooo_codes = [
        "SC_ABSENCE_TOTAL",
        "SC_A_Sick_Long_Term",
        "SC_HOLIDAY",
        "SC_VACATION",
        "SC_LEAVE",
        "SC_UNPAID",
    ]
    ino_codes = [
        "SC_TRAINING_TOTAL",
        "SC_BREAKS",
        "SC_SYSTEM_EXCEPTION",
        "SC_MEETING",
        "SC_COACHING",
    ]

    pivot["OOO Hours"] = sum((_series(code) for code in ooo_codes))
    pivot["In Office Hours"] = sum((_series(code) for code in ino_codes))
    pivot["Base Hours"] = _series("SC_INCLUDED_TIME")

    keep = keys + ["OOO Hours", "In Office Hours", "Base Hours"]
    return pivot[keep].sort_values(keys)


def weekly_shrinkage_from_voice_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame(columns=SHRINK_WEEKLY_FIELDS)
    df = daily.copy()
    df["week"] = pd.to_datetime(df["date"], errors="coerce").dt.date.apply(lambda x: _week_floor(x, "Monday"))
    df["program"] = df["Business Area"].fillna("All").astype(str)
    agg = df.groupby(["week", "program"], as_index=False)[["OOO Hours", "In Office Hours", "Base Hours"]].sum()
    base = agg["Base Hours"].replace({0.0: np.nan})
    agg["ooo_pct"] = np.where(base.gt(0), (agg["OOO Hours"] / base) * 100.0, np.nan)
    agg["ino_pct"] = np.where(base.gt(0), (agg["In Office Hours"] / base) * 100.0, np.nan)
    agg["overall_pct"] = np.where(base.gt(0), ((agg["OOO Hours"] + agg["In Office Hours"]) / base) * 100.0, np.nan)
    agg = agg.rename(
        columns={
            "OOO Hours": "ooo_hours",
            "In Office Hours": "ino_hours",
            "Base Hours": "base_hours",
        }
    )
    return agg[SHRINK_WEEKLY_FIELDS]


def normalize_shrink_weekly(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif data is None:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=SHRINK_WEEKLY_FIELDS)
    rename_map = {}
    for col in df.columns:
        slug = _shrink_slug(col)
        if slug in _SHRINK_COLUMN_ALIASES:
            rename_map[col] = _SHRINK_COLUMN_ALIASES[slug]
    df = df.rename(columns=rename_map)
    for field in SHRINK_WEEKLY_FIELDS:
        if field not in df.columns:
            df[field] = np.nan if field != "program" else "All"
    df["program"] = df["program"].fillna("All").astype(str)
    df["week"] = pd.to_datetime(df["week"], errors="coerce").dt.date
    for col in ["ooo_hours", "ino_hours", "base_hours", "ooo_pct", "ino_pct", "overall_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _compute_shrink_weekly_percentages(df)
    return df[SHRINK_WEEKLY_FIELDS]


def _compute_shrink_weekly_percentages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    base = pd.to_numeric(out["base_hours"], errors="coerce")
    if "ooo_pct" not in out.columns or out["ooo_pct"].isna().all():
        out["ooo_pct"] = (pd.to_numeric(out["ooo_hours"], errors="coerce") / base) * 100.0
    if "ino_pct" not in out.columns or out["ino_pct"].isna().all():
        out["ino_pct"] = (pd.to_numeric(out["ino_hours"], errors="coerce") / base) * 100.0
    if "overall_pct" not in out.columns or out["overall_pct"].isna().all():
        out["overall_pct"] = (
            (pd.to_numeric(out["ooo_hours"], errors="coerce") + pd.to_numeric(out["ino_hours"], errors="coerce"))
            / base
        ) * 100.0
    return out


def merge_shrink_weekly(*frames: pd.DataFrame) -> pd.DataFrame:
    prepared = []
    for frame in frames:
        norm = normalize_shrink_weekly(frame)
        if isinstance(norm, pd.DataFrame) and not norm.empty:
            prepared.append(norm)
    if not prepared:
        return pd.DataFrame(columns=SHRINK_WEEKLY_FIELDS)

    combo = pd.concat(prepared, ignore_index=True)
    for col in ("ooo_hours", "ino_hours", "base_hours", "ooo_pct", "ino_pct", "overall_pct"):
        if col not in combo.columns:
            combo[col] = np.nan

    weights = pd.to_numeric(combo["base_hours"], errors="coerce").fillna(0.0)

    def _num(col: str) -> pd.Series:
        vals = pd.to_numeric(combo[col], errors="coerce")
        mask = (~vals.isna()) & (weights > 0)
        return vals.where(mask, 0.0) * weights.where(mask, 0.0)

    def _den(col: str) -> pd.Series:
        vals = pd.to_numeric(combo[col], errors="coerce")
        mask = (~vals.isna()) & (weights > 0)
        return weights.where(mask, 0.0)

    combo["_num_ooo"] = _num("ooo_pct")
    combo["_den_ooo"] = _den("ooo_pct")
    combo["_num_ino"] = _num("ino_pct")
    combo["_den_ino"] = _den("ino_pct")
    combo["_num_all"] = _num("overall_pct")
    combo["_den_all"] = _den("overall_pct")

    agg = combo.groupby(["week", "program"], as_index=False).agg(
        {
            "ooo_hours": "sum",
            "ino_hours": "sum",
            "base_hours": "sum",
            "_num_ooo": "sum",
            "_den_ooo": "sum",
            "_num_ino": "sum",
            "_den_ino": "sum",
            "_num_all": "sum",
            "_den_all": "sum",
        }
    )

    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace({0.0: np.nan})
        return num / den

    agg["ooo_pct"] = _safe_div(agg["_num_ooo"], agg["_den_ooo"]).astype(float)
    agg["ino_pct"] = _safe_div(agg["_num_ino"], agg["_den_ino"]).astype(float)
    agg["overall_pct"] = _safe_div(agg["_num_all"], agg["_den_all"]).astype(float)
    agg = agg.drop(columns=["_num_ooo", "_den_ooo", "_num_ino", "_den_ino", "_num_all", "_den_all"])
    agg = _compute_shrink_weekly_percentages(agg)
    agg = agg.sort_values(["week", "program"]).reset_index(drop=True)
    return agg[SHRINK_WEEKLY_FIELDS]


def load_shrinkage_weekly() -> pd.DataFrame:
    ensure_shrinkage_schema()
    if not has_dsn():
        return pd.DataFrame(columns=SHRINK_WEEKLY_FIELDS)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT week, program, ooo_hours, ino_hours, base_hours, ooo_pct, ino_pct, overall_pct
            FROM shrinkage_weekly_entries
            ORDER BY week, program
            """
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def save_shrinkage_weekly(df: pd.DataFrame) -> int:
    ensure_shrinkage_schema()
    if not has_dsn():
        return 0
    data = normalize_shrink_weekly(df)
    if data.empty:
        return 0
    data["week"] = pd.to_datetime(data["week"], errors="coerce").dt.date
    with db_conn() as conn:
        conn.execute("DELETE FROM shrinkage_weekly_entries")
        conn.executemany(
            """
            INSERT INTO shrinkage_weekly_entries (
                week, program, ooo_hours, ino_hours, base_hours, ooo_pct, ino_pct, overall_pct
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    row.get("week"),
                    row.get("program"),
                    row.get("ooo_hours"),
                    row.get("ino_hours"),
                    row.get("base_hours"),
                    row.get("ooo_pct"),
                    row.get("ino_pct"),
                    row.get("overall_pct"),
                )
                for _, row in data.iterrows()
            ],
        )
    return int(len(data.index))


def load_shrinkage_raw(kind: str) -> pd.DataFrame:
    ensure_shrinkage_schema()
    if not has_dsn() or not kind:
        return pd.DataFrame()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT payload FROM shrinkage_raw_entries WHERE kind = %s ORDER BY id",
            (kind,),
        )
        rows = cur.fetchall()
    payloads = [row[0] if isinstance(row, tuple) else row for row in rows]
    return pd.DataFrame(payloads) if payloads else pd.DataFrame()


def save_shrinkage_raw(kind: str, df: pd.DataFrame) -> int:
    ensure_shrinkage_schema()
    if not has_dsn() or not kind:
        return 0
    rows = df.to_dict("records") if isinstance(df, pd.DataFrame) else []
    with db_conn() as conn:
        conn.execute("DELETE FROM shrinkage_raw_entries WHERE kind = %s", (kind,))
        if rows:
            conn.executemany(
                "INSERT INTO shrinkage_raw_entries (kind, payload) VALUES (%s, %s)",
                [(kind, Json(_json_safe_row(row))) for row in rows],
            )
    return int(len(rows))


def weekly_avg_active_fte_from_roster(week_start: str = "Monday") -> pd.DataFrame:
    roster = load_roster_supply()
    if isinstance(roster, pd.DataFrame) and (not roster.empty) and {"start_date", "fte"}.issubset(roster.columns):
        def _to_date(val):
            try:
                return pd.to_datetime(val, errors="coerce").date()
            except Exception:
                return None

        r = roster.copy()
        r["sd"] = r["start_date"].apply(_to_date)
        r["ed"] = r.get("end_date", pd.Series([None] * len(r))).apply(_to_date)
        sd_min = min([d for d in r["sd"].dropna()] or [dt.date.today()])
        ed_max = max([d for d in r["ed"].dropna()] or [dt.date.today() + dt.timedelta(days=180)])
        if ed_max < sd_min:
            ed_max = sd_min + dt.timedelta(days=180)
        days = pd.date_range(sd_min, ed_max, freq="D").date
        rows = []
        for _, row in r.iterrows():
            sd = row["sd"] or sd_min
            ed = row["ed"] or ed_max
            fte = float(row.get("fte", 0) or 0)
            if fte <= 0:
                continue
            for day in days:
                if sd <= day <= ed:
                    rows.append({"date": day, "fte": fte})
        if not rows:
            return pd.DataFrame(columns=["week", "avg_active_fte"])
        daily = pd.DataFrame(rows).groupby("date", as_index=False)["fte"].sum()
        daily["week"] = daily["date"].apply(lambda x: _week_floor(x, week_start))
        weekly = daily.groupby("week", as_index=False)["fte"].mean().rename(columns={"fte": "avg_active_fte"})
        return weekly.sort_values("week")

    long = load_roster_long()
    if long is None or long.empty or "date" not in long.columns:
        return pd.DataFrame(columns=["week", "avg_active_fte"])
    df = long.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    brid_col = "BRID" if "BRID" in df.columns else ("employee_id" if "employee_id" in df.columns else None)
    if not brid_col:
        return pd.DataFrame(columns=["week", "avg_active_fte"])
    daily = df.groupby("date", as_index=False)[brid_col].nunique().rename(columns={brid_col: "fte"})
    daily["week"] = daily["date"].apply(lambda x: _week_floor(x, week_start))
    weekly = daily.groupby("week", as_index=False)["fte"].mean().rename(columns={"fte": "avg_active_fte"})
    return weekly.sort_values("week")


def attrition_weekly_from_raw(df_raw: pd.DataFrame, week_start: str = "Monday") -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["week", "leavers_fte", "avg_active_fte", "attrition_pct", "program"])
    df = df_raw.copy()
    if "Resignation Date" not in df.columns:
        if "Reporting Full Date" in df.columns:
            df["Resignation Date"] = df["Reporting Full Date"]
        else:
            return pd.DataFrame(columns=["week", "leavers_fte", "avg_active_fte", "attrition_pct", "program"])
    df = df[~df["Resignation Date"].isna()].copy()
    if "FTE" not in df.columns:
        df["FTE"] = 1.0

    program_series = None
    hc = load_headcount()
    if "BRID" in df.columns and isinstance(hc, pd.DataFrame) and not hc.empty:
        brid_col = _pick_col(hc, "brid", "employee id", "employee_id")
        journey_col = _pick_col(hc, "journey", "business area", "level 0")
        if brid_col and journey_col:
            map_brid = dict(zip(hc[brid_col].astype(str), hc[journey_col].astype(str)))
            program_series = df["BRID"].astype(str).map(lambda x: map_brid.get(x))

    if program_series is None or program_series.isna().all():
        lower = [str(c).strip().lower() for c in df.columns]
        if any(name in lower for name in ["org unit", "business area", "journey"]):
            pick = df.columns[lower.index(next(name for name in ["org unit", "business area", "journey"] if name in lower))]
            program_series = df[pick]
        else:
            program_series = pd.Series(["All"] * len(df))

    df["program"] = program_series.fillna("All").astype(str)
    df["week"] = df["Resignation Date"].apply(lambda x: _week_floor(x, week_start))
    wk = df.groupby(["week", "program"], as_index=False)["FTE"].sum().rename(columns={"FTE": "leavers_fte"})
    den = weekly_avg_active_fte_from_roster(week_start=week_start)
    out = wk.merge(den, on="week", how="left")
    out["attrition_pct"] = (out["leavers_fte"] / out["avg_active_fte"].replace({0: np.nan})) * 100.0
    out["attrition_pct"] = out["attrition_pct"].round(2)
    keep = ["week", "leavers_fte", "avg_active_fte", "attrition_pct", "program"]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan if col != "program" else "All"
    return out[keep].sort_values(["week", "program"])


def load_attrition_weekly() -> pd.DataFrame:
    ensure_shrinkage_schema()
    if not has_dsn():
        return pd.DataFrame(columns=["week", "attrition_pct", "program"])
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT week, program, leavers_fte, avg_active_fte, attrition_pct
            FROM attrition_weekly_entries
            ORDER BY week, program
            """
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def save_attrition_weekly(df: pd.DataFrame) -> int:
    ensure_shrinkage_schema()
    if not has_dsn():
        return 0
    data = pd.DataFrame(df or [])
    if data.empty:
        return 0
    if "program" not in data.columns:
        data["program"] = "All"
    data["week"] = pd.to_datetime(data["week"], errors="coerce").dt.date
    for col in ("leavers_fte", "avg_active_fte", "attrition_pct"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    with db_conn() as conn:
        conn.execute("DELETE FROM attrition_weekly_entries")
        conn.executemany(
            """
            INSERT INTO attrition_weekly_entries (
                week, program, leavers_fte, avg_active_fte, attrition_pct
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (
                    row.get("week"),
                    row.get("program"),
                    row.get("leavers_fte"),
                    row.get("avg_active_fte"),
                    row.get("attrition_pct"),
                )
                for _, row in data.iterrows()
            ],
        )
    return int(len(data.index))


def load_attrition_raw() -> pd.DataFrame:
    ensure_shrinkage_schema()
    if not has_dsn():
        return pd.DataFrame()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT payload FROM attrition_raw_entries ORDER BY id")
        rows = cur.fetchall()
    payloads = [row[0] if isinstance(row, tuple) else row for row in rows]
    return pd.DataFrame(payloads) if payloads else pd.DataFrame()


def save_attrition_raw(df: pd.DataFrame) -> int:
    ensure_shrinkage_schema()
    if not has_dsn():
        return 0
    rows = df.to_dict("records") if isinstance(df, pd.DataFrame) else []
    with db_conn() as conn:
        conn.execute("DELETE FROM attrition_raw_entries")
        if rows:
            conn.executemany(
                "INSERT INTO attrition_raw_entries (payload) VALUES (%s)",
                [(Json(_json_safe_row(row)),) for row in rows],
            )
    return int(len(rows))
