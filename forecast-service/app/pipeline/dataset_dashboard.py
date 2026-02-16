from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.pipeline.capacity_core import required_fte_daily, supply_fte_daily
from app.pipeline.headcount import CHANNEL_LIST, _hcu_cols, _hcu_df
from app.pipeline.ops_store import load_hiring, load_roster, load_timeseries_any
from app.pipeline.settings_store import load_settings


def _today_range(days: int = 56) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=days)
    return start, end


def _normalize_channel(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if "voice" in text or text in {"inbound", "call", "telephony"}:
        return "Voice"
    if "back office" in text or "backoffice" in text or text in {"bo", "backoffice"}:
        return "Back Office"
    if "outbound" in text or "out bound" in text or text in {"ob"}:
        return "Outbound"
    if "blended" in text:
        return "Blended"
    if "message" in text or "msg" in text:
        return "MessageUs"
    if "chat" in text:
        return "Chat"
    return str(value).strip()


def _hc_dim_df() -> pd.DataFrame:
    df = _hcu_df()
    if df is None or df.empty:
        return pd.DataFrame(columns=["Business Area", "Sub Business Area", "Channel", "Location", "Site"])
    cols = _hcu_cols(df)
    out = pd.DataFrame()
    out["Business Area"] = df[cols["ba"]].astype(str) if cols.get("ba") in df.columns else ""
    out["Sub Business Area"] = df[cols["sba"]].astype(str) if cols.get("sba") in df.columns else ""
    out["Channel"] = df[cols["lob"]].astype(str) if cols.get("lob") in df.columns else ""
    out["Location"] = df[cols["loc"]].astype(str) if cols.get("loc") in df.columns else ""
    out["Site"] = df[cols["site"]].astype(str) if cols.get("site") in df.columns else ""
    for col in out.columns:
        out[col] = out[col].fillna("").astype(str).str.strip()
    if "Channel" in out.columns:
        out["Channel"] = out["Channel"].map(_normalize_channel).fillna(out["Channel"])
    if "Site" in out.columns and out["Site"].replace("", pd.NA).dropna().empty:
        # Fallback for headcount files where site is present under alternative column names.
        best_col = None
        best_count = 0
        for col in df.columns:
            key = "".join(ch for ch in str(col).strip().lower() if ch.isalnum())
            if not key:
                continue
            if any(token in key for token in ("site", "building", "locationbuilding", "locationbuildingdescription")):
                series = df[col].astype(str).str.strip()
                count = int(series.replace("", pd.NA).dropna().shape[0])
                if count > best_count:
                    best_count = count
                    best_col = col
        if best_col:
            out["Site"] = df[best_col].astype(str).fillna("").str.strip()
    return out


def _resolve_settings(
    ba: Optional[str],
    sba: Optional[str],
    ch: Optional[str],
    site: Optional[str],
    loc: Optional[str],
) -> dict:
    if ba and sba and ch:
        return load_settings("hier", None, ba, sba, ch, site)
    if loc:
        return load_settings("location", loc, None, None, None, None)
    return load_settings("global", None, None, None, None, None)


def _load_voice(scopes: list[str], pref: str = "auto") -> pd.DataFrame:
    pref = (pref or "auto").lower()
    if pref == "forecast":
        df = load_timeseries_any("voice_forecast", scopes)
        if df.empty:
            df = load_timeseries_any("voice_actual", scopes)
    else:
        df = load_timeseries_any("voice_actual", scopes)
        if df.empty:
            df = load_timeseries_any("voice_forecast", scopes)
    if df.empty:
        df = load_timeseries_any("voice_tactical", scopes)
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = pd.NA
    if "aht_sec" not in out.columns:
        out["aht_sec"] = 300.0
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    return out


def _load_bo(scopes: list[str], pref: str = "auto") -> pd.DataFrame:
    pref = (pref or "auto").lower()
    if pref == "forecast":
        df = load_timeseries_any("bo_forecast", scopes)
        if df.empty:
            df = load_timeseries_any("bo_actual", scopes)
    else:
        df = load_timeseries_any("bo_actual", scopes)
        if df.empty:
            df = load_timeseries_any("bo_forecast", scopes)
    if df.empty:
        df = load_timeseries_any("bo_tactical", scopes)
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "items" not in out.columns:
        out["items"] = out["volume"] if "volume" in out.columns else 0.0
    out["items"] = pd.to_numeric(out["items"], errors="coerce").fillna(0.0)
    if "sut_sec" not in out.columns:
        out["sut_sec"] = pd.NA
    out["sut_sec"] = pd.to_numeric(out["sut_sec"], errors="coerce")
    if "aht_sec" in out.columns:
        aht_sec = pd.to_numeric(out["aht_sec"], errors="coerce")
        # BO uploads are often stored under aht_sec. Use it when sut_sec is blank/zero.
        out["sut_sec"] = out["sut_sec"].where(out["sut_sec"].fillna(0.0) > 0.0, aht_sec)
    out["sut_sec"] = out["sut_sec"].fillna(600.0)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    return out


def dataset_snapshot(
    start_date: Optional[str],
    end_date: Optional[str],
    pref: str,
    ba: list[str],
    sba: list[str],
    ch: list[str],
    loc: list[str],
    site: list[str],
) -> dict[str, Any]:
    try:
        start = pd.to_datetime(start_date).date() if start_date else _today_range(56)[0]
        end = pd.to_datetime(end_date).date() if end_date else _today_range(56)[1]
    except Exception:
        start, end = _today_range(56)

    dims = _hc_dim_df()
    if "Channel" in dims.columns:
        dims["Channel"] = dims["Channel"].replace("", "Voice")
        dims["Channel"] = dims["Channel"].map(_normalize_channel).fillna(dims["Channel"])

    def _filter(col: str, values: list[str]):
        nonlocal dims
        if values:
            if col == "Channel":
                lowered = {_normalize_channel(v).strip().lower() for v in values if str(v).strip()}
            else:
                lowered = {str(v).strip().lower() for v in values if str(v).strip()}
            dims = dims[dims[col].astype(str).str.strip().str.lower().isin(lowered)]

    _filter("Business Area", ba)
    _filter("Sub Business Area", sba)
    if not dims.empty and "Channel" in dims.columns:
        if not dims["Channel"].replace("", pd.NA).dropna().empty and ch:
            dims_before = dims
            _filter("Channel", ch)
            # Keep BA/SBA/site scopes when channel labels are incomplete or mismatched.
            if dims.empty:
                dims = dims_before
    _filter("Location", loc)
    _filter("Site", site)

    if dims.empty:
        return {"rows": [], "chart": []}

    has_site_filter = bool(site)
    channel_list = [_normalize_channel(x).strip() for x in ch if str(x).strip()] if ch else list(CHANNEL_LIST)
    if has_site_filter and "Site" in dims.columns:
        d = dims[["Business Area", "Sub Business Area", "Channel", "Site"]].copy()
        d["Channel"] = d["Channel"].replace("", "Voice")
        d = d[d["Site"].astype(str).str.strip() != ""]
        d["sk"] = (
            d["Business Area"].str.strip()
            + "|"
            + d["Sub Business Area"].str.strip()
            + "|"
            + d["Channel"].str.strip()
            + "|"
            + d["Site"].str.strip()
        ).str.lower()
    else:
        d = dims[["Business Area", "Sub Business Area", "Channel"]].copy()
        d["Channel"] = d["Channel"].replace("", "Voice")
        d["sk"] = (
            d["Business Area"].str.strip()
            + "|"
            + d["Sub Business Area"].str.strip()
            + "|"
            + d["Channel"].str.strip()
        ).str.lower()

    scopes = sorted(d["sk"].dropna().unique().tolist())
    voice = _load_voice(scopes, pref=pref)
    bo = _load_bo(scopes, pref=pref)

    if not voice.empty:
        voice = voice[pd.notna(voice["date"])]
        voice = voice[(voice["date"] >= start) & (voice["date"] <= end)]
    if not bo.empty:
        bo = bo[pd.notna(bo["date"])]
        bo = bo[(bo["date"] >= start) & (bo["date"] <= end)]

    ba_arg = ba[0] if ba else None
    sba_arg = sba[0] if sba else None
    ch_arg = ch[0] if ch else None
    loc_arg = loc[0] if loc else None
    site_arg = site[0] if site else None
    settings = _resolve_settings(ba_arg, sba_arg, ch_arg, site_arg, loc_arg) or {}
    req_df = required_fte_daily(voice, bo, pd.DataFrame(), settings)
    if not req_df.empty:
        req_df = req_df.groupby(["date", "program"], as_index=False)["total_req_fte"].sum()

    roster = load_roster()
    hiring = load_hiring()
    sup_df = supply_fte_daily(roster, hiring)
    if not sup_df.empty:
        sup_df["date"] = pd.to_datetime(sup_df["date"], errors="coerce").dt.date
        sup_df = sup_df[pd.notna(sup_df["date"])]
        sup_df = sup_df[(sup_df["date"] >= start) & (sup_df["date"] <= end)]
        if not sup_df.empty:
            sup_df = sup_df.groupby(["date", "program"], as_index=False)["supply_fte"].sum()

    df = pd.merge(req_df, sup_df, on=["date", "program"], how="outer").fillna(
        {"total_req_fte": 0.0, "supply_fte": 0.0}
    )
    req_fte = pd.to_numeric(df["total_req_fte"], errors="coerce")
    sup_fte = pd.to_numeric(df["supply_fte"], errors="coerce")
    df["total_req_fte"] = req_fte.fillna(0.0)
    df["supply_fte"] = sup_fte.fillna(0.0)
    df["staffing_pct"] = np.nan
    valid_req = req_fte > 0
    if valid_req.any():
        df.loc[valid_req, "staffing_pct"] = (sup_fte[valid_req] / req_fte[valid_req]) * 100.0
    df = df.sort_values(["date", "program"]) if not df.empty else df

    daily = (
        df.groupby("date", as_index=False)[["total_req_fte", "supply_fte"]].sum()
        if not df.empty
        else pd.DataFrame(columns=["date", "total_req_fte", "supply_fte"])
    )
    for col in ["total_req_fte", "supply_fte"]:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")

    return {"rows": df.to_dict("records"), "chart": daily.to_dict("records")}
