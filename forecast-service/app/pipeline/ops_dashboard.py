from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable, Optional

import pandas as pd

from app.pipeline.capacity_core import required_fte_daily, supply_fte_daily, voice_requirements_interval
from app.pipeline.headcount import CHANNEL_LIST, _hcu_cols, _hcu_df
from app.pipeline.ops_store import list_timeseries_scope_keys, load_hiring, load_roster, load_timeseries_any
from app.pipeline.settings_store import load_settings


def _today_range(default_days: int = 28) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=default_days)
    return start, end


def _listify(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        return [str(v) for v in values if str(v).strip()]
    text = str(values).strip()
    if not text:
        return []
    return [text]


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
    return out


def _scope_keys_from_filters(
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> pd.DataFrame:
    df = _hc_dim_df().copy()

    def _apply(col: str, values: list[str]):
        nonlocal df
        if values:
            lowered = {str(v).strip().lower() for v in values if str(v).strip()}
            if lowered:
                df = df[df[col].astype(str).str.strip().str.lower().isin(lowered)]

    _apply("Business Area", ba)
    _apply("Sub Business Area", sba)
    _apply("Channel", ch)
    _apply("Site", site)
    _apply("Location", loc)

    if df.empty:
        return pd.DataFrame(columns=["ba", "sba", "ch", "loc", "site", "sk"])

    ch_list = [str(x).strip() for x in ch if str(x).strip()] if ch else list(CHANNEL_LIST)
    rows = []
    for _, row in df.iterrows():
        ba_v = str(row.get("Business Area", "")).strip()
        sba_v = str(row.get("Sub Business Area", "")).strip()
        loc_v = str(row.get("Location", "")).strip()
        site_v = str(row.get("Site", "")).strip()
        for ch_v in ch_list:
            sk = f"{ba_v}|{sba_v}|{str(ch_v).strip()}".lower()
            rows.append({"ba": ba_v, "sba": sba_v, "ch": str(ch_v), "loc": loc_v, "site": site_v, "sk": sk})
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["sk", "site", "loc"]) if not out.empty else out
    return out[["ba", "sba", "ch", "loc", "site", "sk"]]


def _scopes_from_datasets(
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> pd.DataFrame:
    kinds = [
        "voice_actual",
        "voice_forecast",
        "voice_tactical",
        "bo_actual",
        "bo_forecast",
        "bo_tactical",
    ]
    scope_keys = list_timeseries_scope_keys(kinds)
    if not scope_keys:
        return pd.DataFrame(columns=["ba", "sba", "ch", "loc", "site", "sk"])

    def _norm(values: list[str]) -> set[str]:
        return {str(v).strip().lower() for v in values if str(v).strip()}

    ba_f = _norm(ba)
    sba_f = _norm(sba)
    ch_f = _norm(ch)
    site_f = _norm(site)

    rows = []
    for raw in sorted(set(scope_keys)):
        if not raw or "|" not in raw:
            continue
        if str(raw).strip().lower().startswith("location|"):
            continue
        parts = [p.strip() for p in str(raw or "").split("|")]
        ba_v = parts[0] if len(parts) > 0 else ""
        sba_v = parts[1] if len(parts) > 1 else ""
        ch_v = parts[2] if len(parts) > 2 else ""
        site_v = parts[3] if len(parts) > 3 else ""
        if ba_f and ba_v.strip().lower() not in ba_f:
            continue
        if sba_f and sba_v.strip().lower() not in sba_f:
            continue
        if ch_f and ch_v.strip().lower() not in ch_f:
            continue
        if site_f and site_v.strip().lower() not in site_f:
            continue
        rows.append({"ba": ba_v, "sba": sba_v, "ch": ch_v, "loc": "", "site": site_v, "sk": raw})
    out = pd.DataFrame(rows)
    return out[["ba", "sba", "ch", "loc", "site", "sk"]] if not out.empty else pd.DataFrame(columns=["ba", "sba", "ch", "loc", "site", "sk"])


def _dataset_sites_all() -> list[str]:
    kinds = ["voice_actual", "voice_forecast", "voice_tactical", "bo_actual", "bo_forecast", "bo_tactical"]
    scope_keys = list_timeseries_scope_keys(kinds)
    sites: set[str] = set()
    for raw in scope_keys:
        if not raw or "|" not in raw:
            continue
        if str(raw).strip().lower().startswith("location|"):
            continue
        parts = [p.strip() for p in str(raw or "").split("|")]
        if len(parts) >= 4 and parts[3]:
            sites.add(parts[3])
    if not sites:
        return []
    country_block = {"india", "uk", "united kingdom", "great britain", "england", "scotland", "wales"}
    return sorted([s for s in sites if s and s.strip().lower() not in country_block])


def ops_options(ba: list[str], sba: list[str], ch: list[str], loc: list[str]) -> dict:
    dim = _hc_dim_df()
    opts_ba = sorted(dim["Business Area"].dropna().unique().tolist())
    opts_sba = sorted(dim["Sub Business Area"].dropna().unique().tolist())
    opts_loc = sorted(dim["Location"].dropna().unique().tolist())
    opts_ch = list(CHANNEL_LIST)

    if ba:
        ba_set = {str(v).strip().lower() for v in ba if str(v).strip()}
        opts_sba = sorted(
            dim.loc[dim["Business Area"].astype(str).str.strip().str.lower().isin(ba_set), "Sub Business Area"]
            .dropna()
            .unique()
            .tolist()
        )

    site_dim = dim.copy()
    if ba:
        ba_set = {str(v).strip().lower() for v in ba if str(v).strip()}
        site_dim = site_dim[site_dim["Business Area"].astype(str).str.strip().str.lower().isin(ba_set)]
    if sba:
        sba_set = {str(v).strip().lower() for v in sba if str(v).strip()}
        site_dim = site_dim[site_dim["Sub Business Area"].astype(str).str.strip().str.lower().isin(sba_set)]
    if ch:
        ch_set = {str(v).strip().lower() for v in ch if str(v).strip()}
        site_dim = site_dim[site_dim["Channel"].astype(str).str.strip().str.lower().isin(ch_set)]
    if loc:
        loc_set = {str(v).strip().lower() for v in loc if str(v).strip()}
        site_dim = site_dim[site_dim["Location"].astype(str).str.strip().str.lower().isin(loc_set)]

    site_list = sorted([x for x in site_dim["Site"].dropna().unique().tolist() if x]) if "Site" in site_dim.columns else []
    loc_set = {str(x).strip().lower() for x in opts_loc}
    country_block = {"india", "uk", "united kingdom", "great britain", "england", "scotland", "wales"}
    site_list = [s for s in site_list if s and (sl := str(s).strip().lower()) not in loc_set and sl not in country_block]
    if not site_list:
        ds_map = _scopes_from_datasets(ba, sba, ch, [], loc)
        if not ds_map.empty and "site" in ds_map.columns:
            site_list = sorted([s for s in ds_map["site"].astype(str).dropna().str.strip().unique().tolist() if s])
        if not site_list:
            site_list = _dataset_sites_all()
        loc_set = {str(x).strip().lower() for x in opts_loc}
        site_list = [
            s
            for s in site_list
            if s
            and (sl := str(s).strip().lower()) not in loc_set
            and sl not in country_block
        ]

    return {
        "business_areas": opts_ba,
        "sub_business_areas": opts_sba,
        "locations": opts_loc,
        "sites": site_list,
        "channels": opts_ch,
    }


def _resolve_settings(ba: Optional[str], sba: Optional[str], ch: Optional[str], site: Optional[str], loc: Optional[str]) -> dict:
    if ba and sba and ch:
        settings = load_settings("hier", None, ba, sba, ch, site)
        if settings:
            return settings
        if site:
            return load_settings("hier", None, ba, sba, ch, None)
    if loc:
        return load_settings("location", loc, None, None, None, None)
    return load_settings("global", None, None, None, None, None)


def _load_voice(scopes: list[str]) -> pd.DataFrame:
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


def _load_bo(scopes: list[str]) -> pd.DataFrame:
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
    if "sut_sec" not in out.columns:
        out["sut_sec"] = 600.0
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    return out


def _agg_by_grain(df: pd.DataFrame, date_col: str, grain: str) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce").dt.date
    if grain in ("D", "daily"):
        d["bucket"] = d[date_col]
    elif grain in ("W", "weekly"):
        d["bucket"] = pd.to_datetime(d[date_col]).dt.to_period("W-MON").dt.start_time.dt.date
    elif grain in ("M", "monthly"):
        d["bucket"] = pd.to_datetime(d[date_col]).dt.to_period("M").dt.start_time.dt.date
    elif grain in ("Q", "quarterly"):
        d["bucket"] = pd.to_datetime(d[date_col]).dt.to_period("Q").dt.start_time.dt.date
    elif grain in ("Y", "yearly"):
        d["bucket"] = pd.to_datetime(d[date_col]).dt.to_period("Y").dt.start_time.dt.date
    else:
        d["bucket"] = d[date_col]
    return d


def refresh_ops(
    start_date: Optional[str],
    end_date: Optional[str],
    grain: str,
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> dict:
    try:
        start = pd.to_datetime(start_date).date() if start_date else _today_range(28)[0]
        end = pd.to_datetime(end_date).date() if end_date else _today_range(28)[1]
    except Exception:
        start, end = _today_range(28)

    map_df = _scope_keys_from_filters(ba, sba, ch, site, loc)
    ds_map = _scopes_from_datasets(ba, sba, ch, site, loc)
    if not ds_map.empty:
        map_df = ds_map
    elif map_df.empty:
        map_df = ds_map
    scopes = map_df["sk"].unique().tolist() if not map_df.empty else []

    voice = _load_voice(scopes)
    bo = _load_bo(scopes)

    if not voice.empty:
        voice["date"] = pd.to_datetime(voice["date"], errors="coerce").dt.date
        voice = voice[pd.notna(voice["date"])]
        voice = voice[(voice["date"] >= start) & (voice["date"] <= end)]
    if not bo.empty:
        bo["date"] = pd.to_datetime(bo["date"], errors="coerce").dt.date
        bo = bo[pd.notna(bo["date"])]
        bo = bo[(bo["date"] >= start) & (bo["date"] <= end)]

    ba_arg = ba[0] if ba else None
    sba_arg = sba[0] if sba else None
    ch_arg = ch[0] if ch else None
    loc_arg = loc[0] if loc else None
    site_arg = site[0] if site else None
    settings = _resolve_settings(ba_arg, sba_arg, ch_arg, site_arg, loc_arg)

    req_day = required_fte_daily(voice, bo, pd.DataFrame(), settings)
    if not req_day.empty:
        req_day = req_day.groupby("date", as_index=False)["total_req_fte"].sum()

    roster = load_roster()
    hiring = load_hiring()

    def _maybe_filter(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()

        def _filter(col: str, values: list[str]):
            nonlocal out
            if col in out.columns and values:
                sset = {str(v) for v in values if str(v).strip()}
                out = out[out[col].astype(str).isin(sset)]

        _filter("program", ba)
        _filter("Business Area", ba)
        _filter("Sub Business Area", sba)
        _filter("LOB", ch)
        _filter("Channel", ch)
        _filter("site", site)
        _filter("Site", site)
        _filter("location", loc)
        _filter("Location", loc)
        _filter("country", loc)
        _filter("Country", loc)
        return out

    roster_f = _maybe_filter(roster)
    hiring_f = _maybe_filter(hiring)
    supply = supply_fte_daily(roster_f, hiring_f)
    if not supply.empty:
        supply["date"] = pd.to_datetime(supply["date"], errors="coerce").dt.date
        supply = supply[pd.notna(supply["date"])]
        supply = supply[(supply["date"] >= start) & (supply["date"] <= end)]
        supply = supply.groupby("date", as_index=False)["supply_fte"].sum()

    kpi_req = float(req_day["total_req_fte"].sum()) if not req_day.empty else 0.0
    kpi_sup = float(supply["supply_fte"].sum()) if not supply.empty else 0.0
    kpi_gap = kpi_req - kpi_sup

    line_series = []
    if grain == "interval" and not voice.empty and "interval" in voice.columns and voice["interval"].notna().any():
        vi = voice_requirements_interval(voice, settings)
        if not vi.empty:
            vi = vi.sort_values(["date", "interval"])
            vi["ts"] = pd.to_datetime(vi["date"]).astype(str) + " " + vi["interval"].astype(str)
            line_series = [
                {
                    "name": "Agents Required",
                    "points": [{"x": str(x), "y": float(y)} for x, y in zip(vi["ts"], vi["agents_req"])],
                }
            ]
        else:
            line_series = []
        line_x = [pt["x"] for pt in line_series[0]["points"]] if line_series else []
    else:
        r_df = req_day.copy() if not req_day.empty else pd.DataFrame(columns=["date", "total_req_fte"])
        s_df = supply.copy() if not supply.empty else pd.DataFrame(columns=["date", "supply_fte"])
        if not r_df.empty:
            r_df = _agg_by_grain(r_df, "date", grain).groupby("bucket", as_index=False)["total_req_fte"].sum()
            r_df = r_df.sort_values("bucket")
        if not s_df.empty:
            s_df = _agg_by_grain(s_df, "date", grain).groupby("bucket", as_index=False)["supply_fte"].sum()
            s_df = s_df.sort_values("bucket")
        line_x = []
        if not r_df.empty:
            line_x = [str(x) for x in r_df["bucket"].tolist()]
            line_series.append(
                {"name": "Required FTE", "points": [{"x": str(x), "y": float(y)} for x, y in zip(r_df["bucket"], r_df["total_req_fte"])]}
            )
        if not s_df.empty:
            if not line_x:
                line_x = [str(x) for x in s_df["bucket"].tolist()]
            line_series.append(
                {"name": "Supply FTE", "points": [{"x": str(x), "y": float(y)} for x, y in zip(s_df["bucket"], s_df["supply_fte"])]}
            )

    line = {"x": line_x, "series": line_series}

    bar_labels: list[str] = []
    bar_series = []
    bar_df = pd.DataFrame()
    if not voice.empty:
        v_day = voice.copy()
        if grain == "interval" and "interval" in v_day.columns and v_day["interval"].notna().any():
            v_day["bucket"] = v_day["date"].astype(str) + " " + v_day["interval"].astype(str)
        else:
            v_day = _agg_by_grain(v_day, "date", grain)
        v_agg = v_day.groupby("bucket", as_index=False)["volume"].sum().rename(columns={"volume": "Voice Calls"})
        bar_df = v_agg
    if not bo.empty:
        b_day = _agg_by_grain(bo.copy(), "date", grain)
        b_agg = b_day.groupby("bucket", as_index=False)["items"].sum().rename(columns={"items": "BO Items"})
        bar_df = b_agg if bar_df.empty else pd.merge(bar_df, b_agg, on="bucket", how="outer")
    if not bar_df.empty:
        bar_df = bar_df.fillna(0.0).sort_values("bucket")
        bar_labels = [str(x) for x in bar_df["bucket"].tolist()]
        for col in [c for c in bar_df.columns if c != "bucket"]:
            bar_series.append({"name": col, "values": bar_df[col].astype(float).tolist()})

    pie_df = pd.DataFrame(columns=["ch", "val"])
    if not voice.empty:
        vc = voice.copy()
        if "scope_key" in vc.columns:
            vc["ch"] = vc["scope_key"].astype(str).str.split("|").str[2].fillna("All")
        else:
            vc["ch"] = "All"
        pie_df = vc.groupby("ch", as_index=False)["volume"].sum().rename(columns={"volume": "val"})
    if not bo.empty:
        bc = bo.copy()
        if "scope_key" in bc.columns:
            bc["ch"] = bc["scope_key"].astype(str).str.split("|").str[2].fillna("All")
        else:
            bc["ch"] = "All"
        bo_p = bc.groupby("ch", as_index=False)["items"].sum().rename(columns={"items": "val"})
        pie_df = bo_p if pie_df.empty else pd.concat([pie_df, bo_p]).groupby("ch", as_index=False)["val"].sum()

    pie = {
        "labels": pie_df["ch"].astype(str).tolist() if not pie_df.empty else [],
        "values": pie_df["val"].astype(float).tolist() if not pie_df.empty else [],
    }

    site = {"labels": [], "values": []}
    if not map_df.empty:
        key_map = map_df.drop_duplicates(subset=["sk", "site"])[["sk", "site"]]
        site_df = pd.DataFrame(columns=["site", "val"])
        if "scope_key" in voice.columns:
            v_site = (
                voice.groupby("scope_key", as_index=False)["volume"].sum().merge(key_map, left_on="scope_key", right_on="sk", how="left")
            )
            v_site = v_site.groupby("site", as_index=False)["volume"].sum().rename(columns={"volume": "val"})
            site_df = v_site
        if "scope_key" in bo.columns:
            b_site = bo.groupby("scope_key", as_index=False)["items"].sum().merge(key_map, left_on="scope_key", right_on="sk", how="left")
            b_site = b_site.groupby("site", as_index=False)["items"].sum().rename(columns={"items": "val"})
            site_df = b_site if site_df.empty else pd.concat([site_df, b_site]).groupby("site", as_index=False)["val"].sum()
        if not site_df.empty:
            site = {"labels": site_df["site"].astype(str).tolist(), "values": site_df["val"].astype(float).tolist()}

    waterfall = {
        "labels": ["Required", "Supply", "Gap"],
        "values": [kpi_req, -kpi_sup, kpi_gap],
        "measure": ["relative", "relative", "total"],
    }

    summary_rows: list[dict[str, Any]] = []
    if not map_df.empty:
        v_sum = voice.groupby("scope_key", as_index=False)["volume"].sum() if not voice.empty and "scope_key" in voice.columns else pd.DataFrame(columns=["scope_key", "volume"])
        b_sum = bo.groupby("scope_key", as_index=False)["items"].sum() if not bo.empty and "scope_key" in bo.columns else pd.DataFrame(columns=["scope_key", "items"])
        sk_map = map_df.drop_duplicates(subset=["sk", "ba", "sba", "ch", "site", "loc"]).rename(columns={"sk": "scope_key"})
        merged = sk_map.merge(v_sum, on="scope_key", how="left").merge(b_sum, on="scope_key", how="left")
        merged["volume"] = merged["volume"].fillna(0)
        merged["items"] = merged["items"].fillna(0)
        tbl = merged.groupby(["ba", "sba", "ch", "site", "loc"], as_index=False)[["volume", "items"]].sum()
        summary_rows = tbl.to_dict("records")

    return {
        "kpis": {"required_fte": kpi_req, "supply_fte": kpi_sup, "gap_fte": kpi_gap},
        "line": line,
        "bar": {"labels": bar_labels, "series": bar_series},
        "pie": pie,
        "site": site,
        "waterfall": waterfall,
        "summary": summary_rows,
    }
