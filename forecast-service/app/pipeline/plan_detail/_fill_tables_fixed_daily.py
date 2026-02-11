from __future__ import annotations
from typing import Dict, List

import dash
import numpy as np
import pandas as pd
from dash import dash_table

from plan_store import get_plan
from cap_store import resolve_settings
from cap_db import load_df
from ._grain_cols import day_cols_for_weeks
from ._common import _week_span, _canon_scope, _monday, get_plan_meta, _load_ts_with_fallback, _assemble_voice, _assemble_chat, _assemble_ob, _assemble_bo
from ._calc import (
    _ivl_seconds,
    _voice_interval_calc,
    _chat_interval_calc,
    _ob_interval_calc,
    _daily_from_intervals,
    _bo_daily_calc,
    _fill_tables_fixed,
    get_cached_consolidated_calcs,
)


def _to_frac(value) -> float:
    try:
        v = float(value)
        return v / 100.0 if v > 1.0 else v
    except Exception:
        try:
            text = str(value or "").strip().rstrip("%")
            v = float(text)
            return v / 100.0 if v > 1.0 else v
        except Exception:
            return 0.0


def _planned_shrink(settings: dict, ch_first: str) -> float:
    ch = (ch_first or "").strip().lower()
    if ch in ("back office", "bo"):
        return _to_frac(settings.get("bo_shrinkage_pct", settings.get("shrinkage_pct", 0.30)))
    if ch in ("chat",):
        return _to_frac(settings.get("chat_shrinkage_pct", settings.get("shrinkage_pct", 0.30)))
    if ch in ("outbound", "ob"):
        return _to_frac(settings.get("ob_shrinkage_pct", settings.get("shrinkage_pct", 0.30)))
    return _to_frac(settings.get("voice_shrinkage_pct", settings.get("shrinkage_pct", 0.30)))


# --- UI helper: keep the first return a DataTable (same as Interval filler expects) ---
def _make_upper_table(df: pd.DataFrame, day_cols_meta: List[dict]):
    if not isinstance(df, pd.DataFrame) or df.empty:
        df = pd.DataFrame({"metric": []})
    return dash_table.DataTable(
        id="tbl-upper",
        data=df.to_dict("records"),
        columns=[{"name": "Metric", "id": "metric", "editable": False}] + day_cols_meta,
        editable=False,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"whiteSpace": "pre"},
    )


# --- derive day columns from data (no _week_span dependency) ---
def _derive_day_ids_from_plan(plan: dict):
    weeks_span = _week_span(plan.get("start_week"), plan.get("end_week"))
    full_cols, day_ids = day_cols_for_weeks(weeks_span)
    # Drop the leading Metric column since _make_upper_table prepends it
    day_cols_meta = [c for c in full_cols if str(c.get("id")) != "metric"]
    return day_cols_meta, day_ids


def _fill_tables_fixed_daily(ptype, pid, _fw_cols_unused, _tick, whatif=None):
    """
    Daily view (Voice/Chat/Outbound):
      - FW: Forecast/Actual/Tactical Volume + AHT/SUT (daily roll-ups from interval streams)
      - Upper: FTE/PHC/Service Level (daily roll-ups from interval streams via Erlang)
    Notes:
      • Uses consolidated_calcs(...) so headers/dates/interval strings are normalized.
      • Vectorized build (no per-column inserts) → no pandas fragmentation warnings.
      • Returns the same 13-item tuple shape your callbacks expect (DataTable + FW dict + empties).
    """
    if not pid:
        raise dash.exceptions.PreventUpdate

    plan = get_plan(pid) or {}
    ch_name = (plan.get("channel") or plan.get("lob") or "").split(",")[0].strip()
    _settings = resolve_settings(
        ba=plan.get("vertical"),
        subba=plan.get("sub_ba"),
        lob=ch_name,
        site=(plan.get("site") or plan.get("location") or plan.get("country")),
    )
    ivl_min = int(float(_settings.get("interval_minutes", 30) or 30))
    # Read per-plan lower FW options (e.g., Backlog toggle)
    try:
        meta = get_plan_meta(pid) or {}
    except Exception:
        meta = {}
    def _meta_list(val):
        if isinstance(val, str):
            import json
            try:
                return list(json.loads(val))
            except Exception:
                return []
        if isinstance(val, (list, tuple)):
            return list(val)
        return []
    lower_opts = set(_meta_list(meta.get("fw_lower_options")))
    upper_opts = set(_meta_list(meta.get("upper_options")))
    # Backlog logic is enabled only when a backlog/queue metric is selected
    backlog_enabled = ("backlog" in lower_opts) or ("queue" in lower_opts) or ("req_queue" in upper_opts)

    # Build scope key and assemble RAW uploads
    ch = ch_name.lower()
    sk = _canon_scope(plan.get("vertical"),
                      plan.get("sub_ba"),
                      ch_name,
                      plan.get("site") or plan.get("location") or plan.get("country"))

    is_bo = False
    if ch.startswith("voice"):
        dfF = _assemble_voice(sk, "forecast")
        dfA = _assemble_voice(sk, "actual")
        dfT = _assemble_voice(sk, "tactical")
        weight_col_upload = "volume"
        aht_label = "AHT/SUT"
    elif ch.startswith("chat"):
        dfF = _assemble_chat(sk, "forecast")
        dfA = _assemble_chat(sk, "actual")
        dfT = _assemble_chat(sk, "tactical")
        weight_col_upload = "items"
        aht_label = "AHT/SUT"
    elif ch.startswith("back office") or ch in ("bo","backoffice"):
        # Back Office uses daily items + SUT (no interval calc needed)
        dfF = _assemble_bo(sk, "forecast")
        dfA = _assemble_bo(sk, "actual")
        dfT = _assemble_bo(sk, "tactical")
        weight_col_upload = "items"
        aht_label = "SUT"
        is_bo = True
    else:  # outbound
        dfF = _assemble_ob(sk, "forecast")
        dfA = _assemble_ob(sk, "actual")
        dfT = _assemble_ob(sk, "tactical")
        weight_col_upload = "opc"
        aht_label = "AHT/SUT"

    # --- Build day columns from UI-provided FW columns to ensure alignment ---
    # The caller provides `fw_cols` (weekly/day headers). Use those so our data keys align with the grid.
    try:
        fw_cols = list(_fw_cols_unused or [])
    except Exception:
        fw_cols = []
    day_ids = [str(c.get("id")) for c in fw_cols if str(c.get("id")) != "metric"]
    # Upper table expects column metadata without the leading Metric column
    day_cols_meta = [
        {"name": c.get("name"), "id": c.get("id")}
        for c in fw_cols if str(c.get("id")) != "metric"
    ]

    # Reference date used in several branches
    today = pd.Timestamp('today').date()

    # -------- helpers --------
    def _daily_sum(df: pd.DataFrame, val_col: str) -> dict:
        if df is None or df.empty or "date" not in df.columns or val_col not in df.columns:
            return {}
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.date
        d[val_col] = pd.to_numeric(d[val_col], errors="coerce").fillna(0.0)
        g = d.groupby("date", as_index=False)[val_col].sum()
        return {str(k): float(v) for k, v in zip(g["date"], g[val_col])}

    def _daily_weighted_aht(df: pd.DataFrame, wcol: str, aht_col: str = "aht_sec") -> dict:
        if df is None or df.empty or "date" not in df.columns or wcol not in df.columns or aht_col not in df.columns:
            return {}
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.date
        d[wcol] = pd.to_numeric(d[wcol], errors="coerce").fillna(0.0)
        d[aht_col] = pd.to_numeric(d[aht_col], errors="coerce").fillna(0.0)
        out = {}
        for dd, grp in d.groupby("date"):
            w = grp[wcol].sum()
            if w > 0:
                out[str(dd)] = float((grp[wcol] * grp[aht_col]).sum() / w)
            else:
                # If volume is zero but AHT/SUT exists, use simple mean to avoid default fallback.
                vals = pd.to_numeric(grp[aht_col], errors="coerce").replace({0.0: np.nan}).dropna()
                out[str(dd)] = float(vals.mean()) if not vals.empty else 0.0
        return out

    def _daily_weighted_occ(df: pd.DataFrame) -> dict:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        if not all(c in df.columns for c in ("date","service_level","staff_seconds","occupancy")):
            return {}
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date
        d["staff_seconds"] = pd.to_numeric(d["staff_seconds"], errors="coerce").fillna(0.0)
        d["occupancy"] = pd.to_numeric(d["occupancy"], errors="coerce").fillna(0.0)
        out = {}
        for dd, grp in d.groupby("date"):
            w = grp["staff_seconds"].sum()
            out[str(dd)] = float((grp["staff_seconds"] * grp["occupancy"]).sum() / w) if w > 0 else 0.0
        return out
    # Pull heavy interval/daily calcs from shared cache (computed once per plan/tick)
    calc_bundle = get_cached_consolidated_calcs(
        int(pid),
        settings=_settings,
        version_token=_tick,
        channel=ch,
    ) if pid else {}
    def _from_bundle(key: str) -> pd.DataFrame:
        if not isinstance(calc_bundle, dict):
            return pd.DataFrame()
        val = calc_bundle.get(key)
        return val if isinstance(val, pd.DataFrame) else pd.DataFrame()

    # Compute interval calcs (if present) and then daily rollups for forecast and actual
    ivl_calc_f = pd.DataFrame()
    ivl_calc_a = pd.DataFrame()
    ivl_calc_t = pd.DataFrame()
    day_calc_f = pd.DataFrame()
    day_calc_a = pd.DataFrame()
    day_calc_t = pd.DataFrame()
    weight_col_ivl = "volume" if ch.startswith("voice") else "items"

    if ch.startswith("voice"):
        ivl_calc_f = _from_bundle("voice_ivl_f")
        ivl_calc_a = _from_bundle("voice_ivl_a")
        ivl_calc_t = _from_bundle("voice_ivl_t")
        day_calc_f = _from_bundle("voice_day_f")
        day_calc_a = _from_bundle("voice_day_a")
        day_calc_t = _from_bundle("voice_day_t") if "voice_day_t" in (calc_bundle or {}) else pd.DataFrame()
        if day_calc_f.empty:
            day_calc_f = _from_bundle("voice_day")
    elif ch.startswith("chat"):
        ivl_calc_f = _from_bundle("chat_ivl_f")
        ivl_calc_a = _from_bundle("chat_ivl_a")
        ivl_calc_t = _from_bundle("chat_ivl_t")
        day_calc_f = _from_bundle("chat_day_f")
        day_calc_a = _from_bundle("chat_day_a")
        day_calc_t = _from_bundle("chat_day_t") if "chat_day_t" in (calc_bundle or {}) else pd.DataFrame()
        if day_calc_f.empty:
            day_calc_f = _from_bundle("chat_day")
    elif ch.startswith("back office") or ch in ("bo", "backoffice"):
        day_calc_f = _from_bundle("bo_day_f")
        day_calc_a = _from_bundle("bo_day_a")
        day_calc_t = _from_bundle("bo_day_t")
    else:
        ivl_calc_f = _from_bundle("ob_ivl_f")
        ivl_calc_a = _from_bundle("ob_ivl_a")
        ivl_calc_t = _from_bundle("ob_ivl_t")
        day_calc_f = _from_bundle("ob_day_f")
        day_calc_a = _from_bundle("ob_day_a")
        day_calc_t = _from_bundle("ob_day_t") if "ob_day_t" in (calc_bundle or {}) else pd.DataFrame()
        if day_calc_f.empty:
            day_calc_f = _from_bundle("ob_day")

    # Fallback to local computation if cache miss (keeps legacy behavior intact)
    def _interval_calc(df: pd.DataFrame, channel: str) -> pd.DataFrame:
        if not (isinstance(df, pd.DataFrame) and not df.empty and "interval" in df.columns):
            return pd.DataFrame()
        if channel.startswith("voice"):
            return _voice_interval_calc(df, _settings, ivl_min)
        elif channel.startswith("chat"):
            return _chat_interval_calc(df, _settings, ivl_min)
        elif channel.startswith("back office") or channel in ("bo","backoffice"):
            return pd.DataFrame()
        else:
            x = df.copy()
            if "items" not in x.columns:
                if "opc" in x.columns:
                    x = x.rename(columns={"opc": "items"})
                elif "volume" in x.columns:
                    x = x.rename(columns={"volume": "items"})
            return _ob_interval_calc(x, _settings, ivl_min)

    if ivl_calc_f.empty and isinstance(dfF, pd.DataFrame):
        ivl_calc_f = _interval_calc(dfF, ch)
    if ivl_calc_a.empty and isinstance(dfA, pd.DataFrame):
        ivl_calc_a = _interval_calc(dfA, ch)
    if ivl_calc_t.empty and isinstance(dfT, pd.DataFrame):
        ivl_calc_t = _interval_calc(dfT, ch)

    if day_calc_f.empty:
        if isinstance(ivl_calc_f, pd.DataFrame) and not ivl_calc_f.empty:
            day_calc_f = _daily_from_intervals(ivl_calc_f, _settings, weight_col_ivl)
        elif isinstance(dfF, pd.DataFrame) and not dfF.empty and not ch.startswith("voice"):
            day_calc_f = _bo_daily_calc(dfF, _settings, channel=ch)
    if day_calc_a.empty:
        if isinstance(ivl_calc_a, pd.DataFrame) and not ivl_calc_a.empty:
            day_calc_a = _daily_from_intervals(ivl_calc_a, _settings, weight_col_ivl)
        elif isinstance(dfA, pd.DataFrame) and not dfA.empty and not ch.startswith("voice"):
            day_calc_a = _bo_daily_calc(dfA, _settings, channel=ch)
    if day_calc_t.empty:
        if isinstance(ivl_calc_t, pd.DataFrame) and not ivl_calc_t.empty:
            day_calc_t = _daily_from_intervals(ivl_calc_t, _settings, weight_col_ivl)
        else:
            day_calc_t = pd.DataFrame()

    # Extract metrics into dicts keyed by date
    m_fte_f = (
        {str(pd.to_datetime(r["date"]).date()): float(r.get("fte_req", 0.0))
         for _, r in day_calc_f.iterrows()}
        if isinstance(day_calc_f, pd.DataFrame) and not day_calc_f.empty else {}
    )
    m_fte_a = (
        {str(pd.to_datetime(r["date"]).date()): float(r.get("fte_req", 0.0))
         for _, r in day_calc_a.iterrows()}
        if isinstance(day_calc_a, pd.DataFrame) and not day_calc_a.empty else {}
    )
    m_fte_t = (
        {str(pd.to_datetime(r["date"]).date()): float(r.get("fte_req", 0.0))
         for _, r in day_calc_t.iterrows()}
        if isinstance(day_calc_t, pd.DataFrame) and not day_calc_t.empty else {}
    )
    src_calc = day_calc_f if (isinstance(day_calc_f, pd.DataFrame) and not day_calc_f.empty) else (
        day_calc_a if (isinstance(day_calc_a, pd.DataFrame) and not day_calc_a.empty) else pd.DataFrame()
    )
    def _safe_float0(x) -> float:
        try:
            v = pd.to_numeric(x, errors="coerce")
            if pd.isna(v):
                return 0.0
            return float(v)
        except Exception:
            return 0.0
    m_phc   = (
        {str(pd.to_datetime(r["date"]).date()): _safe_float0(r.get("phc"))
         for _, r in src_calc.iterrows()}
        if isinstance(src_calc, pd.DataFrame) and not src_calc.empty else {}
    )
    m_sl    = (
        {str(pd.to_datetime(r["date"]).date()): _safe_float0(r.get("service_level"))
         for _, r in src_calc.iterrows()}
        if isinstance(src_calc, pd.DataFrame) and not src_calc.empty else {}
    )

    # Daily shrinkage maps from daily uploads (no interval shrinkage feed).
    # BO uses BO activity model; Voice/Chat/OB use voice-style superstate model.
    bo_shr_g = pd.DataFrame()
    bo_ooo_pct_map: dict[str, float] = {}
    bo_ino_pct_map: dict[str, float] = {}
    bo_ov_pct_map: dict[str, float] = {}
    daily_actual_shrink_pct_map: dict[str, float] = {}
    # Daily overtime (hours) map derived from shrinkage raw uploads.
    overtime_daily_map: dict[str, float] = {}

    def _parse_hours_value(val) -> float:
        try:
            v = float(val)
            if np.isnan(v):
                return 0.0
            return float(v)
        except Exception:
            pass
        try:
            s = str(val or "").strip()
            if not s:
                return 0.0
            if ":" in s:
                parts = [p.strip() for p in s.split(":")]
                if len(parts) >= 2:
                    hh = float(parts[0] or 0.0)
                    mm = float(parts[1] or 0.0)
                    ss = float(parts[2] or 0.0) if len(parts) >= 3 else 0.0
                    return max(0.0, hh + (mm / 60.0) + (ss / 3600.0))
            return float(s)
        except Exception:
            return 0.0

    def _apply_scope_filters_raw(df_raw: pd.DataFrame, raw_key: str) -> pd.DataFrame:
        if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
            return pd.DataFrame()
        df = df_raw.copy()
        L = {str(c).strip().lower(): c for c in df.columns}
        c_ba = L.get("business area") or L.get("ba") or L.get("vertical") or L.get("journey")
        c_sba = L.get("sub business area") or L.get("sub_ba") or L.get("sub business area") or L.get("subba")
        c_ch = L.get("channel") or L.get("lob")
        c_site = L.get("site"); c_location = L.get("location"); c_country = L.get("country"); c_city = L.get("city")
        mask = pd.Series(True, index=df.index)
        if c_ba and plan.get("vertical"):
            mask &= df[c_ba].astype(str).str.strip().str.lower().eq(str(plan.get("vertical")).strip().lower())
        if c_sba and plan.get("sub_ba"):
            mask &= df[c_sba].astype(str).str.strip().str.lower().eq(str(plan.get("sub_ba")).strip().lower())
        if c_ch:
            ch_series = df[c_ch].astype(str).str.strip().str.lower()
            rk = str(raw_key or "").strip().lower()
            if rk.endswith("_backoffice"):
                targets = {"back office", "backoffice", "bo", ""}
            elif rk.endswith("_chat"):
                targets = {"chat", "messageus", "message us", ""}
            elif rk.endswith("_outbound"):
                targets = {"outbound", "ob", "out bound", ""}
            else:
                ch_norm = str(ch_name or "").strip().lower()
                if ch_norm in {"chat", "messageus", "message us"}:
                    targets = {"chat", "messageus", "message us", ""}
                elif ch_norm in {"outbound", "ob", "out bound"}:
                    targets = {"outbound", "ob", "out bound", ""}
                elif ch_norm in {"back office", "backoffice", "bo"}:
                    targets = {"back office", "backoffice", "bo", ""}
                else:
                    targets = {"voice", "inbound", "telephony", "volume", ""}
            if ch_series.isin(targets).any():
                mask &= ch_series.isin(targets)
        loc_first = (plan.get("site") or plan.get("location") or plan.get("country") or "").strip()
        if loc_first:
            target = loc_first.strip().lower()
            for col in [c_site, c_location, c_country, c_city]:
                if col and col in df.columns:
                    loc_l = df[col].astype(str).str.strip().str.lower()
                    if loc_l.eq(target).any():
                        mask &= loc_l.eq(target)
                        break
        return df.loc[mask]

    def _accumulate_overtime_daily(raw_key: str) -> None:
        nonlocal overtime_daily_map
        try:
            raw = load_df(raw_key)
        except Exception:
            raw = None
        scoped = _apply_scope_filters_raw(raw, raw_key)
        if not isinstance(scoped, pd.DataFrame) or scoped.empty:
            return
        L = {str(c).strip().lower(): c for c in scoped.columns}
        c_date = L.get("date")
        if not c_date:
            return

        # Voice-like: superstate + hours
        c_state = L.get("superstate") or L.get("state")
        c_hours = L.get("hours") or L.get("duration_hours") or L.get("duration")
        if c_state and c_hours:
            d = scoped[[c_date, c_state, c_hours]].copy()
            d[c_date] = pd.to_datetime(d[c_date], errors="coerce")
            d = d.dropna(subset=[c_date])
            if not d.empty:
                st = d[c_state].astype(str).str.strip().str.upper()
                d = d.loc[st.eq("SC_OVERTIME_DELIVERED")]
                if not d.empty:
                    hrs = d[c_hours].map(_parse_hours_value)
                    tmp = pd.DataFrame({"date": d[c_date].dt.date, "hours": hrs})
                    agg = tmp.groupby("date", as_index=False)["hours"].sum()
                    for _, r in agg.iterrows():
                        k = str(r["date"])
                        overtime_daily_map[k] = overtime_daily_map.get(k, 0.0) + float(r["hours"])
            return

        # Activity-based: look for overtime keywords
        c_act = L.get("activity")
        c_sec = L.get("duration_seconds") or L.get("seconds")
        c_hr = L.get("hours") or L.get("duration_hours")
        if not c_act or not (c_sec or c_hr):
            return
        d = scoped[[c_date, c_act, c_sec or c_hr]].copy()
        d[c_act] = d[c_act].astype(str).str.strip().str.lower()
        d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date
        d = d.dropna(subset=[c_date])
        if d.empty:
            return
        m_ot = d[c_act].str.contains(r"\bover\s*time\b|\bovertime\b|\bot\b|\bot\s*hours\b|\bot\s*hrs\b", regex=True, na=False) | d[c_act].eq("overtime")
        d = d.loc[m_ot]
        if d.empty:
            return
        if c_sec and (c_sec in d.columns):
            hrs = pd.to_numeric(d[c_sec], errors="coerce").fillna(0.0) / 3600.0
        else:
            hrs = d[c_hr].map(_parse_hours_value)
        tmp = pd.DataFrame({"date": d[c_date], "hours": hrs})
        agg = tmp.groupby("date", as_index=False)["hours"].sum()
        for _, r in agg.iterrows():
            k = str(r["date"])
            overtime_daily_map[k] = overtime_daily_map.get(k, 0.0) + float(r["hours"])

    # Choose sources by channel; fall back to voice raw if channel-specific is absent.
    if ch.startswith("back office") or ch in ("bo", "backoffice"):
        _accumulate_overtime_daily("shrinkage_raw_backoffice")
    elif ch.startswith("chat"):
        _accumulate_overtime_daily("shrinkage_raw_chat")
        if not overtime_daily_map:
            _accumulate_overtime_daily("shrinkage_raw_voice")
    elif ch.startswith("outbound") or ch in ("ob", "out bound"):
        _accumulate_overtime_daily("shrinkage_raw_outbound")
        if not overtime_daily_map:
            _accumulate_overtime_daily("shrinkage_raw_voice")
    else:
        _accumulate_overtime_daily("shrinkage_raw_voice")
        if not overtime_daily_map:
            _accumulate_overtime_daily("shrinkage_raw_backoffice")

    if is_bo:
        try:
            raw = load_df("shrinkage_raw_backoffice")
        except Exception:
            raw = None
        try:
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                from app.pipeline.shrinkage_store import summarize_shrinkage_bo
                dsum = summarize_shrinkage_bo(raw)
                ba  = str(plan.get("vertical") or "").strip().lower()
                sba = str(plan.get("sub_ba") or "").strip().lower()
                site = str(plan.get("site") or plan.get("location") or plan.get("country") or "").strip().lower()
                if "Business Area" in dsum.columns:
                    dsum = dsum[dsum["Business Area"].astype(str).str.strip().str.lower().eq(ba) | (ba == "")]
                if "Sub Business Area" in dsum.columns:
                    dsum = dsum[dsum["Sub Business Area"].astype(str).str.strip().str.lower().eq(sba) | (sba == "")]
                if "Channel" in dsum.columns:
                    dsum = dsum[dsum["Channel"].astype(str).str.strip().str.lower().isin(["back office", "bo", "backoffice"])]
                if site and ("Site" in dsum.columns):
                    dsum = dsum[dsum["Site"].astype(str).str.strip().str.lower().eq(site)]
                dsum["date"] = pd.to_datetime(dsum["date"], errors="coerce").dt.date
                keep = [c for c in ["OOO Hours", "In Office Hours", "Base Hours", "TTW Hours"] if c in dsum.columns]
                bo_shr_g = dsum.groupby("date", as_index=False)[keep].sum() if keep else pd.DataFrame()
                for _, r in (bo_shr_g.iterrows() if isinstance(bo_shr_g, pd.DataFrame) and not bo_shr_g.empty else []):
                    try:
                        d = str(pd.to_datetime(r["date"]).date())
                    except Exception:
                        continue
                    base = float(r.get("Base Hours", 0.0) or 0.0)
                    ttw  = float(r.get("TTW Hours",  0.0) or 0.0)
                    ooo  = float(r.get("OOO Hours",  0.0) or 0.0)
                    ino  = float(r.get("In Office Hours", 0.0) or 0.0)
                    ooo_pct = (100.0 * ooo / base) if base > 0 else 0.0
                    ino_pct = (100.0 * ino / ttw)  if ttw  > 0 else 0.0
                    availability = (1.0 - (ooo_pct / 100.0)) * (1.0 - (ino_pct / 100.0))
                    ov_pct = max(0.0, min(100.0, (1.0 - availability) * 100.0))
                    bo_ooo_pct_map[d] = ooo_pct
                    bo_ino_pct_map[d] = ino_pct
                    bo_ov_pct_map[d]  = ov_pct
                daily_actual_shrink_pct_map.update(bo_ov_pct_map)
        except Exception:
            bo_shr_g = pd.DataFrame()
            bo_ooo_pct_map = {}
            bo_ino_pct_map = {}
            bo_ov_pct_map = {}
    else:
        try:
            raw_voice = load_df("shrinkage_raw_voice")
        except Exception:
            raw_voice = None
        try:
            if isinstance(raw_voice, pd.DataFrame) and not raw_voice.empty:
                from app.pipeline.shrinkage_store import summarize_shrinkage_voice
                dsum = summarize_shrinkage_voice(raw_voice)
                if isinstance(dsum, pd.DataFrame) and not dsum.empty:
                    ba = str(plan.get("vertical") or "").strip().lower()
                    sba = str(plan.get("sub_ba") or "").strip().lower()
                    loc = str(plan.get("site") or plan.get("location") or plan.get("country") or "").strip().lower()
                    if "Business Area" in dsum.columns:
                        dsum = dsum[dsum["Business Area"].astype(str).str.strip().str.lower().eq(ba) | (ba == "")]
                    if "Sub Business Area" in dsum.columns:
                        dsum = dsum[dsum["Sub Business Area"].astype(str).str.strip().str.lower().eq(sba) | (sba == "")]
                    if "Channel" in dsum.columns:
                        channel_targets = {"voice"}
                        ch_norm = str(ch_name or "").strip().lower()
                        if ch_norm in {"chat", "messageus", "message us"}:
                            channel_targets = {"chat", "messageus", "message us"}
                        elif ch_norm in {"outbound", "ob", "out bound"}:
                            channel_targets = {"outbound", "ob", "out bound"}
                        ch_series = dsum["Channel"].astype(str).str.strip().str.lower()
                        if ch_series.isin(channel_targets).any():
                            dsum = dsum[ch_series.isin(channel_targets)]
                    if loc and ("Country" in dsum.columns):
                        loc_series = dsum["Country"].astype(str).str.strip().str.lower()
                        if loc_series.eq(loc).any():
                            dsum = dsum[loc_series.eq(loc)]
                    if not dsum.empty:
                        dsum = dsum.copy()
                        dsum["date"] = pd.to_datetime(dsum.get("date"), errors="coerce").dt.date
                        dsum = dsum.dropna(subset=["date"])
                        dsum["OOO Hours"] = pd.to_numeric(dsum.get("OOO Hours"), errors="coerce").fillna(0.0)
                        dsum["In Office Hours"] = pd.to_numeric(dsum.get("In Office Hours"), errors="coerce").fillna(0.0)
                        dsum["Base Hours"] = pd.to_numeric(dsum.get("Base Hours"), errors="coerce").fillna(0.0)
                        g = dsum.groupby("date", as_index=False)[["OOO Hours", "In Office Hours", "Base Hours"]].sum()
                        for _, r in g.iterrows():
                            d = str(pd.to_datetime(r["date"]).date())
                            base = float(r.get("Base Hours", 0.0) or 0.0)
                            ooo = float(r.get("OOO Hours", 0.0) or 0.0)
                            ino = float(r.get("In Office Hours", 0.0) or 0.0)
                            ooo_pct = (100.0 * ooo / base) if base > 0 else 0.0
                            ino_pct = (100.0 * ino / base) if base > 0 else 0.0
                            availability = (1.0 - (ooo_pct / 100.0)) * (1.0 - (ino_pct / 100.0))
                            daily_actual_shrink_pct_map[d] = max(0.0, min(100.0, (1.0 - availability) * 100.0))
        except Exception:
            daily_actual_shrink_pct_map = {}

    # Apply actual shrinkage for current/past days on Actual FTE
    try:
        shr = load_df(f"plan_{pid}_shr")
        act_row = None
        plan_row = None
        if isinstance(shr, pd.DataFrame) and not shr.empty and "metric" in shr.columns:
            m = shr["metric"].astype(str).str.strip().str.lower()
            if (m == "overall shrinkage %").any():
                act_row = shr.loc[m == "overall shrinkage %"].iloc[0].to_dict()
            if (m == "planned shrinkage %").any():
                plan_row = shr.loc[m == "planned shrinkage %"].iloc[0].to_dict()
        today = pd.Timestamp.today().date()
        planned_base = _planned_shrink(_settings, ch)
        for d in list(m_fte_a.keys()):
            try:
                dd = pd.to_datetime(d).date()
            except Exception:
                continue
            if dd > today:
                continue
            if d in daily_actual_shrink_pct_map:
                s_act = _to_frac(daily_actual_shrink_pct_map.get(d, 0.0))
                s_plan = planned_base
            else:
                wk = (dd - pd.to_timedelta(dd.weekday(), unit="D")).date().isoformat()
                s_act = _to_frac(act_row.get(wk)) if isinstance(act_row, dict) and (wk in act_row) else None
                if s_act is None:
                    continue
                s_plan = _to_frac(plan_row.get(wk)) if isinstance(plan_row, dict) and (wk in plan_row) else None
                if s_plan is None:
                    s_plan = planned_base
            denom_old = max(0.01, 1.0 - float(s_plan))
            denom_new = max(0.01, 1.0 - float(s_act))
            m_fte_a[d] = float(m_fte_a.get(d, 0.0)) * (denom_old / denom_new)
    except Exception:
        pass

    # Projected Supply HC — derive from weekly upper and repeat per day (no division)
    m_supply = {d: 0.0 for d in day_ids}
    try:
        weeks = _week_span(plan.get("start_week"), plan.get("end_week"))
        weekly_fw_cols = [{"name": "Metric", "id": "metric", "editable": False}] + [{"name": w, "id": w} for w in weeks]
        weekly = _fill_tables_fixed(ptype, pid, weekly_fw_cols, _tick, whatif=whatif, grain='week')
        (upper_wk, *_rest) = weekly
        upper_df_w = pd.DataFrame(getattr(upper_wk, 'data', None) or [])
        sup_w = {}
        if isinstance(upper_df_w, pd.DataFrame) and not upper_df_w.empty and "metric" in upper_df_w.columns:
            row = upper_df_w[upper_df_w["metric"].astype(str).str.strip().eq("Projected Supply HC")]
            if not row.empty:
                for w in weeks:
                    if w in row.columns:
                        try:
                            sup_w[str(pd.to_datetime(w).date())] = float(pd.to_numeric(row[w], errors='coerce').fillna(0.0).iloc[0])
                        except Exception:
                            sup_w[str(pd.to_datetime(w).date())] = 0.0
        # Fill each day with the weekly headcount value (aligns with weekly/monthly semantics)
        for d in day_ids:
            w = str(_monday(d))
            m_supply[d] = float(sup_w.get(w, 0.0))
    except Exception:
        pass

    # Prepare daily AHT/SUT maps used for BO calcs (computed here to avoid unbound refs)
    try:
        ahtF = _daily_weighted_aht(dfF, weight_col_upload)
    except Exception:
        ahtF = {}
    try:
        ahtA = _daily_weighted_aht(dfA, weight_col_upload)
    except Exception:
        ahtA = {}

    # For Back Office (daily/TAT), derive PHC and a proxy Service Level from supply and SUT
    if is_bo:
        try:
            hrs = float(_settings.get("bo_hours_per_day", _settings.get("hours_per_fte", 8.0)) or 8.0)
            shrink = float(_settings.get("bo_shrinkage_pct", _settings.get("shrinkage_pct", 0.30)) or 0.30)
            util = float(_settings.get("util_bo", 0.85) or 0.85)
            productive_sec = hrs * 3600.0 * max(1e-6, (1.0 - shrink)) * util
        except Exception:
            productive_sec = 8.0 * 3600.0 * 0.7

        # pick daily SUT: Actual for past/today; Forecast for future; else settings default
        def _sut_for_day(d):
            try:
                dd = pd.to_datetime(d).date()
            except Exception:
                dd = None
            try:
                if dd is not None and dd <= today:
                    v = ahtA.get(d, None)
                    if v is None or pd.isna(v) or float(v) <= 0:
                        v = ahtF.get(d, None)
                else:
                    v = ahtF.get(d, None)
                    if v is None or pd.isna(v) or float(v) <= 0:
                        v = ahtA.get(d, None)
                if v is None or pd.isna(v) or float(v) <= 0:
                    v = float(_settings.get("target_sut", _settings.get("budgeted_sut", 600)) or 600.0)
                return float(v)
            except Exception:
                return float(_settings.get("target_sut", _settings.get("budgeted_sut", 600)) or 600.0)

        # Recompute PHC from projected supply only if interval-based PHC is missing.
        # Otherwise keep the sum of interval PHC for accuracy.
        if all(float(m_phc.get(d, 0.0) or 0.0) == 0.0 for d in day_ids):
            # PHC = Supply FTE (per-day) * productive seconds per day / SUT_seconds
            for d in day_ids:
                sut = max(1e-6, _sut_for_day(d))
                sup_fte = float(m_supply.get(d, 0.0) or 0.0)
                m_phc[d] = float((sup_fte * productive_sec) / sut)

    # Linear PHC (non-Erlang) when interval PHC is missing: use FTE @ Forecast and AHT/SUT.
    # PHC = (ivl_sec / AHT) * intervals_per_day * FTE_forecast
    try:
        if not is_bo and all(float(m_phc.get(d, 0.0) or 0.0) == 0.0 for d in day_ids):
            ivl_sec = _ivl_seconds(_settings.get("interval_minutes", 30))
            hours_per_day = float(_settings.get("hours_per_fte", 8.0) or 8.0)
            intervals_per_day = float(hours_per_day * 3600.0) / max(1.0, float(ivl_sec))
            for d in day_ids:
                aht = float(ahtF.get(d, 0.0) or ahtA.get(d, 0.0) or 0.0)
                if aht <= 0:
                    continue
                fte_f = float(m_fte_f.get(d, 0.0) or 0.0)
                m_phc[d] = float((ivl_sec / aht) * intervals_per_day * fte_f)
    except Exception:
        pass

        # Proxy Service Level as Supply/Required (capped to 100%)
        # Required FTE: Actual for past/today; Forecast for future
        for d in day_ids:
            try:
                dd = pd.to_datetime(d).date()
            except Exception:
                dd = None
            req = float(((m_fte_a.get(d) if (dd is not None and dd <= today) else m_fte_f.get(d)) or m_fte_a.get(d) or 0.0))
            sup = float(m_supply.get(d, 0.0) or 0.0)
            if req <= 0:
                m_sl[d] = 100.0 if sup > 0 else 0.0
            else:
                m_sl[d] = float(min(100.0, max(0.0, (sup / req) * 100.0)))

    # Compute variance rows (MTP≈Forecast, Tactical, Budgeted)
    # Budgeted FTE via budget AHT applied to Forecast intervals when possible
    m_fte_b = {}
    try:
        dfB = dfF.copy() if isinstance(dfF, pd.DataFrame) and not dfF.empty else pd.DataFrame()
        if not dfB.empty and "date" in dfB.columns:
            dfB["date"] = pd.to_datetime(dfB["date"], errors="coerce").dt.date.astype(str)
            dfB["aht_sec"] = dfB["date"].map(lambda s: _budget_for_day(s))
            ivl_b = _interval_calc(dfB, ch)
            if isinstance(ivl_b, pd.DataFrame) and not ivl_b.empty:
                day_b = _daily_from_intervals(ivl_b, _settings, weight_col_ivl)
                if isinstance(day_b, pd.DataFrame) and not day_b.empty:
                    m_fte_b = {str(pd.to_datetime(r["date"]).date()): float(r.get("fte_req", 0.0)) for _, r in day_b.iterrows()}
    except Exception:
        m_fte_b = {}

    var_mtp = [ (m_fte_f.get(c, 0.0) - m_fte_a.get(c, 0.0)) for c in day_ids ]
    var_tac = [ (m_fte_t.get(c, 0.0) - m_fte_a.get(c, 0.0)) for c in day_ids ]
    var_bud = [ (m_fte_b.get(c, 0.0) - m_fte_a.get(c, 0.0)) for c in day_ids ]

    # Daily roll-ups used for backlog/queue and FW grid
    volF = _daily_sum(dfF, weight_col_upload)
    volA = _daily_sum(dfA, weight_col_upload)
    volT = _daily_sum(dfT, weight_col_upload)

    backlog_map: dict[str, float] = {}
    queue_map: dict[str, float] = {}
    if backlog_enabled:
        for i, d in enumerate(day_ids):
            try:
                fval = float(volF.get(d, 0.0) or 0.0)
            except Exception:
                fval = 0.0
            try:
                aval = float(volA.get(d, 0.0) or 0.0)
            except Exception:
                aval = 0.0
            bl = max(0.0, aval - fval)
            backlog_map[d] = bl
            prev_bl = float(backlog_map.get(day_ids[i-1], 0.0)) if i > 0 else 0.0
            queue_map[d] = max(0.0, prev_bl + fval)

    req_queue_vals = None
    if "req_queue" in upper_opts and backlog_enabled:
        req_queue_vals = []
        for d in day_ids:
            fval = float(volF.get(d, 0.0) or 0.0)
            qval = float(queue_map.get(d, 0.0) or 0.0)
            base_req = float(m_fte_f.get(d, 0.0) or 0.0)
            req_queue_vals.append((base_req * (qval / fval)) if fval > 0 else 0.0)

    # Apply backlog carryover to next day's forecast (Back Office only)
    backlog_carryover = bool((whatif or {}).get("backlog_carryover", True))
    if backlog_carryover and is_bo and backlog_enabled:
        for i in range(len(day_ids) - 1):
            cur_d = day_ids[i]
            nxt_d = day_ids[i + 1]
            add = float(backlog_map.get(cur_d, 0.0) or 0.0)
            if add:
                volF[nxt_d] = float(volF.get(nxt_d, 0.0) or 0.0) + add

    # Build the upper DataFrame
    upper_payload = {
        "FTE Required @ Forecast Volume":   [m_fte_f.get(c, 0.0) for c in day_ids],
        "FTE Required @ Actual Volume":     [m_fte_a.get(c, 0.0) for c in day_ids],
        "FTE Over/Under MTP Vs Actual":     var_mtp,
        "FTE Over/Under Tactical Vs Actual":var_tac,
        "FTE Over/Under Budgeted Vs Actual":var_bud,
        "Projected Supply HC":              [m_supply.get(c, 0.0) for c in day_ids],
        "Projected Handling Capacity (#)":  [m_phc.get(c, 0.0) for c in day_ids],
        "Projected Service Level":          [m_sl.get(c, 0.0) for c in day_ids],
    }
    if req_queue_vals is not None:
        upper_payload["FTE Required @ Queue"] = req_queue_vals
    upper_df = pd.DataFrame.from_dict(
        upper_payload,
        orient="index", columns=day_ids,
    ).reset_index().rename(columns={"index":"metric"}).fillna(0.0)

    # Round to 1 decimal place for display
    def _round1(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        out = df.copy()
        for c in day_ids:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(1)
        return out

    upper_df = _round1(upper_df[["metric"] + day_ids])
    # Display headcount as whole numbers
    try:
        msk_hc = upper_df["metric"].astype(str).str.strip().eq("Projected Supply HC")
        if msk_hc.any():
            for c in day_ids:
                if c in upper_df.columns:
                    upper_df.loc[msk_hc, c] = pd.to_numeric(upper_df.loc[msk_hc, c], errors="coerce").fillna(0.0).round(0).astype(int)
    except Exception:
        pass

    # Add hover tooltips for shrinkage/base FTE
    try:
        shr = load_df(f"plan_{pid}_shr")
        act_row = None
        plan_row = None
        if isinstance(shr, pd.DataFrame) and not shr.empty and "metric" in shr.columns:
            m = shr["metric"].astype(str).str.strip().str.lower()
            if (m == "overall shrinkage %").any():
                act_row = shr.loc[m == "overall shrinkage %"].iloc[0].to_dict()
            if (m == "planned shrinkage %").any():
                plan_row = shr.loc[m == "planned shrinkage %"].iloc[0].to_dict()
        today = pd.Timestamp.today().date()
        upper_df["__tooltips"] = [{} for _ in range(len(upper_df))]
        for d in day_ids:
            try:
                dd = pd.to_datetime(d).date()
            except Exception:
                continue
            wk = (dd - pd.to_timedelta(dd.weekday(), unit="D")).date().isoformat()
            s_plan = _to_frac(plan_row.get(wk)) if isinstance(plan_row, dict) and (wk in plan_row) else _planned_shrink(_settings, ch)
            if d in daily_actual_shrink_pct_map:
                s_act = _to_frac(daily_actual_shrink_pct_map.get(d, 0.0))
            else:
                s_act = _to_frac(act_row.get(wk)) if isinstance(act_row, dict) and (wk in act_row) else None
            s_act_use = s_act if (dd <= today and s_act is not None) else s_plan
            # Forecast row tooltip
            try:
                f_val = float(upper_df.loc[upper_df["metric"].astype(str).str.strip().eq("FTE Required @ Forecast Volume"), d].iloc[0])
                base_f = f_val * max(0.0, (1.0 - float(s_plan or 0.0)))
                tip_f = f"Shrinkage included. Base FTE (pre-shrink): {base_f:.1f}"
                upper_df.loc[upper_df["metric"].astype(str).str.strip().eq("FTE Required @ Forecast Volume"), "__tooltips"] = \
                    upper_df.loc[upper_df["metric"].astype(str).str.strip().eq("FTE Required @ Forecast Volume"), "__tooltips"].apply(
                        lambda t: {**(t or {}), d: tip_f}
                    )
            except Exception:
                pass
            # Actual row tooltip
            try:
                a_val = float(upper_df.loc[upper_df["metric"].astype(str).str.strip().eq("FTE Required @ Actual Volume"), d].iloc[0])
                base_a = a_val * max(0.0, (1.0 - float(s_act_use or 0.0)))
                tip_a = f"Shrinkage included. Base FTE (pre-shrink): {base_a:.1f}"
                upper_df.loc[upper_df["metric"].astype(str).str.strip().eq("FTE Required @ Actual Volume"), "__tooltips"] = \
                    upper_df.loc[upper_df["metric"].astype(str).str.strip().eq("FTE Required @ Actual Volume"), "__tooltips"].apply(
                        lambda t: {**(t or {}), d: tip_a}
                    )
            except Exception:
                pass
    except Exception:
        pass
    # Add % sign to percentage metrics (1 decimal)
    try:
        msk = upper_df["metric"].astype(str).str.strip().eq("Projected Service Level")
        if msk.any():
            # ensure object dtype before assigning string values
            for c in day_ids:
                if c in upper_df.columns:
                    upper_df[c] = upper_df[c].astype(object)
            for c in day_ids:
                if c in upper_df.columns:
                    try:
                        v = float(pd.to_numeric(upper_df.loc[msk, c], errors="coerce").fillna(0.0).iloc[0])
                    except Exception:
                        v = 0.0
                    upper_df.loc[msk, c] = f"{v:.1f}%"
    except Exception:
        pass
    upper_tbl = _make_upper_table(upper_df, day_cols_meta)

        # -------- FW (Forecast/Tactical/Actual Volume + AHT/SUT + Occupancy/Budget) --------
    # Daily roll-ups (vol* already computed above)
    ahtF = _daily_weighted_aht(dfF, weight_col_upload)
    ahtA = _daily_weighted_aht(dfA, weight_col_upload)
    ahtT = _daily_weighted_aht(dfT, weight_col_upload)   # include tactical AHT

    # Occupancy from per-interval calcs (weighted). Use Actual for past/today, Forecast for future (fallback to settings).
    occ_f = _daily_weighted_occ(ivl_calc_f)
    occ_a = _daily_weighted_occ(ivl_calc_a)
    def _occ_setting_frac(settings: dict, channel: str) -> float:
        try:
            ch = (channel or '').strip().lower()
            if ch.startswith('voice'):
                base = settings.get('occupancy_cap_voice', settings.get('occupancy', 0.85))
            elif ch.startswith('back'):
                base = settings.get('util_bo', settings.get('occupancy', 0.85))
            elif ch.startswith('chat'):
                base = settings.get('util_chat', settings.get('util_bo', 0.85))
            else:
                base = settings.get('util_ob', settings.get('occupancy', 0.85))
            v = float(base if base is not None else 0.85)
            if v > 1.0:
                v = v/100.0
            return max(0.0, min(1.0, v))
        except Exception:
            return 0.85
    occ_setting = _occ_setting_frac(_settings, ch)
    today = pd.Timestamp('today').date()
    occF = {}
    for d in list(day_ids):
        try:
            dd = pd.to_datetime(d).date()
        except Exception:
            dd = None
        if dd is not None and dd <= today:
            val = occ_a.get(d, occ_f.get(d, occ_setting))
        else:
            val = occ_f.get(d, occ_setting)
        try:
            occF[str(d)] = float(pd.to_numeric(val, errors='coerce')) if val is not None else float(occ_setting)
        except Exception:
            occF[str(d)] = float(occ_setting)

    # Budgeted AHT/SUT per week → per day
    def _ts_week_dict(df: pd.DataFrame, val_candidates: list[str]) -> dict:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        d = df.copy()
        if "week" in d.columns:
            d["week"] = pd.to_datetime(d["week"], errors="coerce").dt.date.astype(str)
        elif "date" in d.columns:
            d["week"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
        else:
            return {}
        low = {c.lower(): c for c in d.columns}
        vcol = None
        for c in val_candidates:
            vcol = low.get(c.lower())
            if vcol:
                break
        if not vcol:
            return {}
        d[vcol] = pd.to_numeric(d[vcol], errors="coerce")
        return d.dropna(subset=["week", vcol]).set_index("week")[vcol].astype(float).to_dict()

    if ch.startswith("voice"):
        planned_df = _load_ts_with_fallback("voice_planned_aht", sk)
        if (not isinstance(planned_df, pd.DataFrame)) or planned_df.empty:
            tmp = _load_ts_with_fallback("voice_budget", sk)
            if isinstance(tmp, pd.DataFrame) and not tmp.empty and "budget_aht_sec" in tmp.columns:
                planned_df = tmp.rename(columns={"budget_aht_sec": "aht_sec"})[
                    [c for c in tmp.columns if c in ("date","week","aht_sec")]
                ]
        wk_budget = _ts_week_dict(planned_df, ["aht_sec", "aht", "avg_aht"]) if isinstance(planned_df, pd.DataFrame) else {}
    else:
        planned_df = _load_ts_with_fallback("bo_planned_sut", sk)
        if (not isinstance(planned_df, pd.DataFrame)) or planned_df.empty:
            tmp = _load_ts_with_fallback("bo_budget", sk)
            if isinstance(tmp, pd.DataFrame) and not tmp.empty and "budget_sut_sec" in tmp.columns:
                planned_df = tmp.rename(columns={"budget_sut_sec": "sut_sec"})[
                    [c for c in tmp.columns if c in ("date","week","sut_sec")]
                ]
        wk_budget = _ts_week_dict(planned_df, ["sut_sec", "aht_sec", "sut", "avg_sut"]) if isinstance(planned_df, pd.DataFrame) else {}

    def _budget_for_day(d: str) -> float:
        try:
            w = str(_monday(d))
            return float(wk_budget.get(w, 0.0))
        except Exception:
            return 0.0

    # Build FW DataFrame (conditionally include Backlog)
    fw_data = {
        "Forecast Volume":       [volF.get(c, 0.0) for c in day_ids],
        "Tactical Volume":       [volT.get(c, 0.0) for c in day_ids],
        "Actual Volume":         [volA.get(c, 0.0) for c in day_ids],
        "Budgeted AHT/SUT":      [_budget_for_day(c) for c in day_ids],
        f"Forecast {aht_label}": [ahtF.get(c, 0.0) for c in day_ids],
        f"Tactical {aht_label}": [ahtT.get(c, 0.0) for c in day_ids],
        f"Actual {aht_label}":   [ahtA.get(c, 0.0) for c in day_ids],
        "Occupancy":             [occF.get(c, 0.0) for c in day_ids],
        "Overtime Hours (#)":    [overtime_daily_map.get(c, 0.0) for c in day_ids],
    }

    # Include Backlog/Queue only if plan options selected
    if "backlog" in lower_opts:
        fw_data["Backlog (Items)"] = [backlog_map.get(c, 0.0) for c in day_ids]
    if "queue" in lower_opts:
        fw_data["Queue (Items)"] = [queue_map.get(c, 0.0) for c in day_ids]

    fw_df = pd.DataFrame.from_dict(fw_data, orient="index", columns=day_ids) \
                        .reset_index().rename(columns={"index":"metric"}).fillna(0.0)
    fw_df = _round1(fw_df[["metric"] + day_ids])
    # Format percentage rows with 1 decimal and a % suffix (Occupancy only in FW)
    try:
        msk_occ = fw_df["metric"].astype(str).str.strip().eq("Occupancy")
        if msk_occ.any():
            for c in day_ids:
                if c in fw_df.columns:
                    fw_df[c] = fw_df[c].astype(object)
            for c in day_ids:
                pct = float(occF.get(c, 0.0) * 100.0)
                fw_df.loc[msk_occ, c] = f"{pct:.1f}%"
    except Exception:
        pass

    # ---- Back Office Daily Shrinkage (OOO/INO/Overall) for lower grid (not FW) ----
    shrink_rows = []
    if is_bo:
        try:
            if isinstance(bo_shr_g, pd.DataFrame) and not bo_shr_g.empty:
                def row_map(label, m):
                    return {"metric": label, **{d: float(m.get(d, 0.0)) for d in day_ids}}
                hr_ooo = {str(k): float(v) for k, v in zip(bo_shr_g.get("date", []), bo_shr_g.get("OOO Hours", []))} if "OOO Hours" in bo_shr_g.columns else {}
                hr_ino = {str(k): float(v) for k, v in zip(bo_shr_g.get("date", []), bo_shr_g.get("In Office Hours", []))} if "In Office Hours" in bo_shr_g.columns else {}
                hr_base = {str(k): float(v) for k, v in zip(bo_shr_g.get("date", []), bo_shr_g.get("Base Hours", []))} if "Base Hours" in bo_shr_g.columns else {}
                hr_ttw = {str(k): float(v) for k, v in zip(bo_shr_g.get("date", []), bo_shr_g.get("TTW Hours", []))} if "TTW Hours" in bo_shr_g.columns else {}
                if hr_ooo:  shrink_rows.append(row_map("OOO Shrink Hours (#)", hr_ooo))
                if hr_ino:  shrink_rows.append(row_map("In-Office Shrink Hours (#)", hr_ino))
                if hr_base: shrink_rows.append(row_map("Base Hours (#)", hr_base))
                if hr_ttw:  shrink_rows.append(row_map("TTW Hours (#)", hr_ttw))
                if bo_ooo_pct_map: shrink_rows.append(row_map("OOO Shrinkage %", bo_ooo_pct_map))
                if bo_ino_pct_map: shrink_rows.append(row_map("In-Office Shrinkage %", bo_ino_pct_map))
                if bo_ov_pct_map:  shrink_rows.append(row_map("Overall Shrinkage %", bo_ov_pct_map))
        except Exception:
            pass

    # Build shrinkage lower table records
    if shrink_rows:
        shr_df = pd.DataFrame(shrink_rows)
        # Ensure all day columns exist
        for c in day_ids:
            if c not in shr_df.columns:
                shr_df[c] = 0.0
        # Planned and variance rows (percent)
        try:
            planned_pct_val = _settings.get("bo_shrinkage_pct", _settings.get("shrinkage_pct", 0.0))
            planned_pct_val = float(planned_pct_val or 0.0)
            if planned_pct_val <= 1.0:
                planned_pct_val *= 100.0
        except Exception:
            planned_pct_val = 0.0
        # Find Overall Shrinkage % row to compute variance
        ov_mask = shr_df["metric"].astype(str).str.strip().eq("Overall Shrinkage %")
        var_row = {"metric": "Variance vs Planned"}
        plan_row = {"metric": "Planned Shrinkage %"}
        for d in day_ids:
            plan_row[d] = planned_pct_val
            try:
                ov_val = float(pd.to_numeric(shr_df.loc[ov_mask, d], errors="coerce").fillna(0.0).iloc[0]) if ov_mask.any() else 0.0
            except Exception:
                ov_val = 0.0
            var_row[d] = ov_val - planned_pct_val
        # Round and format
        hours_labels = {"OOO Shrink Hours (#)", "In-Office Shrink Hours (#)", "Base Hours (#)", "TTW Hours (#)"}
        pct_labels   = {"OOO Shrinkage %", "In-Office Shrinkage %", "Overall Shrinkage %", "Planned Shrinkage %", "Variance vs Planned"}
        # Append planned and variance rows
        shr_df = pd.concat([shr_df, pd.DataFrame([plan_row, var_row])], ignore_index=True)
        # Round hours to 1 decimal
        for lab in hours_labels:
            m = shr_df["metric"].astype(str).str.strip().eq(lab)
            if m.any():
                for d in day_ids:
                    shr_df.loc[m, d] = pd.to_numeric(shr_df.loc[m, d], errors="coerce").fillna(0.0).round(1)
        # Round pct to 1 decimal and add % suffix
        for lab in pct_labels:
            m = shr_df["metric"].astype(str).str.strip().eq(lab)
            if m.any():
                for d in day_ids:
                    try:
                        v = float(pd.to_numeric(shr_df.loc[m, d], errors="coerce").fillna(0.0).iloc[0])
                    except Exception:
                        v = 0.0
                    shr_df.loc[m, d] = f"{v:.1f}%"
        shr_records = shr_df.to_dict("records")
    else:
        shr_records = []

    # -------- Return 13-item tuple --------
    empty = []
    return (
        upper_tbl,
        fw_df.to_dict("records"),
        empty, empty, shr_records, empty, empty, empty, empty, empty,
        empty, empty, empty,
    )
