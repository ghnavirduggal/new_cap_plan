from __future__ import annotations

import datetime as dt
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.pipeline.capacity_core import (
    add_week_month_keys,
    bo_erlang_rollups,
    bo_rollups,
    chat_fte_daily,
    min_agents,
    required_fte_daily,
    voice_requirements_interval,
)
from app.pipeline.ops_store import load_timeseries_any
from app.pipeline.settings_store import load_settings

_CONSOLIDATED_CACHE: dict[tuple[str, int, str, int], dict] = {}


def _cache_key(scope_key: str, ivl_min: int, plan_date: dt.date | None, version_token: Any) -> tuple[str, int, str, int]:
    try:
        ver = int(version_token or 0)
    except Exception:
        ver = abs(hash(str(version_token)))
    return (str(scope_key or "global"), int(ivl_min), str(plan_date or ""), ver)


def invalidate_consolidated_cache(plan_key: Any) -> None:
    plan_key = str(plan_key or "global")
    keys = [k for k in _CONSOLIDATED_CACHE.keys() if str(k[0]) == plan_key]
    for key in keys:
        _CONSOLIDATED_CACHE.pop(key, None)


def _canon_scope(ba: Optional[str], sba: Optional[str], ch: Optional[str], site: Optional[str]) -> str:
    def _canon(value: Optional[str]) -> str:
        return (value or "").strip()

    return f"{_canon(ba)}|{_canon(sba)}|{_canon(ch)}|{_canon(site)}"


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


def _load_timeseries(kind: str, scope_key: str) -> pd.DataFrame:
    if not kind:
        return pd.DataFrame()
    return load_timeseries_any(kind, [scope_key])


def _ivl_seconds(ivl_min: int | float | str) -> int:
    try:
        return int(round(float(ivl_min))) * 60
    except Exception:
        return 1800


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def _to_frac(value: Any) -> float:
    try:
        v = float(str(value).strip().replace("%", ""))
        return v / 100.0 if v > 1 else v
    except Exception:
        return 0.0


def _weighted_avg(values: list[float], weights: list[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return 0.0
    num = 0.0
    den = 0.0
    for v, w in zip(values, weights):
        v = _safe_float(v, 0.0)
        w = _safe_float(w, 0.0)
        num += v * w
        den += w
    return num / den if den > 0 else 0.0


def _assemble_voice(scope_key: str, which: str, settings: dict) -> pd.DataFrame:
    df = _load_timeseries(f"voice_{(which or 'forecast').strip().lower()}", scope_key)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "interval", "volume", "aht_sec", "program"])

    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    else:
        out["date"] = pd.NaT
    out["interval"] = out.get("interval")
    out["volume"] = pd.to_numeric(out.get("volume"), errors="coerce")
    out["aht_sec"] = pd.to_numeric(out.get("aht_sec"), errors="coerce")
    if out["aht_sec"].isna().all():
        out["aht_sec"] = float(settings.get("target_aht", settings.get("budgeted_aht", 300)) or 300)
    out["program"] = "Voice"
    return out.dropna(subset=["date"])[["date", "interval", "volume", "aht_sec", "program"]]


def _assemble_chat(scope_key: str, which: str, settings: dict) -> pd.DataFrame:
    df = _load_timeseries(f"chat_{(which or 'forecast').strip().lower()}", scope_key)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "interval", "items", "aht_sec", "program"])

    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    else:
        out["date"] = pd.NaT
    out["interval"] = out.get("interval")
    item_col = "items" if "items" in out.columns else "volume"
    out["items"] = pd.to_numeric(out.get(item_col), errors="coerce")
    out["aht_sec"] = pd.to_numeric(out.get("aht_sec"), errors="coerce")
    if out["aht_sec"].isna().all():
        out["aht_sec"] = float(settings.get("target_aht", settings.get("budgeted_aht", 300)) or 300)
    out["program"] = "Chat"
    return out.dropna(subset=["date"])[["date", "interval", "items", "aht_sec", "program"]]


def _assemble_ob(scope_key: str, which: str, settings: dict) -> pd.DataFrame:
    df = _load_timeseries(f"ob_{(which or 'forecast').strip().lower()}", scope_key)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "interval", "items", "aht_sec", "program"])

    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    else:
        out["date"] = pd.NaT
    out["interval"] = out.get("interval")
    item_col = "items" if "items" in out.columns else "volume"
    out["items"] = pd.to_numeric(out.get(item_col), errors="coerce")
    out["aht_sec"] = pd.to_numeric(out.get("aht_sec"), errors="coerce")
    if out["aht_sec"].isna().all():
        out["aht_sec"] = float(settings.get("target_aht", settings.get("budgeted_aht", 300)) or 300)
    out["program"] = "Outbound"
    return out.dropna(subset=["date"])[["date", "interval", "items", "aht_sec", "program"]]


def _assemble_bo(scope_key: str, which: str, settings: dict) -> pd.DataFrame:
    df = _load_timeseries(f"bo_{(which or 'forecast').strip().lower()}", scope_key)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "interval", "items", "aht_sec", "program"])

    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    else:
        out["date"] = pd.NaT
    out["interval"] = out.get("interval")
    item_col = "items" if "items" in out.columns else "volume"
    out["items"] = pd.to_numeric(out.get(item_col), errors="coerce")
    aht_col = "sut_sec" if "sut_sec" in out.columns else "aht_sec"
    out["aht_sec"] = pd.to_numeric(out.get(aht_col), errors="coerce")
    if out["aht_sec"].isna().all():
        out["aht_sec"] = float(settings.get("target_sut", settings.get("budgeted_sut", 600)) or 600)
    out["program"] = "Back Office"
    return out.dropna(subset=["date"])[["date", "interval", "items", "aht_sec", "program"]]


def _voice_interval_calc(ivl_df: pd.DataFrame, settings: dict, ivl_min: int) -> pd.DataFrame:
    if ivl_df is None or ivl_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "interval",
                "program",
                "volume",
                "aht_sec",
                "agents_req",
                "staff_seconds",
                "phc",
                "service_level",
                "occupancy",
            ]
        )

    base = voice_requirements_interval(ivl_df, settings)
    if base is None or base.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "interval",
                "program",
                "volume",
                "aht_sec",
                "agents_req",
                "staff_seconds",
                "phc",
                "service_level",
                "occupancy",
            ]
        )
    df = base.copy()
    df["volume"] = pd.to_numeric(df.get("calls"), errors="coerce").fillna(0.0)
    df["aht_sec"] = pd.to_numeric(df.get("aht_sec"), errors="coerce").fillna(0.0)
    df["agents_req"] = pd.to_numeric(df.get("agents_req"), errors="coerce").fillna(0.0)
    df["staff_seconds"] = pd.to_numeric(df.get("staff_seconds"), errors="coerce").fillna(0.0)
    df["service_level"] = pd.to_numeric(df.get("service_level"), errors="coerce").fillna(0.0) * 100.0
    df["occupancy"] = pd.to_numeric(df.get("occupancy"), errors="coerce").fillna(0.0)
    df["phc"] = (df["staff_seconds"] / df["aht_sec"].replace({0: np.nan})).fillna(0.0)
    df["program"] = df.get("program", "Voice")
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.date
    df["interval"] = df.get("interval")
    return (
        df[
            [
                "date",
                "interval",
                "program",
                "volume",
                "aht_sec",
                "agents_req",
                "staff_seconds",
                "phc",
                "service_level",
                "occupancy",
            ]
        ]
        .sort_values(["date", "interval", "program"])
        .reset_index(drop=True)
    )


def _chat_interval_calc(ivl_df: pd.DataFrame, settings: dict, ivl_min: int) -> pd.DataFrame:
    if ivl_df is None or ivl_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "interval",
                "program",
                "items",
                "aht_sec",
                "agents_req",
                "staff_seconds",
                "phc",
                "service_level",
                "occupancy",
            ]
        )

    df = ivl_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    ivl_sec = _ivl_seconds(settings.get("interval_minutes", ivl_min))
    target_sl = _safe_float(settings.get("chat_target_sl", settings.get("target_sl", 0.80)), 0.80)
    target_sec = _safe_float(settings.get("chat_sl_seconds", settings.get("sl_seconds", 20)), 20.0)
    occ_cap = settings.get("occupancy_cap_chat", settings.get("util_chat", settings.get("occupancy_cap", 0.85)))
    conc = max(0.1, _safe_float(settings.get("chat_concurrency", 1.5), 1.5))

    rows = []
    for _, row in df.iterrows():
        items = _safe_float(row.get("items") if "items" in df.columns else row.get("volume"), 0.0)
        aht = _safe_float(row.get("aht_sec"), 0.0)
        aht_eff = aht / conc
        agents, sl, occ, _asa = min_agents(items, aht_eff, int(ivl_sec / 60), target_sl, target_sec, occ_cap)
        staff_sec = agents * ivl_sec
        phc = (agents * ivl_sec / max(1e-6, aht_eff)) if aht_eff > 0 else 0.0
        rows.append(
            {
                "date": row["date"],
                "interval": row.get("interval"),
                "program": row.get("program", "Chat"),
                "items": items,
                "aht_sec": aht,
                "agents_req": agents,
                "staff_seconds": staff_sec,
                "phc": phc,
                "service_level": sl * 100.0,
                "occupancy": occ,
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "interval", "program"]).reset_index(drop=True)


def _ob_interval_calc(ivl_df: pd.DataFrame, settings: dict, ivl_min: int) -> pd.DataFrame:
    if ivl_df is None or ivl_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "interval",
                "program",
                "items",
                "aht_sec",
                "agents_req",
                "staff_seconds",
                "phc",
                "service_level",
                "occupancy",
            ]
        )

    df = ivl_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    ivl_sec = _ivl_seconds(settings.get("interval_minutes", ivl_min))
    target_sl = _safe_float(settings.get("ob_target_sl", settings.get("target_sl", 0.80)), 0.80)
    target_sec = _safe_float(settings.get("ob_sl_seconds", settings.get("sl_seconds", 20)), 20.0)
    occ_cap = settings.get("occupancy_cap_ob", settings.get("util_ob", settings.get("occupancy_cap", 0.85)))

    rows = []
    for _, row in df.iterrows():
        items = _safe_float(row.get("items") or row.get("volume"), 0.0)
        aht = _safe_float(row.get("aht_sec"), 0.0)
        agents, sl, occ, _asa = min_agents(items, aht, int(ivl_sec / 60), target_sl, target_sec, occ_cap)
        staff_sec = agents * ivl_sec
        phc = (agents * ivl_sec / max(1e-6, aht)) if aht > 0 else 0.0
        rows.append(
            {
                "date": row["date"],
                "interval": row.get("interval"),
                "program": row.get("program", "Outbound"),
                "items": items,
                "aht_sec": aht,
                "agents_req": agents,
                "staff_seconds": staff_sec,
                "phc": phc,
                "service_level": sl * 100.0,
                "occupancy": occ,
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "interval", "program"]).reset_index(drop=True)


def _bo_daily_calc(bo_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if bo_df is None or bo_df.empty:
        return pd.DataFrame(columns=["date", "program", "items", "aht_sec", "fte_req", "phc", "service_level"])

    df = bo_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    hrs = _safe_float(settings.get("hours_per_fte", 8.0), 8.0)
    shrink = _to_frac(settings.get("bo_shrinkage_pct", settings.get("shrinkage_pct", 0.30)))
    util = _safe_float(settings.get("util_bo", 0.85), 0.85)
    denom_tat = max(1e-6, hrs * 3600.0 * (1.0 - shrink) * util)

    model = str(settings.get("bo_capacity_model", "tat")).lower()
    target_sl = _safe_float(settings.get("target_sl", 0.80), 0.80)
    target_sec = _safe_float(settings.get("sl_seconds", 20), 20.0)
    cov_min = int(round(_safe_float(settings.get("bo_hours_per_day", hrs), hrs) * 60.0))
    occ_cap = settings.get("occupancy_cap", 0.85)

    rows = []
    for _, row in df.iterrows():
        items = _safe_float(row.get("items") or row.get("volume"), 0.0)
        aht = _safe_float(row.get("aht_sec") or row.get("sut_sec") or row.get("sut"), 0.0)
        if model == "tat":
            fte = (items * aht) / denom_tat
            phc = None
            slp = None
        else:
            agents, sl, _occ, _asa = min_agents(items, aht, cov_min, target_sl, target_sec, occ_cap)
            denom = max(1e-6, hrs * 3600.0 * (1.0 - _to_frac(settings.get("shrinkage_pct", 0.30))))
            fte = (agents * cov_min * 60.0) / denom
            phc = (agents * cov_min * 60.0) / max(1e-6, aht) if aht > 0 else 0.0
            slp = sl * 100.0
        rows.append(
            {
                "date": row["date"],
                "program": row.get("program", "Back Office"),
                "items": items,
                "aht_sec": aht,
                "fte_req": fte,
                "phc": phc,
                "service_level": slp,
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "program"]).reset_index(drop=True)


def _chat_daily_calc(chat_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    daily = chat_fte_daily(chat_df, settings)
    if daily is None or daily.empty:
        return pd.DataFrame(columns=["date", "program", "fte_req", "phc", "service_level"])
    out = daily.rename(columns={"chat_fte": "fte_req"}).copy()
    out["phc"] = 0.0
    out["service_level"] = 0.0
    return out[["date", "program", "fte_req", "phc", "service_level"]].sort_values(["date", "program"])


def _ob_daily_calc(ob_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if ob_df is None or ob_df.empty:
        return pd.DataFrame(columns=["date", "program", "fte_req", "phc", "service_level"])
    empty = pd.DataFrame(columns=["date", "program", "calls", "aht_sec"])
    daily = required_fte_daily(empty, empty, ob_df, settings)
    if daily is None or daily.empty:
        return pd.DataFrame(columns=["date", "program", "fte_req", "phc", "service_level"])
    out = daily.rename(columns={"ob_fte": "fte_req"}).copy()
    out["phc"] = 0.0
    out["service_level"] = 0.0
    return out[["date", "program", "fte_req", "phc", "service_level"]].sort_values(["date", "program"])


def _daily_from_intervals(ivl_df: pd.DataFrame, settings: dict, weight_col: str) -> pd.DataFrame:
    if ivl_df is None or ivl_df.empty:
        return pd.DataFrame(columns=["date", "program", "fte_req", "phc", "service_level", "arrival_load"])

    df = ivl_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    hrs = _safe_float(settings.get("hours_per_fte", 8.0), 8.0)
    shrink = _to_frac(settings.get("shrinkage_pct", 0.30))
    denom = max(1e-6, hrs * 3600.0 * (1.0 - shrink))

    grouped = df.groupby(["date", "program"], as_index=False).agg(
        staff_seconds=("staff_seconds", "sum"),
        phc=("phc", "sum"),
        arrival_load=(weight_col, "sum"),
    )

    sl_rows = []
    for (day, program), grp in df.groupby(["date", "program"]):
        w = grp[weight_col].astype(float).values.tolist()
        sls = grp["service_level"].astype(float).values.tolist()
        sl_rows.append((day, program, _weighted_avg(sls, w)))

    sl_df = pd.DataFrame(sl_rows, columns=["date", "program", "service_level"])
    out = grouped.merge(sl_df, on=["date", "program"], how="left")
    out["fte_req"] = out["staff_seconds"] / denom
    return out[["date", "program", "fte_req", "phc", "service_level", "arrival_load"]].sort_values(["date", "program"])


def _weekly_from_daily(day_df: pd.DataFrame) -> pd.DataFrame:
    if day_df is None or day_df.empty:
        return pd.DataFrame(columns=["week", "program", "fte_req", "phc", "service_level"])

    df = day_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["week"] = df["date"].apply(lambda d: d - dt.timedelta(days=d.weekday()))
    if "arrival_load" not in df.columns:
        df["arrival_load"] = 0.0

    rows = []
    for (week, program), grp in df.groupby(["week", "program"], as_index=False):
        fte = pd.to_numeric(grp["fte_req"], errors="coerce").fillna(0.0).sum()
        phc = pd.to_numeric(grp["phc"], errors="coerce").fillna(0.0).sum()
        loads = pd.to_numeric(grp["arrival_load"], errors="coerce").fillna(0.0).values.tolist()
        sls = pd.to_numeric(grp["service_level"], errors="coerce").fillna(0.0).values.tolist()
        sl = _weighted_avg(sls, loads) if sum(loads) > 0 else _weighted_avg(sls, [1.0] * len(sls))
        rows.append({"week": week, "program": program, "fte_req": fte, "phc": phc, "service_level": sl})
    return pd.DataFrame(rows).sort_values(["week", "program"]).reset_index(drop=True)


def _monthly_from_daily(day_df: pd.DataFrame) -> pd.DataFrame:
    if day_df is None or day_df.empty:
        return pd.DataFrame(columns=["month", "program", "fte_req", "phc", "service_level"])

    df = day_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp().dt.date
    if "arrival_load" not in df.columns:
        df["arrival_load"] = 0.0

    rows = []
    for (month, program), grp in df.groupby(["month", "program"], as_index=False):
        fte = pd.to_numeric(grp["fte_req"], errors="coerce").fillna(0.0).sum()
        phc = pd.to_numeric(grp["phc"], errors="coerce").fillna(0.0).sum()
        loads = pd.to_numeric(grp["arrival_load"], errors="coerce").fillna(0.0).values.tolist()
        sls = pd.to_numeric(grp["service_level"], errors="coerce").fillna(0.0).values.tolist()
        sl = _weighted_avg(sls, loads) if sum(loads) > 0 else _weighted_avg(sls, [1.0] * len(sls))
        rows.append({"month": month, "program": program, "fte_req": fte, "phc": phc, "service_level": sl})
    return pd.DataFrame(rows).sort_values(["month", "program"]).reset_index(drop=True)


def _has_intervals(df: pd.DataFrame) -> bool:
    if not isinstance(df, pd.DataFrame) or df.empty or "interval" not in df.columns:
        return False
    try:
        series = df["interval"].replace("", pd.NA)
        return bool(pd.Series(series).notna().any())
    except Exception:
        return False


def consolidated_calcs(
    scope: dict,
    *,
    grain: str = "week",
    plan_date: Optional[dt.date] = None,
    settings: Optional[dict] = None,
    ivl_min_override: Optional[int] = None,
) -> dict:
    scope = scope or {}
    ba = scope.get("business_area") or scope.get("ba")
    sba = scope.get("sub_business_area") or scope.get("sba")
    ch = scope.get("channel") or scope.get("lob")
    site = scope.get("site")
    loc = scope.get("location") or scope.get("country")

    settings = dict(settings or _resolve_settings(ba, sba, ch, site, loc))
    ivl_min = int(float(ivl_min_override if ivl_min_override is not None else settings.get("interval_minutes", 30) or 30))

    scope_key = _canon_scope(ba, sba, ch, site or loc)
    voice_f = _assemble_voice(scope_key, "forecast", settings)
    voice_a = _assemble_voice(scope_key, "actual", settings)
    voice_t = _assemble_voice(scope_key, "tactical", settings)

    chat_f = _assemble_chat(scope_key, "forecast", settings)
    chat_a = _assemble_chat(scope_key, "actual", settings)
    chat_t = _assemble_chat(scope_key, "tactical", settings)

    ob_f = _assemble_ob(scope_key, "forecast", settings)
    ob_a = _assemble_ob(scope_key, "actual", settings)
    ob_t = _assemble_ob(scope_key, "tactical", settings)

    bo_f = _assemble_bo(scope_key, "forecast", settings)
    bo_a = _assemble_bo(scope_key, "actual", settings)
    bo_t = _assemble_bo(scope_key, "tactical", settings)

    res: dict[str, pd.DataFrame] = {}

    voice_ivl_f = _voice_interval_calc(voice_f if _has_intervals(voice_f) else pd.DataFrame(), settings, ivl_min)
    voice_ivl_a = _voice_interval_calc(voice_a if _has_intervals(voice_a) else pd.DataFrame(), settings, ivl_min)
    voice_ivl_t = _voice_interval_calc(voice_t if _has_intervals(voice_t) else pd.DataFrame(), settings, ivl_min)
    res["voice_ivl_f"], res["voice_ivl_a"], res["voice_ivl_t"] = voice_ivl_f, voice_ivl_a, voice_ivl_t
    res["voice_day_f"] = _daily_from_intervals(voice_ivl_f, settings, "volume") if not voice_ivl_f.empty else pd.DataFrame()
    res["voice_day_a"] = _daily_from_intervals(voice_ivl_a, settings, "volume") if not voice_ivl_a.empty else pd.DataFrame()
    res["voice_day_t"] = _daily_from_intervals(voice_ivl_t, settings, "volume") if not voice_ivl_t.empty else pd.DataFrame()
    res["voice_ivl"] = voice_ivl_f
    res["voice_day"] = res["voice_day_f"]
    res["voice_week"] = _weekly_from_daily(res["voice_day"]) if not res["voice_day"].empty else pd.DataFrame()
    res["voice_month"] = _monthly_from_daily(res["voice_day"]) if not res["voice_day"].empty else pd.DataFrame()

    chat_ivl_f = pd.DataFrame()
    chat_ivl_a = pd.DataFrame()
    chat_ivl_t = pd.DataFrame()
    res["chat_ivl_f"], res["chat_ivl_a"], res["chat_ivl_t"] = chat_ivl_f, chat_ivl_a, chat_ivl_t
    res["chat_day_f"] = _chat_daily_calc(chat_f, settings)
    res["chat_day_a"] = _chat_daily_calc(chat_a, settings)
    res["chat_day_t"] = _chat_daily_calc(chat_t, settings)
    res["chat_ivl"] = chat_ivl_f
    res["chat_day"] = res["chat_day_f"]
    res["chat_week"] = _weekly_from_daily(res["chat_day"]) if not res["chat_day"].empty else pd.DataFrame()
    res["chat_month"] = _monthly_from_daily(res["chat_day"]) if not res["chat_day"].empty else pd.DataFrame()

    ob_ivl_f = pd.DataFrame()
    ob_ivl_a = pd.DataFrame()
    ob_ivl_t = pd.DataFrame()
    res["ob_ivl_f"], res["ob_ivl_a"], res["ob_ivl_t"] = ob_ivl_f, ob_ivl_a, ob_ivl_t
    res["ob_day_f"] = _ob_daily_calc(ob_f, settings)
    res["ob_day_a"] = _ob_daily_calc(ob_a, settings)
    res["ob_day_t"] = _ob_daily_calc(ob_t, settings)
    res["ob_ivl"] = ob_ivl_f
    res["ob_day"] = res["ob_day_f"]
    res["ob_week"] = _weekly_from_daily(res["ob_day"]) if not res["ob_day"].empty else pd.DataFrame()
    res["ob_month"] = _monthly_from_daily(res["ob_day"]) if not res["ob_day"].empty else pd.DataFrame()

    model = str(settings.get("bo_capacity_model", "tat")).lower()
    def _bo_day(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "program", "fte_req", "phc", "service_level"])
        roll = bo_erlang_rollups(df, settings)["daily"] if model == "erlang" else bo_rollups(df, settings)["daily"]
        if roll is None or roll.empty:
            return pd.DataFrame(columns=["date", "program", "fte_req", "phc", "service_level"])
        out = roll.rename(columns={"fte_req": "fte_req"}).copy()
        out["phc"] = 0.0
        out["service_level"] = 0.0
        return out[["date", "program", "fte_req", "phc", "service_level"]]

    res["bo_day_f"] = _bo_day(bo_f)
    res["bo_day_a"] = _bo_day(bo_a)
    res["bo_day_t"] = _bo_day(bo_t)
    res["bo_day"] = res["bo_day_f"]
    res["bo_week"] = _weekly_from_daily(res["bo_day"]) if not res["bo_day"].empty else pd.DataFrame()
    res["bo_month"] = _monthly_from_daily(res["bo_day"]) if not res["bo_day"].empty else pd.DataFrame()

    return res


def get_cached_consolidated_calcs(
    scope: dict,
    *,
    settings: dict | None = None,
    plan_date: dt.date | None = None,
    version_token: Any = None,
) -> dict:
    scope = scope or {}
    ba = scope.get("business_area") or scope.get("ba")
    sba = scope.get("sub_business_area") or scope.get("sba")
    ch = scope.get("channel") or scope.get("lob")
    site = scope.get("site")
    loc = scope.get("location") or scope.get("country")
    scope_key = _canon_scope(ba, sba, ch, site or loc)
    effective_settings = dict(settings or _resolve_settings(ba, sba, ch, site, loc))
    ivl_min = int(float(effective_settings.get("interval_minutes", 30) or 30))
    key = _cache_key(scope_key, ivl_min, plan_date, version_token)
    cached = _CONSOLIDATED_CACHE.get(key)
    if cached is not None:
        return cached
    bundle = consolidated_calcs(scope, plan_date=plan_date, settings=effective_settings, ivl_min_override=ivl_min)
    if len(_CONSOLIDATED_CACHE) > 64:
        _CONSOLIDATED_CACHE.pop(next(iter(_CONSOLIDATED_CACHE)), None)
    _CONSOLIDATED_CACHE[key] = bundle
    return bundle


def serialize_bundle(bundle: dict) -> dict:
    payload = {}
    for key, value in (bundle or {}).items():
        if isinstance(value, pd.DataFrame):
            payload[key] = value.to_json(date_format="iso", orient="split")
    return payload
