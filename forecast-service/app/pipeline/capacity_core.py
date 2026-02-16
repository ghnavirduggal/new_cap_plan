from __future__ import annotations

import datetime as dt
import math
import re
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd


MAX_ACCURACY = 0.00001
MAX_LOOPS = 100


def _cacheable(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _to_frac(value) -> float:
    try:
        v = float(value)
        return v / 100.0 if v > 1.0 else v
    except Exception:
        try:
            text = str(value or "").strip()
            if text.endswith("%"):
                text = text[:-1]
            v = float(text)
            return v / 100.0 if v > 1.0 else v
        except Exception:
            return 0.0


def week_floor(day: dt.date | pd.Timestamp | str, week_start: str = "Monday") -> dt.date:
    d = pd.to_datetime(day, errors="coerce").date()
    wd = d.weekday()
    if (week_start or "Monday").lower().startswith("sun"):
        return d - dt.timedelta(days=(wd + 1) % 7)
    return d - dt.timedelta(days=wd)


def add_week_month_keys(df: pd.DataFrame, date_col: str, week_start: str = "Monday") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date
    out["week"] = out[date_col].apply(lambda x: week_floor(x, week_start))
    out["month"] = pd.to_datetime(out[date_col]).dt.to_period("M").dt.start_time.dt.date
    return out


def _ivl_minutes_from_str(value, default: int = 30) -> int:
    try:
        if isinstance(value, (int, float)) and not pd.isna(value):
            v = int(value)
            return v if v > 0 else default
        text = str(value)
        match = re.search(r"(\d{1,2}):(\d{2})\s*(?:-\s*(\d{1,2}):(\d{2}))?", text)
        if match:
            h1, m1 = int(match.group(1)), int(match.group(2))
            if match.group(3):
                h2, m2 = int(match.group(3)), int(match.group(4))
                t1 = h1 * 60 + m1
                t2 = h2 * 60 + m2
                diff = (t2 - t1) % (24 * 60)
                return diff if diff > 0 else default
    except Exception:
        pass
    return int(default)


def erlang_b(traffic: float, agents: int) -> float:
    if agents <= 0:
        return 1.0
    block = 1.0
    for n in range(1, agents + 1):
        block = (traffic * block) / (n + traffic * block)
    return block


def erlang_c(traffic: float, agents: int) -> float:
    if agents <= 0:
        return 1.0
    if traffic <= 0:
        return 0.0
    if agents <= traffic:
        return 1.0
    rho = traffic / agents
    block = erlang_b(traffic, agents)
    denom = 1 - rho + rho * block
    if denom <= 0:
        return 1.0
    return block / denom


def service_level(traffic: float, agents: int, aht_sec: float, target_sec: float) -> float:
    if agents <= 0:
        return 0.0
    if traffic <= 0:
        return 1.0
    pw = erlang_c(traffic, agents)
    gap = agents - traffic
    if gap <= 0:
        return 0.0
    return 1.0 - pw * math.exp(-gap * (target_sec / max(aht_sec, 1e-9)))


def asa(traffic: float, agents: int, aht_sec: float) -> float:
    if agents <= 0 or traffic <= 0 or agents <= traffic:
        return float("inf")
    pw = erlang_c(traffic, agents)
    return (pw * aht_sec) / (agents - traffic)


def offered_load_erlangs(calls: float, aht_sec: float, interval_minutes: int) -> float:
    interval_minutes = max(5, int(interval_minutes or 30))
    return (calls * aht_sec) / (interval_minutes * 60.0)


def _fractional_agents_impl(target_sl: float, target_sec: float, calls_per_hour: float, aht_sec: float) -> float:
    target_sl = float(target_sl or 0.0)
    if target_sl > 1.0:
        target_sl = 1.0
    if calls_per_hour <= 0 or aht_sec <= 0:
        return 0.0
    traffic = (calls_per_hour * aht_sec) / 3600.0
    erlangs = int((calls_per_hour * aht_sec) / 3600.0 + 0.5)
    agents = 1 if erlangs < 1 else int(erlangs)
    util = traffic / agents if agents > 0 else 0.0
    while util >= 1.0:
        agents += 1
        util = traffic / agents
    sl_queued = 0.0
    last_slq = 0.0
    max_iterate = agents * 100
    for _ in range(1, max_iterate + 1):
        last_slq = sl_queued
        util = traffic / agents
        if util < 1.0:
            c = erlang_c(traffic, agents)
            sl_queued = 1.0 - c * math.exp((traffic - agents) * (target_sec / aht_sec))
            if sl_queued < 0.0:
                sl_queued = 0.0
            if sl_queued > 1.0:
                sl_queued = 1.0
            if sl_queued >= target_sl or sl_queued > (1.0 - MAX_ACCURACY):
                break
        if agents < max_iterate:
            agents += 1
    agents_float = float(agents)
    if sl_queued > target_sl:
        one_agent = sl_queued - last_slq
        if one_agent > 0:
            agents_float = (target_sl - last_slq) / one_agent + (agents - 1)
    return agents_float


@lru_cache(maxsize=200_000)
def _fractional_agents_cached(target_sl, target_sec, calls_per_hour, aht_sec) -> float:
    return _fractional_agents_impl(target_sl, target_sec, calls_per_hour, aht_sec)


def fractional_agents(target_sl: float, target_sec: float, calls_per_hour: float, aht_sec: float) -> float:
    try:
        return _fractional_agents_cached(
            _cacheable(target_sl),
            _cacheable(target_sec),
            _cacheable(calls_per_hour),
            _cacheable(aht_sec),
        )
    except Exception:
        return _fractional_agents_impl(target_sl, target_sec, calls_per_hour, aht_sec)


def _min_agents_impl(
    calls: float,
    aht_sec: float,
    ivl_min: int,
    target_sl: float,
    target_sec: float,
    occ_cap: Optional[float] = None,
    asa_cap: Optional[float] = None,
    n_cap: int = 2000,
) -> Tuple[int, float, float, float]:
    traffic = offered_load_erlangs(calls, aht_sec, ivl_min)
    if traffic <= 0:
        return 0, 1.0, 0.0, 0.0
    start = max(1, math.ceil(traffic))
    for agents in range(start, min(start + 1000, n_cap)):
        sl = service_level(traffic, agents, aht_sec, target_sec)
        occ = traffic / agents
        avg_asa = asa(traffic, agents, aht_sec)
        ok = True
        if occ_cap is not None and occ > occ_cap:
            ok = False
        if target_sl is not None and sl < target_sl:
            ok = False
        if asa_cap is not None and avg_asa > asa_cap:
            ok = False
        if ok:
            return agents, sl, occ, avg_asa
    agents = min(start + 1000, n_cap) - 1
    return agents, service_level(traffic, agents, aht_sec, target_sec), traffic / max(agents, 1), asa(traffic, agents, aht_sec)


@lru_cache(maxsize=200_000)
def _min_agents_cached(calls, aht_sec, ivl_min, target_sl, target_sec, occ_cap, asa_cap, n_cap):
    return _min_agents_impl(calls, aht_sec, ivl_min, target_sl, target_sec, occ_cap, asa_cap, n_cap)


def min_agents(
    calls: float,
    aht_sec: float,
    ivl_min: int,
    target_sl: float,
    target_sec: float,
    occ_cap: Optional[float] = None,
    asa_cap: Optional[float] = None,
    n_cap: int = 2000,
) -> Tuple[int, float, float, float]:
    try:
        return _min_agents_cached(
            _cacheable(calls),
            _cacheable(aht_sec),
            _cacheable(ivl_min),
            _cacheable(target_sl),
            _cacheable(target_sec),
            _cacheable(occ_cap),
            _cacheable(asa_cap),
            _cacheable(n_cap),
        )
    except Exception:
        return _min_agents_impl(calls, aht_sec, ivl_min, target_sl, target_sec, occ_cap, asa_cap, n_cap)


def clear_capacity_cache() -> None:
    _min_agents_cached.cache_clear()
    _fractional_agents_cached.cache_clear()


def voice_requirements_interval(voice_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if not isinstance(voice_df, pd.DataFrame) or voice_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "interval",
                "program",
                "calls",
                "aht_sec",
                "A_erlangs",
                "agents_req",
                "service_level",
                "occupancy",
                "asa_sec",
                "staff_seconds",
            ]
        )

    df = voice_df.copy()
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date")
    interval_col = cols.get("interval") or cols.get("interval_start") or cols.get("time")
    calls_col = cols.get("calls") or cols.get("volume")
    aht_col = cols.get("aht_sec") or cols.get("aht (sec)") or cols.get("aht")
    program_col = cols.get("program") or cols.get("business area")

    if not all([date_col, interval_col, calls_col, aht_col]):
        return pd.DataFrame(
            columns=[
                "date",
                "interval",
                "program",
                "calls",
                "aht_sec",
                "A_erlangs",
                "agents_req",
                "service_level",
                "occupancy",
                "asa_sec",
                "staff_seconds",
            ]
        )

    ivl_min_default = int(settings.get("interval_minutes", 30) or 30)
    target_sl = float(settings.get("target_sl", 0.8) or 0.8)
    target_sec = float(settings.get("sl_seconds", 20) or 20)
    occ_cap = _to_frac(settings.get("occupancy_cap_voice", 0.85) or 0.85)

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.date,
            "interval": df[interval_col].astype(str),
            "calls": pd.to_numeric(df[calls_col], errors="coerce"),
            "aht_sec": pd.to_numeric(df[aht_col], errors="coerce"),
            "program": (df[program_col].astype(str) if program_col else "All"),
        }
    ).dropna(subset=["date", "interval", "calls", "aht_sec"])

    if out.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "interval",
                "program",
                "calls",
                "aht_sec",
                "A_erlangs",
                "agents_req",
                "service_level",
                "occupancy",
                "asa_sec",
                "staff_seconds",
            ]
        )

    interval_vals = out["interval"].astype(str)
    uniq_ivl = pd.unique(interval_vals)
    ivl_map = {v: _ivl_minutes_from_str(v, ivl_min_default) for v in uniq_ivl}
    out["_ivl_sec"] = interval_vals.map(ivl_map).fillna(ivl_min_default).astype(int) * 60

    uniq = out[["calls", "aht_sec"]].drop_duplicates()
    metrics = []
    for row in uniq.itertuples(index=False):
        calls = float(row.calls or 0.0)
        aht = float(row.aht_sec or 0.0)
        calls_per_hour = calls
        traffic = offered_load_erlangs(calls, aht, 60)
        agents = fractional_agents(target_sl, target_sec, calls_per_hour, aht)
        sl = service_level(traffic, int(math.ceil(agents)) if agents > 0 else 0, aht, target_sec)
        occ = (traffic / agents) if agents > 0 else 0.0
        avg_asa = asa(traffic, int(math.ceil(agents)) if agents > 0 else 0, aht)
        metrics.append((row.calls, row.aht_sec, traffic, agents, sl, occ, avg_asa))

    metrics_df = pd.DataFrame(
        metrics,
        columns=[
            "calls",
            "aht_sec",
            "A_erlangs",
            "agents_req",
            "service_level",
            "occupancy",
            "asa_sec",
        ],
    )
    res = out.merge(metrics_df, on=["calls", "aht_sec"], how="left")
    res["staff_seconds"] = res["agents_req"] * res["_ivl_sec"]
    return res[
        [
            "date",
            "interval",
            "program",
            "calls",
            "aht_sec",
            "A_erlangs",
            "agents_req",
            "service_level",
            "occupancy",
            "asa_sec",
            "staff_seconds",
        ]
    ]


def voice_rollups(voice_ivl: pd.DataFrame, settings: dict, week_start: str = "Monday") -> dict:
    if voice_ivl is None or voice_ivl.empty:
        empty = pd.DataFrame(columns=["date", "program", "fte_req"])
        return {"interval": pd.DataFrame(), "daily": empty, "weekly": empty, "monthly": empty}

    shrink = _to_frac(settings.get("shrinkage_pct", 0.30) or 0.30)
    hours = float(settings.get("hours_per_fte", 8.0) or 8.0)
    denom = hours * 3600.0 * (1.0 - shrink)

    base = voice_ivl.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.date
    daily = base.groupby(["date", "program"], as_index=False)["staff_seconds"].sum()
    daily["fte_req"] = daily["staff_seconds"] / max(1e-6, denom)
    daily = daily[["date", "program", "fte_req"]].sort_values(["date", "program"])

    wk = add_week_month_keys(daily, "date", week_start)
    weekly = wk.groupby(["week", "program"], as_index=False)["fte_req"].sum().rename(columns={"week": "start_week"})
    monthly = wk.groupby(["month", "program"], as_index=False)["fte_req"].sum().rename(columns={"month": "month_start"})
    return {"interval": base, "daily": daily, "weekly": weekly, "monthly": monthly}


def bo_rollups(bo_df: pd.DataFrame, settings: dict, week_start: str = "Monday") -> dict:
    if bo_df is None or bo_df.empty:
        empty = pd.DataFrame(columns=["date", "program", "fte_req"])
        return {"daily": empty, "weekly": empty, "monthly": empty}

    bo_hpd = float(settings.get("bo_hours_per_day", settings.get("hours_per_fte", 8.0)) or 8.0)
    bo_wpd = float(settings.get("bo_workdays_per_week", 5.0) or 5.0)
    shrink = _to_frac(settings.get("bo_shrinkage_pct", settings.get("shrinkage_pct", 0.30)) or 0.30)
    util = float(settings.get("util_bo", 0.85) or 0.85)
    # BO linear FTE formula with utilization in denominator:
    # Daily  : items*sut / (bo_hpd*3600*(1-shrink)*util)
    # Weekly : items*sut / (bo_hpd*bo_wpd*3600*(1-shrink)*util)
    # Monthly: items*sut / (bo_hpd*bo_wpd*3600*(52/12)*(1-shrink)*util)
    denom_day = bo_hpd * 3600.0 * (1.0 - shrink) * util
    weekly_hours = bo_hpd * bo_wpd
    monthly_hours = weekly_hours * (52.0 / 12.0)
    denom_week = weekly_hours * 3600.0 * (1.0 - shrink) * util
    denom_month = monthly_hours * 3600.0 * (1.0 - shrink) * util

    df = bo_df.copy()
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date")
    items_col = cols.get("items") or cols.get("volume")
    aht_col = cols.get("aht_sec") or cols.get("sut_sec") or cols.get("sut")
    program_col = cols.get("program") or cols.get("business area")

    if not all([date_col, items_col, aht_col]):
        empty = pd.DataFrame(columns=["date", "program", "fte_req"])
        return {"daily": empty, "weekly": empty, "monthly": empty}

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.date,
            "items": pd.to_numeric(df[items_col], errors="coerce"),
            "aht_sec": pd.to_numeric(df[aht_col], errors="coerce"),
            "program": (df[program_col].astype(str) if program_col else "All"),
        }
    ).dropna(subset=["date", "items", "aht_sec"])

    out["work_seconds"] = out["items"] * out["aht_sec"]
    daily = out.groupby(["date", "program"], as_index=False)["work_seconds"].sum()
    daily["fte_req"] = daily["work_seconds"] / max(1e-6, denom_day)
    daily = daily[["date", "program", "fte_req"]].sort_values(["date", "program"])

    wk = add_week_month_keys(daily, "date", week_start)
    weekly = wk.groupby(["week", "program"], as_index=False)["work_seconds"].sum().rename(columns={"week": "start_week"})
    weekly["fte_req"] = weekly["work_seconds"] / max(1e-6, denom_week)
    weekly = weekly.drop(columns=["work_seconds"])
    monthly = wk.groupby(["month", "program"], as_index=False)["work_seconds"].sum().rename(columns={"month": "month_start"})
    monthly["fte_req"] = monthly["work_seconds"] / max(1e-6, denom_month)
    monthly = monthly.drop(columns=["work_seconds"])
    return {"daily": daily, "weekly": weekly, "monthly": monthly}


def chat_fte_daily(chat_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Daily FTE for Chat using Erlang C with concurrency adjustments."""
    if not isinstance(chat_df, pd.DataFrame) or chat_df.empty:
        return pd.DataFrame(columns=["date", "program", "chat_fte"])

    hrs = float(settings.get("hours_per_fte", 8.0) or 8.0)
    shrink = float(settings.get("chat_shrinkage_pct", settings.get("shrinkage_pct", 0.30)) or 0.30)
    target_sl = float(settings.get("chat_target_sl", settings.get("target_sl", 0.8)) or 0.8)
    target_sec = float(settings.get("chat_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
    occ_cap = settings.get("occupancy_cap_chat")
    if occ_cap is None:
        occ_cap = settings.get("util_chat", settings.get("occupancy_cap_voice", settings.get("occupancy_cap", 0.85)))
    occ_cap = float(occ_cap or 0.85)
    conc = float(settings.get("chat_concurrency", 1.5) or 1.0)
    default_interval = float(settings.get("chat_interval_minutes", settings.get("interval_minutes", 30)) or 30.0)
    coverage_min = float(settings.get("chat_coverage_minutes", hrs * 60.0) or (hrs * 60.0))

    d = chat_df.copy()
    cols = {str(c).strip().lower(): c for c in d.columns}
    date_col = cols.get("date")
    items_col = cols.get("items") or cols.get("volume")
    aht_col = cols.get("aht_sec") or cols.get("aht")
    program_col = cols.get("program") or cols.get("business area") or cols.get("journey")
    interval_col = cols.get("interval") or cols.get("time")
    ivl_minutes_col = cols.get("interval_minutes") or cols.get("interval_mins")

    if not date_col or not items_col:
        return pd.DataFrame(columns=["date", "program", "chat_fte"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(d[date_col], errors="coerce").dt.date,
            "items": pd.to_numeric(d[items_col], errors="coerce"),
        }
    )

    default_aht = float(settings.get("chat_aht_sec", settings.get("target_aht", 240)) or 240.0)
    if aht_col:
        out["aht_sec"] = pd.to_numeric(d[aht_col], errors="coerce").fillna(default_aht)
    else:
        out["aht_sec"] = default_aht

    if program_col:
        out["program"] = d[program_col].astype(str).replace("", "Chat")
    else:
        out["program"] = "Chat"

    out["interval_minutes"] = coverage_min
    if interval_col:
        out["interval_minutes"] = d[interval_col].map(lambda val: _ivl_minutes_from_str(val, default_interval)).fillna(
            coverage_min
        )
    if ivl_minutes_col:
        out["interval_minutes"] = pd.to_numeric(d[ivl_minutes_col], errors="coerce").fillna(out["interval_minutes"])

    out = out.dropna(subset=["date"]).fillna({"items": 0.0, "aht_sec": default_aht})
    if out.empty:
        return pd.DataFrame(columns=["date", "program", "chat_fte"])

    eff_conc = max(conc, 1e-6)
    out["_calls"] = pd.to_numeric(out["items"], errors="coerce").fillna(0.0)
    out["_aht_eff"] = pd.to_numeric(out["aht_sec"], errors="coerce").fillna(default_aht) / eff_conc
    out["_ivl_min"] = pd.to_numeric(out["interval_minutes"], errors="coerce").fillna(coverage_min)
    out["_ivl_min"] = out["_ivl_min"].where(out["_ivl_min"] > 0, coverage_min)
    out["_ivl_min"] = out["_ivl_min"].clip(lower=1.0)

    out["_staff_seconds"] = 0.0
    mask = (out["_calls"] > 0) & (out["_aht_eff"] > 0)
    if mask.any():
        uniq = out.loc[mask, ["_calls", "_aht_eff", "_ivl_min"]].drop_duplicates()
        rows = []
        for row in uniq.itertuples(index=False):
            calls = float(row._calls or 0.0)
            aht = float(row._aht_eff or 0.0)
            ivl_min = float(row._ivl_min or coverage_min)
            agents, _sl, _occ, _asa = min_agents(calls, aht, int(round(ivl_min)), target_sl, target_sec, occ_cap)
            rows.append((row._calls, row._aht_eff, row._ivl_min, agents))
        agents_df = pd.DataFrame(rows, columns=["_calls", "_aht_eff", "_ivl_min", "_agents"])
        out = out.merge(agents_df, on=["_calls", "_aht_eff", "_ivl_min"], how="left")
        out["_staff_seconds"] = out["_agents"].fillna(0.0) * out["_ivl_min"] * 60.0
    agg = out.groupby(["date", "program"], as_index=False)["_staff_seconds"].sum()
    denom = hrs * 3600.0 * max(1e-6, (1.0 - shrink))
    agg["chat_fte"] = agg["_staff_seconds"] / denom if denom > 0 else 0.0
    return agg[["date", "program", "chat_fte"]]


def bo_erlang_rollups(bo_df: pd.DataFrame, settings: dict, week_start: str = "Monday") -> dict:
    if bo_df is None or bo_df.empty:
        empty = pd.DataFrame(columns=["date", "program", "fte_req"])
        return {"daily": empty, "weekly": empty, "monthly": empty}

    hours = float(settings.get("hours_per_fte", 8.0) or 8.0)
    shrink = float(settings.get("bo_shrinkage_pct", settings.get("shrinkage_pct", 0.30)) or 0.30)
    util = float(settings.get("util_bo", settings.get("occupancy_cap", 0.85)) or 0.85)
    bo_hpd = float(settings.get("bo_hours_per_day", hours) or hours)
    coverage_min = max(1.0, bo_hpd * 60.0)

    target_sl = float(settings.get("bo_target_sl", settings.get("target_sl", 0.8)) or 0.8)
    target_sec = float(settings.get("bo_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
    occ_cap = float(util or 0.85)

    df = bo_df.copy()
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date")
    items_col = cols.get("items") or cols.get("volume")
    aht_col = cols.get("aht_sec") or cols.get("sut_sec") or cols.get("sut")
    program_col = cols.get("program") or cols.get("business area")

    if not items_col:
        empty = pd.DataFrame(columns=["date", "program", "fte_req"])
        return {"daily": empty, "weekly": empty, "monthly": empty}

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col] if date_col else df.get("date"), errors="coerce").dt.date,
            "items": pd.to_numeric(df[items_col], errors="coerce").fillna(0.0),
        }
    )
    if aht_col:
        out["aht_sec"] = pd.to_numeric(df[aht_col], errors="coerce").fillna(np.nan)
    else:
        out["aht_sec"] = np.nan
    if program_col:
        out["program"] = df[program_col].astype(str).replace("", "Back Office")
    else:
        out["program"] = "Back Office"
    out = out.dropna(subset=["date"]).copy()

    def _agg(group: pd.DataFrame) -> pd.Series:
        items = pd.to_numeric(group["items"], errors="coerce").fillna(0.0)
        aht = pd.to_numeric(group["aht_sec"], errors="coerce")
        items_sum = float(items.sum())
        if items_sum <= 0:
            waht = float(settings.get("target_sut", settings.get("budgeted_sut", 600)) or 600.0)
        else:
            waht = float((items * aht.fillna(0.0)).sum() / max(1.0, items_sum))
        return pd.Series({"items": items_sum, "aht_sec": waht})

    grouped = out.groupby(["date", "program"], as_index=False).apply(_agg).reset_index(drop=True)
    grouped["date"] = pd.to_datetime(grouped["date"], errors="coerce").dt.date
    grouped["calls"] = grouped["items"]
    grouped["interval_minutes"] = coverage_min
    grouped["program"] = grouped["program"].astype(str)

    if grouped.empty:
        daily = pd.DataFrame(columns=["date", "program", "fte_req"])
    else:
        uniq = grouped[["calls", "aht_sec"]].drop_duplicates()
        rows = []
        for row in uniq.itertuples(index=False):
            agents, _sl, _occ, _asa = min_agents(
                float(row.calls), float(row.aht_sec), int(round(coverage_min)), target_sl, target_sec, occ_cap
            )
            rows.append((row.calls, row.aht_sec, agents))
        agents_df = pd.DataFrame(rows, columns=["calls", "aht_sec", "fte_req"])
        daily = grouped.merge(agents_df, on=["calls", "aht_sec"], how="left")
        daily = daily[["date", "program", "fte_req"]]

    denom = hours * 3600.0 * max(1e-6, (1.0 - shrink))
    daily["fte_req"] = daily["fte_req"].apply(lambda v: float(v or 0.0) / max(1e-6, denom))

    wk = add_week_month_keys(daily, "date", week_start)
    weekly = wk.groupby(["week", "program"], as_index=False)["fte_req"].sum().rename(columns={"week": "start_week"})
    monthly = wk.groupby(["month", "program"], as_index=False)["fte_req"].sum().rename(columns={"month": "month_start"})
    return {"daily": daily, "weekly": weekly, "monthly": monthly}


def required_fte_daily(voice_df: pd.DataFrame, bo_df: pd.DataFrame, ob_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    frames = []

    try:
        voice_ivl = voice_requirements_interval(voice_df, settings)
        voice_day = voice_rollups(voice_ivl, settings)["daily"].rename(columns={"fte_req": "voice_fte"})
        frames.append(voice_day)
    except Exception:
        pass

    def _bo_daily_tat(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=["date", "program", "bo_fte"])
        d = df.copy()
        cols = {c.lower(): c for c in d.columns}
        date_col = cols.get("date")
        program_col = cols.get("program") or cols.get("journey") or cols.get("ba") or cols.get("business area")
        items_col = cols.get("items") or cols.get("txns") or cols.get("transactions") or cols.get("volume")
        sut_col = cols.get("sut") or cols.get("sut_sec") or cols.get("aht_sec") or cols.get("avg_sut")

        if not sut_col:
            sut = float(settings.get("target_sut", settings.get("budgeted_sut", 600)) or 600.0)
            d["_sut_sec_"] = float(sut)
            sut_col = "_sut_sec_"
        if not items_col:
            return pd.DataFrame(columns=["date", "program", "bo_fte"])

        d[sut_col] = pd.to_numeric(d[sut_col], errors="coerce").fillna(0.0)
        d[items_col] = pd.to_numeric(d[items_col], errors="coerce").fillna(0.0)
        d["date"] = pd.to_datetime(d[date_col] if date_col else d.get("date"), errors="coerce").dt.date

        bo_hpd = float(settings.get("bo_hours_per_day", settings.get("hours_per_fte", 8.0)) or 8.0)
        bo_shr = _to_frac(settings.get("bo_shrinkage_pct", settings.get("shrinkage_pct", 0.0)))
        util_bo = float(settings.get("util_bo", 0.85) or 0.85)
        # BO daily FTE formula includes utilization in denominator.
        denom_day = max(bo_hpd * 3600.0 * (1.0 - bo_shr) * util_bo, 1e-6)

        d["bo_fte"] = (d[items_col] * d[sut_col]) / denom_day
        if not program_col:
            d["_program_"] = "Back Office"
            program_col = "_program_"
        grouped = d.groupby(["date", program_col], as_index=False)["bo_fte"].sum()
        return grouped.rename(columns={program_col: "program"})

    try:
        model = str(settings.get("bo_capacity_model", "tat")).lower()
        if model == "tat":
            frames.append(_bo_daily_tat(bo_df, settings))
        elif model == "erlang":
            frames.append(bo_erlang_rollups(bo_df, settings)["daily"].rename(columns={"fte_req": "bo_fte"}))
        else:
            frames.append(bo_rollups(bo_df, settings)["daily"].rename(columns={"fte_req": "bo_fte"}))
    except Exception:
        pass

    if isinstance(ob_df, pd.DataFrame) and not ob_df.empty:
        d = ob_df.copy()
        cols = {c.lower(): c for c in d.columns}
        date_col = cols.get("date")
        program_col = cols.get("program") or cols.get("business area") or cols.get("journey")
        opc_col = cols.get("opc") or cols.get("dials") or cols.get("calls") or cols.get("attempts")
        conn_col = cols.get("connect_rate") or cols.get("connect%") or cols.get("connect pct") or cols.get("connect")
        rpc_col = cols.get("rpc")
        rpc_rate_col = cols.get("rpc_rate") or cols.get("rpc%") or cols.get("rpc pct")
        aht_col = cols.get("aht_sec") or cols.get("talk_sec") or cols.get("avg_talk_sec")
        interval_col = cols.get("interval") or cols.get("time")
        ivl_min_col = cols.get("interval_minutes") or cols.get("interval_mins")

        d["date"] = pd.to_datetime(d[date_col] if date_col else d.get("date"), errors="coerce").dt.date
        if opc_col:
            d["_opc"] = pd.to_numeric(d[opc_col], errors="coerce").fillna(0.0)
        else:
            d["_opc"] = 0.0
        if conn_col:
            d["_conn"] = d[conn_col].apply(_to_frac)
        else:
            d["_conn"] = 0.0
        if rpc_col:
            d["_rpc"] = pd.to_numeric(d[rpc_col], errors="coerce").fillna(0.0)
        else:
            d["_rpc"] = 0.0
        if rpc_rate_col:
            d["_rpc_rate"] = d[rpc_rate_col].apply(_to_frac)
        else:
            d["_rpc_rate"] = 0.0
        if aht_col:
            d["_aht"] = pd.to_numeric(d[aht_col], errors="coerce").fillna(0.0)
        else:
            d["_aht"] = float(settings.get("ob_aht_sec", 240) or 240.0)
        if interval_col:
            d["_interval"] = d[interval_col].astype(str)
        else:
            d["_interval"] = ""
        if ivl_min_col:
            d["_interval_minutes"] = pd.to_numeric(d[ivl_min_col], errors="coerce").fillna(
                float(settings.get("interval_minutes", 30) or 30)
            )
        else:
            d["_interval_minutes"] = float(settings.get("interval_minutes", 30) or 30)

        target_sl_ob = float(settings.get("ob_target_sl", settings.get("target_sl", 0.8)) or 0.8)
        target_sec_ob = float(settings.get("ob_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
        occ_cap_ob = float(settings.get("util_ob", settings.get("occupancy_cap_voice", 0.85)) or 0.85)
        shrink_ob = float(settings.get("ob_shrinkage_pct", settings.get("shrinkage_pct", 0.3)) or 0.3)
        hours = float(settings.get("hours_per_fte", 8.0) or 8.0)
        coverage_min = float(settings.get("interval_minutes", 30) or 30)

        d["_contacts"] = d["_opc"] * d["_conn"] * (d["_rpc"] if d["_rpc"].sum() else d["_rpc_rate"])
        if not program_col:
            d["_program"] = "Outbound"
            program_col = "_program"
        else:
            d[program_col] = d[program_col].astype(str).replace("", "Outbound")

        d["_staff_seconds"] = 0.0
        d["_calls"] = pd.to_numeric(d["_contacts"], errors="coerce").fillna(0.0)
        d["_aht_eff"] = pd.to_numeric(d["_aht"], errors="coerce").fillna(0.0)
        d["_ivl_min"] = pd.to_numeric(d["_interval_minutes"], errors="coerce").fillna(coverage_min)
        d["_ivl_min"] = d["_ivl_min"].where(d["_ivl_min"] > 0, coverage_min)
        d["_ivl_min"] = d["_ivl_min"].clip(lower=1.0)
        mask = (d["_calls"] > 0) & (d["_aht_eff"] > 0)
        if mask.any():
            uniq = d.loc[mask, ["_calls", "_aht_eff", "_ivl_min"]].drop_duplicates()
            rows = []
            for row in uniq.itertuples(index=False):
                calls = float(row._calls or 0.0)
                aht = float(row._aht_eff or 0.0)
                ivl_min = float(row._ivl_min or coverage_min)
                agents, _sl, _occ, _asa = min_agents(
                    calls, aht, int(round(ivl_min)), target_sl_ob, target_sec_ob, occ_cap_ob
                )
                rows.append((row._calls, row._aht_eff, row._ivl_min, agents))
            agents_df = pd.DataFrame(rows, columns=["_calls", "_aht_eff", "_ivl_min", "_agents"])
            d = d.merge(agents_df, on=["_calls", "_aht_eff", "_ivl_min"], how="left")
            d["_staff_seconds"] = d["_agents"].fillna(0.0) * d["_ivl_min"] * 60.0
        grouped = d.groupby(["date", program_col], as_index=False)["_staff_seconds"].sum()
        grouped["date"] = pd.to_datetime(grouped["date"], errors="coerce").dt.date
        grouped = grouped.dropna(subset=["date"])
        denom = hours * 3600.0 * max(1e-6, (1.0 - shrink_ob))
        grouped["ob_fte"] = grouped["_staff_seconds"] / denom if denom > 0 else 0.0
        grouped = grouped.rename(columns={program_col: "program"})[["date", "program", "ob_fte"]]
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(columns=["date", "program", "voice_fte", "bo_fte", "ob_fte", "total_req_fte"])

    out = frames[0]
    for frame in frames[1:]:
        out = pd.merge(out, frame, on=["date", "program"], how="outer")
    for col in ["voice_fte", "bo_fte", "ob_fte"]:
        if col not in out:
            out[col] = 0.0
    out["total_req_fte"] = out[["voice_fte", "bo_fte", "ob_fte"]].fillna(0).sum(axis=1)
    return out.fillna(0)


def _norm_col_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _row_pick(row: pd.Series, names: list[str]) -> object:
    norm_map = {_norm_col_key(col): col for col in row.index}
    for name in names:
        key = _norm_col_key(name)
        col = norm_map.get(key)
        if col is not None:
            return row.get(col)
    return None


def _to_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        num = float(value)
        if not np.isfinite(num):
            return None
        return num
    except Exception:
        return None


def _row_fte(row: pd.Series, default: float = 0.0) -> float:
    fte_val = _row_pick(
        row,
        [
            "fte",
            "supply_fte",
            "projected_supply_fte",
            "projected fte",
        ],
    )
    fte = _to_float(fte_val)
    if fte is not None and fte >= 0:
        return float(fte)

    hc_val = _row_pick(
        row,
        [
            "projected_supply_hc",
            "projected supply hc",
            "projected_hc",
            "projected hc",
            "supply_hc",
            "supply hc",
            "hc",
            "headcount",
        ],
    )
    hc = _to_float(hc_val)
    if hc is not None and hc >= 0:
        # HC-to-FTE conversion is 1:1 unless an explicit FTE field is provided.
        return float(hc)
    return float(default)


def _row_program(row: pd.Series) -> str:
    val = _row_pick(row, ["program", "lob", "channel", "line of business", "queue", "work_type"])
    text = str(val or "").strip()
    return text or "WFM"


def _row_is_active(row: pd.Series) -> bool:
    status = str(_row_pick(row, ["status", "current_status"]) or "").strip().lower()
    if status and status not in {"active", "a"}:
        return False

    leave_val = _row_pick(row, ["is_leave", "leave", "on_leave"])
    leave_text = str(leave_val or "").strip().lower()
    if leave_text in {"true", "1", "yes", "y"}:
        return False
    if isinstance(leave_val, bool) and leave_val:
        return False

    entry = str(_row_pick(row, ["entry", "shift", "schedule"]) or "").strip().lower()
    if entry in {"leave", "l", "off", "pto"}:
        return False
    return True


def supply_fte_daily(roster: pd.DataFrame, hiring: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    today = dt.date.today()
    horizon = today + dt.timedelta(days=28)
    date_list = [today + dt.timedelta(days=i) for i in range((horizon - today).days + 1)]

    if isinstance(roster, pd.DataFrame) and not roster.empty:
        roster_df = roster.copy()
        norm_cols = {_norm_col_key(c): c for c in roster_df.columns}
        date_col = norm_cols.get("date")

        if date_col:
            roster_df[date_col] = pd.to_datetime(roster_df[date_col], errors="coerce").dt.date
            for _, row in roster_df.iterrows():
                day = row.get(date_col)
                if not isinstance(day, dt.date):
                    continue
                if not _row_is_active(row):
                    continue
                fte = _row_fte(row, default=1.0)
                if fte <= 0:
                    continue
                rows.append({"date": day, "program": _row_program(row), "supply_fte": fte})
        else:
            for _, row in roster_df.iterrows():
                if not _row_is_active(row):
                    continue
                start_val = _row_pick(row, ["start_date", "start date", "date", "start_week"])
                end_val = _row_pick(row, ["end_date", "end date"])
                try:
                    start_date = pd.to_datetime(start_val, errors="coerce").date()
                except Exception:
                    start_date = today
                if not isinstance(start_date, dt.date):
                    start_date = today
                if str(end_val or "").strip():
                    try:
                        end_date = pd.to_datetime(end_val, errors="coerce").date()
                    except Exception:
                        end_date = horizon
                else:
                    end_date = horizon
                if not isinstance(end_date, dt.date):
                    end_date = horizon
                fte = _row_fte(row, default=1.0)
                if fte <= 0:
                    continue
                for day in date_list:
                    if start_date <= day <= end_date:
                        rows.append({"date": day, "program": _row_program(row), "supply_fte": fte})

    if isinstance(hiring, pd.DataFrame) and not hiring.empty:
        for _, row in hiring.iterrows():
            if not _row_is_active(row):
                continue
            raw = str(_row_pick(row, ["start_week", "week", "date", "start_date"]) or "").strip()
            if not raw:
                continue
            try:
                if re.match(r"^\d{1,2}-\d{1,2}-\d{4}$", raw):
                    week_start = pd.to_datetime(raw, dayfirst=True).date()
                else:
                    week_start = pd.to_datetime(raw, errors="coerce").date()
            except Exception:
                continue
            if not isinstance(week_start, dt.date):
                continue
            fte = _row_fte(row, default=0.0)
            if fte <= 0:
                continue
            for offset in range(7):
                rows.append(
                    {
                        "date": week_start + dt.timedelta(days=offset),
                        "program": _row_program(row),
                        "supply_fte": fte,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["date", "program", "supply_fte"])

    supply = pd.DataFrame(rows)
    supply["date"] = pd.to_datetime(supply["date"], errors="coerce").dt.date
    supply = supply[pd.notna(supply["date"])]
    if supply.empty:
        return pd.DataFrame(columns=["date", "program", "supply_fte"])
    supply["supply_fte"] = pd.to_numeric(supply["supply_fte"], errors="coerce").fillna(0.0)
    supply = supply.groupby(["date", "program"], as_index=False)["supply_fte"].sum()
    return supply
