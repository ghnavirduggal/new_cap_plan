from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable, Optional
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from app.pipeline.capacity_core import chat_fte_daily, required_fte_daily, supply_fte_daily, voice_requirements_interval
from app.pipeline.headcount import CHANNEL_LIST, _hcu_cols, _hcu_df
from app.pipeline.ops_store import (
    list_timeseries_scope_keys,
    list_timeseries_export_scopes,
    load_hiring,
    load_roster,
    load_timeseries_any,
)
from app.pipeline.timeseries_store import scope_file_keys
from app.pipeline.settings_store import load_settings


def _today_range(default_days: int = 28) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=default_days)
    return start, end


logger = logging.getLogger(__name__)
OPS_CACHE_TTL_SEC = 300
OPS_CACHE_MAX = 200
_OPS_CACHE: dict[tuple, tuple[float, dict]] = {}
OPS_ASYNC_JOB_TTL_SEC = 900
_OPS_JOBS: dict[tuple, dict] = {}
_OPS_JOBS_LOCK = threading.Lock()
_OPS_EXECUTOR = ThreadPoolExecutor(max_workers=2)
OPS_BASE_CACHE_TTL_SEC = 300
OPS_BASE_CACHE_MAX = 50
_OPS_BASE_CACHE: dict[tuple, tuple[float, dict]] = {}
_OPS_BASE_JOBS: dict[tuple, dict] = {}
_OPS_BASE_LOCK = threading.Lock()

OPS_TS_KINDS = [
    "voice_actual",
    "voice_forecast",
    "voice_tactical",
    "bo_actual",
    "bo_forecast",
    "bo_tactical",
    "chat_actual",
    "chat_forecast",
    "chat_tactical",
    "ob_actual",
    "ob_forecast",
    "ob_tactical",
]


def _ops_cache_key(
    start: date,
    end: date,
    grain: str,
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> tuple:
    return (
        str(start),
        str(end),
        str(grain),
        tuple(sorted({str(v).strip() for v in ba if str(v).strip()})),
        tuple(sorted({str(v).strip() for v in sba if str(v).strip()})),
        tuple(sorted({str(v).strip() for v in ch if str(v).strip()})),
        tuple(sorted({str(v).strip() for v in site if str(v).strip()})),
        tuple(sorted({str(v).strip() for v in loc if str(v).strip()})),
    )


def _ops_base_key(
    start: date,
    end: date,
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> tuple:
    return (
        str(start),
        str(end),
        tuple(sorted({str(v).strip() for v in ba if str(v).strip()})),
        tuple(sorted({str(v).strip() for v in sba if str(v).strip()})),
        tuple(sorted({str(v).strip() for v in ch if str(v).strip()})),
        tuple(sorted({str(v).strip() for v in site if str(v).strip()})),
        tuple(sorted({str(v).strip() for v in loc if str(v).strip()})),
    )


def _ops_cache_get(key: tuple) -> Optional[dict]:
    entry = _OPS_CACHE.get(key)
    if not entry:
        return None
    ts, payload = entry
    if (time.time() - ts) > OPS_CACHE_TTL_SEC:
        _OPS_CACHE.pop(key, None)
        return None
    return payload


def _ops_base_cache_get(key: tuple) -> Optional[dict]:
    entry = _OPS_BASE_CACHE.get(key)
    if not entry:
        return None
    ts, payload = entry
    if (time.time() - ts) > OPS_BASE_CACHE_TTL_SEC:
        _OPS_BASE_CACHE.pop(key, None)
        return None
    return payload


def _ops_base_cache_peek(key: tuple) -> tuple[Optional[dict], Optional[float], bool]:
    entry = _OPS_BASE_CACHE.get(key)
    if not entry:
        return None, None, False
    ts, payload = entry
    stale = (time.time() - ts) > OPS_BASE_CACHE_TTL_SEC
    return payload, ts, stale


def _ops_cache_peek(key: tuple) -> tuple[Optional[dict], Optional[float], bool]:
    entry = _OPS_CACHE.get(key)
    if not entry:
        return None, None, False
    ts, payload = entry
    stale = (time.time() - ts) > OPS_CACHE_TTL_SEC
    return payload, ts, stale


def _ops_cache_set(key: tuple, payload: dict) -> None:
    if len(_OPS_CACHE) >= OPS_CACHE_MAX:
        _OPS_CACHE.clear()
    _OPS_CACHE[key] = (time.time(), payload)


def _ops_base_cache_set(key: tuple, payload: dict) -> None:
    if len(_OPS_BASE_CACHE) >= OPS_BASE_CACHE_MAX:
        _OPS_BASE_CACHE.clear()
    _OPS_BASE_CACHE[key] = (time.time(), payload)


def _run_ops_job(
    key: tuple,
    start_date: Optional[str],
    end_date: Optional[str],
    grain: str,
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> None:
    t0 = time.perf_counter()
    logger.info(
        "ops_dashboard job started grain=%s start=%s end=%s ba=%s sba=%s ch=%s site=%s loc=%s",
        grain,
        start_date,
        end_date,
        len(ba or []),
        len(sba or []),
        len(ch or []),
        len(site or []),
        len(loc or []),
    )
    try:
        refresh_ops(start_date, end_date, grain, ba, sba, ch, site, loc)
        status = "ready"
        err = None
    except Exception as exc:
        status = "error"
        err = str(exc)
        logger.exception("ops_dashboard async job failed")
    finally:
        t1 = time.perf_counter()
        logger.info(
            "ops_dashboard job finished status=%s elapsed=%.3fs",
            status,
            (t1 - t0),
        )
    with _OPS_JOBS_LOCK:
        _OPS_JOBS[key] = {
            "status": status,
            "started_at": _OPS_JOBS.get(key, {}).get("started_at"),
            "finished_at": time.time(),
            "error": err,
        }


def _run_ops_base_job(
    key: tuple,
    start_date: Optional[str],
    end_date: Optional[str],
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> None:
    t0 = time.perf_counter()
    logger.info(
        "ops_dashboard base job started start=%s end=%s ba=%s sba=%s ch=%s site=%s loc=%s",
        start_date,
        end_date,
        len(ba or []),
        len(sba or []),
        len(ch or []),
        len(site or []),
        len(loc or []),
    )
    try:
        base_payload = _compute_ops_base(start_date, end_date, ba, sba, ch, site, loc)
        _ops_base_cache_set(key, base_payload)
        status = "ready"
        err = None
    except Exception as exc:
        status = "error"
        err = str(exc)
        logger.exception("ops_dashboard base job failed")
    finally:
        t1 = time.perf_counter()
        logger.info(
            "ops_dashboard base job finished status=%s elapsed=%.3fs",
            status,
            (t1 - t0),
        )
    with _OPS_BASE_LOCK:
        _OPS_BASE_JOBS[key] = {
            "status": status,
            "started_at": _OPS_BASE_JOBS.get(key, {}).get("started_at"),
            "finished_at": time.time(),
            "error": err,
        }


def _start_ops_job(
    key: tuple,
    start_date: Optional[str],
    end_date: Optional[str],
    grain: str,
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> None:
    now = time.time()
    with _OPS_JOBS_LOCK:
        existing = _OPS_JOBS.get(key)
        if existing and existing.get("status") == "running":
            started_at = existing.get("started_at") or 0.0
            if (now - started_at) < OPS_ASYNC_JOB_TTL_SEC:
                return
        _OPS_JOBS[key] = {"status": "running", "started_at": now, "finished_at": None, "error": None}
    _OPS_EXECUTOR.submit(_run_ops_job, key, start_date, end_date, grain, ba, sba, ch, site, loc)


def _start_ops_base_job(
    key: tuple,
    start_date: Optional[str],
    end_date: Optional[str],
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> None:
    now = time.time()
    with _OPS_BASE_LOCK:
        existing = _OPS_BASE_JOBS.get(key)
        if existing and existing.get("status") == "running":
            started_at = existing.get("started_at") or 0.0
            if (now - started_at) < OPS_ASYNC_JOB_TTL_SEC:
                return
        _OPS_BASE_JOBS[key] = {"status": "running", "started_at": now, "finished_at": None, "error": None}
    _OPS_EXECUTOR.submit(_run_ops_base_job, key, start_date, end_date, ba, sba, ch, site, loc)


def refresh_ops_async(
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

    cache_key = _ops_cache_key(start, end, grain, ba, sba, ch, site, loc)
    cached, cached_ts, stale = _ops_cache_peek(cache_key)

    if cached is not None and not stale:
        return {
            "status": "ready",
            "data": cached,
            "updated_at": cached_ts,
        }

    _start_ops_job(cache_key, start_date, end_date, grain, ba, sba, ch, site, loc)

    with _OPS_JOBS_LOCK:
        job = _OPS_JOBS.get(cache_key, {})
        status = job.get("status", "refreshing")
        err = job.get("error")

    return {
        "status": "refreshing" if status in {"running", "refreshing"} else status,
        "data": cached,
        "updated_at": cached_ts,
        "error": err,
    }


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
    if "Channel" in out.columns:
        out["Channel"] = out["Channel"].map(_normalize_channel).fillna(out["Channel"])
    if "Site" in out.columns and out["Site"].replace("", pd.NA).dropna().empty:
        # Fallback: pick any column that looks like a site/building field with values.
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
            if col == "Channel":
                lowered = {_normalize_channel(v).strip().lower() for v in values if str(v).strip()}
            else:
                lowered = {str(v).strip().lower() for v in values if str(v).strip()}
            if lowered:
                df = df[df[col].astype(str).str.strip().str.lower().isin(lowered)]

    _apply("Business Area", ba)
    _apply("Sub Business Area", sba)
    if not df.empty and "Channel" in df.columns:
        if not df["Channel"].replace("", pd.NA).dropna().empty and ch:
            df_before = df
            _apply("Channel", ch)
            if df.empty:
                df = df_before
    _apply("Site", site)
    _apply("Location", loc)

    if df.empty:
        return pd.DataFrame(columns=["ba", "sba", "ch", "loc", "site", "sk"])

    if ch:
        ch_list = [_normalize_channel(x).strip() for x in ch if str(x).strip()]
    else:
        ch_list = list(CHANNEL_LIST)
    rows = []
    for _, row in df.iterrows():
        ba_v = str(row.get("Business Area", "")).strip()
        sba_v = str(row.get("Sub Business Area", "")).strip()
        loc_v = str(row.get("Location", "")).strip()
        site_v = str(row.get("Site", "")).strip()
        for ch_v in ch_list:
            ch_key = str(ch_v).strip()
            if site_v:
                sk = f"{ba_v}|{sba_v}|{ch_key}|{site_v}".lower()
            else:
                sk = f"{ba_v}|{sba_v}|{ch_key}".lower()
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
        "chat_actual",
        "chat_forecast",
        "chat_tactical",
        "ob_actual",
        "ob_forecast",
        "ob_tactical",
    ]
    scope_keys = list_timeseries_scope_keys(kinds)
    if not scope_keys:
        return pd.DataFrame(columns=["ba", "sba", "ch", "loc", "site", "sk"])

    def _norm(values: list[str], *, is_channel: bool = False) -> set[str]:
        if is_channel:
            return {_normalize_channel(v).strip().lower() for v in values if str(v).strip()}
        return {str(v).strip().lower() for v in values if str(v).strip()}

    ba_f = _norm(ba)
    sba_f = _norm(sba)
    ch_f = _norm(ch, is_channel=True)
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
        ch_v_norm = _normalize_channel(ch_v).strip().lower()
        site_v = parts[3] if len(parts) > 3 else ""
        if ba_f and ba_v.strip().lower() not in ba_f:
            continue
        if sba_f and sba_v.strip().lower() not in sba_f:
            continue
        if ch_f and ch_v_norm not in ch_f:
            continue
        if site_f and site_v.strip().lower() not in site_f:
            continue
        rows.append({"ba": ba_v, "sba": sba_v, "ch": ch_v, "loc": "", "site": site_v, "sk": raw})
    out = pd.DataFrame(rows)
    return out[["ba", "sba", "ch", "loc", "site", "sk"]] if not out.empty else pd.DataFrame(columns=["ba", "sba", "ch", "loc", "site", "sk"])


def _dataset_sites_all() -> list[str]:
    kinds = [
        "voice_actual",
        "voice_forecast",
        "voice_tactical",
        "bo_actual",
        "bo_forecast",
        "bo_tactical",
        "chat_actual",
        "chat_forecast",
        "chat_tactical",
        "ob_actual",
        "ob_forecast",
        "ob_tactical",
    ]
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


def ops_options(ba: list[str], sba: list[str], ch: list[str], loc: list[str], debug: bool = False) -> dict:
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
    debug_counts = {"start": int(len(site_dim.index))}
    if ba:
        ba_set = {str(v).strip().lower() for v in ba if str(v).strip()}
        site_dim = site_dim[site_dim["Business Area"].astype(str).str.strip().str.lower().isin(ba_set)]
    debug_counts["after_ba"] = int(len(site_dim.index))
    if sba:
        sba_set = {str(v).strip().lower() for v in sba if str(v).strip()}
        site_dim = site_dim[site_dim["Sub Business Area"].astype(str).str.strip().str.lower().isin(sba_set)]
    debug_counts["after_sba"] = int(len(site_dim.index))
    if ch and "Channel" in site_dim.columns:
        if not site_dim["Channel"].replace("", pd.NA).dropna().empty:
            ch_set = {_normalize_channel(v).strip().lower() for v in ch if str(v).strip()}
            site_dim_before = site_dim
            filtered = site_dim[site_dim["Channel"].astype(str).str.strip().str.lower().isin(ch_set)]
            site_dim = filtered if not filtered.empty else site_dim_before
    debug_counts["after_ch"] = int(len(site_dim.index))
    if loc:
        loc_set = {str(v).strip().lower() for v in loc if str(v).strip()}
        site_dim = site_dim[site_dim["Location"].astype(str).str.strip().str.lower().isin(loc_set)]
    debug_counts["after_loc"] = int(len(site_dim.index))

    raw_site_list = (
        sorted([x for x in site_dim["Site"].dropna().unique().tolist() if x])
        if "Site" in site_dim.columns
        else []
    )
    loc_set = {str(x).strip().lower() for x in opts_loc}
    country_block = {"india", "uk", "united kingdom", "great britain", "england", "scotland", "wales"}
    site_list = [s for s in raw_site_list if s and (sl := str(s).strip().lower()) not in country_block]
    if loc_set:
        site_list_loc = [s for s in site_list if s and (sl := str(s).strip().lower()) not in loc_set]
        if site_list_loc:
            site_list = site_list_loc
    if not site_list:
        ds_map = _scopes_from_datasets(ba, sba, ch, [], loc)
        if not ds_map.empty and "site" in ds_map.columns:
            site_list = sorted([s for s in ds_map["site"].astype(str).dropna().str.strip().unique().tolist() if s])
    if not site_list:
        site_list = _dataset_sites_all()
    site_list = [s for s in site_list if s and (sl := str(s).strip().lower()) not in country_block]

    result = {
        "business_areas": opts_ba,
        "sub_business_areas": opts_sba,
        "locations": opts_loc,
        "sites": site_list,
        "channels": opts_ch,
    }
    if debug:
        raw_cols = _hcu_cols(_hcu_df()) if not dim.empty else {}
        result["debug"] = {
            "filters": {"ba": ba, "sba": sba, "ch": ch, "loc": loc},
            "headcount_rows": int(len(dim.index)),
            "headcount_cols": list(_hcu_df().columns) if not dim.empty else [],
            "headcount_mapping": raw_cols,
            "site_dim_counts": debug_counts,
            "channel_values_present": bool(
                "Channel" in dim.columns and not dim["Channel"].replace("", pd.NA).dropna().empty
            ),
            "site_values_present": bool(
                "Site" in dim.columns and not dim["Site"].replace("", pd.NA).dropna().empty
            ),
            "channel_sample": sorted(dim["Channel"].dropna().unique().tolist())[:20] if "Channel" in dim.columns else [],
            "site_sample": sorted(dim["Site"].dropna().unique().tolist())[:20] if "Site" in dim.columns else [],
            "site_list_count": len(site_list),
            "site_list_sample": site_list[:20],
        }
    return result


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


def _load_voice(scopes: list[str], start: Optional[date] = None, end: Optional[date] = None) -> pd.DataFrame:
    df = load_timeseries_any("voice_actual", scopes, start_date=start, end_date=end, batch=True)
    if df.empty:
        df = load_timeseries_any("voice_forecast", scopes, start_date=start, end_date=end, batch=True)
    if df.empty:
        df = load_timeseries_any("voice_tactical", scopes, start_date=start, end_date=end, batch=True)
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = out["items"] if "items" in out.columns else pd.NA
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    if "aht_sec" not in out.columns:
        out["aht_sec"] = pd.NA
    out["aht_sec"] = pd.to_numeric(out["aht_sec"], errors="coerce")
    if "sut_sec" in out.columns:
        sut_sec = pd.to_numeric(out["sut_sec"], errors="coerce")
        # Some voice uploads put handle time under sut_sec; use it if aht_sec is missing/zero.
        out["aht_sec"] = out["aht_sec"].where(out["aht_sec"].fillna(0.0) > 0.0, sut_sec)
    out["aht_sec"] = out["aht_sec"].fillna(300.0)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    return out


def _load_bo(scopes: list[str], start: Optional[date] = None, end: Optional[date] = None) -> pd.DataFrame:
    df = load_timeseries_any("bo_actual", scopes, start_date=start, end_date=end, batch=True)
    if df.empty:
        df = load_timeseries_any("bo_forecast", scopes, start_date=start, end_date=end, batch=True)
    if df.empty:
        df = load_timeseries_any("bo_tactical", scopes, start_date=start, end_date=end, batch=True)
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
        # Many BO uploads store handle time under aht_sec; use it when sut_sec is missing/zero.
        out["sut_sec"] = out["sut_sec"].where(out["sut_sec"].fillna(0.0) > 0.0, aht_sec)
    out["sut_sec"] = out["sut_sec"].fillna(600.0)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    return out


def _load_chat(scopes: list[str], start: Optional[date] = None, end: Optional[date] = None) -> pd.DataFrame:
    df = load_timeseries_any("chat_actual", scopes, start_date=start, end_date=end, batch=True)
    if df.empty:
        df = load_timeseries_any("chat_forecast", scopes, start_date=start, end_date=end, batch=True)
    if df.empty:
        df = load_timeseries_any("chat_tactical", scopes, start_date=start, end_date=end, batch=True)
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "items" not in out.columns:
        out["items"] = out["volume"] if "volume" in out.columns else 0.0
    out["items"] = pd.to_numeric(out["items"], errors="coerce").fillna(0.0)
    if "aht_sec" not in out.columns:
        out["aht_sec"] = pd.NA
    out["aht_sec"] = pd.to_numeric(out["aht_sec"], errors="coerce")
    if "sut_sec" in out.columns:
        sut_sec = pd.to_numeric(out["sut_sec"], errors="coerce")
        out["aht_sec"] = out["aht_sec"].where(out["aht_sec"].fillna(0.0) > 0.0, sut_sec)
    out["aht_sec"] = out["aht_sec"].fillna(240.0)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    return out


def _load_ob(scopes: list[str], start: Optional[date] = None, end: Optional[date] = None) -> pd.DataFrame:
    df = load_timeseries_any("ob_actual", scopes, start_date=start, end_date=end, batch=True)
    if df.empty:
        df = load_timeseries_any("ob_forecast", scopes, start_date=start, end_date=end, batch=True)
    if df.empty:
        df = load_timeseries_any("ob_tactical", scopes, start_date=start, end_date=end, batch=True)
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "opc" not in out.columns:
        if "volume" in out.columns:
            out["opc"] = out["volume"]
        elif "items" in out.columns:
            out["opc"] = out["items"]
        else:
            out["opc"] = 0.0
    out["opc"] = pd.to_numeric(out["opc"], errors="coerce").fillna(0.0)
    if "connect_rate" not in out.columns:
        out["connect_rate"] = 1.0
    out["connect_rate"] = pd.to_numeric(out["connect_rate"], errors="coerce")
    out["connect_rate"] = out["connect_rate"].where(out["connect_rate"].fillna(0.0) > 0.0, 1.0)
    out["connect_rate"] = out["connect_rate"].where(out["connect_rate"] <= 1.0, out["connect_rate"] / 100.0)
    out["connect_rate"] = out["connect_rate"].fillna(1.0)
    if "rpc_rate" not in out.columns:
        out["rpc_rate"] = 1.0
    out["rpc_rate"] = pd.to_numeric(out["rpc_rate"], errors="coerce")
    out["rpc_rate"] = out["rpc_rate"].where(out["rpc_rate"].fillna(0.0) > 0.0, 1.0)
    out["rpc_rate"] = out["rpc_rate"].where(out["rpc_rate"] <= 1.0, out["rpc_rate"] / 100.0)
    out["rpc_rate"] = out["rpc_rate"].fillna(1.0)
    if "rpc" not in out.columns:
        out["rpc"] = out["opc"] * out["rpc_rate"]
    out["rpc"] = pd.to_numeric(out["rpc"], errors="coerce").fillna(0.0)
    if "aht_sec" not in out.columns:
        out["aht_sec"] = pd.NA
    out["aht_sec"] = pd.to_numeric(out["aht_sec"], errors="coerce")
    if "sut_sec" in out.columns:
        sut_sec = pd.to_numeric(out["sut_sec"], errors="coerce")
        out["aht_sec"] = out["aht_sec"].where(out["aht_sec"].fillna(0.0) > 0.0, sut_sec)
    out["aht_sec"] = out["aht_sec"].fillna(240.0)
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


def _add_bucket(
    df: pd.DataFrame,
    date_col: str,
    grain: str,
    interval_col: str = "interval",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date
    if grain == "interval" and interval_col in out.columns and out[interval_col].notna().any():
        out["bucket"] = out[date_col].astype(str) + " " + out[interval_col].astype(str)
        return out
    return _agg_by_grain(out, date_col, grain)


def _filter_scope_df(
    df: pd.DataFrame,
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    def _filter(col: str, values: list[str], *, is_channel: bool = False):
        nonlocal out
        if col in out.columns and values:
            if is_channel:
                sset = {_normalize_channel(v).strip().lower() for v in values if str(v).strip()}
                col_norm = out[col].astype(str).map(_normalize_channel).str.strip().str.lower()
                out = out[col_norm.isin(sset)]
            else:
                sset = {str(v).strip().lower() for v in values if str(v).strip()}
                out = out[out[col].astype(str).str.strip().str.lower().isin(sset)]

    has_ba_col = any(col in out.columns for col in ("Business Area", "business_area", "ba"))
    if has_ba_col:
        _filter("Business Area", ba)
        _filter("business_area", ba)
        _filter("ba", ba)
    else:
        # Legacy fallback: some old supply tables stored BA in `program`.
        _filter("program", ba)

    _filter("Sub Business Area", sba)
    _filter("sub_business_area", sba)
    _filter("sub_ba", sba)

    has_channel_col = any(col in out.columns for col in ("LOB", "Channel", "lob", "channel"))
    _filter("LOB", ch, is_channel=True)
    _filter("Channel", ch, is_channel=True)
    _filter("lob", ch, is_channel=True)
    _filter("channel", ch, is_channel=True)
    if not has_channel_col:
        _filter("program", ch, is_channel=True)

    _filter("site", site)
    _filter("Site", site)
    _filter("site_name", site)
    _filter("location", loc)
    _filter("Location", loc)
    _filter("loc", loc)
    _filter("country", loc)
    _filter("Country", loc)
    return out


def _filter_map_df_by_exports(map_df: pd.DataFrame, export_scopes: set[str]) -> pd.DataFrame:
    if map_df is None or map_df.empty or not export_scopes:
        return map_df
    sks = map_df["sk"].dropna().astype(str).unique().tolist()
    keep = {
        sk
        for sk in sks
        if any(scope_key.lower() in export_scopes for scope_key in scope_file_keys(sk))
    }
    logger.info("ops_dashboard export scope filter total=%s keep=%s", len(sks), len(keep))
    if not keep:
        return map_df.iloc[0:0].copy()
    return map_df[map_df["sk"].isin(keep)].copy()


def _workload_channel_df(
    voice: pd.DataFrame,
    bo: pd.DataFrame,
    chat: pd.DataFrame,
    ob: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    def _from_df(df: pd.DataFrame, value_col: str, default_channel: str) -> pd.DataFrame:
        if df is None or df.empty or value_col not in df.columns:
            return pd.DataFrame(columns=["ch", "val"])
        work = df.copy()
        work["val"] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
        if "scope_key" in work.columns:
            work["ch"] = work["scope_key"].astype(str).str.split("|").str[2].fillna(default_channel)
        else:
            work["ch"] = default_channel
        work["ch"] = work["ch"].map(_normalize_channel).fillna(work["ch"]).replace("", default_channel)
        return work.groupby("ch", as_index=False)["val"].sum()

    parts.append(_from_df(voice, "volume", "Voice"))
    parts.append(_from_df(bo, "items", "Back Office"))
    parts.append(_from_df(chat, "items", "Chat"))
    if ob is not None and not ob.empty:
        ob_value_col = "opc" if "opc" in ob.columns else ("volume" if "volume" in ob.columns else None)
        if ob_value_col:
            parts.append(_from_df(ob, ob_value_col, "Outbound"))

    non_empty = [p for p in parts if p is not None and not p.empty]
    if not non_empty:
        return pd.DataFrame(columns=["ch", "val"])
    out = pd.concat(non_empty, ignore_index=True)
    out = out.groupby("ch", as_index=False)["val"].sum()
    out = out[out["val"] > 0].sort_values("val", ascending=False).reset_index(drop=True)
    return out


def _build_weight_df(
    voice: pd.DataFrame,
    bo: pd.DataFrame,
    chat: pd.DataFrame,
    ob: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if voice is not None and not voice.empty and "volume" in voice.columns:
        frames.append(voice.groupby("date", as_index=False)["volume"].sum().rename(columns={"volume": "weight"}))
    if bo is not None and not bo.empty and "items" in bo.columns:
        frames.append(bo.groupby("date", as_index=False)["items"].sum().rename(columns={"items": "weight"}))
    if chat is not None and not chat.empty and "items" in chat.columns:
        frames.append(chat.groupby("date", as_index=False)["items"].sum().rename(columns={"items": "weight"}))
    if ob is not None and not ob.empty:
        ob_value_col = "opc" if "opc" in ob.columns else ("volume" if "volume" in ob.columns else None)
        if ob_value_col:
            frames.append(ob.groupby("date", as_index=False)[ob_value_col].sum().rename(columns={ob_value_col: "weight"}))

    if not frames:
        return pd.DataFrame(columns=["date", "weight"])
    out = pd.concat(frames, ignore_index=True)
    out = out.groupby("date", as_index=False)["weight"].sum()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    return out


def _required_fte_all_channels(
    voice: pd.DataFrame,
    bo: pd.DataFrame,
    chat: pd.DataFrame,
    ob: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    req = required_fte_daily(voice, bo, ob, settings)
    req_day = (
        req.groupby("date", as_index=False)["total_req_fte"].sum()
        if isinstance(req, pd.DataFrame) and not req.empty and "total_req_fte" in req.columns
        else pd.DataFrame(columns=["date", "total_req_fte"])
    )
    chat_req = chat_fte_daily(chat, settings)
    if isinstance(chat_req, pd.DataFrame) and not chat_req.empty and "chat_fte" in chat_req.columns:
        chat_day = chat_req.groupby("date", as_index=False)["chat_fte"].sum().rename(columns={"chat_fte": "total_req_fte"})
        if req_day.empty:
            req_day = chat_day
        else:
            req_day = req_day.merge(chat_day, on="date", how="outer", suffixes=("_base", "_chat"))
            req_day["total_req_fte"] = req_day[["total_req_fte_base", "total_req_fte_chat"]].fillna(0.0).sum(axis=1)
            req_day = req_day[["date", "total_req_fte"]]
    return req_day


def _compute_ops_base(
    start_date: Optional[str],
    end_date: Optional[str],
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> dict:
    t0 = time.perf_counter()
    try:
        start = pd.to_datetime(start_date).date() if start_date else _today_range(28)[0]
        end = pd.to_datetime(end_date).date() if end_date else _today_range(28)[1]
    except Exception:
        start, end = _today_range(28)

    selected_channels = [_normalize_channel(v).strip() for v in ch if str(v).strip()]
    selected_channels = [v for v in selected_channels if v]

    map_df = _scope_keys_from_filters(ba, sba, ch, site, loc)
    ds_map = _scopes_from_datasets(ba, sba, ch, site, loc)
    if not ds_map.empty:
        map_df = ds_map
    elif map_df.empty:
        map_df = ds_map
    else:
        export_scopes = list_timeseries_export_scopes(OPS_TS_KINDS)
        if export_scopes:
            map_df = _filter_map_df_by_exports(map_df, export_scopes)
    scopes = map_df["sk"].unique().tolist() if not map_df.empty else []
    t_scopes = time.perf_counter()

    voice = _load_voice(scopes, start, end)
    t_voice = time.perf_counter()
    bo = _load_bo(scopes, start, end)
    t_bo = time.perf_counter()
    chat = _load_chat(scopes, start, end)
    t_chat = time.perf_counter()
    ob = _load_ob(scopes, start, end)
    t_ob = time.perf_counter()

    ba_arg = ba[0] if ba else None
    sba_arg = sba[0] if sba else None
    ch_arg = ch[0] if ch else None
    loc_arg = loc[0] if loc else None
    site_arg = site[0] if site else None
    settings = _resolve_settings(ba_arg, sba_arg, ch_arg, site_arg, loc_arg)
    t_settings = time.perf_counter()

    req_day = _required_fte_all_channels(voice, bo, chat, ob, settings)
    t_req = time.perf_counter()

    roster = load_roster()
    t_roster = time.perf_counter()
    hiring = load_hiring()
    t_hiring = time.perf_counter()
    roster_f = _filter_scope_df(roster, ba, sba, ch, site, loc)
    hiring_f = _filter_scope_df(hiring, ba, sba, ch, site, loc)
    supply = supply_fte_daily(roster_f, hiring_f)
    if not supply.empty:
        supply["date"] = pd.to_datetime(supply["date"], errors="coerce").dt.date
        supply = supply[pd.notna(supply["date"])]
        supply = supply[(supply["date"] >= start) & (supply["date"] <= end)]
        supply = supply.groupby("date", as_index=False)["supply_fte"].sum()
    t_supply = time.perf_counter()

    weight_df = _build_weight_df(voice, bo, chat, ob)
    t_weight = time.perf_counter()

    logger.info(
        "ops_dashboard base timings scopes=%.3fs voice=%.3fs bo=%.3fs chat=%.3fs ob=%.3fs settings=%.3fs req=%.3fs roster=%.3fs hiring=%.3fs supply=%.3fs weight=%.3fs total=%.3fs scopes=%s voice_rows=%s bo_rows=%s chat_rows=%s ob_rows=%s roster_rows=%s hiring_rows=%s",
        (t_scopes - t0),
        (t_voice - t_scopes),
        (t_bo - t_voice),
        (t_chat - t_bo),
        (t_ob - t_chat),
        (t_settings - t_ob),
        (t_req - t_settings),
        (t_roster - t_req),
        (t_hiring - t_roster),
        (t_supply - t_hiring),
        (t_weight - t_supply),
        (t_weight - t0),
        len(scopes),
        int(len(voice.index)) if isinstance(voice, pd.DataFrame) else 0,
        int(len(bo.index)) if isinstance(bo, pd.DataFrame) else 0,
        int(len(chat.index)) if isinstance(chat, pd.DataFrame) else 0,
        int(len(ob.index)) if isinstance(ob, pd.DataFrame) else 0,
        int(len(roster.index)) if isinstance(roster, pd.DataFrame) else 0,
        int(len(hiring.index)) if isinstance(hiring, pd.DataFrame) else 0,
    )

    return {
        "start": start,
        "end": end,
        "map_df": map_df,
        "scopes": scopes,
        "selected_channels": selected_channels,
        "voice": voice,
        "bo": bo,
        "chat": chat,
        "ob": ob,
        "settings": settings,
        "req_day": req_day,
        "supply": supply,
        "weight_df": weight_df,
    }


def _weighted_avg(df: pd.DataFrame, value_col: str, weight_df: pd.DataFrame) -> float:
    if df is None or df.empty or value_col not in df.columns:
        return 0.0
    merged = df.copy()
    if not weight_df.empty:
        merged = merged.merge(weight_df, on="date", how="left")
    else:
        merged["weight"] = 1.0
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    merged[value_col] = pd.to_numeric(merged[value_col], errors="coerce").fillna(0.0)
    wsum = float(merged["weight"].sum())
    if wsum > 0:
        return float((merged[value_col] * merged["weight"]).sum() / wsum)
    return float(merged[value_col].mean()) if not merged.empty else 0.0


def _weighted_avg_with_grain(
    df: pd.DataFrame,
    value_col: str,
    weight_df: pd.DataFrame,
    grain: str,
) -> float:
    if df is None or df.empty or value_col not in df.columns:
        return 0.0
    if grain in ("interval", "D", "daily"):
        return _weighted_avg(df, value_col, weight_df)
    merged = df.copy()
    if weight_df is not None and not weight_df.empty and "weight" in weight_df.columns:
        merged = merged.merge(weight_df, on="date", how="left")
    else:
        merged["weight"] = 1.0
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    merged[value_col] = pd.to_numeric(merged[value_col], errors="coerce").fillna(0.0)
    merged = _agg_by_grain(merged, "date", grain)
    if merged.empty:
        return 0.0

    def _bucket_avg(group: pd.DataFrame) -> float:
        wsum = float(group["weight"].sum())
        if wsum > 0:
            return float((group[value_col] * group["weight"]).sum() / wsum)
        return float(group[value_col].mean()) if not group.empty else 0.0

    bucket_vals = merged.groupby("bucket").apply(_bucket_avg)
    if isinstance(bucket_vals, pd.Series):
        bucket_vals = bucket_vals.reset_index(name="avg")
    else:
        bucket_vals = bucket_vals.rename(columns={0: "avg"})
    if bucket_vals.empty:
        return 0.0
    return float(bucket_vals["avg"].mean())


def _avg_aht_with_grain(voice: pd.DataFrame, grain: str) -> float:
    if voice is None or voice.empty or "volume" not in voice.columns or "aht_sec" not in voice.columns:
        return 0.0
    df = voice.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df["aht_sec"] = pd.to_numeric(df["aht_sec"], errors="coerce").fillna(0.0)
    if grain in ("interval", "D", "daily"):
        if grain == "interval" and "interval" in df.columns and df["interval"].notna().any():
            df["bucket"] = df["date"].astype(str) + " " + df["interval"].astype(str)
            bucket_vals = df.groupby("bucket").apply(
                lambda g: float((g["volume"] * g["aht_sec"]).sum() / max(1e-6, g["volume"].sum()))
            )
            if isinstance(bucket_vals, pd.Series):
                return float(bucket_vals.mean()) if not bucket_vals.empty else 0.0
            return 0.0
        total_calls = float(df["volume"].sum())
        if total_calls <= 0:
            return 0.0
        numer = float((df["volume"] * df["aht_sec"]).sum())
        return float(numer / max(total_calls, 1e-6))

    daily = df.groupby("date", as_index=False).apply(
        lambda g: pd.Series(
            {
                "aht_sec": float((g["volume"] * g["aht_sec"]).sum() / max(1e-6, g["volume"].sum())),
                "weight": float(g["volume"].sum()),
            }
        )
    )
    if daily.empty:
        return 0.0
    daily = _agg_by_grain(daily, "date", grain)

    def _bucket_avg(group: pd.DataFrame) -> float:
        wsum = float(group["weight"].sum())
        if wsum > 0:
            return float((group["aht_sec"] * group["weight"]).sum() / wsum)
        return float(group["aht_sec"].mean()) if not group.empty else 0.0

    bucket_vals = daily.groupby("bucket").apply(_bucket_avg)
    if isinstance(bucket_vals, pd.Series):
        bucket_vals = bucket_vals.reset_index(name="avg")
    else:
        bucket_vals = bucket_vals.rename(columns={0: "avg"})
    if bucket_vals.empty:
        return 0.0
    return float(bucket_vals["avg"].mean())


def _avg_sut_with_grain(bo: pd.DataFrame, grain: str) -> float:
    if bo is None or bo.empty or "items" not in bo.columns:
        return 0.0
    handle_col = "sut_sec" if "sut_sec" in bo.columns else ("aht_sec" if "aht_sec" in bo.columns else None)
    if not handle_col:
        return 0.0
    df = bo.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["items"] = pd.to_numeric(df["items"], errors="coerce").fillna(0.0)
    df[handle_col] = pd.to_numeric(df[handle_col], errors="coerce").fillna(0.0)
    if grain in ("interval", "D", "daily"):
        total_items = float(df["items"].sum())
        if total_items <= 0:
            return 0.0
        numer = float((df["items"] * df[handle_col]).sum())
        return float(numer / max(total_items, 1e-6))

    daily = df.groupby("date", as_index=False).apply(
        lambda g: pd.Series(
            {
                "sut_sec": float((g["items"] * g[handle_col]).sum() / max(1e-6, g["items"].sum())),
                "weight": float(g["items"].sum()),
            }
        )
    )
    if daily.empty:
        return 0.0
    daily = _agg_by_grain(daily, "date", grain)

    def _bucket_avg(group: pd.DataFrame) -> float:
        wsum = float(group["weight"].sum())
        if wsum > 0:
            return float((group["sut_sec"] * group["weight"]).sum() / wsum)
        return float(group["sut_sec"].mean()) if not group.empty else 0.0

    bucket_vals = daily.groupby("bucket").apply(_bucket_avg)
    if isinstance(bucket_vals, pd.Series):
        bucket_vals = bucket_vals.reset_index(name="avg")
    else:
        bucket_vals = bucket_vals.rename(columns={0: "avg"})
    if bucket_vals.empty:
        return 0.0
    return float(bucket_vals["avg"].mean())


def _gap_df_with_grain(
    req_day: pd.DataFrame,
    supply: pd.DataFrame,
    voice: pd.DataFrame,
    settings: dict,
    weight_df: pd.DataFrame,
    grain: str,
) -> pd.DataFrame:
    if grain == "interval" and voice is not None and not voice.empty and "interval" in voice.columns and voice["interval"].notna().any():
        vi = voice_requirements_interval(voice, settings)
        if isinstance(vi, pd.DataFrame) and not vi.empty:
            vi = (
                vi.groupby(["date", "interval"], as_index=False)["agents_req"]
                .sum()
                .sort_values(["date", "interval"])
            )
            vi["date"] = pd.to_datetime(vi["date"], errors="coerce").dt.date
            vi["total_req_fte"] = pd.to_numeric(vi["agents_req"], errors="coerce").fillna(0.0)
            if supply is not None and not supply.empty:
                supply_day = supply.copy()
                supply_day["date"] = pd.to_datetime(supply_day["date"], errors="coerce").dt.date
                interval_counts = (
                    vi.groupby("date", as_index=False)["interval"]
                    .nunique()
                    .rename(columns={"interval": "intervals"})
                )
                supply_day = supply_day.merge(interval_counts, on="date", how="left")
                supply_day["intervals"] = pd.to_numeric(supply_day["intervals"], errors="coerce").fillna(0.0)
                denom = supply_day["intervals"].replace({0: pd.NA})
                supply_day["supply_fte"] = (supply_day["supply_fte"] / denom).fillna(0.0)
                vi = vi.merge(supply_day[["date", "supply_fte"]], on="date", how="left")
            if "supply_fte" not in vi.columns:
                vi["supply_fte"] = 0.0
            vi["supply_fte"] = pd.to_numeric(vi["supply_fte"], errors="coerce").fillna(0.0)
            vi["gap_fte"] = vi["total_req_fte"] - vi["supply_fte"]
            vi = _add_bucket(vi, "date", "interval", interval_col="interval")
            return vi[["bucket", "total_req_fte", "supply_fte", "gap_fte"]]

    r_df = _bucketed_weighted_avg_series(req_day, "total_req_fte", weight_df, grain) if not req_day.empty else pd.DataFrame(columns=["bucket", "total_req_fte"])
    s_df = _bucketed_weighted_avg_series(supply, "supply_fte", weight_df, grain) if not supply.empty else pd.DataFrame(columns=["bucket", "supply_fte"])
    gap_df = pd.merge(r_df, s_df, on="bucket", how="outer").fillna(0.0)
    gap_df["gap_fte"] = gap_df["total_req_fte"] - gap_df["supply_fte"]
    return gap_df


def _top_volume_with_grain(voice: pd.DataFrame, grain: str, limit: int = 5) -> list[dict[str, Any]]:
    if voice is None or voice.empty or "volume" not in voice.columns:
        return []
    vday = voice.copy()
    vday["date"] = pd.to_datetime(vday["date"], errors="coerce").dt.date
    vday["volume"] = pd.to_numeric(vday["volume"], errors="coerce").fillna(0.0)
    vday = _add_bucket(vday, "date", grain)
    if vday.empty:
        return []
    vday = vday.groupby("bucket", as_index=False)["volume"].sum().sort_values("volume", ascending=False).head(limit)
    return [{"date": str(row["bucket"]), "volume": float(row["volume"])} for _, row in vday.iterrows()]


def _top_workload_with_grain(
    voice: pd.DataFrame,
    bo: pd.DataFrame,
    chat: pd.DataFrame,
    ob: pd.DataFrame,
    grain: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    top_voice = _top_volume_with_grain(voice, grain, limit=limit)
    if top_voice:
        return top_voice
    if bo is not None and not bo.empty and "items" in bo.columns:
        bday = bo.copy()
        bday["date"] = pd.to_datetime(bday["date"], errors="coerce").dt.date
        bday["items"] = pd.to_numeric(bday["items"], errors="coerce").fillna(0.0)
        bday = _add_bucket(bday, "date", grain)
        if not bday.empty:
            bday = bday.groupby("bucket", as_index=False)["items"].sum().sort_values("items", ascending=False).head(limit)
            return [{"date": str(row["bucket"]), "volume": float(row["items"])} for _, row in bday.iterrows()]
    if chat is not None and not chat.empty and "items" in chat.columns:
        cday = chat.copy()
        cday["date"] = pd.to_datetime(cday["date"], errors="coerce").dt.date
        cday["items"] = pd.to_numeric(cday["items"], errors="coerce").fillna(0.0)
        cday = _add_bucket(cday, "date", grain)
        if not cday.empty:
            cday = cday.groupby("bucket", as_index=False)["items"].sum().sort_values("items", ascending=False).head(limit)
            return [{"date": str(row["bucket"]), "volume": float(row["items"])} for _, row in cday.iterrows()]
    if ob is not None and not ob.empty:
        col = "opc" if "opc" in ob.columns else ("volume" if "volume" in ob.columns else None)
        if col:
            oday = ob.copy()
            oday["date"] = pd.to_datetime(oday["date"], errors="coerce").dt.date
            oday[col] = pd.to_numeric(oday[col], errors="coerce").fillna(0.0)
            oday = _add_bucket(oday, "date", grain)
            if not oday.empty:
                oday = oday.groupby("bucket", as_index=False)[col].sum().sort_values(col, ascending=False).head(limit)
                return [{"date": str(row["bucket"]), "volume": float(row[col])} for _, row in oday.iterrows()]
    return []


def _bucketed_weighted_avg_series(
    df: pd.DataFrame,
    value_col: str,
    weight_df: pd.DataFrame,
    grain: str,
) -> pd.DataFrame:
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=["bucket", value_col])
    if grain in ("D", "daily"):
        out = df.copy()
        out["bucket"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        return out.groupby("bucket", as_index=False)[value_col].mean()
    merged = df.copy()
    if weight_df is not None and not weight_df.empty and "weight" in weight_df.columns:
        merged = merged.merge(weight_df, on="date", how="left")
    else:
        merged["weight"] = 1.0
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    merged[value_col] = pd.to_numeric(merged[value_col], errors="coerce").fillna(0.0)
    merged = _agg_by_grain(merged, "date", grain)
    if merged.empty:
        return pd.DataFrame(columns=["bucket", value_col])

    def _bucket_avg(group: pd.DataFrame) -> float:
        wsum = float(group["weight"].sum())
        if wsum > 0:
            return float((group[value_col] * group["weight"]).sum() / wsum)
        return float(group[value_col].mean()) if not group.empty else 0.0

    out = merged.groupby("bucket").apply(_bucket_avg)
    if isinstance(out, pd.Series):
        out = out.reset_index(name=value_col)
    else:
        out = out.rename(columns={0: value_col})
    return out


def _compute_weighted_kpis(base: dict, grain: str) -> tuple[float, float, float, float]:
    req_day = base.get("req_day", pd.DataFrame())
    supply = base.get("supply", pd.DataFrame())
    weight_df = base.get("weight_df", pd.DataFrame())
    kpi_req = _weighted_avg_with_grain(req_day, "total_req_fte", weight_df, grain) if not req_day.empty else 0.0
    kpi_sup = _weighted_avg_with_grain(supply, "supply_fte", weight_df, grain) if not supply.empty else 0.0
    kpi_gap = kpi_req - kpi_sup
    coverage_pct = (kpi_sup / kpi_req * 100.0) if kpi_req > 0 else 0.0
    return kpi_req, kpi_sup, kpi_gap, coverage_pct


def _compute_kpis_insights(base: dict, grain: str) -> dict:
    req_day = base.get("req_day", pd.DataFrame())
    supply = base.get("supply", pd.DataFrame())
    voice = base.get("voice", pd.DataFrame())
    bo = base.get("bo", pd.DataFrame())
    chat = base.get("chat", pd.DataFrame())
    ob = base.get("ob", pd.DataFrame())
    settings = base.get("settings", {})
    weight_df = base.get("weight_df", pd.DataFrame())

    kpi_req, kpi_sup, kpi_gap, coverage_pct = _compute_weighted_kpis(base, grain)

    total_calls = float(voice["volume"].sum()) if not voice.empty and "volume" in voice.columns else 0.0
    total_items = float(bo["items"].sum()) if not bo.empty and "items" in bo.columns else 0.0
    total_items += float(chat["items"].sum()) if not chat.empty and "items" in chat.columns else 0.0
    total_items += float(ob["opc"].sum()) if not ob.empty and "opc" in ob.columns else 0.0
    avg_aht = _avg_aht_with_grain(voice, grain)
    if avg_aht <= 0.0:
        avg_aht = _avg_sut_with_grain(bo, grain)
    if avg_aht <= 0.0 and not chat.empty and "items" in chat.columns and "aht_sec" in chat.columns:
        avg_aht = _avg_sut_with_grain(chat, grain)

    gap_df = _gap_df_with_grain(req_day, supply, voice, settings, weight_df, grain)
    worst_gap = {}
    best_gap = {}
    if not gap_df.empty:
        peak_req_row = gap_df.sort_values("total_req_fte", ascending=False).head(1)
        peak_sup_row = gap_df.sort_values("supply_fte", ascending=False).head(1)
        worst = gap_df.sort_values("gap_fte", ascending=False).head(1)
        best = gap_df.sort_values("gap_fte", ascending=True).head(1)
        peak_required = {}
        peak_supply = {}
        if not peak_req_row.empty:
            peak_required = {"date": str(peak_req_row.iloc[0]["bucket"]), "value": float(peak_req_row.iloc[0]["total_req_fte"])}
        if not peak_sup_row.empty:
            peak_supply = {"date": str(peak_sup_row.iloc[0]["bucket"]), "value": float(peak_sup_row.iloc[0]["supply_fte"])}
        if not worst.empty:
            worst_gap = {"date": str(worst.iloc[0]["bucket"]), "value": float(worst.iloc[0]["gap_fte"])}
        if not best.empty:
            best_gap = {"date": str(best.iloc[0]["bucket"]), "value": float(best.iloc[0]["gap_fte"])}
    else:
        peak_required = {}
        peak_supply = {}

    top_shortfalls: list[dict[str, Any]] = []
    if not gap_df.empty:
        short = gap_df[gap_df["gap_fte"] > 0].sort_values("gap_fte", ascending=False).head(5)
        for _, row in short.iterrows():
            top_shortfalls.append(
                {
                    "date": str(row["bucket"]),
                    "required_fte": float(row["total_req_fte"]),
                    "supply_fte": float(row["supply_fte"]),
                    "gap_fte": float(row["gap_fte"]),
                }
            )

    top_volume_days = _top_workload_with_grain(voice, bo, chat, ob, grain)

    return {
        "kpis": {"required_fte": kpi_req, "supply_fte": kpi_sup, "gap_fte": kpi_gap},
        "insights": {
            "coverage_pct": coverage_pct,
            "total_calls": total_calls,
            "total_items": total_items,
            "avg_aht_sec": avg_aht,
            "peak_required": peak_required,
            "peak_supply": peak_supply,
            "worst_gap": worst_gap,
            "best_gap": best_gap,
            "top_shortfalls": top_shortfalls,
            "top_volume_days": top_volume_days,
        },
    }


def _compute_line_part(base: dict, grain: str) -> dict:
    voice = base.get("voice", pd.DataFrame())
    settings = base.get("settings", {})
    req_day = base.get("req_day", pd.DataFrame())
    supply = base.get("supply", pd.DataFrame())
    weight_df = base.get("weight_df", pd.DataFrame())

    line_series: list[dict[str, Any]] = []
    if grain == "interval" and not voice.empty and "interval" in voice.columns and voice["interval"].notna().any():
        vi = voice_requirements_interval(voice, settings)
        if not vi.empty:
            if "date" in vi.columns and "interval" in vi.columns:
                vi = (
                    vi.groupby(["date", "interval"], as_index=False)["agents_req"]
                    .sum()
                    .sort_values(["date", "interval"])
                )
            vi["ts"] = pd.to_datetime(vi["date"]).astype(str) + " " + vi["interval"].astype(str)
            line_series = [
                {
                    "name": "Agents Required",
                    "points": [{"x": str(x), "y": float(y)} for x, y in zip(vi["ts"], vi["agents_req"])],
                }
            ]
        line_x = [pt["x"] for pt in line_series[0]["points"]] if line_series else []
    else:
        r_df = req_day.copy() if not req_day.empty else pd.DataFrame(columns=["date", "total_req_fte"])
        s_df = supply.copy() if not supply.empty else pd.DataFrame(columns=["date", "supply_fte"])
        if not r_df.empty:
            r_df = _bucketed_weighted_avg_series(r_df, "total_req_fte", weight_df, grain)
            r_df = r_df.sort_values("bucket")
        if not s_df.empty:
            s_df = _bucketed_weighted_avg_series(s_df, "supply_fte", weight_df, grain)
            s_df = s_df.sort_values("bucket")
        line_x: list[str] = []
        if not r_df.empty:
            line_x = [str(x) for x in r_df["bucket"].tolist()]
            line_series.append(
                {
                    "name": "Required FTE",
                    "points": [{"x": str(x), "y": float(y)} for x, y in zip(r_df["bucket"], r_df["total_req_fte"])],
                }
            )
        if not s_df.empty:
            if not line_x:
                line_x = [str(x) for x in s_df["bucket"].tolist()]
            line_series.append(
                {
                    "name": "Supply FTE",
                    "points": [{"x": str(x), "y": float(y)} for x, y in zip(s_df["bucket"], s_df["supply_fte"])],
                }
            )

    return {"line": {"x": line_x, "series": line_series}}


def _compute_bar_part(base: dict, grain: str) -> dict:
    voice = base.get("voice", pd.DataFrame())
    bo = base.get("bo", pd.DataFrame())
    chat = base.get("chat", pd.DataFrame())
    ob = base.get("ob", pd.DataFrame())

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
    if not chat.empty:
        c_day = _agg_by_grain(chat.copy(), "date", grain)
        c_agg = c_day.groupby("bucket", as_index=False)["items"].sum().rename(columns={"items": "Chat Items"})
        bar_df = c_agg if bar_df.empty else pd.merge(bar_df, c_agg, on="bucket", how="outer")
    if not ob.empty:
        ob_col = "opc" if "opc" in ob.columns else ("volume" if "volume" in ob.columns else None)
        if ob_col:
            o_day = _agg_by_grain(ob.copy(), "date", grain)
            o_agg = o_day.groupby("bucket", as_index=False)[ob_col].sum().rename(columns={ob_col: "Outbound OPC"})
            bar_df = o_agg if bar_df.empty else pd.merge(bar_df, o_agg, on="bucket", how="outer")
    if not bar_df.empty:
        bar_df = bar_df.fillna(0.0).sort_values("bucket")
        bar_labels = [str(x) for x in bar_df["bucket"].tolist()]
        for col in [c for c in bar_df.columns if c != "bucket"]:
            bar_series.append({"name": col, "values": bar_df[col].astype(float).tolist()})
    return {"bar": {"labels": bar_labels, "series": bar_series}}


def _compute_pie_part(base: dict) -> dict:
    voice = base.get("voice", pd.DataFrame())
    bo = base.get("bo", pd.DataFrame())
    chat = base.get("chat", pd.DataFrame())
    ob = base.get("ob", pd.DataFrame())
    pie_df = _workload_channel_df(voice, bo, chat, ob)
    selected_channels = [str(v).strip() for v in (base.get("selected_channels") or []) if str(v).strip()]
    if len(selected_channels) == 1 and not pie_df.empty:
        selected = _normalize_channel(selected_channels[0]).strip() or selected_channels[0]
        total = float(pd.to_numeric(pie_df["val"], errors="coerce").fillna(0.0).sum())
        pie_df = pd.DataFrame([{"ch": selected, "val": total}]) if total > 0 else pd.DataFrame(columns=["ch", "val"])

    return {
        "pie": {
            "labels": pie_df["ch"].astype(str).tolist() if not pie_df.empty else [],
            "values": pie_df["val"].astype(float).tolist() if not pie_df.empty else [],
        }
    }


def _compute_site_part(base: dict) -> dict:
    map_df = base.get("map_df", pd.DataFrame())
    voice = base.get("voice", pd.DataFrame())
    bo = base.get("bo", pd.DataFrame())
    chat = base.get("chat", pd.DataFrame())
    ob = base.get("ob", pd.DataFrame())
    site = {"labels": [], "values": []}
    if not map_df.empty:
        key_map = map_df.drop_duplicates(subset=["sk", "site"])[["sk", "site"]]
        site_df = pd.DataFrame(columns=["site", "val"])
        if "scope_key" in voice.columns:
            v_site = (
                voice.groupby("scope_key", as_index=False)["volume"]
                .sum()
                .merge(key_map, left_on="scope_key", right_on="sk", how="left")
            )
            v_site = v_site.groupby("site", as_index=False)["volume"].sum().rename(columns={"volume": "val"})
            site_df = v_site
        if "scope_key" in bo.columns:
            b_site = (
                bo.groupby("scope_key", as_index=False)["items"]
                .sum()
                .merge(key_map, left_on="scope_key", right_on="sk", how="left")
            )
            b_site = b_site.groupby("site", as_index=False)["items"].sum().rename(columns={"items": "val"})
            site_df = b_site if site_df.empty else pd.concat([site_df, b_site]).groupby("site", as_index=False)["val"].sum()
        if "scope_key" in chat.columns:
            c_site = (
                chat.groupby("scope_key", as_index=False)["items"]
                .sum()
                .merge(key_map, left_on="scope_key", right_on="sk", how="left")
            )
            c_site = c_site.groupby("site", as_index=False)["items"].sum().rename(columns={"items": "val"})
            site_df = c_site if site_df.empty else pd.concat([site_df, c_site]).groupby("site", as_index=False)["val"].sum()
        ob_col = "opc" if "opc" in ob.columns else ("volume" if "volume" in ob.columns else None)
        if ob_col and "scope_key" in ob.columns:
            o_site = (
                ob.groupby("scope_key", as_index=False)[ob_col]
                .sum()
                .merge(key_map, left_on="scope_key", right_on="sk", how="left")
            )
            o_site = o_site.groupby("site", as_index=False)[ob_col].sum().rename(columns={ob_col: "val"})
            site_df = o_site if site_df.empty else pd.concat([site_df, o_site]).groupby("site", as_index=False)["val"].sum()
        if not site_df.empty:
            site = {"labels": site_df["site"].astype(str).tolist(), "values": site_df["val"].astype(float).tolist()}
    return {"site": site}


def _compute_waterfall_part(base: dict, grain: str) -> dict:
    kpi_req, kpi_sup, kpi_gap, _coverage = _compute_weighted_kpis(base, grain)
    return {
        "waterfall": {
            "labels": ["Required", "Supply", "Gap"],
            "values": [kpi_req, -kpi_sup, kpi_gap],
            "measure": ["relative", "relative", "total"],
        }
    }


def _compute_summary_part(base: dict) -> dict:
    map_df = base.get("map_df", pd.DataFrame())
    voice = base.get("voice", pd.DataFrame())
    bo = base.get("bo", pd.DataFrame())
    chat = base.get("chat", pd.DataFrame())
    ob = base.get("ob", pd.DataFrame())
    summary_rows: list[dict[str, Any]] = []
    if not map_df.empty:
        v_sum = (
            voice.groupby("scope_key", as_index=False)["volume"].sum()
            if not voice.empty and "scope_key" in voice.columns
            else pd.DataFrame(columns=["scope_key", "volume"])
        )
        b_sum = (
            bo.groupby("scope_key", as_index=False)["items"].sum()
            if not bo.empty and "scope_key" in bo.columns
            else pd.DataFrame(columns=["scope_key", "items"])
        )
        c_sum = (
            chat.groupby("scope_key", as_index=False)["items"].sum().rename(columns={"items": "chat_items"})
            if not chat.empty and "scope_key" in chat.columns
            else pd.DataFrame(columns=["scope_key", "chat_items"])
        )
        ob_col = "opc" if "opc" in ob.columns else ("volume" if "volume" in ob.columns else None)
        o_sum = (
            ob.groupby("scope_key", as_index=False)[ob_col].sum().rename(columns={ob_col: "outbound_opc"})
            if ob_col and not ob.empty and "scope_key" in ob.columns
            else pd.DataFrame(columns=["scope_key", "outbound_opc"])
        )
        sk_map = map_df.drop_duplicates(subset=["sk", "ba", "sba", "ch", "site", "loc"]).rename(columns={"sk": "scope_key"})
        merged = (
            sk_map.merge(v_sum, on="scope_key", how="left")
            .merge(b_sum, on="scope_key", how="left")
            .merge(c_sum, on="scope_key", how="left")
            .merge(o_sum, on="scope_key", how="left")
        )
        merged["volume"] = merged["volume"].fillna(0)
        merged["items"] = merged["items"].fillna(0)
        merged["chat_items"] = merged["chat_items"].fillna(0)
        merged["outbound_opc"] = merged["outbound_opc"].fillna(0)
        merged["workload"] = merged[["volume", "items", "chat_items", "outbound_opc"]].sum(axis=1)
        tbl = merged.groupby(
            ["ba", "sba", "ch", "site", "loc"], as_index=False
        )[["volume", "items", "chat_items", "outbound_opc", "workload"]].sum()
        summary_rows = tbl.to_dict("records")
    return {"summary": summary_rows}


def refresh_ops_part(
    part: str,
    start_date: Optional[str],
    end_date: Optional[str],
    grain: str,
    ba: list[str],
    sba: list[str],
    ch: list[str],
    site: list[str],
    loc: list[str],
) -> dict:
    if not part:
        return {"status": "error", "error": "Missing part."}

    try:
        start = pd.to_datetime(start_date).date() if start_date else _today_range(28)[0]
        end = pd.to_datetime(end_date).date() if end_date else _today_range(28)[1]
    except Exception:
        start, end = _today_range(28)

    base_key = _ops_base_key(start, end, ba, sba, ch, site, loc)
    cached_base, cached_ts, stale = _ops_base_cache_peek(base_key)

    if cached_base is None or stale:
        _start_ops_base_job(base_key, start_date, end_date, ba, sba, ch, site, loc)

    def _compute_part_payload(base: dict) -> dict:
        if part == "kpis":
            return _compute_kpis_insights(base, grain)
        if part == "line":
            return _compute_line_part(base, grain)
        if part == "bar":
            return _compute_bar_part(base, grain)
        if part == "pie":
            return _compute_pie_part(base)
        if part == "site":
            return _compute_site_part(base)
        if part == "waterfall":
            return _compute_waterfall_part(base, grain)
        if part == "summary":
            return _compute_summary_part(base)
        return {"error": f"Unknown part '{part}'."}

    if cached_base is not None and not stale:
        payload = _compute_part_payload(cached_base)
        status = "ready" if "error" not in payload else "error"
        return {"status": status, "data": payload if status == "ready" else None, "updated_at": cached_ts, "error": payload.get("error")}

    with _OPS_BASE_LOCK:
        job = _OPS_BASE_JOBS.get(base_key, {})
        job_status = job.get("status", "refreshing")
        job_error = job.get("error")

    data = _compute_part_payload(cached_base) if cached_base is not None else None
    status = "refreshing" if job_status in {"running", "refreshing"} else job_status
    if status == "error":
        return {"status": "error", "data": data, "updated_at": cached_ts, "error": job_error or (data or {}).get("error")}
    return {"status": status, "data": data, "updated_at": cached_ts, "error": None}


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
    t0 = time.perf_counter()
    try:
        start = pd.to_datetime(start_date).date() if start_date else _today_range(28)[0]
        end = pd.to_datetime(end_date).date() if end_date else _today_range(28)[1]
    except Exception:
        start, end = _today_range(28)

    cache_key = _ops_cache_key(start, end, grain, ba, sba, ch, site, loc)
    cached = _ops_cache_get(cache_key)
    if cached is not None:
        logger.info("ops_dashboard cache_hit scope=%s dates=%s..%s grain=%s", len(ba or []), start, end, grain)
        return cached

    base = _compute_ops_base(start_date, end_date, ba, sba, ch, site, loc)

    result: dict[str, Any] = {}
    for payload in (
        _compute_kpis_insights(base, grain),
        _compute_line_part(base, grain),
        _compute_bar_part(base, grain),
        _compute_pie_part(base),
        _compute_site_part(base),
        _compute_waterfall_part(base, grain),
        _compute_summary_part(base),
    ):
        result.update(payload)

    t_end = time.perf_counter()
    voice_df = base.get("voice", pd.DataFrame())
    bo_df = base.get("bo", pd.DataFrame())
    chat_df = base.get("chat", pd.DataFrame())
    ob_df = base.get("ob", pd.DataFrame())
    logger.info(
        "ops_dashboard timings total=%.3fs scopes=%s voice_rows=%s bo_rows=%s chat_rows=%s ob_rows=%s",
        (t_end - t0),
        len(base.get("scopes") or []),
        int(len(voice_df.index)) if isinstance(voice_df, pd.DataFrame) else 0,
        int(len(bo_df.index)) if isinstance(bo_df, pd.DataFrame) else 0,
        int(len(chat_df.index)) if isinstance(chat_df, pd.DataFrame) else 0,
        int(len(ob_df.index)) if isinstance(ob_df, pd.DataFrame) else 0,
    )
    _ops_cache_set(cache_key, result)
    return result
