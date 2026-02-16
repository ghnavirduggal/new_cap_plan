from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from app.pipeline.headcount import load_headcount
from app.pipeline.ops_store import load_hiring, load_roster, load_timeseries_any, normalize_timeseries_rows
from app.pipeline.roster_store import load_roster_long, load_roster_wide
from app.pipeline.settings_store import load_holidays, load_settings
from app.pipeline.timeseries_store import load_timeseries_csv, save_timeseries, _exports_dir, _safe_component


def _merge_timeseries_frames(df_db: pd.DataFrame, df_csv: pd.DataFrame) -> pd.DataFrame:
    if df_db is None or df_db.empty:
        return df_csv if isinstance(df_csv, pd.DataFrame) else pd.DataFrame()
    if df_csv is None or df_csv.empty:
        return df_db
    merged = pd.concat([df_db, df_csv], ignore_index=True)
    if "date" not in merged.columns:
        return merged
    merged["_order"] = range(len(merged))
    if "interval" in merged.columns:
        interval_norm = (
            merged["interval"]
            .astype(str)
            .replace("nan", "")
            .replace("NaT", "")
            .str.strip()
        )
        merged["__interval_norm"] = interval_norm
        merged = merged.sort_values("_order").drop_duplicates(
            subset=["date", "__interval_norm"], keep="last"
        )
        merged = merged.drop(columns=["__interval_norm"])
    else:
        merged = merged.sort_values("_order").drop_duplicates(subset=["date"], keep="last")
    return merged.drop(columns=["_order"])


def load_timeseries(kind: str, scope_key: str) -> pd.DataFrame:
    if not kind or not scope_key:
        return pd.DataFrame()
    df_db = pd.DataFrame()
    try:
        df_db = load_timeseries_any(kind, [scope_key])
    except Exception:
        df_db = pd.DataFrame()
    df_csv_raw = load_timeseries_csv(kind, scope_key)
    df_csv = pd.DataFrame()
    if isinstance(df_csv_raw, pd.DataFrame) and not df_csv_raw.empty:
        try:
            df_csv = normalize_timeseries_rows(kind, df_csv_raw.to_dict("records"))
        except Exception:
            df_csv = df_csv_raw
    merged = _merge_timeseries_frames(df_db, df_csv)
    if isinstance(merged, pd.DataFrame) and not merged.empty:
        return merged
    df_prefix = _load_timeseries_csv_prefix(kind, scope_key)
    if isinstance(df_prefix, pd.DataFrame) and not df_prefix.empty:
        return df_prefix
    return pd.DataFrame()


def push_forecast_to_planning(
    scope_key: str,
    channel: str,
    forecast_df: pd.DataFrame,
    *,
    default_aht_sec: Optional[float] = None,
    mode: str = "append",
) -> tuple[bool, str, Optional[int]]:
    """
    Push forecast volume (+AHT/SUT when available) into planning timeseries.
    Expects forecast_df to contain 'date' and 'volume' columns; optional
    'interval' and 'aht_sec'/'sut_sec' columns will be preserved.
    """
    if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
        return False, "No forecast rows to save.", None

    low_cols = {str(c).strip().lower(): c for c in forecast_df.columns}
    c_date = low_cols.get("date")
    c_vol = low_cols.get("volume")
    if not c_date or not c_vol:
        return False, "Forecast must include 'date' and 'volume' columns.", None

    df = forecast_df.copy()
    df[c_date] = pd.to_datetime(df[c_date], errors="coerce").dt.date.astype(str)
    df = df.dropna(subset=[c_date, c_vol])

    ch_low = (channel or "").strip().lower()
    vol_kind = None
    aht_kind = None
    if ch_low in ("voice", "call", "telephony"):
        vol_kind = "voice_forecast_volume"
        aht_kind = "voice_forecast_aht"
    elif ch_low in ("back office", "bo", "backoffice"):
        vol_kind = "bo_forecast_volume"
        aht_kind = "bo_forecast_sut"
    elif ch_low in ("chat", "messaging", "messageus", "message us"):
        vol_kind = "chat_forecast_volume"
        aht_kind = "chat_forecast_aht"
    elif ch_low in ("outbound", "ob", "out bound"):
        vol_kind = "ob_forecast_calls"
        aht_kind = None

    if not vol_kind:
        return False, f"Unsupported channel '{channel}' for planning push.", None

    extra_cols: list[str] = []
    for cand in ("interval", "interval_start", "time"):
        if cand in low_cols:
            extra_cols.append(low_cols[cand])
            break

    try:
        save_timeseries(
            vol_kind,
            scope_key,
            df[[c_date, c_vol] + extra_cols].rename(columns={c_date: "date", c_vol: "volume"}),
            mode=mode,
        )
        if aht_kind:
            target_aht_col = "aht_sec" if "aht" in aht_kind else "sut_sec"
            picked = None
            for cand in (
                "aht_sec",
                "sut_sec",
                "aht",
                "sut",
                "aht/sut",
                "aht_sut",
                "forecast_aht",
                "actual_aht",
                "tactical_aht",
                "forecast_sut",
                "actual_sut",
                "tactical_sut",
                "avg_handle_time",
                "avg_handle_time_sec",
                "avg_talk_sec",
                "talk_sec",
            ):
                if cand in low_cols:
                    picked = low_cols[cand]
                    break
            if picked or default_aht_sec is not None:
                aht_df = df[[c_date] + extra_cols].copy()
                if picked:
                    aht_vals = pd.to_numeric(df[picked], errors="coerce")
                    if default_aht_sec is not None:
                        aht_vals = aht_vals.fillna(float(default_aht_sec))
                    aht_df[target_aht_col] = aht_vals
                else:
                    aht_df[target_aht_col] = float(default_aht_sec)
                save_timeseries(
                    aht_kind,
                    scope_key,
                    aht_df.rename(columns={c_date: "date"}),
                    mode=mode,
                )
        return True, "Forecast saved and pushed to planning.", None
    except Exception as exc:
        return False, f"Failed to push forecast: {exc}", None


def _load_timeseries_csv_prefix(kind: str, scope_key: str) -> pd.DataFrame:
    parts = [p.strip() for p in str(scope_key or "").split("|")]
    if len(parts) < 3 or str(scope_key).lower().startswith("location|"):
        return pd.DataFrame()
    ba, sba, ch = parts[:3]
    safe_kind = _safe_component(kind)
    prefix = "timeseries_" + safe_kind + "_hier_" + "_".join(
        _safe_component(p) for p in (ba, sba, ch)
    ) + "_"
    outdir = _exports_dir()
    frames = []
    for path in outdir.glob(f"{prefix}*.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        # Compact scope filenames cannot be reverse-mapped from filename tokens.
        # Probe candidate 4-part scope keys and aggregate matches.
        probe_keys: list[str] = [f"{ba}|{sba}|{ch}", f"{ba}|{sba}|{ch}|all"]
        try:
            hc = load_headcount()
            if isinstance(hc, pd.DataFrame) and not hc.empty:
                cols = _headcount_cols(hc)
                ba_col = cols.get("ba")
                sba_col = cols.get("sba")
                lob_col = cols.get("lob")
                site_col = cols.get("site")

                def _norm(val: object) -> str:
                    return " ".join(str(val or "").strip().lower().split())

                subset = hc.copy()
                if ba_col and ba_col in subset.columns and ba:
                    subset = subset[subset[ba_col].map(_norm) == _norm(ba)]
                if sba_col and sba_col in subset.columns and sba:
                    subset = subset[subset[sba_col].map(_norm) == _norm(sba)]
                if lob_col and lob_col in subset.columns and ch:
                    subset = subset[subset[lob_col].map(_norm) == _norm(ch)]
                if site_col and site_col in subset.columns:
                    for site in subset[site_col].dropna().astype(str).map(str.strip):
                        if site:
                            probe_keys.append(f"{ba}|{sba}|{ch}|{site}")
        except Exception:
            pass

        seen: set[str] = set()
        for key in probe_keys:
            k = key.strip().lower()
            if not key or k in seen:
                continue
            seen.add(k)
            try:
                df_probe = load_timeseries_csv(kind, key)
            except Exception:
                df_probe = pd.DataFrame()
            if df_probe is None or df_probe.empty:
                continue
            try:
                df_probe = normalize_timeseries_rows(kind, df_probe.to_dict("records"))
            except Exception:
                pass
            if not df_probe.empty:
                frames.append(df_probe)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "date" not in out.columns:
        return out
    out["_order"] = range(len(out))
    if "interval" in out.columns:
        interval_norm = (
            out["interval"]
            .astype(str)
            .replace("nan", "")
            .replace("NaT", "")
            .str.strip()
        )
        out["__interval_norm"] = interval_norm
        out = out.sort_values("_order").drop_duplicates(subset=["date", "__interval_norm"], keep="last")
        out = out.drop(columns=["__interval_norm"])
    else:
        out = out.sort_values("_order").drop_duplicates(subset=["date"], keep="last")
    return out.drop(columns=["_order"])


def resolve_settings(
    ba: Optional[str] = None,
    subba: Optional[str] = None,
    lob: Optional[str] = None,
    site: Optional[str] = None,
    for_date: Optional[str] = None,
):
    # Match the resolution order used in plan_detail/planning_calc.
    if ba and subba and lob:
        if site:
            try:
                from app.pipeline import settings_store as _settings_store

                key = _settings_store._scope_key("hier", None, ba, subba, lob, site)
                path = _settings_store._exports_dir() / f"settings_{key}.json"
                key_lower = _settings_store._scope_key(
                    "hier",
                    None,
                    (ba or "").lower() if ba else None,
                    (subba or "").lower() if subba else None,
                    (lob or "").lower() if lob else None,
                    (site or "").lower() if site else None,
                )
                path_lower = _settings_store._exports_dir() / f"settings_{key_lower}.json"
                if path.exists() or path_lower.exists():
                    return load_settings("hier", None, ba, subba, lob, site, for_date)
            except Exception:
                pass
        # Fallback to site-agnostic settings for BA/SubBA/LOB
        return load_settings("hier", None, ba, subba, lob, None, for_date)
    return load_settings("global", None, None, None, None, None, for_date)


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
