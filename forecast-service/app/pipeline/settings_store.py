from __future__ import annotations

import datetime as dt
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd

DEFAULT_SETTINGS = {
    "interval_minutes": 30,
    "hours_per_fte": 8.0,
    "shrinkage_pct": 0.30,
    "target_sl": 0.80,
    "sl_seconds": 20,
    "occupancy_cap_voice": 0.85,
    "util_bo": 0.85,
    "util_ob": 0.85,
    "chat_shrinkage_pct": 0.30,
    "ob_shrinkage_pct": 0.30,
    "util_chat": 0.85,
    "chat_concurrency": 1.5,
    "bo_capacity_model": "tat",
    "bo_tat_days": 5,
    "bo_workdays_per_week": 5,
    "bo_hours_per_day": 8.0,
    "bo_shrinkage_pct": 0.30,
    "nesting_weeks": 0,
    "sda_weeks": 0,
    "nesting_productivity_pct": [],
    "nesting_aht_uplift_pct": [],
    "sda_productivity_pct": [],
    "sda_aht_uplift_pct": [],
    "throughput_train_pct": 100.0,
    "throughput_nest_pct": 100.0,
    "cross_skill_efficiency_pct": 0.85,
    "max_lend_pct": 0.60,
    "lock_critical_teams": [],
}


def _normalize_effective_date(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        try:
            return dt.datetime.strptime(value, "%Y%m%d").date()
        except Exception:
            try:
                return dt.date.fromisoformat(value)
            except Exception:
                try:
                    return pd.to_datetime(value, errors="coerce").date()
                except Exception:
                    return None


def _format_effective_date(value: dt.date) -> str:
    return value.strftime("%Y%m%d")


@lru_cache(maxsize=2048)
def _read_settings_cached(path_str: str, mtime_ns: int) -> dict:
    try:
        data = json.loads(Path(path_str).read_text())
    except Exception:
        return DEFAULT_SETTINGS.copy()
    if not isinstance(data, dict):
        return DEFAULT_SETTINGS.copy()
    merged = DEFAULT_SETTINGS.copy()
    merged.update(data)
    return merged


def _read_settings(path: Path) -> dict:
    # Cache by (path, mtime) so the per-week settings resolution in plan detail
    # stops re-reading and re-parsing the same JSON file on every iteration.
    # A save bumps the mtime, so the cache self-invalidates.
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return DEFAULT_SETTINGS.copy()
    # Return a fresh copy so callers can mutate without corrupting the cache.
    return dict(_read_settings_cached(str(path), mtime_ns))


@lru_cache(maxsize=1024)
def _versioned_settings_files_cached(key: str, dir_mtime_ns: int) -> tuple[tuple[dt.date, Path], ...]:
    out: list[tuple[dt.date, Path]] = []
    prefix = f"settings_{key}_"
    for path in _exports_dir().glob(f"{prefix}*.json"):
        stem = path.stem
        suffix = stem[len(prefix) :]
        if not suffix:
            continue
        try:
            eff_date = dt.datetime.strptime(suffix, "%Y%m%d").date()
        except Exception:
            continue
        out.append((eff_date, path))
    return tuple(sorted(out, key=lambda item: item[0]))


def _versioned_settings_files(key: str) -> list[tuple[dt.date, Path]]:
    # Cache the directory scan by (key, dir-mtime). Adding/removing a versioned
    # settings file changes the directory mtime and invalidates the cache.
    try:
        dir_mtime_ns = _exports_dir().stat().st_mtime_ns
    except OSError:
        dir_mtime_ns = 0
    return list(_versioned_settings_files_cached(key, dir_mtime_ns))


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _exports_dir() -> Path:
    outdir = _repo_root() / "exports"
    outdir.mkdir(exist_ok=True)
    return outdir


def _safe_component(value: Optional[str]) -> str:
    if value is None:
        return "all"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return cleaned or "all"


def _scope_key(scope_type: str, location: Optional[str], ba: Optional[str], sba: Optional[str], channel: Optional[str], site: Optional[str]) -> str:
    scope = (scope_type or "global").strip().lower()
    if scope == "location":
        return f"location_{_safe_component(location)}"
    if scope == "hier":
        return (
            f"hier_{_safe_component(ba)}_"
            f"{_safe_component(sba)}_{_safe_component(channel)}_{_safe_component(site)}"
        )
    return "global"


def load_settings(
    scope_type: str,
    location: Optional[str],
    ba: Optional[str],
    sba: Optional[str],
    channel: Optional[str],
    site: Optional[str],
    for_date: Optional[str] = None,
) -> dict:
    key = _scope_key(scope_type, location, ba, sba, channel, site)
    target_date = _normalize_effective_date(for_date) or dt.date.today()
    candidates = _versioned_settings_files(key)
    if not candidates:
        key_lower = _scope_key(
            scope_type,
            (location or "").lower() if location else None,
            (ba or "").lower() if ba else None,
            (sba or "").lower() if sba else None,
            (channel or "").lower() if channel else None,
            (site or "").lower() if site else None,
        )
        candidates = _versioned_settings_files(key_lower)
    best_path = None
    best_date: Optional[dt.date] = None
    for eff_date, path in candidates:
        if eff_date <= target_date and (best_date is None or eff_date > best_date):
            best_date = eff_date
            best_path = path
    if best_path:
        return _read_settings(best_path)
    if candidates:
        # If no file is before the target, return the earliest available versioned settings.
        return _read_settings(candidates[0][1])
    base_path = _exports_dir() / f"settings_{key}.json"
    if not base_path.exists():
        key_lower = _scope_key(
            scope_type,
            (location or "").lower() if location else None,
            (ba or "").lower() if ba else None,
            (sba or "").lower() if sba else None,
            (channel or "").lower() if channel else None,
            (site or "").lower() if site else None,
        )
        base_path = _exports_dir() / f"settings_{key_lower}.json"
        if not base_path.exists():
            return DEFAULT_SETTINGS.copy()
    return _read_settings(base_path)


def save_settings(
    scope_type: str,
    location: Optional[str],
    ba: Optional[str],
    sba: Optional[str],
    channel: Optional[str],
    site: Optional[str],
    settings: dict,
    effective_date: Optional[str] = None,
) -> dict:
    key = _scope_key(
        scope_type,
        (location or "").lower() if location else None,
        (ba or "").lower() if ba else None,
        (sba or "").lower() if sba else None,
        (channel or "").lower() if channel else None,
        (site or "").lower() if site else None,
    )
    eff_date = _normalize_effective_date(effective_date) or dt.date.today()
    filename = f"settings_{key}_{_format_effective_date(eff_date)}.json"
    path = _exports_dir() / filename
    merged = DEFAULT_SETTINGS.copy()
    merged.update(settings or {})
    path.write_text(json.dumps(merged, indent=2))
    base_path = _exports_dir() / f"settings_{key}.json"
    try:
        base_path.write_text(json.dumps(merged, indent=2))
    except Exception:
        pass
    return merged


def load_holidays(scope_type: str, location: Optional[str], ba: Optional[str], sba: Optional[str], channel: Optional[str], site: Optional[str]) -> pd.DataFrame:
    key = _scope_key(scope_type, location, ba, sba, channel, site)
    path = _exports_dir() / f"holidays_{key}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df


def save_holidays(scope_type: str, location: Optional[str], ba: Optional[str], sba: Optional[str], channel: Optional[str], site: Optional[str], df: pd.DataFrame) -> pd.DataFrame:
    key = _scope_key(scope_type, location, ba, sba, channel, site)
    path = _exports_dir() / f"holidays_{key}.csv"
    df.to_csv(path, index=False)
    return df
