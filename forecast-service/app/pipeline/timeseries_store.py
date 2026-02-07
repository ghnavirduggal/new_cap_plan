from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


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


def _canonical_scope_key(scope_key: str) -> str:
    if not scope_key:
        return "global"
    raw = str(scope_key).strip()
    lower = raw.lower()
    if lower.startswith("location|"):
        _, loc = raw.split("|", 1)
        return f"location_{_safe_component(loc)}"
    if "|" in raw:
        parts = raw.split("|")
        while len(parts) < 4:
            parts.append("all")
        ba, sba, channel, site = parts[:4]
        return "hier_" + "_".join(
            _safe_component(part) for part in (ba, sba, channel, site)
        )
    return _safe_component(raw)


def load_timeseries_csv(kind: str, scope_key: str) -> pd.DataFrame:
    if not kind:
        return pd.DataFrame()
    safe_kind = _safe_component(kind)
    safe_scope = _canonical_scope_key(scope_key)
    path = _exports_dir() / f"timeseries_{safe_kind}_{safe_scope}.csv"
    if not path.exists():
        # Case-insensitive fallback (Windows mounts can preserve case, Linux lookups are strict).
        try:
            target = path.name.lower()
            for cand in _exports_dir().glob(f"timeseries_{safe_kind}_*.csv"):
                if cand.name.lower() == target:
                    path = cand
                    break
        except Exception:
            return pd.DataFrame()
        if not path.exists():
            return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_timeseries(kind: str, scope_key: str, df: pd.DataFrame, mode: str = "append") -> dict:
    if not kind:
        return {"status": "missing_kind"}
    outdir = _exports_dir()
    safe_kind = _safe_component(kind)
    safe_scope = _canonical_scope_key(scope_key)
    path = outdir / f"timeseries_{safe_kind}_{safe_scope}.csv"
    if df is None or df.empty:
        return {"status": "empty", "rows": 0, "path": str(path)}
    try:
        from app.pipeline.ops_store import save_timeseries_rows

        save_timeseries_rows(kind, scope_key, df.to_dict("records"), mode=mode)
    except Exception:
        pass
    if mode != "replace" and path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
    if "date" in df.columns:
        df = df.copy()
        df["_order"] = range(len(df))
        if "interval" in df.columns:
            interval_norm = (
                df["interval"]
                .astype(str)
                .replace("nan", "")
                .replace("NaT", "")
                .str.strip()
            )
            df["__interval_norm"] = interval_norm
            df = df.sort_values("_order").drop_duplicates(
                subset=["date", "__interval_norm"], keep="last"
            )
            df = df.drop(columns=["__interval_norm"])
        else:
            df = df.sort_values("_order").drop_duplicates(subset=["date"], keep="last")
        df = df.drop(columns=["_order"])
    df.to_csv(path, index=False)
    return {"status": "saved", "rows": len(df.index), "path": str(path)}
