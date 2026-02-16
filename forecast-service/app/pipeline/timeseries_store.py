from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

_SCOPE_FILE_MAX_LEN = 160
_SCOPE_HASH_LEN = 12


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _exports_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_dir = str(os.getenv("CAP_EXPORTS_DIR") or "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(_repo_root() / "exports")
    candidates.append(Path(tempfile.gettempdir()) / "cap_exports")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _exports_dir() -> Path:
    for outdir in _exports_candidates():
        try:
            outdir.mkdir(exist_ok=True, parents=True)
            probe = outdir / ".perm_probe"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            return outdir
        except Exception:
            continue
    # Final fallback keeps behavior deterministic even when probes fail.
    outdir = _exports_candidates()[-1]
    outdir.mkdir(exist_ok=True, parents=True)
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


def _compact_scope_token(token: str) -> str:
    value = str(token or "").strip("_")
    if not value:
        return "global"
    if len(value) <= _SCOPE_FILE_MAX_LEN:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:_SCOPE_HASH_LEN]
    suffix = f"__h{digest}"
    keep = max(_SCOPE_FILE_MAX_LEN - len(suffix), 1)
    prefix = value[:keep].rstrip("_")
    if not prefix:
        prefix = value[:keep]
    return f"{prefix}{suffix}"


def scope_file_keys(scope_key: str) -> list[str]:
    canonical = _canonical_scope_key(scope_key)
    compact = _compact_scope_token(canonical)
    keys = [compact]
    if canonical != compact:
        keys.append(canonical)
    return keys


def load_timeseries_csv(kind: str, scope_key: str) -> pd.DataFrame:
    if not kind:
        return pd.DataFrame()
    safe_kind = _safe_component(kind)
    for safe_scope in scope_file_keys(scope_key):
        filename = f"timeseries_{safe_kind}_{safe_scope}.csv"
        for outdir in _exports_candidates():
            path = outdir / filename
            if not path.exists():
                # Case-insensitive fallback (Windows mounts can preserve case, Linux lookups are strict).
                try:
                    target = path.name.lower()
                    for cand in outdir.glob(f"timeseries_{safe_kind}_*.csv"):
                        if cand.name.lower() == target:
                            path = cand
                            break
                except Exception:
                    continue
            if not path.exists():
                continue
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return pd.DataFrame()


def save_timeseries(kind: str, scope_key: str, df: pd.DataFrame, mode: str = "append") -> dict:
    if not kind:
        return {"status": "missing_kind"}
    outdir = _exports_dir()
    safe_kind = _safe_component(kind)
    safe_scope = scope_file_keys(scope_key)[0]
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
