from __future__ import annotations

import re
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from app.pipeline.postgres import db_conn, has_dsn, ensure_ops_schema
from app.pipeline.timeseries_store import load_timeseries_csv


logger = logging.getLogger(__name__)


def _channel_scope_aliases(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return [""]
    low = _loose_part(raw)
    aliases: list[str] = [low]
    if low in {"back office", "backoffice", "bo"}:
        aliases.extend(["back office", "backoffice", "bo"])
    elif low in {"outbound", "out bound", "ob"}:
        aliases.extend(["outbound", "out bound", "ob"])
    elif low in {"voice", "inbound", "call", "telephony"}:
        aliases.extend(["voice", "inbound"])
    elif low in {"chat", "message us", "messageus", "messaging"}:
        aliases.extend(["chat", "message us", "messageus", "messaging"])
    out: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        key = _loose_part(alias)
        if key in seen:
            continue
        seen.add(key)
        out.append(alias)
    return out


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


def list_timeseries_export_scopes(kinds: list[str]) -> set[str]:
    if not kinds:
        return set()
    out: set[str] = set()
    exports_dir = _exports_dir()
    for kind in kinds:
        safe_kind = _safe_component(kind)
        prefix = f"timeseries_{safe_kind}_"
        for path in exports_dir.glob(f"{prefix}*.csv"):
            stem = path.stem
            if stem.startswith(prefix):
                safe_scope = stem[len(prefix) :]
                if safe_scope:
                    out.add(safe_scope.lower())
    return out


def _norm_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _norm_hash_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, (float, int)):
        return str(value)
    return str(value).strip()


def _row_hash_from_parts(parts: list) -> str:
    payload = json.dumps([_norm_hash_value(p) for p in parts], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _timeseries_row_hash(row: dict) -> str:
    return _row_hash_from_parts(
        [
            row.get("date"),
            row.get("interval"),
            row.get("volume"),
            row.get("aht_sec"),
            row.get("sut_sec"),
            row.get("items"),
        ]
    )


def _pick_col(columns: Iterable[str], *candidates: str) -> Optional[str]:
    lookup = {_norm_col(col): col for col in columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in lookup:
            return lookup[key]
    return None


def _parse_date_series(series: pd.Series) -> pd.Series:
    s = pd.Series(series)
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            return pd.to_datetime(s, errors="coerce").dt.date
    except Exception:
        pass

    sample = s.dropna().astype(str).str.strip()
    if sample.empty:
        return pd.to_datetime(s, errors="coerce").dt.date

    iso_mask = sample.str.match(r"^\\d{4}-\\d{1,2}-\\d{1,2}$")
    if iso_mask.any() and iso_mask.mean() > 0.5:
        return pd.to_datetime(s, errors="coerce", format="%Y-%m-%d").dt.date

    dash_mask = sample.str.match(r"^\\d{1,2}-\\d{1,2}-\\d{2,4}$")
    if dash_mask.any() and dash_mask.mean() > 0.5:
        parts = sample[dash_mask].str.extract(r"^(\\d{1,2})-(\\d{1,2})-(\\d{2,4})$")
        first = pd.to_numeric(parts[0], errors="coerce")
        fmt = "%d-%m-%Y" if (first > 12).any() else "%m-%d-%Y"
        return pd.to_datetime(s, errors="coerce", format=fmt).dt.date

    slash_mask = sample.str.match(r"^\\d{1,2}/\\d{1,2}/\\d{2,4}$")
    if slash_mask.any() and slash_mask.mean() > 0.5:
        parts = sample[slash_mask].str.extract(r"^(\\d{1,2})/(\\d{1,2})/(\\d{2,4})$")
        first = pd.to_numeric(parts[0], errors="coerce")
        fmt = "%d/%m/%Y" if (first > 12).any() else "%m/%d/%Y"
        return pd.to_datetime(s, errors="coerce", format=fmt).dt.date

    return pd.to_datetime(s, errors="coerce").dt.date


def _to_seconds_value(value):
    try:
        if value is None:
            return np.nan
        if value is pd.NA:
            return np.nan
        try:
            if pd.isna(value):
                return np.nan
        except Exception:
            pass
        if isinstance(value, pd.Timedelta):
            return float(value.total_seconds())
        if isinstance(value, (np.integer, int)):
            return float(value)
        if isinstance(value, (np.floating, float)):
            v = float(value)
            if not np.isfinite(v):
                return np.nan
            if 0.0 < abs(v) < 1.0:
                return float(v * 86400.0)
            return v
        if hasattr(value, "hour") and hasattr(value, "minute") and hasattr(value, "second"):
            return float(int(value.hour) * 3600 + int(value.minute) * 60 + int(value.second))
        s = str(value).strip()
        if not s:
            return np.nan
        if ":" in s:
            parts = [p.strip() for p in s.split(":")]
            if len(parts) == 3:
                h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
                return (h * 3600.0) + (m * 60.0) + sec
            if len(parts) == 2:
                m = float(parts[0]); sec = float(parts[1])
                return (m * 60.0) + sec
        v = float(s.replace(",", ""))
        if 0.0 < abs(v) < 1.0:
            return float(v * 86400.0)
        return v
    except Exception:
        return np.nan


def _to_seconds_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(pd.Series(series).map(_to_seconds_value), errors="coerce")


def normalize_scope_key(scope_key: str) -> str:
    raw = str(scope_key or "").strip()
    if not raw:
        return "global"
    if raw.lower() == "global":
        return "global"
    parts = [p.strip() for p in raw.split("|") if p is not None]
    if not parts:
        return "global"
    if parts[0].lower() == "location":
        loc = parts[1] if len(parts) > 1 else ""
        return f"location|{loc.strip().lower()}"
    lower = [p.strip().lower() for p in parts if p is not None]
    if len(lower) >= 4:
        return "|".join(lower[:4])
    if len(lower) >= 3:
        return "|".join(lower[:3])
    return lower[0] if lower else "global"


def _loose_part(value: str) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = s.replace(",", " ")
    s = " ".join(s.split())
    return s


def normalize_scope_key_loose(scope_key: str) -> str:
    raw = str(scope_key or "").strip()
    if not raw:
        return "global"
    if raw.lower() == "global":
        return "global"
    parts = [p.strip() for p in raw.split("|") if p is not None]
    if not parts:
        return "global"
    if parts[0].lower() == "location":
        loc = parts[1] if len(parts) > 1 else ""
        return f"location|{_loose_part(loc)}"
    lower = [_loose_part(p) for p in parts if p is not None]
    if len(lower) >= 4:
        return "|".join(lower[:4])
    if len(lower) >= 3:
        return "|".join(lower[:3])
    return lower[0] if lower else "global"


def normalize_timeseries_rows(kind: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    if df.empty:
        return df
    cols = list(df.columns)
    date_col = _pick_col(cols, "date", "ds", "day")
    interval_col = _pick_col(cols, "interval", "interval start", "interval_start", "time")

    volume_col = _pick_col(
        cols,
        "volume",
        "actual volume/forecast volume",
        "forecast volume/actual volume",
        "actual/forecast volume",
        "forecast/actual volume",
        "forecast volume",
        "actual volume",
        "interval_forecast",
        "interval forecast",
        "daily_forecast",
        "daily forecast",
        "forecast",
        "calls",
        "items",
        "count",
    )
    aht_col = _pick_col(
        cols,
        "aht_sec",
        "aht",
        "aht/sut",
        "aht_sut",
        "aht sut",
        "actual aht/forecast aht",
        "forecast aht/actual aht",
        "actual/forecast aht",
        "forecast/actual aht",
        "aht sec",
        "aht secs",
        "aht seconds",
        "aht (sec)",
        "aht (secs)",
        "aht (seconds)",
        "forecast aht",
        "forecast aht/sut",
        "actual aht",
        "actual aht/sut",
        "tactical aht",
        "tactical aht/sut",
        "avg aht",
        "avg handle time",
        "avg handle time sec",
        "avg handle time secs",
        "average handle time",
        "average handle time sec",
        "average handle time secs",
        "average handle time seconds",
    )
    sut_col = _pick_col(
        cols,
        "sut_sec",
        "sut sec",
        "sut seconds",
        "sut (sec)",
        "sut",
        "forecast sut",
        "forecast sut (sec)",
        "actual sut",
        "actual sut (sec)",
        "tactical sut",
        "tactical sut (sec)",
    )
    items_col = _pick_col(cols, "items", "transactions", "txns")

    out = pd.DataFrame()
    if date_col:
        out["date"] = _parse_date_series(df[date_col])
    else:
        out["date"] = pd.NaT
    if interval_col:
        out["interval"] = df[interval_col].astype(str).str.strip()
    else:
        out["interval"] = None

    if volume_col:
        out["volume"] = pd.to_numeric(df[volume_col], errors="coerce")
    else:
        out["volume"] = None
    if aht_col:
        out["aht_sec"] = _to_seconds_series(df[aht_col])
    else:
        out["aht_sec"] = None
    if sut_col:
        out["sut_sec"] = _to_seconds_series(df[sut_col])
    else:
        out["sut_sec"] = None

    if items_col:
        out["items"] = pd.to_numeric(df[items_col], errors="coerce")
    else:
        out["items"] = None

    if out["items"].isna().all() and out["volume"].notna().any():
        out["items"] = out["volume"]

    return out.dropna(subset=["date"]).copy()


def _payload_rows(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("Rows") or payload.get("records")
        return rows if isinstance(rows, list) else []
    if isinstance(payload, list):
        return payload
    return []


def load_upload_rows(kind: str, scope_key: str) -> list[dict]:
    if not kind or not has_dsn():
        return []
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT payload FROM uploads WHERE kind = %s AND scope_key = %s",
                (kind, scope_key),
            )
            row = cur.fetchone()
            if not row:
                cur.execute(
                    "SELECT payload FROM uploads WHERE kind = %s AND lower(scope_key) = lower(%s)",
                    (kind, scope_key),
                )
                row = cur.fetchone()
    except Exception:
        return []
    if not row:
        return []
    payload = row[0]
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except Exception:
            pass
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return []
    return _payload_rows(payload)


def _split_metric_kind(kind: str) -> tuple[Optional[str], Optional[str]]:
    for suffix in ("_volume", "_aht", "_sut", "_opc", "_connect_rate", "_rpc", "_rpc_rate"):
        if kind.endswith(suffix):
            return kind[: -len(suffix)], suffix
    return None, None


def load_timeseries_from_uploads(kind: str, scope_key: str) -> pd.DataFrame:
    rows = load_upload_rows(kind, scope_key)
    if rows:
        return normalize_timeseries_rows(kind, rows)
    base_kind, metric = _split_metric_kind(kind)
    if not base_kind or not metric:
        return pd.DataFrame()
    base_rows = load_upload_rows(base_kind, scope_key)
    if not base_rows:
        return pd.DataFrame()
    df = normalize_timeseries_rows(base_kind, base_rows)
    if df.empty:
        return df
    if metric == "_volume":
        # Match combined-to-metric split behavior.
        value_col = "items" if base_kind.startswith(("bo_", "chat_")) else "volume"
        cols = ["date"]
        if "interval" in df.columns:
            cols.append("interval")
        if value_col in df.columns:
            cols.append(value_col)
        return df[cols].copy()
    if metric == "_aht":
        cols = ["date"]
        if "aht_sec" in df.columns:
            cols.append("aht_sec")
        return df[cols].copy()
    if metric == "_sut":
        cols = ["date"]
        if "sut_sec" in df.columns:
            cols.append("sut_sec")
        return df[cols].copy()
    return pd.DataFrame()


def _hashable_timeseries_df(kind: str, rows_or_df: object) -> pd.DataFrame:
    if isinstance(rows_or_df, pd.DataFrame):
        df = rows_or_df.copy()
    elif isinstance(rows_or_df, Sequence):
        df = pd.DataFrame(rows_or_df)
    else:
        df = pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if "date" not in df.columns or "interval" not in df.columns or "volume" not in df.columns:
        df = normalize_timeseries_rows(kind, df.to_dict("records"))
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
    if "interval" in df.columns:
        df["interval"] = df["interval"].astype("string")
    cols = [c for c in ("date", "interval", "volume", "aht_sec", "sut_sec", "items") if c in df.columns]
    df = df[cols].copy()
    for col in cols:
        df[col] = df[col].fillna("")
    sort_cols = [c for c in ("date", "interval") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def timeseries_dataset_hash(kind: str, rows_or_df: object) -> str:
    df = _hashable_timeseries_df(kind, rows_or_df)
    if df.empty:
        return ""
    hashed = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.sha1(hashed.tobytes()).hexdigest()


def get_latest_timeseries_hash(kind: str, scope_key_norm: str) -> Optional[str]:
    if not has_dsn():
        return None
    ensure_ops_schema()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT dataset_hash
            FROM timeseries_upload_meta
            WHERE kind = %s AND scope_key_norm = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (kind, scope_key_norm),
        )
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def record_timeseries_upload_hash(
    kind: str,
    scope_key_norm: str,
    dataset_hash: str,
    row_count: int,
) -> None:
    if not has_dsn() or not dataset_hash:
        return
    ensure_ops_schema()
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO timeseries_upload_meta (kind, scope_key_norm, dataset_hash, row_count)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (kind, scope_key_norm, dataset_hash)
            DO UPDATE SET row_count = EXCLUDED.row_count
            """,
            (kind, scope_key_norm, dataset_hash, int(row_count)),
        )


def save_timeseries_rows(kind: str, scope_key: str, rows: list[dict], mode: str = "append") -> int:
    if not kind:
        return 0
    if not has_dsn():
        return 0
    ensure_ops_schema()
    df = normalize_timeseries_rows(kind, rows)
    if df.empty:
        return 0

    scope_norm = normalize_scope_key(scope_key)
    df = df.copy()
    df["row_hash"] = df.apply(lambda r: _timeseries_row_hash(r.to_dict()), axis=1)

    if mode == "replace":
        with db_conn() as conn:
            conn.execute(
                "DELETE FROM timeseries_entries WHERE kind = %s AND scope_key_norm = %s",
                (kind, scope_norm),
            )
    else:
        existing_hashes: set[str] = set()
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT row_hash
                    FROM timeseries_entries
                    WHERE kind = %s AND scope_key_norm = %s AND row_hash IS NOT NULL
                    """,
                    (kind, scope_norm),
                )
                existing_hashes = {row[0] for row in cur.fetchall() if row and row[0]}
        except Exception:
            existing_hashes = set()

        if mode == "append":
            if existing_hashes:
                df = df[~df["row_hash"].isin(existing_hashes)].copy()
        elif mode == "override":
            if existing_hashes:
                to_delete = df["row_hash"].dropna().unique().tolist()
                if to_delete:
                    with db_conn() as conn:
                        conn.executemany(
                            """
                            DELETE FROM timeseries_entries
                            WHERE kind = %s AND scope_key_norm = %s AND row_hash = %s
                            """,
                            [(kind, scope_norm, h) for h in to_delete],
                        )

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                kind,
                scope_key or "global",
                scope_norm,
                row.get("date"),
                row.get("interval"),
                row.get("volume"),
                row.get("aht_sec"),
                row.get("sut_sec"),
                row.get("items"),
                row.get("row_hash"),
            )
        )
    with db_conn() as conn:
        conn.executemany(
            """
            INSERT INTO timeseries_entries (
                kind,
                scope_key,
                scope_key_norm,
                date,
                interval,
                volume,
                aht_sec,
                sut_sec,
                items,
                row_hash
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            records,
        )
    return len(records)


def preview_timeseries_rows(kind: str, scope_key: str, rows: list[dict]) -> dict:
    if not kind:
        return {"rows": 0, "duplicates": 0}
    df = normalize_timeseries_rows(kind, rows)
    if df.empty:
        return {"rows": 0, "duplicates": 0}
    df = df.copy()
    df["row_hash"] = df.apply(lambda r: _timeseries_row_hash(r.to_dict()), axis=1)
    scope_norm = normalize_scope_key(scope_key)
    existing_hashes: set[str] = set()
    if has_dsn():
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT row_hash
                    FROM timeseries_entries
                    WHERE kind = %s AND scope_key_norm = %s AND row_hash IS NOT NULL
                    """,
                    (kind, scope_norm),
                )
                existing_hashes = {row[0] for row in cur.fetchall() if row and row[0]}
        except Exception:
            existing_hashes = set()
    dup_count = int(df["row_hash"].isin(existing_hashes).sum()) if existing_hashes else 0
    return {
        "rows": int(len(df.index)),
        "duplicates": dup_count,
        "dataset_hash": timeseries_dataset_hash(kind, df),
    }


def load_timeseries_any(
    kind: str,
    scopes: list[str],
    start_date: Optional[object] = None,
    end_date: Optional[object] = None,
    batch: bool = False,
) -> pd.DataFrame:
    if not kind or not scopes:
        return pd.DataFrame()

    t0 = time.perf_counter()

    try:
        start = pd.to_datetime(start_date, errors="coerce").date() if start_date else None
    except Exception:
        start = None
    try:
        end = pd.to_datetime(end_date, errors="coerce").date() if end_date else None
    except Exception:
        end = None

    def _filter_dates(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "date" not in df.columns:
            return df
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        out = out[pd.notna(out["date"])]
        if start:
            out = out[out["date"] >= start]
        if end:
            out = out[out["date"] <= end]
        return out

    def _scope_norms_for(scope: str) -> list[str]:
        scope_norm = normalize_scope_key(scope)
        scope_norm_loose = normalize_scope_key_loose(scope)
        scope_norms: list[str] = []
        if scope_norm:
            scope_norms.append(scope_norm)
        if scope_norm_loose and scope_norm_loose not in scope_norms:
            scope_norms.append(scope_norm_loose)
        # Legacy variant: some uploads stored with trailing empty site segment
        for val in list(scope_norms):
            if val and val.count("|") == 2 and not val.endswith("|"):
                legacy = val + "|"
                if legacy not in scope_norms:
                    scope_norms.append(legacy)
            if val and val.endswith("|") and val.count("|") >= 3:
                trimmed = val.rstrip("|")
                if trimmed and trimmed not in scope_norms:
                    scope_norms.append(trimmed)
        try:
            raw = str(scope or "").strip()
            if raw and "|" in raw and not raw.lower().startswith("location|"):
                parts = [p.strip() for p in raw.split("|")]
                while len(parts) < 4:
                    parts.append("")
                ba, sba, channel, site = parts[:4]
                for ch_alias in _channel_scope_aliases(channel):
                    alt_raw = f"{ba}|{sba}|{ch_alias}|{site}" if site else f"{ba}|{sba}|{ch_alias}"
                    for cand in (normalize_scope_key(alt_raw), normalize_scope_key_loose(alt_raw)):
                        if cand and cand not in scope_norms:
                            scope_norms.append(cand)
                        if cand and cand.count("|") == 3 and cand.endswith("|"):
                            trimmed = cand.rstrip("|")
                            if trimmed and trimmed not in scope_norms:
                                scope_norms.append(trimmed)
        except Exception:
            pass
        return scope_norms

    if not has_dsn():
        frames = []
        for scope_key in scopes:
            df_raw = load_timeseries_csv(kind, scope_key)
            df = normalize_timeseries_rows(kind, df_raw.to_dict("records")) if not df_raw.empty else pd.DataFrame()
            if not df.empty:
                df["scope_key"] = scope_key
                df = _filter_dates(df)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    frames = []
    scope_norm_map = {scope: _scope_norms_for(scope) for scope in scopes}

    df_all = pd.DataFrame()
    batch_deduped = False
    if batch:
        all_norms = sorted({n for norms in scope_norm_map.values() for n in norms if n})
        if all_norms:
            placeholders = ",".join(["%s"] * len(all_norms))
            where = f"kind = %s AND scope_key_norm IN ({placeholders})"
            params: list[object] = [kind, *all_norms]
            if start:
                where += " AND date >= %s"
                params.append(start)
            if end:
                where += " AND date <= %s"
                params.append(end)
            t_query = time.perf_counter()
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    SELECT DISTINCT ON (scope_key_norm, date, interval)
                        scope_key,
                        scope_key_norm,
                        date,
                        interval,
                        volume,
                        aht_sec,
                        sut_sec,
                        items
                    FROM timeseries_entries
                    WHERE {where}
                    ORDER BY scope_key_norm, date, interval NULLS FIRST, created_at DESC
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
            t_fetch = time.perf_counter()
            df_all = pd.DataFrame(rows, columns=cols)
            batch_deduped = True
            t_df = time.perf_counter()
            logger.info(
                "timeseries batch kind=%s norms=%s rows=%s query=%.3fs df=%.3fs",
                kind,
                len(all_norms),
                len(rows),
                (t_fetch - t_query),
                (t_df - t_fetch),
            )

    for scope in scopes:
        scope_norms = scope_norm_map.get(scope) or []
        if not scope_norms:
            continue

        df = pd.DataFrame()
        if batch and not df_all.empty:
            df = df_all[df_all["scope_key_norm"].isin(scope_norms)].copy()
        elif not batch:
            with db_conn() as conn:
                cur = conn.cursor()
                if len(scope_norms) == 1:
                    where = "kind = %s AND scope_key_norm = %s"
                    params: list[object] = [kind, scope_norms[0]]
                else:
                    placeholders = ",".join(["%s"] * len(scope_norms))
                    where = f"kind = %s AND scope_key_norm IN ({placeholders})"
                    params = [kind, *scope_norms]
                if start:
                    where += " AND date >= %s"
                    params.append(start)
                if end:
                    where += " AND date <= %s"
                    params.append(end)
                cur.execute(
                    f"""
                    SELECT scope_key, date, interval, volume, aht_sec, sut_sec, items, created_at
                    FROM timeseries_entries
                    WHERE {where}
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
            df = pd.DataFrame(rows, columns=cols)

        if "created_at" in df.columns and not df.empty and not batch_deduped:
            try:
                df = df.sort_values("created_at")
            except Exception:
                pass
            if "date" in df.columns:
                if "interval" in df.columns:
                    interval_norm = (
                        df["interval"]
                        .astype(str)
                        .replace("nan", "")
                        .replace("NaT", "")
                        .str.strip()
                    )
                    df["__interval_norm"] = interval_norm
                    df = df.drop_duplicates(subset=["date", "__interval_norm"], keep="last").drop(columns=["__interval_norm"])
                else:
                    df = df.drop_duplicates(subset=["date"], keep="last")
            df = df.drop(columns=["created_at", "scope_key_norm"], errors="ignore")

        # CSV fallback for same scope (helps when DB write fails).
        df_csv = pd.DataFrame()
        try:
            df_raw = load_timeseries_csv(kind, scope)
            if not df_raw.empty:
                df_csv = normalize_timeseries_rows(kind, df_raw.to_dict("records"))
                if not df_csv.empty:
                    df_csv["scope_key"] = scope
                    df_csv = _filter_dates(df_csv)
        except Exception:
            df_csv = pd.DataFrame()

        if not df.empty:
            df = _filter_dates(df)
            if df_csv.empty:
                frames.append(df)
            else:
                merged = pd.concat([df, df_csv], ignore_index=True)
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
                merged = merged.drop(columns=["_order"])
                merged = _filter_dates(merged)
                frames.append(merged)
            continue

        df_upload = load_timeseries_from_uploads(kind, scope)
        if not df_upload.empty:
            if "scope_key" not in df_upload.columns:
                df_upload = df_upload.copy()
                df_upload["scope_key"] = scope
            df_upload = _filter_dates(df_upload)
            frames.append(df_upload)
            continue
        if not df_csv.empty:
            frames.append(df_csv)
            continue

        prefixes = []
        for sn in scope_norms:
            parts = sn.split("|")
            if len(parts) >= 3 and not sn.startswith("location|"):
                prefixes.append("|".join(parts[:3]))
        prefixes = list(dict.fromkeys(prefixes))
        for prefix in prefixes:
            where = "kind = %s AND scope_key_norm LIKE %s"
            params: list[object] = [kind, f"{prefix}|%"]
            if start:
                where += " AND date >= %s"
                params.append(start)
            if end:
                where += " AND date <= %s"
                params.append(end)
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    SELECT scope_key, date, interval, volume, aht_sec, sut_sec, items, created_at
                    FROM timeseries_entries
                    WHERE {where}
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
            df_prefix = pd.DataFrame(rows, columns=cols)
            if "created_at" in df_prefix.columns and not df_prefix.empty:
                try:
                    df_prefix = df_prefix.sort_values("created_at")
                except Exception:
                    pass
                if "date" in df_prefix.columns:
                    if "interval" in df_prefix.columns:
                        interval_norm = (
                            df_prefix["interval"]
                            .astype(str)
                            .replace("nan", "")
                            .replace("NaT", "")
                            .str.strip()
                        )
                        df_prefix["__interval_norm"] = interval_norm
                        df_prefix = df_prefix.drop_duplicates(subset=["date", "__interval_norm"], keep="last").drop(columns=["__interval_norm"])
                    else:
                        df_prefix = df_prefix.drop_duplicates(subset=["date"], keep="last")
                df_prefix = df_prefix.drop(columns=["created_at"], errors="ignore")
            df_prefix = _filter_dates(df_prefix)
            if not df_prefix.empty:
                frames.append(df_prefix)
                break

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    t_end = time.perf_counter()
    logger.info(
        "timeseries load kind=%s scopes=%s batch=%s rows=%s total=%.3fs",
        kind,
        len(scopes),
        bool(batch),
        int(len(result.index)) if isinstance(result, pd.DataFrame) else 0,
        (t_end - t0),
    )
    return result


def list_timeseries_scope_keys(kinds: list[str]) -> list[str]:
    if not kinds:
        return []
    if not has_dsn():
        return []
    placeholders = ", ".join(["%s"] * len(kinds))
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT DISTINCT scope_key FROM timeseries_entries WHERE kind IN ({placeholders})",
            tuple(kinds),
        )
        rows = cur.fetchall()
    return sorted({(row[0] if isinstance(row, tuple) else row.get("scope_key")) for row in rows if row})


def load_roster() -> pd.DataFrame:
    if not has_dsn():
        return pd.DataFrame()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT start_date, end_date, fte, program, status
            FROM roster_entries
            """
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        return df

    # Fallback for environments where roster uploads are stored only as payload rows.
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT payload FROM roster_long_entries ORDER BY date, id")
        payload_rows = cur.fetchall()
    payloads = [row[0] if isinstance(row, tuple) else row for row in payload_rows if row]
    if payloads:
        return pd.DataFrame(payloads)
    return pd.DataFrame()


def load_hiring() -> pd.DataFrame:
    if not has_dsn():
        return pd.DataFrame()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT start_week, fte, program
            FROM hiring_entries
            """
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)
