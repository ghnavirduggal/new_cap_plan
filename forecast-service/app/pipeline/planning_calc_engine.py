from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, Tuple

_MAX_WORKERS = max(1, int(os.getenv("PLAN_CALC_WORKERS", "4")))
_REALTIME = os.getenv("PLAN_CALC_REALTIME", "1").strip().lower() not in {"0", "false", "no"}
_CACHE_LIMIT = max(32, int(os.getenv("PLAN_CALC_CACHE_LIMIT", "128")))
_EXECUTOR: ThreadPoolExecutor | None = None

if not _REALTIME:
    _EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="plan-calc")

_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}
_JOBS: Dict[str, Dict[str, Any]] = {}
_PLAN_KEYS: Dict[str, set[str]] = {}


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize(value[k]) for k in sorted(value.keys())}
    if isinstance(value, (list, tuple, set)):
        return [_normalize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fw_signature(cols: Iterable[Any] | None) -> list[str]:
    sig: list[str] = []
    for col in cols or []:
        if isinstance(col, dict):
            col_id = col.get("id") or col.get("name")
        else:
            col_id = col
        if isinstance(col_id, (dict, list, tuple, set)):
            sig.append(json.dumps(_normalize(col_id), sort_keys=True))
        else:
            sig.append(str(col_id))
    return sig


def _make_key(
    plan_key: str,
    *,
    grain: str,
    fw_cols: Iterable[Any] | None,
    whatif: dict | None,
    interval_date: str | None,
    plan_type: str | None,
    version_token: Any,
    extra: dict | None,
) -> str:
    payload = dict(
        plan_key=str(plan_key or "global"),
        grain=str(grain or "week"),
        version=str(version_token or 0),
        fw=_fw_signature(fw_cols),
        whatif=_normalize(whatif or {}),
        interval=str(interval_date or ""),
        plan_type=str(plan_type or ""),
        extra=_normalize(extra or {}),
    )
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{plan_key}:{digest}"


def _public_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {k: v for k, v in meta.items() if k not in {"future"}}
    if "future" in meta:
        cleaned["future"] = None
    return cleaned


def _record_job(plan_key: str, key: str, status: str, **updates: Any) -> Dict[str, Any]:
    meta = dict(_JOBS.get(key) or {})
    meta.update(status=status, key=key, **updates)
    _JOBS[key] = meta
    _PLAN_KEYS.setdefault(str(plan_key or "global"), set()).add(key)
    return _public_meta(meta)


def ensure_plan_calc(
    plan_key: Any,
    *,
    grain: str,
    fw_cols: Iterable[Any] | None,
    whatif: dict | None,
    interval_date: str | None,
    plan_type: str | None,
    version_token: Any,
    builder: Callable[[], Tuple],
    extra: dict | None = None,
) -> tuple[Any, str, dict]:
    plan_key = str(plan_key or "global")
    key = _make_key(
        plan_key,
        grain=grain,
        fw_cols=fw_cols,
        whatif=whatif,
        interval_date=interval_date,
        plan_type=plan_type,
        version_token=version_token,
        extra=extra,
    )

    if _REALTIME:
        with _LOCK:
            if key in _CACHE:
                meta = _record_job(plan_key, key, "ready", finished=time.time())
                return _CACHE[key], "ready", meta

            existing = _JOBS.get(key)
            if existing and existing.get("status") == "running":
                return None, "running", _public_meta(existing)

            started = time.time()
            _record_job(plan_key, key, "running", started=started)

        try:
            result = builder()
        except Exception as exc:
            finished = time.time()
            with _LOCK:
                meta = _record_job(
                    plan_key,
                    key,
                    "failed",
                    error=repr(exc),
                    finished=finished,
                    duration=max(0.0, finished - started),
                )
            return None, "failed", meta

        finished = time.time()
        with _LOCK:
            _CACHE[key] = result
            meta = _record_job(
                plan_key,
                key,
                "ready",
                finished=finished,
                duration=max(0.0, finished - started),
            )
        _trim_cache()
        return result, "ready", meta

    with _LOCK:
        if key in _CACHE:
            meta = _record_job(plan_key, key, "ready", finished=time.time())
            return _CACHE[key], "ready", meta

        existing = _JOBS.get(key)
        if not existing or existing.get("status") != "running":
            started = time.time()
            future = _EXECUTOR.submit(_run_job, plan_key, key, builder, started) if _EXECUTOR else None
            meta = _record_job(plan_key, key, "running", started=started, future=future)
        else:
            meta = _public_meta(existing)
    return None, meta.get("status", "running"), meta


def _run_job(plan_key: str, key: str, builder: Callable[[], Tuple], started: float | None = None) -> None:
    started_at = started or time.time()
    try:
        result = builder()
    except Exception as exc:
        finished = time.time()
        with _LOCK:
            _record_job(
                plan_key,
                key,
                "failed",
                error=repr(exc),
                finished=finished,
                duration=max(0.0, finished - started_at),
            )
        return
    finished = time.time()
    with _LOCK:
        _CACHE[key] = result
        _record_job(
            plan_key,
            key,
            "ready",
            finished=finished,
            duration=max(0.0, finished - started_at),
        )
        _trim_cache()


def _trim_cache() -> None:
    if len(_CACHE) <= _CACHE_LIMIT:
        return
    for key in list(_CACHE.keys())[: max(1, len(_CACHE) - _CACHE_LIMIT)]:
        _CACHE.pop(key, None)


def mark_plan_dirty(plan_key: Any) -> None:
    plan_key = str(plan_key or "global")
    with _LOCK:
        keys = list(_PLAN_KEYS.get(plan_key, set()))
        for key in keys:
            _CACHE.pop(key, None)
            meta = _JOBS.pop(key, {})
            fut = meta.get("future")
            try:
                if fut and hasattr(fut, "cancel"):
                    fut.cancel()
            except Exception:
                pass
        _PLAN_KEYS[plan_key] = set()
    try:
        from app.pipeline.planning_calc import invalidate_consolidated_cache

        invalidate_consolidated_cache(plan_key)
    except Exception:
        pass
