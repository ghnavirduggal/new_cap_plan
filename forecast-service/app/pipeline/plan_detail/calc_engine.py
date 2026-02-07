from __future__ import annotations
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, Tuple

_MAX_WORKERS = max(1, int(os.getenv("PLAN_CALC_WORKERS", "2")))
_REALTIME = os.getenv("PLAN_CALC_REALTIME", "1").strip().lower() not in {"0", "false", "no"}
_EXECUTOR: ThreadPoolExecutor | None = None
if not _REALTIME:
    # Background workers remain available when realtime mode is disabled.
    _EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="plan-calc")

_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}
_JOBS: Dict[str, Dict[str, Any]] = {}
_PLAN_KEYS: Dict[int, set[str]] = {}
_DEP_LOCK = threading.Lock()
_DEP_TOKENS: Dict[int, Dict[str, int]] = {}
_KNOWN_DEPS = {
    "timeseries",
    "settings",
    "plan_meta",
    "roster",
    "newhire",
    "shrinkage",
    "attrition",
    "plan_tables",
    "whatif",
}

_LOG = logging.getLogger("plan-calc")


def _normalize(value: Any) -> Any:
    """Convert nested objects into JSON-friendly deterministic structures."""
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
    pid: int,
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
        pid=int(pid),
        grain=str(grain or "week"),
        version=str(version_token or 0),
        fw=_fw_signature(fw_cols),
        whatif=_normalize(whatif or {}),
        interval=str(interval_date or ""),
        plan_type=str(plan_type or ""),
        extra=_normalize(extra or {}),
    )
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{pid}:{digest}"


def _public_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {k: v for k, v in meta.items() if k not in {"future"}}
    if "future" in meta:
        cleaned["future"] = None
    return cleaned


def _record_job(pid: int, key: str, status: str, **updates: Any) -> Dict[str, Any]:
    meta = dict(_JOBS.get(key) or {})
    meta.update(status=status, key=key, **updates)
    _JOBS[key] = meta
    _PLAN_KEYS.setdefault(int(pid), set()).add(key)
    return _public_meta(meta)


def ensure_plan_calc(
    pid: Any,
    *,
    grain: str,
    fw_cols: Iterable[Any] | None,
    whatif: dict | None,
    interval_date: str | None,
    plan_type: str | None,
    version_token: Any,
    builder: Callable[[], Tuple],
    extra: dict | None = None,
):
    """
    Return cached results when available. In realtime mode we run the builder inline,
    otherwise we queue a worker job so Dash callbacks remain non-blocking.
    """
    try:
        pid_int = int(pid)
    except Exception:
        return None, "missing", {"reason": "invalid plan id"}

    key = _make_key(
        pid_int,
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
                meta = _record_job(pid_int, key, "ready", finished=time.time())
                return _CACHE[key], "ready", meta

            existing = _JOBS.get(key)
            if existing and existing.get("status") == "running":
                return None, "running", _public_meta(existing)

            started = time.time()
            _record_job(pid_int, key, "running", started=started)

        try:
            result = builder()
        except Exception as exc:
            _LOG.exception("plan calc failed (realtime): pid=%s key=%s grain=%s", pid_int, key, grain)
            finished = time.time()
            with _LOCK:
                meta = _record_job(
                    pid_int,
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
                pid_int,
                key,
                "ready",
                finished=finished,
                duration=max(0.0, finished - started),
            )
        return result, "ready", meta

    with _LOCK:
        if key in _CACHE:
            meta = _record_job(pid_int, key, "ready", finished=time.time())
            return _CACHE[key], "ready", meta

        existing = _JOBS.get(key)
        if not existing or existing.get("status") != "running":
            started = time.time()
            future = _EXECUTOR.submit(_run_job, pid_int, key, builder, started) if _EXECUTOR else None
            meta = _record_job(pid_int, key, "running", started=started, future=future)
        else:
            meta = _public_meta(existing)
    return None, meta.get("status", "running"), meta


def _run_job(pid: int, key: str, builder: Callable[[], Tuple], started: float | None = None):
    started_at = started or time.time()
    try:
        result = builder()
    except Exception as exc:
        _LOG.exception("plan calc failed (async): pid=%s key=%s", pid, key)
        finished = time.time()
        with _LOCK:
            _record_job(
                pid,
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
            pid,
            key,
            "ready",
            finished=finished,
            duration=max(0.0, finished - started_at),
        )


def mark_plan_dirty(pid: Any):
    """Mark plan dependencies dirty and drop pending jobs so new calc keys are used."""
    _bump_deps(pid, None)
    _clear_jobs(pid)
    _invalidate_consolidated_if_needed(pid, None)


def mark_plan_dirty_deps(pid: Any, deps: Any):
    """Mark specific dependencies dirty."""
    _bump_deps(pid, deps)
    _clear_jobs(pid)
    _invalidate_consolidated_if_needed(pid, deps)


def dep_snapshot(pid: Any, deps: list[str]) -> dict:
    """Return a stable snapshot of dependency tokens for cache keys."""
    try:
        pid_int = int(pid)
    except Exception:
        return {}
    with _DEP_LOCK:
        tokens = _DEP_TOKENS.get(pid_int, {})
        return {dep: int(tokens.get(dep, 0)) for dep in deps}


def dep_snapshot_all(pid: Any) -> dict:
    """Return all dependency tokens for a plan (debug/diagnostics)."""
    try:
        pid_int = int(pid)
    except Exception:
        return {}
    with _DEP_LOCK:
        tokens = _DEP_TOKENS.get(pid_int, {})
        return {str(k): int(v) for k, v in tokens.items()}


def _normalize_deps(deps: Any) -> list[str]:
    if deps is None:
        return sorted(_KNOWN_DEPS)
    if isinstance(deps, str):
        deps = [deps]
    if not isinstance(deps, (list, tuple, set)):
        return sorted(_KNOWN_DEPS)
    out = []
    for dep in deps:
        dep_str = str(dep or "").strip().lower()
        if dep_str:
            out.append(dep_str)
    return sorted(set(out)) or sorted(_KNOWN_DEPS)


def _ensure_dep_map(pid_int: int) -> Dict[str, int]:
    current = _DEP_TOKENS.get(pid_int)
    if current is None:
        current = {}
        _DEP_TOKENS[pid_int] = current
    return current


def _bump_deps(pid: Any, deps: Any) -> None:
    try:
        pid_int = int(pid)
    except Exception:
        return
    dep_list = _normalize_deps(deps)
    with _DEP_LOCK:
        token_map = _ensure_dep_map(pid_int)
        for dep in dep_list:
            token_map[dep] = int(token_map.get(dep, 0)) + 1


def _clear_jobs(pid: Any) -> None:
    try:
        pid_int = int(pid)
    except Exception:
        return
    with _LOCK:
        keys = list(_PLAN_KEYS.get(pid_int, set()))
        for key in keys:
            meta = _JOBS.pop(key, {})
            fut = meta.get("future")
            try:
                if fut and hasattr(fut, "cancel"):
                    fut.cancel()
            except Exception:
                pass
        _PLAN_KEYS[pid_int] = set()


def _invalidate_consolidated_if_needed(pid: Any, deps: Any) -> None:
    dep_list = _normalize_deps(deps)
    if not {"timeseries", "settings", "plan_meta"}.intersection(dep_list):
        return
    try:
        from ._calc import invalidate_consolidated_cache
        invalidate_consolidated_cache(pid)
    except Exception:
        pass
    try:
        pid_int = int(pid)
    except Exception:
        return
    with _LOCK:
        keys = list(_PLAN_KEYS.get(pid_int, set()))
        for key in keys:
            _CACHE.pop(key, None)
            meta = _JOBS.pop(key, {})
            fut = meta.get("future")
            try:
                if fut and hasattr(fut, "cancel"):
                    fut.cancel()
            except Exception:
                pass
        _PLAN_KEYS[pid_int] = set()
    # Also drop consolidated rollup cache (stored in plan_detail/_calc.py)
    try:
        from ._calc import invalidate_consolidated_cache
        invalidate_consolidated_cache(pid_int)
    except Exception:
        pass


def is_job_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except Exception:
        return False
    with _LOCK:
        keys = _PLAN_KEYS.get(pid_int, set())
        return any((_JOBS.get(k) or {}).get("status") == "running" for k in keys)


def describe_jobs() -> dict:
    with _LOCK:
        return json.loads(json.dumps(_JOBS, default=str))
