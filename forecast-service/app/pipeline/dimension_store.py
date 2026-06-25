"""Flexible-dimension registry (Phase 1).

A small, global, ordered list of custom dimensions planners can tag plans with
(e.g. tenure band, language, line-of-business, customer tier). System dimensions
(business area / sub-business-area / channel / site) are implicit and NOT stored
here. See docs/FLEXIBLE_DIMENSIONS_DESIGN.md.

The registry is a single JSON document persisted to the exports dir — no DB
migration, dependency-free. Dimension *values* live on each plan in
hierarchy_json.dimensions (handled in planning_store), not here.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

_KEY_RE = re.compile(r"[^a-z0-9_]+")
_REGISTRY_FILENAME = "dimension_registry.json"
_MAX_DIMENSIONS = 24
_MAX_VALUES = 200
# 'segment' is the legacy Phase-0 dimension with its own dedicated storage and UI
# panel/filter; reserve the key so a registry entry can't shadow it and become a
# dead, non-functional row (the UI filters 'segment' out of the generic surfaces).
_RESERVED_KEYS = {"segment"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _candidates() -> list[Path]:
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


def _registry_dir() -> Path:
    for base in _candidates():
        try:
            base.mkdir(exist_ok=True, parents=True)
            probe = base / ".perm_probe"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            return base
        except Exception:
            continue
    fallback = _candidates()[-1]
    fallback.mkdir(exist_ok=True, parents=True)
    return fallback


def _registry_path() -> Path:
    return _registry_dir() / _REGISTRY_FILENAME


def slug(value: Any) -> str:
    """Normalize a dimension key to a stable slug: lower-case, [a-z0-9_]."""
    s = _KEY_RE.sub("_", str(value or "").strip().lower()).strip("_")
    return s[:48]


def normalize_registry(raw: Any) -> list[dict]:
    """Coerce arbitrary input into a clean, ordered, de-duplicated registry.

    Each entry: {key, label, order, values}. Invalid/empty-key entries are
    dropped; duplicate keys keep the first occurrence. Order is reassigned
    densely (1..N) by the incoming sequence so callers can reorder by position.
    """
    items = raw if isinstance(raw, list) else []
    out: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = slug(item.get("key") or item.get("label"))
        if not key or key in seen or key in _RESERVED_KEYS:
            continue
        seen.add(key)
        label = str(item.get("label") or key).strip()[:64] or key
        values_raw = item.get("values")
        values: list[str] = []
        if isinstance(values_raw, (list, tuple)):
            vseen: set[str] = set()
            for v in values_raw:
                sv = str(v or "").strip()[:120]
                if sv and sv.lower() not in vseen:
                    vseen.add(sv.lower())
                    values.append(sv)
                if len(values) >= _MAX_VALUES:
                    break
        out.append({"key": key, "label": label, "values": values})
        if len(out) >= _MAX_DIMENSIONS:
            break
    for idx, entry in enumerate(out):
        entry["order"] = idx + 1
    return out


def load_registry() -> list[dict]:
    """Return the registered dimensions (empty list if none/unreadable)."""
    path = _registry_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(payload, dict):
        payload = payload.get("dimensions")
    return normalize_registry(payload)


def save_registry(raw: Any) -> list[dict]:
    """Validate + persist the registry. Returns the normalized list."""
    registry = normalize_registry(raw)
    path = _registry_path()
    try:
        path.write_text(json.dumps({"dimensions": registry}))
    except Exception:
        pass
    return registry


def registry_keys(registry: Optional[list[dict]] = None) -> list[str]:
    reg = registry if registry is not None else load_registry()
    return [d["key"] for d in reg]


def normalize_dimensions(raw: Any, registry: Optional[list[dict]] = None) -> dict:
    """Clean a plan's {key: value} dimension map.

    Keys are slugged; blank values are dropped. When a registry is supplied,
    only registered keys are kept (so deleting a dimension hides stale values
    without a destructive plan rewrite). Without a registry, all slugged keys
    pass through (back-compat / API ingest before a registry exists).
    """
    if not isinstance(raw, dict):
        return {}
    allowed: Optional[set[str]] = None
    if registry is not None:
        allowed = set(registry_keys(registry))
    out: dict[str, str] = {}
    for k, v in raw.items():
        key = slug(k)
        if not key:
            continue
        if allowed is not None and key not in allowed:
            continue
        val = str(v or "").strip()[:120]
        if val:
            out[key] = val
    return out
