"""Per-scope dimension sidecar (flexible dimensions, Phase 2).

Maps a normalized timeseries scope_key (``ba|sba|channel|site``) to a custom
dimension map (e.g. ``{"tenure": "Tenured", "language": "Spanish"}``). This is
the non-destructive "Option B" sidecar from docs/FLEXIBLE_DIMENSIONS_DESIGN.md:
the timeseries scope_key itself is NOT rewritten — demand stays exactly where it
is — and rollups can OPT IN to grouping scopes by a registered dimension by
joining this map. The default BA/SBA/Channel/Site rollup is untouched.

Stored as a single JSON document in the exports dir (mirrors accuracy_store /
dimension_store): no DB migration, dependency-free.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.pipeline import dimension_store
from app.pipeline.ops_store import normalize_scope_key

_FILENAME = "scope_dimensions_map.json"


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


def _dir() -> Path:
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


def _path() -> Path:
    return _dir() / _FILENAME


def _norm_key(scope_key: Any) -> str:
    try:
        return normalize_scope_key(str(scope_key or "").strip())
    except Exception:
        return str(scope_key or "").strip().lower()


def _clean_dims(raw: Any, registry: Any = None) -> dict:
    """Slug keys, drop blanks; with a registry, keep only registered keys."""
    return dimension_store.normalize_dimensions(raw, registry)


def load_all() -> dict[str, dict]:
    """Return the full {scope_key_norm: {dim_key: value}} map."""
    path = _path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict] = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            dims = {str(dk).strip().lower(): str(dv).strip()
                    for dk, dv in v.items() if str(dk).strip() and str(dv).strip()}
            if dims:
                out[str(k).strip().lower()] = dims
    return out


def load(scope_key: str) -> dict:
    """Dimensions for one scope (empty dict if none)."""
    return load_all().get(_norm_key(scope_key), {})


def save(scope_key: str, dimensions: Any, registry: Any = None) -> dict:
    """Replace one scope's dimension map; blank map removes the entry.

    Returns the cleaned dimensions that were stored.
    """
    key = _norm_key(scope_key)
    if not key or key == "global":
        return {}
    dims = _clean_dims(dimensions, registry)
    data = load_all()
    if dims:
        data[key] = dims
    else:
        data.pop(key, None)
    try:
        _path().write_text(json.dumps(data))
    except Exception:
        pass
    return dims
