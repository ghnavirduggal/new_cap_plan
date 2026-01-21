from __future__ import annotations

import importlib
import sys

_BASE = "app.pipeline.plan_detail"


def _alias(name: str) -> None:
    module = importlib.import_module(f"{_BASE}.{name}")
    sys.modules[f"{__name__}.{name}"] = module


for _mod in (
    "_common",
    "_calc",
    "_fill_tables_fixed_monthly",
    "_fill_tables_fixed_daily",
    "_fill_tables_fixed_interval",
    "_grain_cols",
    "calc_engine",
):
    try:
        _alias(_mod)
    except Exception:
        pass

__all__ = [m.split(".")[-1] for m in sys.modules if m.startswith(f"{__name__}.")]
