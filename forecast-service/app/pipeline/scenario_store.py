"""Saved what-if scenarios per plan.

A *scenario* is a named, reusable snapshot of the what-if dials (overrides) plus
the active window (start/end week). Where the live What-If is a single transient
record that the plan recomputes against, scenarios let a planner keep several
durable cases — "base", "downside", "peak-season" — save the current dials as one,
re-apply any of them later, and compare them side by side.

Persistence reuses the same per-plan table store as the rest of the plan
(planning_store.{load,save}_plan_table), so scenarios inherit the plan's
Postgres/local persistence with no new storage wiring.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Optional

import pandas as pd

from app.pipeline.planning_store import load_plan_table, save_plan_table

_TABLE = "scenarios"
# Keys we persist per scenario row.
_FIELDS = (
    "scenario_id",
    "name",
    "note",
    "overrides",
    "start_week",
    "end_week",
    "created_by",
    "created_ts",
    "updated_ts",
)


def _now() -> str:
    return pd.Timestamp.utcnow().isoformat(timespec="seconds")


def _coerce_overrides(raw: Any) -> dict:
    """Overrides may come back as a dict (JSON store) or a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            val = json.loads(raw)
            return val if isinstance(val, dict) else {}
        except Exception:
            return {}
    return {}


def _row_to_scenario(row: dict) -> dict:
    return {
        "scenario_id": str(row.get("scenario_id") or ""),
        "name": str(row.get("name") or ""),
        "note": str(row.get("note") or ""),
        "overrides": _coerce_overrides(row.get("overrides")),
        "start_week": str(row.get("start_week") or ""),
        "end_week": str(row.get("end_week") or ""),
        "created_by": str(row.get("created_by") or ""),
        "created_ts": str(row.get("created_ts") or ""),
        "updated_ts": str(row.get("updated_ts") or ""),
    }


def list_scenarios(plan_id: int) -> list[dict]:
    rows = load_plan_table(int(plan_id), _TABLE) or []
    out = [_row_to_scenario(r) for r in rows if isinstance(r, dict) and r.get("scenario_id")]
    # Most-recently-updated first so the newest case surfaces at the top.
    out.sort(key=lambda s: s.get("updated_ts") or s.get("created_ts") or "", reverse=True)
    return out


def get_scenario(plan_id: int, scenario_id: str) -> Optional[dict]:
    sid = str(scenario_id or "")
    for s in list_scenarios(plan_id):
        if s["scenario_id"] == sid:
            return s
    return None


def _persist(plan_id: int, scenarios: list[dict]) -> None:
    rows = []
    for s in scenarios:
        rows.append(
            {
                "scenario_id": s["scenario_id"],
                "name": s["name"],
                "note": s.get("note", ""),
                # Store overrides as JSON text so it round-trips through both the
                # Postgres JSON store and any flat/CSV fallback unchanged.
                "overrides": json.dumps(s.get("overrides") or {}),
                "start_week": s.get("start_week", ""),
                "end_week": s.get("end_week", ""),
                "created_by": s.get("created_by", ""),
                "created_ts": s.get("created_ts", ""),
                "updated_ts": s.get("updated_ts", ""),
            }
        )
    save_plan_table(int(plan_id), _TABLE, rows)


def save_scenario(
    plan_id: int,
    *,
    scenario_id: Optional[str] = None,
    name: str,
    note: str = "",
    overrides: Optional[dict] = None,
    start_week: str = "",
    end_week: str = "",
    actor: str = "",
) -> dict:
    """Create a new scenario or update an existing one (matched by scenario_id)."""
    name = (name or "").strip() or "Untitled scenario"
    overrides = overrides if isinstance(overrides, dict) else {}
    scenarios = list_scenarios(plan_id)
    now = _now()

    sid = str(scenario_id or "").strip()
    if sid:
        for s in scenarios:
            if s["scenario_id"] == sid:
                s.update(
                    name=name,
                    note=note or "",
                    overrides=overrides,
                    start_week=start_week or "",
                    end_week=end_week or "",
                    updated_ts=now,
                )
                _persist(plan_id, scenarios)
                return s
        # scenario_id supplied but not found — fall through and create it.

    sid = sid or uuid.uuid4().hex[:12]
    rec = {
        "scenario_id": sid,
        "name": name,
        "note": note or "",
        "overrides": overrides,
        "start_week": start_week or "",
        "end_week": end_week or "",
        "created_by": actor or "",
        "created_ts": now,
        "updated_ts": now,
    }
    scenarios.append(rec)
    _persist(plan_id, scenarios)
    return rec


def delete_scenario(plan_id: int, scenario_id: str) -> bool:
    sid = str(scenario_id or "")
    scenarios = list_scenarios(plan_id)
    remaining = [s for s in scenarios if s["scenario_id"] != sid]
    if len(remaining) == len(scenarios):
        return False
    _persist(plan_id, remaining)
    return True
