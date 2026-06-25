"""Plan approval workflow — a small state machine + audit trail.

States: draft → submitted → approved | rejected, with reopen back to draft.
Transitions are attributed (actor + timestamp + note) and logged to the existing
planning activity feed, giving an audit trail of who moved the plan and when.
This module is the pure state machine; the IO (meta + activity) lives in the API.
"""
from __future__ import annotations

from typing import Optional

STATES = ("draft", "submitted", "approved", "rejected")

# action -> (states it is allowed from, resulting state)
_TRANSITIONS = {
    "submit": ({"draft", "rejected"}, "submitted"),
    "approve": ({"submitted"}, "approved"),
    "reject": ({"submitted"}, "rejected"),
    "reopen": ({"submitted", "approved", "rejected"}, "draft"),
}
# Actions that change the approved/rejected outcome — gated to admins when AUTHZ on.
DECISION_ACTIONS = {"approve", "reject"}


def normalize_state(current: Optional[str]) -> str:
    s = (current or "draft").strip().lower()
    return s if s in STATES else "draft"


def transition(current: Optional[str], action: str) -> Optional[str]:
    """Return the new state for `action` from `current`, or None if not allowed."""
    cur = normalize_state(current)
    spec = _TRANSITIONS.get((action or "").strip().lower())
    if not spec:
        return None
    allowed_from, to = spec
    if cur not in allowed_from:
        return None
    return to


def available_actions(current: Optional[str]) -> list[str]:
    cur = normalize_state(current)
    return [a for a, (allowed_from, _to) in _TRANSITIONS.items() if cur in allowed_from]
