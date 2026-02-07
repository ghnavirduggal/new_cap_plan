from __future__ import annotations


def get_user_role(_user: str | None) -> str:
    raise NotImplementedError("Auth module not configured.")


def can_delete_plans(_role: str | None) -> bool:
    raise NotImplementedError("Auth module not configured.")
