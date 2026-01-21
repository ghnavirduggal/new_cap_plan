from __future__ import annotations

from types import SimpleNamespace


request = SimpleNamespace(url=None, referrer=None)
session = {}
current_app = SimpleNamespace(secret_key=None)


def has_request_context() -> bool:
    return False
