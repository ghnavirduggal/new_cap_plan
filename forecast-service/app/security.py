"""Lightweight authentication/authorization for the forecast service.

Design goals:
- No third-party dependency: HS256 JWT verification is implemented with the
  standard library so it works even when PyJWT isn't installed.
- Fail-open by default: everything is gated behind env flags so existing
  deployments keep working until auth is explicitly enabled.

Env flags:
- AUTH_ENABLED=1        require a valid token on protected routes (401 otherwise)
- AUTHZ_ENABLED=1       enforce per-plan ownership checks (403 otherwise)
- AUTH_JWT_SECRET=...   shared HS256 signing secret (must match the token issuer)
- TRUST_PROXY_AUTH=1    accept identity from X-Forwarded-Email when no token is
                        present (only safe behind a proxy that strips inbound copies)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import HTTPException, Request


def _flag(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def auth_enabled() -> bool:
    return _flag("AUTH_ENABLED")


def authz_enabled() -> bool:
    return _flag("AUTHZ_ENABLED")


def trust_proxy_auth() -> bool:
    return _flag("TRUST_PROXY_AUTH")


def _secret() -> str:
    return os.getenv("AUTH_JWT_SECRET", "")


def _proxy_shared_secret() -> str:
    return os.getenv("PROXY_SHARED_SECRET", "")


def proxy_request_verified(request: Request) -> bool:
    """Defense-in-depth for the token-mint oracle and proxy-trusted identity.

    When PROXY_SHARED_SECRET is set, proxy-provided identity (and the
    /api/auth/token mint endpoint) is only honored if the request carries the
    matching X-Proxy-Auth header — a value the trusted reverse proxy injects and
    strips from inbound client requests. This stops an attacker who reaches the
    backend port directly (or spoofs X-Forwarded-Email) from minting a token.
    When the secret is unset, returns True to preserve existing behavior.
    """
    secret = _proxy_shared_secret()
    if not secret:
        return True
    provided = (request.headers.get("x-proxy-auth") or "").strip()
    return bool(provided) and hmac.compare_digest(provided, secret)


@dataclass
class Principal:
    email: str
    sub: str = ""
    roles: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles

    @property
    def key(self) -> str:
        return (self.email or self.sub or "").strip().lower()


def _b64url_decode(segment: str) -> bytes:
    pad = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + pad)


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def verify_jwt(token: str, secret: str) -> Optional[dict]:
    """Verify an HS256 JWT and return its claims, or None if invalid/expired."""
    if not token or not secret:
        return None
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError:
        return None
    try:
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != "HS256":
            return None
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(sig_b64)):
            return None
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return None
    # Require exp: a token without an expiry would be valid forever, so reject it
    # rather than treating a missing exp as "never expires".
    exp = payload.get("exp")
    if exp is None:
        return None
    try:
        if time.time() > float(exp):
            return None
    except (TypeError, ValueError):
        return None
    return payload


def mint_jwt(email: str, secret: str, *, sub: str = "", roles: Optional[list[str]] = None, ttl_seconds: int = 3600) -> str:
    """Issue a short-lived HS256 token. Used by the local/dev token endpoint."""
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": sub or email,
        "email": email,
        "roles": roles or [],
        "iat": now,
        "exp": now + int(ttl_seconds),
    }
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"


def principal_from_request(request: Request) -> Optional[Principal]:
    secret = _secret()
    token = None
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
    if not token:
        token = request.cookies.get("session_token")
    if token and secret:
        claims = verify_jwt(token, secret)
        if claims:
            roles = claims.get("roles") or []
            if not isinstance(roles, (list, tuple)):
                roles = []
            return Principal(
                email=str(claims.get("email") or claims.get("sub") or ""),
                sub=str(claims.get("sub") or ""),
                roles=tuple(str(r) for r in roles),
            )
    # Only honor proxy-provided identity when explicitly told the upstream proxy
    # is trusted (and strips client-supplied copies of these headers), and — when
    # a PROXY_SHARED_SECRET is configured — only if the request proves it came
    # through that proxy.
    if trust_proxy_auth() and proxy_request_verified(request):
        email = (
            request.headers.get("x-forwarded-email")
            or request.headers.get("x-email")
            or request.headers.get("x-user-email")
            or ""
        ).strip()
        if email:
            return Principal(email=email)
    return None


# Paths reachable without a token even when AUTH_ENABLED.
_EXEMPT_PREFIXES = ("/health", "/api/auth/token", "/docs", "/openapi.json", "/redoc")


def is_exempt_path(path: str) -> bool:
    return any(path == p or path.startswith(p) for p in _EXEMPT_PREFIXES)


def require_user(request: Request) -> Principal:
    """FastAPI dependency: the authenticated principal (or anonymous if auth off)."""
    principal = getattr(request.state, "principal", None)
    if principal is not None:
        return principal
    if auth_enabled():
        raise HTTPException(status_code=401, detail="Authentication required.")
    return Principal(email="anonymous")


def authorize_plan(plan: Optional[dict], principal: Principal) -> None:
    """Raise 403 if the principal may not access this plan (when AUTHZ enabled)."""
    if not authz_enabled():
        return
    if principal.is_admin:
        return
    if not isinstance(plan, dict):
        return
    owner = str(plan.get("owner") or plan.get("created_by") or "").strip().lower()
    if owner and owner == principal.key:
        return
    raise HTTPException(status_code=403, detail="Not authorized for this plan.")
