# CAP-CONNECT (rebuild)

This is a clean rebuild with a simplified, easy-to-debug structure.

## Stack
- Frontend: Next.js (apps/web)
- API: Go (services/api)
- Calculation Engine: Go (services/engine)
- Forecast Models: Python/FastAPI (services/models)
- Forecast Workspace: Python/FastAPI (apps/forecast-service)
- Data: Postgres + Redis

## Quick start
```bash
cd cap-planner
cp .env.example .env

docker compose up --build
```

## Ports
- Web: http://localhost:3000
- API: http://localhost:8080
- Engine: http://localhost:8081
- Models: http://localhost:8000
- Forecast Service: http://localhost:8082
- Postgres: localhost:5432
- Redis: localhost:6379

## Notes
- Planning workspace is intentionally excluded in this rebuild (as requested).
- Uploads are stored in Postgres as JSON.
- Use CSV or XLSX/XLS uploads in Settings and Shrinkage pages.

## Configuration

All tunables live in `.env` (see `.env.example` for the full list with
defaults). Highlights:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated CORS allowlist (never `*` with credentials). |
| `POSTGRES_POOL` / `_MIN` / `_MAX` | `1` / `1` / `16` | forecast-service connection pool sizing (`POSTGRES_POOL=0` disables). |
| `MAX_REQUEST_BYTES` / `MAX_BODY_BYTES` | 50 MiB | Request-body size caps (forecast-service / Go API). |
| `ENABLE_DEBUG_ENDPOINTS` | `0` | Exposes `/api/planning/plan/debug*` (off in production). |
| `HEADCOUNT_CACHE_TTL` | `30` | Headcount snapshot cache TTL (seconds). |
| `AUTH_ENABLED` | `0` | Require a valid session token on protected routes. |
| `AUTHZ_ENABLED` | `0` | Enforce per-plan ownership. |
| `AUTH_JWT_SECRET` | — | Shared HS256 secret (identical across Go API + forecast-service). |
| `TRUST_PROXY_AUTH` | `0` | Trust upstream `X-Forwarded-Email` and enable token minting. |
| `AUTH_TOKEN_TTL` | `3600` | Minted session-token lifetime (seconds). |

## Authentication

Auth is **off by default** — with none of the `AUTH_*` flags set the app behaves
exactly as before. When enabled it uses short-lived HS256 JWTs that both the Go
API and the Python forecast-service verify with the same `AUTH_JWT_SECRET`, so a
single token works across both services.

### Reverse-proxy (SSO) deployment — recommended

Put a trusted reverse proxy (oauth2-proxy, nginx with auth_request, Cloudflare
Access, etc.) in front that authenticates the user and forwards their identity
as `X-Forwarded-Email`. The flow:

1. Proxy authenticates the user (OIDC/SAML/whatever) and injects
   `X-Forwarded-Email`, **stripping any client-supplied copy** of that header.
2. The browser calls `POST /api/auth/token`; the forecast-service reads the
   trusted header and mints a JWT signed with `AUTH_JWT_SECRET`.
3. `web/lib/api.ts` caches that token and attaches it as `Authorization: Bearer`
   on every request, refreshing once on a 401.
4. Both backends verify the token; per-plan authorization uses the plan `owner`.

Required env (set on **both** backends):

```env
AUTH_ENABLED=1
AUTHZ_ENABLED=1
TRUST_PROXY_AUTH=1
AUTH_JWT_SECRET=<a long random shared secret>
CORS_ORIGINS=https://your-app-host
```

> **Security:** `TRUST_PROXY_AUTH` makes the services trust `X-Forwarded-Email`.
> Only enable it behind a proxy that overwrites/strips that header on inbound
> requests — otherwise a client can spoof identity. Never expose the backend
> ports (8080/8082) directly to the internet; route all traffic through the proxy.

A ready-to-adapt oauth2-proxy + nginx example is in
[`deploy/reverse-proxy/`](deploy/reverse-proxy/).
