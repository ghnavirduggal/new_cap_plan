# Architecture Overview

## Services
- `services/api`: Go API for auth (token mint + verification), the current
  user, and global/scoped settings.
- `services/engine`: Go calculation engine stub (planning excluded for now).
- `services/models`: Python FastAPI for forecasting models (`POST /forecast/run`).
- `forecast-service`: Python FastAPI workspace — the authoritative backend for
  uploads (timeseries/shrinkage/attrition), forecasting, and planning. The web
  app's Next.js rewrites route `/api/forecast/*`, `/api/planning/*`, and
  `/api/uploads/*` here.
- `apps/web`: Next.js frontend.

## Data
- Postgres stores settings/uploads as JSONB.
- Redis reserved for caching and session data.

## Key API Endpoints
Go API (`services/api`):
- `GET /health`
- `POST /api/auth/token` (mint a session JWT from a trusted proxy identity)
- `GET /api/user`
- `GET /api/settings?scope_type=&scope_key=`
- `POST /api/settings`

forecast-service (Python) — reached via the web rewrites:
- `POST /api/uploads/timeseries` (and `/preview`)
- `POST /api/forecast/*`, `POST /api/planning/*`

## Frontend Routes
- `/` Home
- `/forecast`
- `/shrinkage`
- `/settings`
- `/budget`
- `/ops`
- `/help`

Planning workspace is intentionally excluded from this rebuild.
