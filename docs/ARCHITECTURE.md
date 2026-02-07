# Architecture Overview

## Services
- `services/api`: Go API for settings, uploads, and orchestration.
- `services/engine`: Go calculation engine stub (planning excluded for now).
- `services/models`: Python FastAPI for forecasting models.
- `apps/web`: Next.js frontend.

## Data
- Postgres stores settings/uploads as JSONB.
- Redis reserved for caching and session data.

## Key API Endpoints
- `GET /health`
- `GET /api/settings?scope_type=&scope_key=`
- `POST /api/settings`
- `POST /api/uploads/timeseries`
- `POST /api/uploads/shrinkage`
- `POST /api/uploads/attrition`
- `POST /api/forecast/run`

## Frontend Routes
- `/` Home
- `/forecast`
- `/shrinkage`
- `/settings`
- `/budget`
- `/ops`
- `/help`

Planning workspace is intentionally excluded from this rebuild.
