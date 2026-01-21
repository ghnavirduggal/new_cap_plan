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
