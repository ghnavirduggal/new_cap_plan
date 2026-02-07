# Forecast Service

FastAPI wrapper around the existing Python forecasting pipeline used by the Dash app.

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Endpoints

- `GET /health`
- `GET /api/forecast/config`
- `POST /api/forecast/config`
- `POST /api/forecast/volume-summary` (multipart file)
- `POST /api/forecast/smoothing`
- `POST /api/forecast/smoothing/auto`
- `POST /api/forecast/phase1`
- `POST /api/forecast/phase2`
- `POST /api/forecast/transformations/apply`
- `POST /api/forecast/daily-interval`
