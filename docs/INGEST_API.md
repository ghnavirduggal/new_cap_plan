# Ingest API (v1)

A stable, documented REST endpoint for pushing volume/AHT **actuals**, **forecasts**,
and **tactical** series into CAP-CONNECT programmatically — so refreshes from a WFM
/ ACD / data-warehouse job don't require manual file uploads.

It reuses the same normalization and persistence as the interactive upload screen,
so ingested data flows into Plan Detail, Ops, BvA, and capacity rollups exactly like
an uploaded file would.

## Authentication

Two options:

1. **API key (recommended for automation).** Set `INGEST_API_KEY` to a long random
   value and send it as the `X-API-Key` header. No interactive login required.
2. **Session token.** If `INGEST_API_KEY` is unset, the endpoint falls back to the
   normal session auth (the same bearer/proxy identity the UI uses).

In a fully open dev setup (no auth configured) the endpoint is reachable without
credentials, like the rest of the app.

## Discover the contract

```
GET /api/ingest/v1/schema
X-API-Key: <key>
```

Returns the valid `kinds`, `channel_types`, `metrics`, `modes`, scope fields, row
fields, and a worked example — so an integration can self-serve.

## Push data

```
POST /api/ingest/v1/timeseries
Content-Type: application/json
X-API-Key: <key>
```

Identify the **series** one of two ways:

- explicit `kind` — one of `voice_forecast`, `voice_actual`, `voice_tactical`,
  `bo_*`, `chat_*`, `ob_*`, **or**
- `channel_type` (`voice` | `bo` | `chat` | `ob`) **+** `metric`
  (`forecast` | `actual` | `tactical`) — the server joins them into the `kind`.

Identify the **scope** with `business_area` / `sub_business_area` / `channel` / `site`
(or pass a raw `scope_key` like `"Cards|Servicing|Voice"`).

`mode` is `append` (default — overlapping dates are replaced) or `replace`.

Optionally include a `dimensions` object (e.g. `{"tenure": "Tenured", "language": "Spanish"}`)
to tag this scope with custom flexible-dimension values. It is stored in a
non-destructive sidecar keyed by the scope (it does **not** change the `scope_key`
or any calculation) and lets rollups group demand by a registered dimension. When
a dimension registry is configured, only registered keys are kept. See
[FLEXIBLE_DIMENSIONS_DESIGN.md](FLEXIBLE_DIMENSIONS_DESIGN.md).

### Row fields

| Field | Required | Notes |
|-------|----------|-------|
| `date` | yes | `YYYY-MM-DD` |
| `interval` | no | `HH:MM`, for interval-grain voice/chat data |
| `volume` | yes | contacts/calls handled; for Back Office use `items` |
| `aht` | no | average handle time (seconds); for Back Office use `sut` |

Up to 200,000 rows per request.

### Example

```bash
curl -X POST https://your-host/api/ingest/v1/timeseries \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $INGEST_API_KEY" \
  -d '{
    "business_area": "Cards",
    "sub_business_area": "Servicing",
    "channel": "Voice",
    "channel_type": "voice",
    "metric": "actual",
    "mode": "append",
    "rows": [
      {"date": "2026-06-01", "interval": "09:00", "volume": 120, "aht": 280},
      {"date": "2026-06-01", "interval": "09:30", "volume": 138, "aht": 295}
    ]
  }'
```

### Response

```json
{
  "status": "saved",
  "kind": "voice_actual",
  "scope_key": "Cards|Servicing|Voice",
  "mode": "append",
  "rows_ingested": 2,
  "date_range": {"from": "2026-06-01", "to": "2026-06-01"}
}
```

If the exact same batch was already ingested, the response is
`{"status": "unchanged", "unchanged": true, ...}` (idempotent — safe to retry).

## Errors

| Status | Meaning |
|--------|---------|
| 400 | Missing/invalid `kind` or `channel_type`+`metric`, empty `rows`, bad `mode` |
| 401 | `INGEST_API_KEY` is set but the `X-API-Key` header is missing/wrong |
| 413 | More than 200,000 rows in one request |

## Scheduling a pull

CAP-CONNECT does not poll external systems itself. To automate a refresh, run a
small job on your side (cron, Airflow, a Lambda) that reads from your WFM/ACD/DWH
and `POST`s to `/api/ingest/v1/timeseries` on a schedule.
