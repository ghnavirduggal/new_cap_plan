CREATE TABLE IF NOT EXISTS settings (
  scope_type TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  data JSONB NOT NULL DEFAULT '{}',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (scope_type, scope_key)
);

CREATE TABLE IF NOT EXISTS uploads (
  kind TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  payload JSONB NOT NULL DEFAULT '{}',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (kind, scope_key)
);

CREATE TABLE IF NOT EXISTS audit_log (
  id BIGSERIAL PRIMARY KEY,
  action TEXT NOT NULL,
  details JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Forecast run history (DB source of truth for core outputs)
CREATE TABLE IF NOT EXISTS forecast_runs (
  id BIGSERIAL PRIMARY KEY,
  run_type TEXT NOT NULL,
  scope_key TEXT,
  scope_key_norm TEXT,
  meta JSONB,
  created_by TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS forecast_run_steps (
  id BIGSERIAL PRIMARY KEY,
  run_id BIGINT NOT NULL REFERENCES forecast_runs(id) ON DELETE CASCADE,
  step TEXT NOT NULL,
  payload JSONB NOT NULL,
  row_count INTEGER,
  dataset_hash TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
