package main

import (
  "context"
  "encoding/json"
  "errors"
  "fmt"
  "io"
  "log"
  "net/http"
  "os"
  "strings"
  "time"

  "github.com/go-chi/chi/v5"
  "github.com/go-chi/cors"
  "github.com/jackc/pgx/v5/pgxpool"
  "github.com/redis/go-redis/v9"
)

type server struct {
  db        *pgxpool.Pool
  redis     *redis.Client
  engineURL string
  modelsURL string
}

type settingsRequest struct {
  ScopeType string         `json:"scope_type"`
  ScopeKey  string         `json:"scope_key"`
  Data      map[string]any `json:"data"`
}

type timeseriesRequest struct {
  Kind     string           `json:"kind"`
  ScopeKey string           `json:"scope_key"`
  Mode     string           `json:"mode"`
  Rows     []map[string]any `json:"rows"`
}

type simpleResponse struct {
  Status  string `json:"status"`
  Message string `json:"message"`
}

type uploadPayload struct {
  Rows []map[string]any `json:"rows"`
}

func main() {
  ctx := context.Background()
  dbURL := envOrDefault("DATABASE_URL", "postgres://cap:cap@localhost:5432/cap_planner?sslmode=disable")
  pool, err := connectDB(ctx, dbURL)
  if err != nil {
    log.Fatalf("db connect failed: %v", err)
  }

  redisURL := envOrDefault("REDIS_URL", "redis://localhost:6379/0")
  redisOpt, err := redis.ParseURL(redisURL)
  if err != nil {
    log.Fatalf("redis url invalid: %v", err)
  }
  rdb := redis.NewClient(redisOpt)
  if err := rdb.Ping(ctx).Err(); err != nil {
    log.Printf("redis ping failed: %v", err)
  }

  srv := &server{
    db:        pool,
    redis:     rdb,
    engineURL: envOrDefault("ENGINE_URL", "http://localhost:8081"),
    modelsURL: envOrDefault("MODELS_URL", "http://localhost:8000"),
  }

  r := chi.NewRouter()
  r.Use(cors.Handler(cors.Options{
    AllowedOrigins:   splitCSV(envOrDefault("CORS_ORIGINS", "http://localhost:3000")),
    AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
    AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
    AllowCredentials: true,
    MaxAge:           300,
  }))

  r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
    respondJSON(w, http.StatusOK, map[string]any{"status": "ok"})
  })

  r.Route("/api", func(r chi.Router) {
    r.Get("/user", srv.handleUser)
    r.Get("/settings", srv.handleGetSettings)
    r.Post("/settings", srv.handleSaveSettings)

    r.Post("/uploads/timeseries", srv.handleSaveTimeseries)
    r.Get("/uploads/timeseries", srv.handleGetTimeseries)

    r.Post("/uploads/shrinkage", srv.handleSaveShrinkage)
    r.Get("/uploads/shrinkage", srv.handleGetShrinkage)

    r.Post("/uploads/attrition", srv.handleSaveAttrition)
    r.Get("/uploads/attrition", srv.handleGetAttrition)

    r.Post("/forecast/run", srv.handleRunForecast)
  })

  addr := envOrDefault("API_ADDR", ":8080")
  log.Printf("api listening on %s", addr)
  log.Fatal(http.ListenAndServe(addr, r))
}

func (s *server) handleUser(w http.ResponseWriter, r *http.Request) {
  name := headerUser(r)
  email := headerEmail(r)
  photo := headerPhoto(r)
  if name == "" && email != "" {
    name = strings.Split(email, "@")[0]
  }
  if name == "" {
    name = envOrDefault("DEFAULT_USER", "")
  }
  respondJSON(w, http.StatusOK, map[string]string{
    "name":      name,
    "email":     email,
    "photo_url": photo,
  })
}

func headerUser(r *http.Request) string {
  for _, key := range []string{"X-Forwarded-User", "X-User", "X-Email", "X-Forwarded-Email"} {
    val := strings.TrimSpace(r.Header.Get(key))
    if val == "" {
      continue
    }
    if strings.Contains(val, "@") {
      return strings.Split(val, "@")[0]
    }
    return val
  }
  return ""
}

func headerEmail(r *http.Request) string {
  for _, key := range []string{"X-Email", "X-Forwarded-Email", "X-User-Email"} {
    val := strings.TrimSpace(r.Header.Get(key))
    if val == "" {
      continue
    }
    return val
  }
  return ""
}

func headerPhoto(r *http.Request) string {
  for _, key := range []string{"X-User-Photo", "X-Forwarded-Photo", "X-Photo-Url", "X-User-Avatar"} {
    val := strings.TrimSpace(r.Header.Get(key))
    if val == "" {
      continue
    }
    return val
  }
  return ""
}

func connectDB(ctx context.Context, dbURL string) (*pgxpool.Pool, error) {
  const maxAttempts = 20
  for attempt := 1; attempt <= maxAttempts; attempt++ {
    pool, err := pgxpool.New(ctx, dbURL)
    if err == nil {
      if pingErr := pool.Ping(ctx); pingErr == nil {
        return pool, nil
      }
      pool.Close()
    }
    log.Printf("db not ready (attempt %d/%d). retrying...", attempt, maxAttempts)
    time.Sleep(2 * time.Second)
  }
  return nil, fmt.Errorf("db connection failed after retries")
}

func (s *server) handleGetSettings(w http.ResponseWriter, r *http.Request) {
  scopeType := strings.TrimSpace(r.URL.Query().Get("scope_type"))
  scopeKey := strings.TrimSpace(r.URL.Query().Get("scope_key"))
  if scopeType == "" {
    scopeType = "global"
  }
  if scopeKey == "" {
    scopeKey = "global"
  }
  var payload []byte
  err := s.db.QueryRow(r.Context(), `
    SELECT data
      FROM settings
     WHERE scope_type=$1 AND scope_key=$2
  `, scopeType, scopeKey).Scan(&payload)
  if err != nil {
    if errors.Is(err, context.Canceled) {
      respondJSON(w, http.StatusRequestTimeout, simpleResponse{Status: "error", Message: "request canceled"})
      return
    }
    respondJSON(w, http.StatusOK, map[string]any{"scope_type": scopeType, "scope_key": scopeKey, "data": map[string]any{}})
    return
  }
  var data map[string]any
  _ = json.Unmarshal(payload, &data)
  respondJSON(w, http.StatusOK, map[string]any{"scope_type": scopeType, "scope_key": scopeKey, "data": data})
}

func (s *server) handleSaveSettings(w http.ResponseWriter, r *http.Request) {
  var req settingsRequest
  if err := decodeJSON(r.Body, &req); err != nil {
    respondJSON(w, http.StatusBadRequest, simpleResponse{Status: "error", Message: err.Error()})
    return
  }
  if req.ScopeType == "" {
    req.ScopeType = "global"
  }
  if req.ScopeKey == "" {
    req.ScopeKey = "global"
  }
  payload, _ := json.Marshal(req.Data)
  _, err := s.db.Exec(r.Context(), `
    INSERT INTO settings (scope_type, scope_key, data, updated_at)
    VALUES ($1, $2, $3, now())
    ON CONFLICT (scope_type, scope_key) DO UPDATE
      SET data=EXCLUDED.data, updated_at=now()
  `, req.ScopeType, req.ScopeKey, payload)
  if err != nil {
    respondJSON(w, http.StatusInternalServerError, simpleResponse{Status: "error", Message: "failed to save settings"})
    return
  }
  respondJSON(w, http.StatusOK, simpleResponse{Status: "ok", Message: "settings saved"})
}

func (s *server) handleSaveTimeseries(w http.ResponseWriter, r *http.Request) {
  var req timeseriesRequest
  if err := decodeJSON(r.Body, &req); err != nil {
    respondJSON(w, http.StatusBadRequest, simpleResponse{Status: "error", Message: err.Error()})
    return
  }
  if req.Kind == "" {
    respondJSON(w, http.StatusBadRequest, simpleResponse{Status: "error", Message: "kind required"})
    return
  }
  if req.ScopeKey == "" {
    req.ScopeKey = "global"
  }
  incoming := uploadPayload{Rows: req.Rows}
  payload := incoming
  if strings.ToLower(strings.TrimSpace(req.Mode)) == "append" {
    if existing, ok := s.loadUpload(r.Context(), req.Kind, req.ScopeKey); ok {
      payload.Rows = mergeRows(existing.Rows, incoming.Rows)
    }
  }
  if err := s.saveUpload(r.Context(), req.Kind, req.ScopeKey, payload); err != nil {
    respondJSON(w, http.StatusInternalServerError, simpleResponse{Status: "error", Message: "failed to save"})
    return
  }
  respondJSON(w, http.StatusOK, simpleResponse{Status: "ok", Message: "saved"})
}

func (s *server) handleGetTimeseries(w http.ResponseWriter, r *http.Request) {
  kind := strings.TrimSpace(r.URL.Query().Get("kind"))
  scopeKey := strings.TrimSpace(r.URL.Query().Get("scope_key"))
  if kind == "" {
    respondJSON(w, http.StatusBadRequest, simpleResponse{Status: "error", Message: "kind required"})
    return
  }
  if scopeKey == "" {
    scopeKey = "global"
  }
  payload, ok := s.loadUpload(r.Context(), kind, scopeKey)
  if !ok {
    payload = uploadPayload{Rows: []map[string]any{}}
  }
  respondJSON(w, http.StatusOK, map[string]any{"kind": kind, "scope_key": scopeKey, "payload": payload})
}

func (s *server) handleSaveShrinkage(w http.ResponseWriter, r *http.Request) {
  s.handleSaveUploadKind(w, r, "shrinkage")
}

func (s *server) handleGetShrinkage(w http.ResponseWriter, r *http.Request) {
  s.handleGetUploadKind(w, r, "shrinkage")
}

func (s *server) handleSaveAttrition(w http.ResponseWriter, r *http.Request) {
  s.handleSaveUploadKind(w, r, "attrition")
}

func (s *server) handleGetAttrition(w http.ResponseWriter, r *http.Request) {
  s.handleGetUploadKind(w, r, "attrition")
}

func (s *server) handleSaveUploadKind(w http.ResponseWriter, r *http.Request, kind string) {
  var req timeseriesRequest
  if err := decodeJSON(r.Body, &req); err != nil {
    respondJSON(w, http.StatusBadRequest, simpleResponse{Status: "error", Message: err.Error()})
    return
  }
  if req.ScopeKey == "" {
    req.ScopeKey = "global"
  }
  payload := uploadPayload{Rows: req.Rows}
  if strings.ToLower(strings.TrimSpace(req.Mode)) == "append" {
    if existing, ok := s.loadUpload(r.Context(), kind, req.ScopeKey); ok {
      payload.Rows = mergeRows(existing.Rows, payload.Rows)
    }
  }
  if err := s.saveUpload(r.Context(), kind, req.ScopeKey, payload); err != nil {
    respondJSON(w, http.StatusInternalServerError, simpleResponse{Status: "error", Message: "failed to save"})
    return
  }
  respondJSON(w, http.StatusOK, simpleResponse{Status: "ok", Message: "saved"})
}

func (s *server) handleGetUploadKind(w http.ResponseWriter, r *http.Request, kind string) {
  scopeKey := strings.TrimSpace(r.URL.Query().Get("scope_key"))
  if scopeKey == "" {
    scopeKey = "global"
  }
  payload, ok := s.loadUpload(r.Context(), kind, scopeKey)
  if !ok {
    payload = uploadPayload{Rows: []map[string]any{}}
  }
  respondJSON(w, http.StatusOK, map[string]any{"kind": kind, "scope_key": scopeKey, "payload": payload})
}

func (s *server) handleRunForecast(w http.ResponseWriter, r *http.Request) {
  body, _ := io.ReadAll(r.Body)
  req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, s.modelsURL+"/forecast/run", strings.NewReader(string(body)))
  if err != nil {
    respondJSON(w, http.StatusInternalServerError, simpleResponse{Status: "error", Message: "failed to create request"})
    return
  }
  req.Header.Set("Content-Type", "application/json")
  client := &http.Client{Timeout: 20 * time.Second}
  resp, err := client.Do(req)
  if err != nil {
    respondJSON(w, http.StatusBadGateway, simpleResponse{Status: "error", Message: "models service unavailable"})
    return
  }
  defer resp.Body.Close()
  w.Header().Set("Content-Type", "application/json")
  w.WriteHeader(resp.StatusCode)
  _, _ = io.Copy(w, resp.Body)
}

func (s *server) saveUpload(ctx context.Context, kind, scopeKey string, payload uploadPayload) error {
  data, _ := json.Marshal(payload)
  _, err := s.db.Exec(ctx, `
    INSERT INTO uploads (kind, scope_key, payload, updated_at)
    VALUES ($1, $2, $3, now())
    ON CONFLICT (kind, scope_key) DO UPDATE
      SET payload=EXCLUDED.payload, updated_at=now()
  `, kind, scopeKey, data)
  return err
}

func (s *server) loadUpload(ctx context.Context, kind, scopeKey string) (uploadPayload, bool) {
  var raw []byte
  err := s.db.QueryRow(ctx, `
    SELECT payload
      FROM uploads
     WHERE kind=$1 AND scope_key=$2
  `, kind, scopeKey).Scan(&raw)
  if err != nil {
    return uploadPayload{}, false
  }
  var payload uploadPayload
  _ = json.Unmarshal(raw, &payload)
  return payload, true
}

func mergeRows(existing []map[string]any, incoming []map[string]any) []map[string]any {
  if len(existing) == 0 {
    return incoming
  }
  if len(incoming) == 0 {
    return existing
  }
  index := map[string]int{}
  out := append([]map[string]any{}, existing...)
  for i, row := range out {
    index[rowKey(row)] = i
  }
  for _, row := range incoming {
    key := rowKey(row)
    if idx, ok := index[key]; ok {
      out[idx] = row
      continue
    }
    index[key] = len(out)
    out = append(out, row)
  }
  return out
}

func rowKey(row map[string]any) string {
  date := strings.TrimSpace(fmt.Sprint(row["date"]))
  interval := strings.TrimSpace(fmt.Sprint(row["interval"]))
  if interval != "" && interval != "<nil>" {
    return date + "|" + interval
  }
  return date
}

func decodeJSON(r io.Reader, out any) error {
  dec := json.NewDecoder(r)
  dec.DisallowUnknownFields()
  if err := dec.Decode(out); err != nil {
    return err
  }
  return nil
}

func respondJSON(w http.ResponseWriter, status int, payload any) {
  w.Header().Set("Content-Type", "application/json")
  w.WriteHeader(status)
  _ = json.NewEncoder(w).Encode(payload)
}

func envOrDefault(key, fallback string) string {
  val := os.Getenv(key)
  if val == "" {
    return fallback
  }
  return val
}

func splitCSV(value string) []string {
  parts := strings.Split(value, ",")
  out := []string{}
  for _, part := range parts {
    trimmed := strings.TrimSpace(part)
    if trimmed != "" {
      out = append(out, trimmed)
    }
  }
  if len(out) == 0 {
    return []string{"*"}
  }
  return out
}
