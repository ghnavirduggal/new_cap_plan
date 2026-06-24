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
  "strconv"
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

type simpleResponse struct {
  Status  string `json:"status"`
  Message string `json:"message"`
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
  r.Use(limitBody(maxBodyBytes()))
  r.Use(authMiddleware)

  r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
    respondJSON(w, http.StatusOK, map[string]any{"status": "ok"})
  })

  r.Route("/api", func(r chi.Router) {
    r.Post("/auth/token", handleIssueToken)
    r.Get("/user", srv.handleUser)
    r.Get("/settings", srv.handleGetSettings)
    r.Post("/settings", srv.handleSaveSettings)

    // NOTE: /api/uploads/* and /api/forecast/* are served by the Python
    // forecast-service (the web Next.js rewrites route them there), so the Go
    // upload/forecast-run handlers were unreachable and have been removed. If
    // the Go service should instead own these paths, drop the matching rewrites
    // in web/next.config.js and reinstate the handlers.
  })

  addr := envOrDefault("API_ADDR", ":8080")
  log.Printf("api listening on %s", addr)
  httpSrv := &http.Server{
    Addr:              addr,
    Handler:           r,
    ReadHeaderTimeout: 10 * time.Second,
    ReadTimeout:       60 * time.Second,
    WriteTimeout:      120 * time.Second,
    IdleTimeout:       120 * time.Second,
  }
  log.Fatal(httpSrv.ListenAndServe())
}

func (s *server) handleUser(w http.ResponseWriter, r *http.Request) {
  // Prefer the verified principal from the auth middleware. Fall back to proxy
  // headers only when the upstream proxy is explicitly trusted.
  var name, email, photo string
  if p := principalFromContext(r); p != nil {
    email = p.Email
  } else if trustProxyAuth() {
    name = headerUser(r)
    email = headerEmail(r)
    photo = headerPhoto(r)
  }
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
    // Fail closed: never fall back to a wildcard origin (especially with
    // AllowCredentials enabled). Default to the local dev frontend only.
    return []string{"http://localhost:3000"}
  }
  return out
}

// maxBodyBytes caps the request body to mitigate memory-exhaustion DoS from
// unbounded uploads. Overridable via MAX_BODY_BYTES.
func maxBodyBytes() int64 {
  if raw := os.Getenv("MAX_BODY_BYTES"); raw != "" {
    if n, err := strconv.ParseInt(raw, 10, 64); err == nil && n > 0 {
      return n
    }
  }
  return 50 << 20 // 50 MiB
}

func limitBody(n int64) func(http.Handler) http.Handler {
  return func(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
      r.Body = http.MaxBytesReader(w, r.Body, n)
      next.ServeHTTP(w, r)
    })
  }
}
