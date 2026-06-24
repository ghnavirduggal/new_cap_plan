package main

import (
  "encoding/json"
  "log"
  "net/http"
  "os"
  "strings"
  "time"

  "github.com/go-chi/chi/v5"
  "github.com/go-chi/cors"
)

type CalcRequest struct {
  PlanID int    `json:"plan_id"`
  Grain  string `json:"grain"`
}

type CalcResponse struct {
  Status  string `json:"status"`
  Message string `json:"message"`
  Results []any  `json:"results,omitempty"`
}

func main() {
  r := chi.NewRouter()
  // Never combine a wildcard origin with credentials. Use an explicit
  // allowlist from CORS_ORIGINS (comma-separated); default to the dev frontend.
  r.Use(cors.Handler(cors.Options{
    AllowedOrigins:   corsOrigins(),
    AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
    AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
    AllowCredentials: false,
    MaxAge:           300,
  }))

  r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
    respondJSON(w, http.StatusOK, map[string]any{"status": "ok"})
  })

  r.Post("/calc", func(w http.ResponseWriter, r *http.Request) {
    _ = json.NewDecoder(r.Body).Decode(&CalcRequest{})
    respondJSON(w, http.StatusNotImplemented, CalcResponse{
      Status:  "error",
      Message: "calculation engine not implemented yet",
    })
  })

  addr := envOrDefault("ENGINE_ADDR", ":8081")
  log.Printf("engine listening on %s", addr)
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

func corsOrigins() []string {
  raw := envOrDefault("CORS_ORIGINS", "http://localhost:3000")
  out := []string{}
  for _, part := range strings.Split(raw, ",") {
    if trimmed := strings.TrimSpace(part); trimmed != "" && trimmed != "*" {
      out = append(out, trimmed)
    }
  }
  if len(out) == 0 {
    return []string{"http://localhost:3000"}
  }
  return out
}

func envOrDefault(key, fallback string) string {
  val := os.Getenv(key)
  if val == "" {
    return fallback
  }
  return val
}

func respondJSON(w http.ResponseWriter, status int, payload any) {
  w.Header().Set("Content-Type", "application/json")
  w.WriteHeader(status)
  _ = json.NewEncoder(w).Encode(payload)
}
