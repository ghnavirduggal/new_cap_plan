package main

import (
  "encoding/json"
  "log"
  "net/http"
  "os"

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
  r.Use(cors.Handler(cors.Options{
    AllowedOrigins:   []string{"*"},
    AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
    AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
    AllowCredentials: true,
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
  log.Fatal(http.ListenAndServe(addr, r))
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
