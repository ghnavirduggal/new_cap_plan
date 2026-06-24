package main

import (
  "context"
  "crypto/hmac"
  "crypto/sha256"
  "encoding/base64"
  "encoding/json"
  "errors"
  "net/http"
  "os"
  "strings"
  "time"
)

// Minimal stdlib-only HS256 JWT verification + a token mint endpoint. Avoids a
// third-party dependency so the service builds without fetching new modules.

type principal struct {
  Email string
  Roles []string
}

type ctxKey string

const principalCtxKey ctxKey = "principal"

func envFlag(key string) bool {
  switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
  case "1", "true", "yes", "on":
    return true
  }
  return false
}

func authEnabled() bool    { return envFlag("AUTH_ENABLED") }
func trustProxyAuth() bool { return envFlag("TRUST_PROXY_AUTH") }
func authSecret() string   { return os.Getenv("AUTH_JWT_SECRET") }

func verifyJWT(token, secret string) (map[string]any, error) {
  parts := strings.Split(token, ".")
  if len(parts) != 3 {
    return nil, errors.New("malformed token")
  }
  var header struct {
    Alg string `json:"alg"`
  }
  headerBytes, err := base64.RawURLEncoding.DecodeString(parts[0])
  if err != nil {
    return nil, err
  }
  if err := json.Unmarshal(headerBytes, &header); err != nil {
    return nil, err
  }
  if header.Alg != "HS256" {
    return nil, errors.New("unexpected alg")
  }
  mac := hmac.New(sha256.New, []byte(secret))
  mac.Write([]byte(parts[0] + "." + parts[1]))
  expected := mac.Sum(nil)
  sig, err := base64.RawURLEncoding.DecodeString(parts[2])
  if err != nil {
    return nil, err
  }
  if !hmac.Equal(expected, sig) {
    return nil, errors.New("bad signature")
  }
  payloadBytes, err := base64.RawURLEncoding.DecodeString(parts[1])
  if err != nil {
    return nil, err
  }
  var claims map[string]any
  if err := json.Unmarshal(payloadBytes, &claims); err != nil {
    return nil, err
  }
  if exp, ok := claims["exp"].(float64); ok {
    if time.Now().Unix() > int64(exp) {
      return nil, errors.New("token expired")
    }
  }
  return claims, nil
}

func mintJWT(email, secret string, ttl time.Duration) string {
  now := time.Now().Unix()
  header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`))
  payloadBytes, _ := json.Marshal(map[string]any{
    "sub":   email,
    "email": email,
    "roles": []string{},
    "iat":   now,
    "exp":   now + int64(ttl.Seconds()),
  })
  payload := base64.RawURLEncoding.EncodeToString(payloadBytes)
  mac := hmac.New(sha256.New, []byte(secret))
  mac.Write([]byte(header + "." + payload))
  return header + "." + payload + "." + base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func principalFromRequest(r *http.Request) *principal {
  secret := authSecret()
  token := ""
  if ah := r.Header.Get("Authorization"); len(ah) > 7 && strings.EqualFold(ah[:7], "bearer ") {
    token = strings.TrimSpace(ah[7:])
  }
  if token == "" {
    if c, err := r.Cookie("session_token"); err == nil {
      token = c.Value
    }
  }
  if token != "" && secret != "" {
    if claims, err := verifyJWT(token, secret); err == nil {
      email, _ := claims["email"].(string)
      if email == "" {
        email, _ = claims["sub"].(string)
      }
      var roles []string
      if rs, ok := claims["roles"].([]any); ok {
        for _, x := range rs {
          if s, ok := x.(string); ok {
            roles = append(roles, s)
          }
        }
      }
      return &principal{Email: email, Roles: roles}
    }
  }
  // Only trust proxy-provided identity when explicitly enabled.
  if trustProxyAuth() {
    if email := headerEmail(r); email != "" {
      return &principal{Email: email}
    }
  }
  return nil
}

func isExemptPath(path string) bool {
  for _, p := range []string{"/health", "/api/auth/token"} {
    if path == p {
      return true
    }
  }
  return false
}

func authMiddleware(next http.Handler) http.Handler {
  return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    p := principalFromRequest(r)
    if authEnabled() && p == nil && r.Method != http.MethodOptions && !isExemptPath(r.URL.Path) {
      respondJSON(w, http.StatusUnauthorized, simpleResponse{Status: "error", Message: "authentication required"})
      return
    }
    if p != nil {
      r = r.WithContext(context.WithValue(r.Context(), principalCtxKey, p))
    }
    next.ServeHTTP(w, r)
  })
}

func principalFromContext(r *http.Request) *principal {
  if p, ok := r.Context().Value(principalCtxKey).(*principal); ok {
    return p
  }
  return nil
}

func handleIssueToken(w http.ResponseWriter, r *http.Request) {
  secret := authSecret()
  if secret == "" || !trustProxyAuth() {
    respondJSON(w, http.StatusNotFound, simpleResponse{Status: "error", Message: "not found"})
    return
  }
  email := headerEmail(r)
  if email == "" {
    respondJSON(w, http.StatusUnauthorized, simpleResponse{Status: "error", Message: "no upstream identity"})
    return
  }
  ttl := 3600
  if raw := os.Getenv("AUTH_TOKEN_TTL"); raw != "" {
    if n, err := time.ParseDuration(raw + "s"); err == nil {
      ttl = int(n.Seconds())
    }
  }
  token := mintJWT(email, secret, time.Duration(ttl)*time.Second)
  respondJSON(w, http.StatusOK, map[string]string{"token": token, "token_type": "bearer", "email": email})
}
