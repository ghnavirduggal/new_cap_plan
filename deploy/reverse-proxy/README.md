# Reverse-proxy auth example

A minimal, adapt-me example for fronting CAP-CONNECT with
[oauth2-proxy](https://oauth2-proxy.github.io/oauth2-proxy/) + nginx so the
backends receive a trusted `X-Forwarded-Email` and mint session JWTs.

Files:
- `nginx.conf` — terminates TLS, authenticates every request via oauth2-proxy
  (`auth_request`), **strips client-supplied identity headers**, and injects the
  authenticated `X-Forwarded-Email` before proxying to the app.
- `oauth2-proxy.cfg` — oauth2-proxy settings (point it at your OIDC provider).

## How it fits together

```
browser ──TLS──> nginx ──auth_request──> oauth2-proxy ──OIDC──> your IdP
                   │  (on success, nginx sets X-Forwarded-Email)
                   ├─ /            ─> web         (Next.js, :3000)
                   ├─ /api/auth/*  ─> forecast    (:8082, mints the JWT)
                   ├─ /api/forecast|planning|uploads ─> forecast (:8082)
                   └─ /api/*       ─> api         (Go, :8080)
```

Set on **both** backends (api + forecast-service):

```env
AUTH_ENABLED=1
AUTHZ_ENABLED=1
TRUST_PROXY_AUTH=1
AUTH_JWT_SECRET=<a long random shared secret, identical on both>
CORS_ORIGINS=https://your-app-host
```

> The single most important rule: nginx must **clear** any inbound
> `X-Forwarded-Email` / `X-Email` / `X-Forwarded-User` from the client and set it
> only from the oauth2-proxy auth response. The provided `nginx.conf` does this.
> Never publish the backend ports (8080/8082) directly.
