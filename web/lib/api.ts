import { logChange } from "./activity";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
const FORECAST_BASE = process.env.NEXT_PUBLIC_FORECAST_URL || API_BASE;
const BROWSER_API_BASE = process.env.NEXT_PUBLIC_BROWSER_API_URL || "";
const BROWSER_FORECAST_BASE = process.env.NEXT_PUBLIC_BROWSER_FORECAST_URL || "";

// The forecast service hosts the bulk of the protected resources, so mint the
// session token there too — otherwise, in a split-host deployment, the token
// would be minted against the API host but sent as a bearer to the forecast
// host (only safe if the two share signing keys).
function isForecastScoped(path: string) {
  return (
    path.startsWith("/api/forecast") ||
    path.startsWith("/api/planning") ||
    path.startsWith("/api/uploads") ||
    path.startsWith("/api/users") ||
    path.startsWith("/api/auth") ||
    path === "/api/user"
  );
}

function resolveBase(path: string) {
  if (typeof window !== "undefined") {
    if (isForecastScoped(path)) {
      return BROWSER_FORECAST_BASE;
    }
    return BROWSER_API_BASE;
  }
  if (isForecastScoped(path)) {
    return FORECAST_BASE;
  }
  return API_BASE;
}

// --- Auth token handling -------------------------------------------------
// The backend mints a short-lived session JWT at /api/auth/token (only when
// auth is configured server-side; otherwise it 404s and we proceed without a
// token, so unauthenticated/dev setups are unaffected). The token is fetched
// once, cached, attached as a bearer header, and refreshed on a 401.
let tokenPromise: Promise<string | null> | null = null;

async function fetchToken(): Promise<string | null> {
  try {
    const res = await fetch(`${resolveBase("/api/auth/token")}/api/auth/token`, {
      method: "POST",
      cache: "no-store",
      credentials: "include"
    });
    if (!res.ok) return null;
    const data = await res.json();
    return typeof data?.token === "string" ? data.token : null;
  } catch {
    return null;
  }
}

function getToken(): Promise<string | null> {
  if (typeof window === "undefined") return Promise.resolve(null);
  if (!tokenPromise) tokenPromise = fetchToken();
  return tokenPromise;
}

async function authHeaders(): Promise<Record<string, string>> {
  const token = await getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// Drop the cached session token (and any client-readable session cookie) so a
// signed-out tab can't keep making authenticated calls until the next reload.
export function clearAuthToken(): void {
  tokenPromise = null;
  if (typeof document !== "undefined") {
    document.cookie = "session_token=; Path=/; Max-Age=0; SameSite=Strict";
  }
}

async function fetchWithAuth(input: string, init: RequestInit): Promise<Response> {
  const withAuth = async (): Promise<Response> => {
    const headers = { ...(init.headers as Record<string, string>), ...(await authHeaders()) };
    return fetch(input, { ...init, headers });
  };
  let res = await withAuth();
  if (res.status === 401 && typeof window !== "undefined") {
    // Token may be missing/expired — refresh once and retry.
    tokenPromise = fetchToken();
    res = await withAuth();
  }
  return res;
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetchWithAuth(`${resolveBase(path)}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`GET ${path} failed (${res.status})`);
  }
  return res.json();
}

export async function apiPost<T>(path: string, body: any, options?: { signal?: AbortSignal }): Promise<T> {
  const res = await fetchWithAuth(`${resolveBase(path)}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: options?.signal,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST ${path} failed (${res.status}): ${text}`);
  }
  if (shouldLogChange(path)) {
    logChange(`Saved ${path.replace("/api/", "")}`, path);
  }
  return res.json();
}

/**
 * POST that returns the raw Response (for binary/blob downloads such as
 * exports). Routes through fetchWithAuth so the bearer token is attached and
 * a 401 is retried once — a plain fetch() would skip auth and 401 whenever
 * AUTH_ENABLED is set.
 */
export async function apiPostRaw(path: string, body: any): Promise<Response> {
  const res = await fetchWithAuth(`${resolveBase(path)}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `POST ${path} failed (${res.status})`);
  }
  return res;
}

export async function apiPatch<T>(path: string, body: any): Promise<T> {
  const res = await fetchWithAuth(`${resolveBase(path)}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`PATCH ${path} failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function apiPostForm<T>(path: string, formData: FormData): Promise<T> {
  const res = await fetchWithAuth(`${resolveBase(path)}${path}`, {
    method: "POST",
    body: formData
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST ${path} failed (${res.status}): ${text}`);
  }
  return res.json();
}

function shouldLogChange(path: string) {
  const lower = path.toLowerCase();
  if (lower.includes("/uploads")) return true;
  if (lower.includes("/settings")) return true;
  if (lower.includes("/shrinkage")) return true;
  if (lower.includes("/attrition")) return true;
  return false;
}
