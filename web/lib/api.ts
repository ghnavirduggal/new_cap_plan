import { logChange } from "./activity";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
const FORECAST_BASE = process.env.NEXT_PUBLIC_FORECAST_URL || API_BASE;
const BROWSER_API_BASE = process.env.NEXT_PUBLIC_BROWSER_API_URL || "";
const BROWSER_FORECAST_BASE = process.env.NEXT_PUBLIC_BROWSER_FORECAST_URL || "";

function resolveBase(path: string) {
  if (typeof window !== "undefined") {
    if (path === "/api/user") {
      return "";
    }
    if (path.startsWith("/api/forecast") || path.startsWith("/api/planning") || path.startsWith("/api/uploads")) {
      return BROWSER_FORECAST_BASE;
    }
    return BROWSER_API_BASE;
  }
  if (path.startsWith("/api/forecast") || path.startsWith("/api/planning") || path.startsWith("/api/uploads")) {
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

export async function apiPost<T>(path: string, body: any): Promise<T> {
  const res = await fetchWithAuth(`${resolveBase(path)}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
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
