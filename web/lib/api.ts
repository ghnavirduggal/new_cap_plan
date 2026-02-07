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

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${resolveBase(path)}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`GET ${path} failed (${res.status})`);
  }
  return res.json();
}

export async function apiPost<T>(path: string, body: any): Promise<T> {
  const res = await fetch(`${resolveBase(path)}${path}`, {
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
  const res = await fetch(`${resolveBase(path)}${path}`, {
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
