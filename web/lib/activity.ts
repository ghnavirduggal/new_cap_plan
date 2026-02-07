export type ActivityType = "visit" | "change";

export type ActivityEntry = {
  id: string;
  type: ActivityType;
  label: string;
  path?: string;
  at: string;
};

const STORAGE_KEY = "cap.activity.log";
const MAX_ENTRIES = 100;

export function logPageVisit(path: string, label: string) {
  logActivity({ type: "visit", label, path });
}

export function logChange(label: string, path?: string) {
  logActivity({ type: "change", label, path });
}

export function getActivity(): ActivityEntry[] {
  if (typeof window === "undefined") return [];
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    const data = JSON.parse(raw);
    if (Array.isArray(data)) {
      return data as ActivityEntry[];
    }
  } catch {
    return [];
  }
  return [];
}

function logActivity(entry: Omit<ActivityEntry, "id" | "at"> & { id?: string; at?: string }) {
  if (typeof window === "undefined") return;
  const existing = getActivity();
  const next: ActivityEntry = {
    id: entry.id ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    type: entry.type,
    label: entry.label,
    path: entry.path,
    at: entry.at ?? new Date().toISOString()
  };
  const merged = [next, ...existing].slice(0, MAX_ENTRIES);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
  window.dispatchEvent(new CustomEvent("cap-activity"));
}
