"use client";

export type ForecastStore = {
  volumeSummary?: any;
  seasonality?: any;
  prophet?: any;
  smoothing?: any;
  finalSmoothed?: any;
  phase1?: any;
  phase2?: any;
  phase2Adjusted?: any;
  phase2Basis?: string;
  volumeSplitEdit?: any;
  transformations?: any;
  dailyInterval?: any;
  originalData?: any;
  intervalData?: any;
  distributionOverride?: any;
  metadata?: {
    lastUpdated?: string;
    // Set when a localStorage quota overflow forced intermediate results to be
    // dropped, so consumers can detect (and surface) the data loss instead of
    // silently showing "no data".
    trimmed?: boolean;
    trimmedAt?: string;
  };
};

const STORAGE_KEY = "capconnect.forecast.store";

function safeParse(value: string | null): ForecastStore {
  if (!value) return {};
  try {
    return JSON.parse(value) as ForecastStore;
  } catch {
    return {};
  }
}

function trimVolumeSummary(summary: any) {
  if (!summary || typeof summary !== "object") return summary;
  const {
    message,
    summary: summaryRows,
    categories,
    pivot,
    volume_split,
    iq_summary,
    // Keep the fields downstream pages depend on (daily-interval's `normalized`
    // fallback and the `holidays` mapping used by Prophet/Phase2). They are far
    // smaller than the IQ tables and dropping them silently broke later steps.
    normalized,
    holidays
  } = summary as any;
  return { message, summary: summaryRows, categories, pivot, volume_split, iq_summary, normalized, holidays };
}

function trimForecastStore(store: ForecastStore): ForecastStore {
  const next: ForecastStore = { ...store };
  if (next.volumeSummary) {
    next.volumeSummary = trimVolumeSummary(next.volumeSummary);
  }
  delete next.seasonality;
  delete next.prophet;
  delete next.smoothing;
  delete next.phase1;
  delete next.phase2;
  delete next.phase2Adjusted;
  delete next.volumeSplitEdit;
  delete next.transformations;
  delete next.dailyInterval;
  delete next.originalData;
  delete next.intervalData;
  return next;
}

export function loadForecastStore(): ForecastStore {
  if (typeof window === "undefined") return {};
  return safeParse(window.localStorage.getItem(STORAGE_KEY));
}

export function saveForecastStore(update: Partial<ForecastStore>): ForecastStore {
  if (typeof window === "undefined") return {};
  const current = loadForecastStore();
  const next = {
    ...current,
    ...update,
    metadata: {
      ...current.metadata,
      lastUpdated: new Date().toISOString()
    }
  };
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    return next;
  } catch (err) {
    // Quota overflow — drop the bulky intermediate tables, but record that we
    // did so the UI can warn the user their step data was discarded.
    const trimmed = trimForecastStore(next);
    trimmed.metadata = { ...trimmed.metadata, trimmed: true, trimmedAt: new Date().toISOString() };
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
      return trimmed;
    } catch {
      const minimal: ForecastStore = {
        metadata: { ...next.metadata, trimmed: true, trimmedAt: new Date().toISOString() }
      };
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(minimal));
      return minimal;
    }
  }
}

export function clearForecastStore() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(STORAGE_KEY);
}
