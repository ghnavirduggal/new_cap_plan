"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import AppShell from "../../_components/AppShell";
import DataTable from "../../_components/DataTable";
import EditableTable from "../../_components/EditableTable";
import { useGlobalLoader } from "../../_components/GlobalLoader";
import LineChart from "../../_components/LineChart";
import { useToast } from "../../_components/ToastProvider";
import { apiGet, apiPost } from "../../../lib/api";
import { loadForecastStore, saveForecastStore } from "../forecast-store";

function toCsv(rows: Array<Record<string, any>>) {
  if (!rows.length) return "";
  const columns = Array.from(
    rows.reduce<Set<string>>((set, row) => {
      Object.keys(row || {}).forEach((key) => set.add(key));
      return set;
    }, new Set<string>())
  );
  const escape = (val: any) => {
    if (val === null || val === undefined) return "";
    const str = String(val).replace(/"/g, '""');
    return /[",\n]/.test(str) ? `"${str}"` : str;
  };
  const header = columns.join(",");
  const body = rows.map((row) => columns.map((col) => escape(row[col])).join(",")).join("\n");
  return `${header}\n${body}`;
}

function downloadCsv(filename: string, rows: Array<Record<string, any>>) {
  const csv = toCsv(rows);
  if (!csv) return false;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  return true;
}

function buildForecastMonth(year: string, month: string) {
  if (!year || !month) return "";
  const monthMap: Record<string, string> = {
    jan: "01",
    feb: "02",
    mar: "03",
    apr: "04",
    may: "05",
    jun: "06",
    jul: "07",
    aug: "08",
    sep: "09",
    oct: "10",
    nov: "11",
    dec: "12"
  };
  const trimmed = month.toString().trim();
  const lower = trimmed.toLowerCase().slice(0, 3);
  const safeMonth = monthMap[lower] || trimmed.padStart(2, "0");
  return `${year}-${safeMonth}-01`;
}

function parseNumber(value: any) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const cleaned = value.replace(/[%,$]/g, "").trim();
  const num = Number(cleaned);
  return Number.isFinite(num) ? num : null;
}

function normalizeDateValue(value: any) {
  if (!value) return "";
  const asString = String(value).trim();
  if (/^\d{4}-\d{2}-\d{2}/.test(asString)) return asString.slice(0, 10);
  const dt = new Date(asString);
  if (!Number.isNaN(dt.valueOf())) return dt.toISOString().slice(0, 10);
  return asString;
}

function resolveChannelKind(channel: string) {
  const raw = (channel || "").trim().toLowerCase();
  const isVoice = ["voice", "call", "telephony"].includes(raw);
  const isBO = ["back office", "backoffice", "bo"].includes(raw);
  const isChat = ["chat", "messaging", "messageus", "message us"].includes(raw);
  const isOB = ["outbound", "ob", "out bound"].includes(raw);
  let kind = "";
  if (isVoice) kind = "voice_forecast";
  else if (isBO) kind = "bo_forecast";
  else if (isChat) kind = "chat_forecast";
  else if (isOB) kind = "ob_forecast";
  return { kind, isVoice, isBO, isChat, isOB };
}

function getDefaultAhtFromSettings(settings: Record<string, any> | null, channel: string) {
  if (!settings) return null;
  const raw = (channel || "").trim().toLowerCase();
  if (["back office", "backoffice", "bo"].includes(raw)) {
    return (
      parseNumber(settings.last_actual_sut_sec) ??
      parseNumber(settings.last_budget_sut_sec) ??
      parseNumber(settings.target_sut) ??
      parseNumber(settings.budgeted_sut) ??
      null
    );
  }
  if (["chat", "messaging", "messageus", "message us"].includes(raw)) {
    return (
      parseNumber(settings.last_actual_aht_sec) ??
      parseNumber(settings.last_budget_aht_sec) ??
      parseNumber(settings.chat_aht_sec) ??
      parseNumber(settings.target_aht) ??
      parseNumber(settings.budgeted_aht) ??
      null
    );
  }
  if (["outbound", "ob", "out bound"].includes(raw)) {
    return (
      parseNumber(settings.last_actual_aht_sec) ??
      parseNumber(settings.last_budget_aht_sec) ??
      parseNumber(settings.ob_aht_sec) ??
      parseNumber(settings.target_aht) ??
      parseNumber(settings.budgeted_aht) ??
      null
    );
  }
  return (
    parseNumber(settings.last_actual_aht_sec) ??
    parseNumber(settings.last_budget_aht_sec) ??
    parseNumber(settings.target_aht) ??
    parseNumber(settings.budgeted_aht) ??
    null
  );
}

function hasIntervalColumn(rows: Array<Record<string, any>>) {
  if (!rows.length) return false;
  const keys = Object.keys(rows[0] || {});
  return keys.some((key) => {
    const normalized = key.toLowerCase().replace(/[^a-z0-9]/g, "");
    return ["interval", "time", "timeslot", "intervalstart", "starttime"].includes(normalized);
  });
}

export default function DailyIntervalPage() {
  const { notify } = useToast();
  const { setLoading: setGlobalLoading } = useGlobalLoader();

  const [transformResult, setTransformResult] = useState<any>(null);
  const [volumeSummary, setVolumeSummary] = useState<any>(null);
  const [dailyResult, setDailyResult] = useState<any>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [originalData, setOriginalData] = useState<Array<Record<string, any>>>([]);
  const [intervalData, setIntervalData] = useState<Array<Record<string, any>>>([]);
  const [distributionOverride, setDistributionOverride] = useState<Array<Record<string, any>>>([]);
  const [selectedYear, setSelectedYear] = useState<string>("");
  const [selectedMonth, setSelectedMonth] = useState<string>("");
  const [selectedGroup, setSelectedGroup] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [groupLevel, setGroupLevel] = useState("forecast_group");
  const [pushBA, setPushBA] = useState("");
  const [pushSBA, setPushSBA] = useState("");
  const [pushChannel, setPushChannel] = useState("Voice");
  const [pushSite, setPushSite] = useState("");
  const [pushGranularity, setPushGranularity] = useState("auto");
  const [pushAhtSec, setPushAhtSec] = useState("");
  const [pushOptions, setPushOptions] = useState({
    businessAreas: [] as string[],
    subBusinessAreas: [] as string[],
    sites: [] as string[],
    channels: ["Voice", "Back Office", "Outbound", "Blended", "Chat", "MessageUs"] as string[]
  });
  const [loading, setLoading] = useState(false);
  const [runComplete, setRunComplete] = useState(false);
  const [savedToDisk, setSavedToDisk] = useState(false);
  const [runTriggered, setRunTriggered] = useState(false);
  const [dupPromptOpen, setDupPromptOpen] = useState(false);
  const [dupPromptCount, setDupPromptCount] = useState(0);
  const [dupPromptPayload, setDupPromptPayload] = useState<Record<string, any> | null>(null);
  const [autoLoadedOriginal, setAutoLoadedOriginal] = useState(false);

  const loadPushOptions = async (params?: { ba?: string }) => {
    const query = new URLSearchParams();
    if (params?.ba) query.set("ba", params.ba);
    try {
      const res = await apiGet<{
        business_areas?: string[];
        sub_business_areas?: string[];
        sites?: string[];
        channels?: string[];
      }>(`/api/forecast/headcount/options${query.toString() ? `?${query.toString()}` : ""}`);
      setPushOptions((prev) => ({
        businessAreas: res.business_areas ?? prev.businessAreas,
        subBusinessAreas: res.sub_business_areas ?? (params?.ba ? [] : prev.subBusinessAreas),
        sites: res.sites ?? (params?.ba ? [] : prev.sites),
        channels: res.channels ?? prev.channels
      }));
    } catch {
      return;
    }
  };

  useEffect(() => {
    const store = loadForecastStore();
    if (store.transformations) setTransformResult(store.transformations);
    if (store.volumeSummary) setVolumeSummary(store.volumeSummary);
    if (store.dailyInterval) {
      setDailyResult(store.dailyInterval);
      setAnalysis(store.dailyInterval?.results?.analysis || null);
    }
    if (store.originalData) setOriginalData(store.originalData);
    if (store.intervalData) setIntervalData(store.intervalData);
    if (store.distributionOverride) setDistributionOverride(store.distributionOverride);
  }, []);

  useEffect(() => {
    void loadPushOptions();
  }, []);

  useEffect(() => {
    if (!pushBA.trim()) return;
    void loadPushOptions({ ba: pushBA });
  }, [pushBA]);

  useEffect(() => {
    const needsDefault = parseNumber(pushAhtSec) === null;
    if (!needsDefault) return;
    if (!pushBA.trim() || !pushSBA.trim() || !pushChannel.trim()) return;
    const params = new URLSearchParams();
    params.set("scope_type", "hier");
    params.set("ba", pushBA.trim());
    params.set("sba", pushSBA.trim());
    params.set("channel", pushChannel.trim());
    if (pushSite.trim()) params.set("site", pushSite.trim());
    apiGet<{ settings?: Record<string, any> }>(`/api/forecast/settings?${params.toString()}`)
      .then((res) => {
        const candidate = getDefaultAhtFromSettings(res?.settings ?? null, pushChannel);
        if (candidate !== null && parseNumber(pushAhtSec) === null) {
          setPushAhtSec(String(candidate));
        }
      })
      .catch(() => {});
  }, [pushBA, pushSBA, pushChannel, pushSite, pushAhtSec]);

  useEffect(() => {
    setPushBA((prev) => {
      if (prev || !pushOptions.businessAreas.length) return prev;
      return pushOptions.businessAreas[0] || prev;
    });
  }, [pushOptions.businessAreas]);

  useEffect(() => {
    setPushSBA((prev) => {
      if (prev || !pushOptions.subBusinessAreas.length) return prev;
      return pushOptions.subBusinessAreas[0] || prev;
    });
  }, [pushOptions.subBusinessAreas]);

  useEffect(() => {
    setPushSite((prev) => {
      if (prev || !pushOptions.sites.length) return prev;
      return pushOptions.sites[0] || prev;
    });
  }, [pushOptions.sites]);

  useEffect(() => {
    setPushChannel((prev) => {
      if (!pushOptions.channels.length) return prev;
      if (pushOptions.channels.includes(prev)) return prev;
      return pushOptions.channels[0] || prev;
    });
  }, [pushOptions.channels]);

  useEffect(() => {
    if (originalData.length || intervalData.length || volumeSummary?.normalized?.length) return;
    apiGet<{ rows?: Array<Record<string, any>> }>("/api/forecast/original-data")
      .then((res) => {
        const rows = res.rows || [];
        if (!rows.length) return;
        setOriginalData(rows);
        if (hasIntervalColumn(rows)) {
          setIntervalData(rows);
        }
        setAutoLoadedOriginal(true);
      })
      .catch(() => {
        setAutoLoadedOriginal(false);
      });
  }, [originalData.length, intervalData.length, volumeSummary?.normalized?.length]);

  const effectiveOriginalData = useMemo(() => {
    if (originalData.length) return originalData;
    if (volumeSummary?.normalized?.length) return volumeSummary.normalized;
    return [];
  }, [originalData, volumeSummary]);

  const effectiveIntervalData = useMemo(() => {
    if (intervalData.length) return intervalData;
    return hasIntervalColumn(effectiveOriginalData) ? effectiveOriginalData : [];
  }, [intervalData, effectiveOriginalData]);

  const hasOriginalSource = effectiveOriginalData.length > 0;
  const hasIntervalSource = effectiveIntervalData.length > 0;
  const originalSourceLabel = volumeSummary?.normalized?.length
    ? "Volume Summary upload"
    : autoLoadedOriginal
      ? "saved Volume Summary export"
      : originalData.length
        ? "cached original data"
        : "";

  useEffect(() => {
    if (dailyResult?.results?.analysis) {
      setAnalysis(dailyResult.results.analysis);
    }
  }, [dailyResult]);

  useEffect(() => {
    if (dailyResult?.results?.distribution?.length) {
      setDistributionOverride(dailyResult.results.distribution.map((row: any) => ({ ...row })));
    }
  }, [dailyResult]);

  useEffect(() => {
    if (dailyResult?.results?.daily?.length || dailyResult?.results?.interval?.length) {
      setRunComplete(true);
      return;
    }
    if (!dailyResult) setRunComplete(false);
  }, [dailyResult]);

  useEffect(() => {
    if (distributionOverride.length) {
      saveForecastStore({ distributionOverride });
    }
  }, [distributionOverride]);

  useEffect(() => {
    setGlobalLoading(loading);
    return () => setGlobalLoading(false);
  }, [loading, setGlobalLoading]);

  const transformData = useMemo(() => {
    return (
      transformResult?.results?.final ||
      transformResult?.results?.processed ||
      []
    );
  }, [transformResult]);

  const groupOptions = useMemo<string[]>(() => {
    if (!transformData.length) return [];
    const cols =
      groupLevel === "business_area"
        ? ["business_area", "Business_Area"]
        : ["forecast_group", "Forecast_Group", "category", "Category"];
    const col = cols.find((c) => transformData[0]?.[c] !== undefined);
    if (!col) return [];
    return Array.from(
      new Set(
        transformData
          .map((row: Record<string, any>) => String(row[col] ?? "").trim())
          .filter(Boolean)
      )
    );
  }, [transformData, groupLevel]);

  const modelOptions = useMemo<string[]>(() => {
    if (!transformData.length) return [];
    return Array.from(
      new Set(
        transformData
          .map((row: Record<string, any>) => String(row.Model ?? "").trim())
          .filter(Boolean)
      )
    );
  }, [transformData]);

  const yearOptions = useMemo<string[]>(() => {
    if (!transformData.length) return [];
    return Array.from(
      new Set(
        transformData
          .map((row: Record<string, any>) => String(row.Year ?? "").trim())
          .filter(Boolean)
      )
    );
  }, [transformData]);

  const monthOptions = useMemo<string[]>(() => {
    if (!transformData.length) return [];
    return Array.from(
      new Set(
        transformData
          .map((row: Record<string, any>) => String(row.Month ?? "").trim())
          .filter(Boolean)
      )
    );
  }, [transformData]);

  const distributionTotal = useMemo(() => {
    return distributionOverride.reduce((sum, row) => {
      const val = parseNumber(row.Distribution_Pct ?? row.distribution_pct);
      return sum + (val ?? 0);
    }, 0);
  }, [distributionOverride]);

  useEffect(() => {
    if (!selectedYear && yearOptions.length) setSelectedYear(yearOptions[0]);
    if (!selectedMonth && monthOptions.length) setSelectedMonth(monthOptions[0]);
    if (!selectedGroup && groupOptions.length) setSelectedGroup(groupOptions[0]);
    if (!selectedModel && modelOptions.length) setSelectedModel(modelOptions[0]);
  }, [yearOptions, monthOptions, groupOptions, modelOptions, selectedYear, selectedMonth, selectedGroup, selectedModel]);

  useEffect(() => {
    if (groupOptions.length && !groupOptions.includes(selectedGroup)) {
      setSelectedGroup(groupOptions[0]);
    }
  }, [groupOptions, selectedGroup]);

  const executePushToPlan = async (payload: Record<string, any>, mode?: string) => {
    setLoading(true);
    try {
      const finalPayload = mode ? { ...payload, mode } : payload;
      const res = await apiPost<{ ok?: boolean; message?: string }>("/api/forecast/push-to-plan", finalPayload);
      if (res?.ok === false) {
        notify("warning", res?.message || "Push to plan failed.");
        return;
      }
      notify("success", res?.message || "Forecast pushed to plan.");
    } catch (err: any) {
      notify("error", err?.message || "Push to plan failed.");
    } finally {
      setLoading(false);
    }
  };

  const handlePushToPlan = async () => {
    if (!dailyResult?.results) {
      notify("warning", "Run the daily/interval forecast first.");
      return;
    }
    if (!pushBA.trim() || !pushSBA.trim() || !pushChannel.trim()) {
      notify("warning", "Select Business Area, Sub BA, and Channel before pushing.");
      return;
    }
    const { kind, isVoice, isBO, isChat, isOB } = resolveChannelKind(pushChannel);
    if (!kind) {
      notify("warning", "Unsupported channel for push.");
      return;
    }
    const granularity = (pushGranularity || "auto").trim().toLowerCase();
    const useDaily = granularity === "auto" ? isBO : granularity === "daily";
    if (isVoice && useDaily) {
      notify("warning", "Voice forecasts must be pushed at interval granularity.");
      return;
    }
    if (isBO && !useDaily) {
      notify("warning", "Back Office forecasts must be pushed at daily granularity.");
      return;
    }
    const sourceRows = useDaily ? dailyResult?.results?.daily : dailyResult?.results?.interval;
    if (!Array.isArray(sourceRows) || !sourceRows.length) {
      notify("warning", "No forecast rows found for the selected granularity.");
      return;
    }
    if (!useDaily && (isVoice || isChat || isOB) && !hasIntervalColumn(sourceRows)) {
      notify("warning", "No interval column found. Re-run the forecast to generate interval output.");
      return;
    }
    const ahtValue = parseNumber(pushAhtSec);
    if ((isVoice || isBO || isChat) && ahtValue === null) {
      notify("warning", "Enter Forecast AHT/SUT (seconds) before pushing.");
      return;
    }
    const payload: Record<string, any> = {
      results: dailyResult?.results ?? null,
      ba: pushBA.trim(),
      sba: pushSBA.trim(),
      channel: pushChannel.trim(),
      site: pushSite.trim(),
      granularity: pushGranularity
    };
    if (ahtValue !== null) payload.aht_sec = ahtValue;

    const scopeKey = [pushBA.trim(), pushSBA.trim(), pushChannel.trim(), pushSite.trim()]
      .filter(Boolean)
      .join("|");
    let previewKind = "";
    if (isVoice) previewKind = "voice_forecast_volume";
    else if (isBO) previewKind = "bo_forecast_volume";
    else if (isChat) previewKind = "chat_forecast_volume";
    else if (isOB) previewKind = "ob_forecast_calls";

    if (previewKind) {
      try {
        const preview = await apiPost<any>("/api/uploads/timeseries/preview", {
          kind: previewKind,
          scope_key: scopeKey || "global",
          rows: sourceRows
        });
        if (preview?.duplicates) {
          setDupPromptCount(Number(preview.duplicates) || 0);
          setDupPromptPayload(payload);
          setDupPromptOpen(true);
          return;
        }
      } catch {
        // If preview fails, fall back to pushing as-is.
      }
    }

    await executePushToPlan(payload);
  };

  const buildDailyInterval = async (useOverride: boolean) => {
    if (!transformData.length) {
      notify("warning", "Apply transformations before building daily intervals.");
      return;
    }
    const forecastMonth = buildForecastMonth(selectedYear, selectedMonth);
    if (!forecastMonth) {
      notify("warning", "Select a year and month first.");
      return;
    }
    const overridePayload = useOverride ? distributionOverride : null;
    if (useOverride && !overridePayload?.length) {
      notify("warning", "Build a distribution before applying edits.");
      return;
    }
    const originalPayload = effectiveOriginalData;
    if (!originalPayload.length && !useOverride) {
      notify("warning", "Original data not found. Run Volume Summary first.");
      return;
    }
    const intervalPayload = hasIntervalSource ? effectiveIntervalData : [];
    setLoading(true);
    try {
      const res = await apiPost<any>("/api/forecast/daily-interval", {
        transform_df: transformData,
        interval_df: intervalPayload,
        forecast_month: forecastMonth,
        group_value: selectedGroup || null,
        model_value: selectedModel || null,
        distribution_override: overridePayload,
        original_data: originalPayload,
        holidays: volumeSummary?.holidays || null,
        group_level: groupLevel
      });
      setDailyResult(res);
      setAnalysis(res?.results?.analysis || null);
      setRunComplete(true);
      setRunTriggered(true);
      setSavedToDisk(false);
      saveForecastStore({ dailyInterval: res });
      if (!res.results || !res.results.daily?.length) {
        notify("warning", res.status || "Daily interval returned no rows.");
      } else {
        notify("success", hasIntervalSource ? "Daily + interval forecast ready." : "Daily forecast ready.");
      }
    } catch (err: any) {
      setRunTriggered(false);
      notify("error", err?.message || "Daily interval forecast failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveToDisk = async () => {
    const dailyRows = dailyResult?.results?.daily || [];
    const intervalRows = dailyResult?.results?.interval || [];
    if (!dailyRows.length && !intervalRows.length) {
      notify("warning", "Run daily/interval forecast first.");
      return;
    }
    setLoading(true);
    try {
      const res = await apiPost<any>("/api/forecast/save/daily-interval", {
        results: dailyResult?.results || {},
        meta: dailyResult?.meta || {}
      });
      const path = res?.paths?.[0] ? `Saved to ${res.paths[0]}` : "Saved to disk.";
      notify("success", path);
      setSavedToDisk(true);
    } catch (err: any) {
      notify("error", err?.message || "Save to disk failed.");
    } finally {
      setLoading(false);
    }
  };

  const showResults = runTriggered && runComplete;

  return (
    <AppShell crumbs="CAP-CONNECT / Forecast / Daily & Interval">
      {dupPromptOpen ? (
        <div className="ws-modal-backdrop">
          <div className="ws-modal ws-modal-sm">
            <div className="ws-modal-header" style={{ background: "#2f3747", color: "white" }}>
              <h3>Duplicate Forecast Rows</h3>
              <button
                type="button"
                className="btn btn-light closeOptions"
                onClick={() => {
                  setDupPromptOpen(false);
                  setDupPromptPayload(null);
                }}
              >
                ✕
              </button>
            </div>
            <div className="ws-modal-body">
              <p>
                {dupPromptCount} row(s) already exist for this scope. Choose how to proceed.
              </p>
            </div>
            <div className="ws-modal-footer">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => {
                  if (!dupPromptPayload) return;
                  setDupPromptOpen(false);
                  void executePushToPlan(dupPromptPayload, "override");
                  setDupPromptPayload(null);
                }}
              >
                Override Duplicates
              </button>
              <button
                type="button"
                className="btn btn-light"
                onClick={() => {
                  if (!dupPromptPayload) return;
                  setDupPromptOpen(false);
                  void executePushToPlan(dupPromptPayload, "append");
                  setDupPromptPayload(null);
                }}
              >
                Append New Only
              </button>
              <button
                type="button"
                className="btn btn-light"
                onClick={() => {
                  setDupPromptOpen(false);
                  setDupPromptPayload(null);
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}
      <div className="forecast-page forecast-volume-page">
        <div className="forecast-page-header">
          <div>
            <h1 className="forecast-page-title">Daily Interval Forecast</h1>
            <p className="forecast-page-subtitle">Follow the steps to keep moving forward.</p>
          </div>
          <Link className="forecast-back-link" href="/forecast/transformation-projects">
            Back to models
          </Link>
        </div>

        <div className="forecast-stepper">
          <div className="forecast-step-pill">📊 Volume Summary</div>
          <span className="forecast-step-arrow">→</span>
          <div className="forecast-step-pill">⚙️ Transformation Projects</div>
          <span className="forecast-step-arrow">→</span>
          <div className="forecast-step-pill active">⏱️ Daily Interval Forecast</div>
        </div>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Source data</div>
          {hasOriginalSource ? (
            <>
              <div className="forecast-section-note">
                Using original data from the {originalSourceLabel || "Volume Summary upload"}.
              </div>
              <div className="forecast-section-note">
                {hasIntervalSource
                  ? "Interval columns detected. Daily + interval forecast is enabled."
                  : "Interval columns not detected. Daily-only forecast will run."}
              </div>
            </>
          ) : (
            <div className="forecast-section-note">
              Run Volume Summary first to load original data for this step.
            </div>
          )}
        </section>

        {!transformData.length ? (
          <section className="forecast-section-block">
            <div className="forecast-section-title">Next step</div>
            <div className="forecast-section-note">
              Load a transformed forecast from the previous step to unlock forecasting options.
            </div>
          </section>
        ) : null}

        {transformData.length ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Forecast dates</div>
          <div className="forecast-form-row">
            <div>
              <div className="label">Select year</div>
              <input
                className="input"
                type="text"
                list="interval-years"
                value={selectedYear}
                onChange={(event) => setSelectedYear(event.target.value)}
                placeholder="Select year"
              />
              <datalist id="interval-years">
                {yearOptions.map((option) => (
                  <option key={option} value={option} />
                ))}
              </datalist>
            </div>
            <div>
              <div className="label">Select month</div>
              <input
                className="input"
                type="text"
                list="interval-months"
                value={selectedMonth}
                onChange={(event) => setSelectedMonth(event.target.value)}
                placeholder="Select month"
              />
              <datalist id="interval-months">
                {monthOptions.map((option) => (
                  <option key={option} value={option} />
                ))}
              </datalist>
            </div>
          </div>
        </section>
        ) : null}

        {transformData.length ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Use transformed forecast</div>
          <div className="forecast-form-row">
            <div>
              <div className="label">Transformed run</div>
              <input
                className="input"
                type="text"
                value={transformData.length ? "Loaded" : "Not loaded"}
                readOnly
              />
            </div>
            <div>
              <div className="label">Group level</div>
              <select
                className="select"
                value={groupLevel}
                onChange={(event) => setGroupLevel(event.target.value)}
              >
                <option value="forecast_group">Forecast Group</option>
                <option value="business_area">Business Area</option>
              </select>
            </div>
            <div>
              <div className="label">Forecast group</div>
              <input
                className="input"
                type="text"
                list="interval-groups"
                value={selectedGroup}
                onChange={(event) => setSelectedGroup(event.target.value)}
                placeholder="Select forecast group"
              />
              <datalist id="interval-groups">
                {groupOptions.map((option) => (
                  <option key={option} value={option} />
                ))}
              </datalist>
            </div>
            <div>
              <div className="label">Model</div>
              <input
                className="input"
                type="text"
                list="interval-models"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                placeholder="Select model"
              />
              <datalist id="interval-models">
                {modelOptions.map((option) => (
                  <option key={option} value={option} />
                ))}
              </datalist>
            </div>
          </div>
          <div className="forecast-actions-row">
            <button className="btn btn-primary" type="button" disabled={loading} onClick={() => buildDailyInterval(false)}>
              {hasIntervalSource ? "Build daily + interval" : "Build daily forecast"}
            </button>
            {showResults ? (
              <>
                <button className="btn btn-outline" type="button" disabled={loading} onClick={() => buildDailyInterval(true)}>
                  Apply distribution edits
                </button>
                <button
                  className="btn btn-outline"
                  type="button"
                  onClick={() =>
                    dailyResult?.results?.daily
                      ? downloadCsv("daily-forecast.csv", dailyResult.results.daily)
                      : notify("warning", "Run daily forecast first.")
                  }
                >
                  Download daily
                </button>
                {hasIntervalSource || dailyResult?.results?.interval?.length ? (
                  <button
                    className="btn btn-outline"
                    type="button"
                    onClick={() =>
                      dailyResult?.results?.interval
                        ? downloadCsv("interval-forecast.csv", dailyResult.results.interval)
                        : notify("warning", "Run interval forecast first.")
                    }
                  >
                    Download interval
                  </button>
                ) : null}
                <button className="btn btn-outline" type="button" disabled={loading} onClick={handleSaveToDisk}>
                  Save to disk
                </button>
              </>
            ) : null}
          </div>
          {savedToDisk ? <div className="forecast-section-note">Saved. You can now push to plan.</div> : null}
        </section>
        ) : null}

        {effectiveOriginalData.length ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Original Data Loaded</div>
          <DataTable data={effectiveOriginalData} />
        </section>
        ) : null}

        {effectiveIntervalData.length ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Interval Data Loaded</div>
          <DataTable data={effectiveIntervalData} />
        </section>
        ) : null}

        {transformData.length ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Transformed Forecast Loaded</div>
          <DataTable data={transformData} />
        </section>
        ) : null}

        {showResults ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Edit Forecast Distribution</div>
          {distributionOverride.length ? (
            <>
              <EditableTable
                data={distributionOverride}
                editableColumns={["Distribution_Pct", "distribution_pct"]}
                onChange={(rows) => setDistributionOverride(rows)}
              />
              <div className="forecast-muted">Total: {distributionTotal.toFixed(1)}%</div>
            </>
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>
        ) : null}

        {showResults ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Final Daily Forecast</div>
          {dailyResult?.results?.daily?.length ? (
            <DataTable data={dailyResult.results.daily} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>
        ) : null}

        {showResults ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Final Interval Forecast</div>
          {dailyResult?.results?.interval?.length ? (
            <DataTable data={dailyResult.results.interval} />
          ) : hasIntervalSource ? (
            <div className="forecast-placeholder-bar" />
          ) : (
            <div className="forecast-section-note">Interval forecast skipped (daily-only data source).</div>
          )}
          {dailyResult?.interval_summary ? (
            <div className="forecast-muted">{dailyResult.interval_summary}</div>
          ) : null}
        </section>
        ) : null}

        {showResults && analysis?.monthly_forecast ? (
          <div className="forecast-muted">
            Monthly forecast value: {Number(analysis.monthly_forecast).toLocaleString()}
          </div>
        ) : null}
        {showResults && analysis?.holiday_info ? (
          <div className="forecast-section-note">{analysis.holiday_info}</div>
        ) : null}

        {showResults && analysis?.charts?.length
          ? analysis.charts.map((chart: any, idx: number) => (
              <section key={`chart-${idx}`} className="forecast-section-block">
                <div className="forecast-section-title">{chart.title}</div>
                {chart.chart ? <LineChart data={chart.chart} /> : <div className="forecast-chart-placeholder" />}
              </section>
            ))
          : null}

        {showResults && analysis?.tables?.length
          ? analysis.tables.map((table: any, idx: number) => (
              <section key={`table-${idx}`} className="forecast-section-block">
                <div className="forecast-section-title">{table.title}</div>
                {table.rows?.length ? <DataTable data={table.rows} /> : <div className="forecast-placeholder-bar" />}
              </section>
            ))
          : null}

        {showResults && analysis?.monthly_tables?.length
          ? analysis.monthly_tables.map((table: any, idx: number) => (
              <section key={`monthly-${idx}`} className="forecast-section-block">
                <div className="forecast-section-title">{table.title}</div>
                {table.rows?.length ? <DataTable data={table.rows} /> : <div className="forecast-placeholder-bar" />}
              </section>
            ))
          : null}

        {showResults ? (
          <section className="forecast-section-block">
            <div className="forecast-section-title">Push to capacity plan</div>
            <div className="forecast-form-row">
              <div>
                <div className="label">Business Area</div>
                <select className="select" value={pushBA} onChange={(event) => setPushBA(event.target.value)}>
                  <option value="">Select BA</option>
                  {pushOptions.businessAreas.map((ba) => (
                    <option key={ba} value={ba}>
                      {ba}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <div className="label">Sub BA</div>
                <select className="select" value={pushSBA} onChange={(event) => setPushSBA(event.target.value)}>
                  <option value="">Select Sub BA</option>
                  {pushOptions.subBusinessAreas.map((sba) => (
                    <option key={sba} value={sba}>
                      {sba}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <div className="label">Channel</div>
                <select
                  className="select"
                  value={pushChannel}
                  onChange={(event) => setPushChannel(event.target.value)}
                >
                  {pushOptions.channels.map((channel) => (
                    <option key={channel} value={channel}>
                      {channel}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <div className="label">Site</div>
                <select className="select" value={pushSite} onChange={(event) => setPushSite(event.target.value)}>
                  <option value="">Select Site</option>
                  {pushOptions.sites.map((site) => (
                    <option key={site} value={site}>
                      {site}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <div className="label">Granularity</div>
                <select
                  className="select"
                  value={pushGranularity}
                  onChange={(event) => setPushGranularity(event.target.value)}
                >
                  <option value="auto">Auto</option>
                  <option value="interval">Interval</option>
                  <option value="daily">Daily</option>
                </select>
              </div>
              <div>
                <div className="label">Forecast AHT/SUT (sec)</div>
                <input
                  className="input"
                  type="number"
                  min="0"
                  step="1"
                  value={pushAhtSec}
                  onChange={(event) => setPushAhtSec(event.target.value)}
                  placeholder="e.g. 300"
                />
              </div>
            </div>
            <div className="forecast-actions-row">
              <button className="btn btn-primary" type="button" onClick={handlePushToPlan}>
                Push forecast to plan
              </button>
            </div>
          </section>
        ) : null}

        <section className="forecast-section-block">
          <div className="forecast-section-title">Tips</div>
          <div className="forecast-section-note">Date column is auto-detected (Date/ds/timestamp).</div>
          <div className="forecast-section-note">Interval like 09:00-09:30 or 09:00 works.</div>
          <div className="forecast-section-note">Volume aliases: volume, calls, items, count.</div>
        </section>
      </div>
    </AppShell>
  );
}
