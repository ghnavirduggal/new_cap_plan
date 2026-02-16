"use client";

import { useEffect, useMemo, useState } from "react";
import DataTable from "../_components/DataTable";
import { useGlobalLoader } from "../_components/GlobalLoader";
import { useToast } from "../_components/ToastProvider";
import { apiGet, apiPost } from "../../lib/api";
import { parseExcelFile } from "../../lib/excel";

type UploadState = {
  rows: Record<string, any>[];
  message: string;
};

type UploadKey =
  | "voiceForecast"
  | "voiceActual"
  | "voiceTactical"
  | "boForecast"
  | "boActual"
  | "boTactical"
  | "chatForecast"
  | "chatActual"
  | "chatTactical"
  | "obForecast"
  | "obActual"
  | "obTactical";

type PendingUpload = {
  key: UploadKey;
  kind: string;
  rows: Record<string, any>[];
  scopeKey: string;
};

type ScopeMode = "global" | "location" | "hier";

type ScopeState = {
  location: string;
  businessArea: string;
  subBusinessArea: string;
  channel: string;
  site: string;
};

type HeadcountOptions = {
  businessAreas: string[];
  subBusinessAreas: string[];
  locations: string[];
  sites: string[];
  channels: string[];
};

type SettingsState = {
  intervalMinutes: number;
  hoursPerFte: number;
  shrinkagePct: number;
  targetSlPct: number;
  slSeconds: number;
  occupancyCapVoicePct: number;
  utilBoPct: number;
  utilObPct: number;
  chatShrinkPct: number;
  obShrinkPct: number;
  utilChatPct: number;
  chatConcurrency: number;
  boCapacityModel: "tat" | "erlang";
  boTatDays: number;
  boWorkdaysPerWeek: number;
  boHoursPerDay: number;
  boShrinkPct: number;
  nestingWeeks: number;
  sdaWeeks: number;
  nestingLoginPct: number[];
  nestingAhtPct: number[];
  sdaLoginPct: number[];
  sdaAhtPct: number[];
  throughputTrainPct: number;
  throughputNestPct: number;
};

const EMPTY_UPLOADS: Record<UploadKey, UploadState> = {
  voiceForecast: { rows: [], message: "" },
  voiceActual: { rows: [], message: "" },
  voiceTactical: { rows: [], message: "" },
  boForecast: { rows: [], message: "" },
  boActual: { rows: [], message: "" },
  boTactical: { rows: [], message: "" },
  chatForecast: { rows: [], message: "" },
  chatActual: { rows: [], message: "" },
  chatTactical: { rows: [], message: "" },
  obForecast: { rows: [], message: "" },
  obActual: { rows: [], message: "" },
  obTactical: { rows: [], message: "" }
};

const DEFAULT_SCOPE: ScopeState = {
  location: "",
  businessArea: "",
  subBusinessArea: "",
  channel: "Voice",
  site: ""
};

const DEFAULT_SETTINGS: SettingsState = {
  intervalMinutes: 30,
  hoursPerFte: 8,
  shrinkagePct: 30,
  targetSlPct: 80,
  slSeconds: 20,
  occupancyCapVoicePct: 85,
  utilBoPct: 85,
  utilObPct: 85,
  chatShrinkPct: 30,
  obShrinkPct: 30,
  utilChatPct: 85,
  chatConcurrency: 1.5,
  boCapacityModel: "tat",
  boTatDays: 5,
  boWorkdaysPerWeek: 5,
  boHoursPerDay: 8,
  boShrinkPct: 30,
  nestingWeeks: 0,
  sdaWeeks: 0,
  nestingLoginPct: [],
  nestingAhtPct: [],
  sdaLoginPct: [],
  sdaAhtPct: [],
  throughputTrainPct: 100,
  throughputNestPct: 100
};

const FORECAST_TABS = [
  { key: "voice", label: "Voice" },
  { key: "bo", label: "Back Office" },
  { key: "chat", label: "Chat" },
  { key: "ob", label: "Outbound" }
] as const;

const TACTICAL_TABS = [
  { key: "voice", label: "Voice" },
  { key: "bo", label: "Back Office" },
  { key: "chat", label: "Chat" },
  { key: "ob", label: "Outbound" }
] as const;

function toNumberList(value: any): number[] {
  if (value === null || value === undefined) return [];
  if (Array.isArray(value)) {
    return value
      .map((entry) => Number(String(entry).replace("%", "").trim()))
      .filter((entry) => Number.isFinite(entry));
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((entry) => Number(entry.replace("%", "").trim()))
      .filter((entry) => Number.isFinite(entry));
  }
  const num = Number(value);
  return Number.isFinite(num) ? [num] : [];
}

function ensureLength(values: number[], length: number, fallback: number) {
  const next = [...values];
  while (next.length < length) next.push(fallback);
  return next.slice(0, length);
}

function toCsv(rows: Array<Record<string, any>>, columnsOverride?: string[]) {
  const columnsSet = rows.reduce<Set<string>>((set, row) => {
    Object.keys(row || {}).forEach((key) => set.add(key));
    return set;
  }, new Set<string>());
  const columns =
    columnsOverride && columnsOverride.length ? columnsOverride : Array.from(columnsSet);
  if (!columns.length) return "";
  const escape = (val: any) => {
    if (val === null || val === undefined) return "";
    const str = String(val).replace(/"/g, '""');
    return /[",\n]/.test(str) ? `"${str}"` : str;
  };
  const header = columns.join(",");
  const body = rows.map((row) => columns.map((col) => escape(row[col])).join(",")).join("\n");
  return body ? `${header}\n${body}` : `${header}\n`;
}

function downloadCsv(filename: string, rows: Array<Record<string, any>>, columnsOverride?: string[]) {
  const csv = toCsv(rows, columnsOverride);
  if (!csv) return;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

type SettingsClientProps = {
  initialScopeMode?: ScopeMode;
  initialScope?: Partial<ScopeState>;
  lockScope?: boolean;
  embedded?: boolean;
  onChanged?: () => void;
};

export default function SettingsClient({
  initialScopeMode,
  initialScope,
  lockScope,
  embedded,
  onChanged
}: SettingsClientProps) {
  const { setLoading } = useGlobalLoader();
  const { notify } = useToast();
  const [scopeMode, setScopeMode] = useState<ScopeMode>(initialScopeMode ?? "global");
  const [scope, setScope] = useState<ScopeState>({ ...DEFAULT_SCOPE, ...(initialScope || {}) });
  const [options, setOptions] = useState<HeadcountOptions>({
    businessAreas: [],
    subBusinessAreas: [],
    locations: [],
    sites: [],
    channels: ["Voice", "Back Office", "Outbound", "Blended", "Chat", "MessageUs"]
  });
  const [uploads, setUploads] = useState(EMPTY_UPLOADS);
  const [saveUploadModalOpen, setSaveUploadModalOpen] = useState(false);
  const [saveUploadPending, setSaveUploadPending] = useState<PendingUpload | null>(null);
  const [activeTab, setActiveTab] = useState<(typeof FORECAST_TABS)[number]["key"]>("voice");
  const [activeTacticalTab, setActiveTacticalTab] = useState<(typeof TACTICAL_TABS)[number]["key"]>("voice");
  const [settings, setSettings] = useState<SettingsState>(DEFAULT_SETTINGS);
  const [settingsMessage, setSettingsMessage] = useState("");
  const [holidaysRows, setHolidaysRows] = useState<Record<string, any>[]>([]);
  const [holidaysMessage, setHolidaysMessage] = useState("");
  const [headcountRows, setHeadcountRows] = useState<Record<string, any>[]>([]);
  const [headcountMessage, setHeadcountMessage] = useState("");
  const scopeLocked = Boolean(lockScope);

  useEffect(() => {
    if (!scopeLocked) return;
    if (initialScopeMode) {
      setScopeMode(initialScopeMode);
    }
  }, [initialScopeMode, scopeLocked]);

  useEffect(() => {
    if (!scopeLocked || !initialScope) return;
    setScope((prev) => ({ ...prev, ...initialScope }));
  }, [initialScope, scopeLocked]);

  const scopeKey = useMemo(() => {
    if (scopeMode === "location") {
      return `location|${scope.location.trim()}`.trim();
    }
    if (scopeMode === "hier") {
      return `${scope.businessArea.trim()}|${scope.subBusinessArea.trim()}|${scope.channel.trim()}|${
        scope.site.trim()
      }`;
    }
    return "global";
  }, [scopeMode, scope]);

  const scopePayload = useMemo(() => {
    return {
      scope_type: scopeMode,
      location: scope.location.trim() || null,
      business_area: scope.businessArea.trim() || null,
      sub_business_area: scope.subBusinessArea.trim() || null,
      channel: scope.channel.trim() || null,
      site: scope.site.trim() || null
    };
  }, [scopeMode, scope]);

  const notifySettingsUpdated = () => {
    onChanged?.();
    if (typeof window === "undefined") return;
    const event = new CustomEvent("settingsUpdated", {
      detail: { scope: scopePayload }
    });
    window.dispatchEvent(event);
  };

  useEffect(() => {
    setSettings((prev) => {
      return {
        ...prev,
        nestingLoginPct: ensureLength(prev.nestingLoginPct, prev.nestingWeeks, 100),
        nestingAhtPct: ensureLength(prev.nestingAhtPct, prev.nestingWeeks, 0),
        sdaLoginPct: ensureLength(prev.sdaLoginPct, prev.sdaWeeks, 100),
        sdaAhtPct: ensureLength(prev.sdaAhtPct, prev.sdaWeeks, 0)
      };
    });
  }, [settings.nestingWeeks, settings.sdaWeeks]);

  useEffect(() => {
    void loadHeadcountOptions();
  }, []);

  useEffect(() => {
    void loadHeadcountOptions({
      ba: scope.businessArea,
      sba: scope.subBusinessArea,
      location: scope.location
    });
  }, [scope.businessArea, scope.subBusinessArea, scope.location]);

  useEffect(() => {
    if (scopeMode === "location" && !scope.location.trim()) {
      return;
    }
    if (scopeMode === "hier" && (!scope.businessArea.trim() || !scope.subBusinessArea.trim())) {
      return;
    }
    void loadSettings();
    void loadHolidays();
  }, [scopeMode, scope.location, scope.businessArea, scope.subBusinessArea, scope.channel, scope.site]);

  useEffect(() => {
    setScope((prev) => {
      if (!options.channels.length) return prev;
      if (scopeLocked && prev.channel.trim()) return prev;
      if (options.channels.includes(prev.channel)) return prev;
      return { ...prev, channel: options.channels[0] };
    });
  }, [options.channels, scopeLocked]);

  const updateUpload = (key: UploadKey, patch: Partial<UploadState>) => {
    setUploads((prev) => ({
      ...prev,
      [key]: { ...prev[key], ...patch }
    }));
  };

  const handleFile = async (key: UploadKey, file?: File | null) => {
    if (!file) return;
    setLoading(true);
    try {
      const rows = await parseFile(file);
      updateUpload(key, {
        rows,
        message: rows.length ? `Loaded ${rows.length} rows from ${file.name}.` : "No rows detected."
      });
    } catch (error) {
      updateUpload(key, { rows: [], message: "Could not parse file." });
    } finally {
      setLoading(false);
    }
  };

  const persistTimeseries = async (
    key: UploadKey,
    kind: string,
    rows: Record<string, any>[],
    mode: "append" | "replace",
    scopeKeyOverride?: string,
    useLoader: boolean = true
  ) => {
    if (!rows.length) {
      updateUpload(key, { message: "No rows to save." });
      return;
    }
    if (!scopeKeyOverride && !validateScope()) return;
    const targetScope = (scopeKeyOverride ?? scopeKey).trim() || "global";
    if (useLoader) setLoading(true);
    try {
      await apiPost("/api/uploads/timeseries", {
        kind,
        scope_key: targetScope,
        mode,
        rows
      });
      updateUpload(key, {
        message: `Saved ${rows.length} rows for ${targetScope} (${mode}).`
      });
      notify("success", "Upload saved.");
      notifySettingsUpdated();
    } catch (error: any) {
      updateUpload(key, { message: "Save failed." });
      notify("error", error?.message || "Save failed.");
    } finally {
      if (useLoader) setLoading(false);
    }
  };

  const saveTimeseries = async (key: UploadKey, kind: string) => {
    const rows = uploads[key].rows;
    if (!rows.length) {
      updateUpload(key, { message: "No rows to save." });
      return;
    }
    if (!validateScope()) return;
    setSaveUploadPending({ key, kind, rows, scopeKey });
    setSaveUploadModalOpen(true);
  };

  const handleSaveUploadChoice = async (mode: "append" | "replace") => {
    const pending = saveUploadPending;
    if (!pending) {
      setSaveUploadModalOpen(false);
      return;
    }
    setSaveUploadModalOpen(false);
    setSaveUploadPending(null);
    await persistTimeseries(pending.key, pending.kind, pending.rows, mode, pending.scopeKey, true);
  };

  const saveUploadScopeLabel = saveUploadPending?.scopeKey?.trim() || "global";

  const validateScope = () => {
    if (scopeMode === "location" && !scope.location.trim()) {
      notify("warning", "Select a location to save in location scope.");
      return false;
    }
    if (scopeMode === "hier" && (!scope.businessArea.trim() || !scope.subBusinessArea.trim() || !scope.channel.trim())) {
      notify("warning", "Select Business Area, Sub Business Area, and Channel for hierarchical scope.");
      return false;
    }
    return true;
  };

  const loadHeadcountOptions = async (params?: { ba?: string; sba?: string; location?: string }) => {
    const query = new URLSearchParams();
    if (params?.ba) query.set("ba", params.ba);
    if (params?.sba) query.set("sba", params.sba);
    if (params?.location) query.set("location", params.location);
    try {
      const res = await apiGet<{ business_areas?: string[]; sub_business_areas?: string[]; locations?: string[]; sites?: string[]; channels?: string[] }>(
        `/api/forecast/headcount/options${query.toString() ? `?${query.toString()}` : ""}`
      );
      setOptions((prev) => ({
        businessAreas: res.business_areas ?? prev.businessAreas,
        subBusinessAreas: res.sub_business_areas ?? (params?.ba ? [] : prev.subBusinessAreas),
        locations: res.locations ?? prev.locations,
        sites: res.sites ?? (params?.ba ? [] : prev.sites),
        channels: res.channels ?? prev.channels
      }));
    } catch {
      return;
    }
  };

  const loadSettings = async () => {
    setLoading(true);
    try {
      const query = new URLSearchParams();
      query.set("scope_type", scopeMode);
      if (scopeMode === "location" && scope.location.trim()) {
        query.set("location", scope.location.trim());
      }
      if (scopeMode === "hier") {
        if (scope.businessArea.trim()) query.set("ba", scope.businessArea.trim());
        if (scope.subBusinessArea.trim()) query.set("sba", scope.subBusinessArea.trim());
        if (scope.channel.trim()) query.set("channel", scope.channel.trim());
        if (scope.site.trim()) query.set("site", scope.site.trim());
      }
      const res = await apiGet<{ settings?: Record<string, any> }>(`/api/forecast/settings?${query.toString()}`);
      applySettings(res.settings || {});
    } catch {
      applySettings({});
    } finally {
      setLoading(false);
    }
  };

  const applySettings = (data: Record<string, any>) => {
    const nestingWeeks = Number(data.nesting_weeks ?? data.default_nesting_weeks ?? DEFAULT_SETTINGS.nestingWeeks) || 0;
    const sdaWeeks = Number(data.sda_weeks ?? data.default_sda_weeks ?? DEFAULT_SETTINGS.sdaWeeks) || 0;
    const nestingLogin = ensureLength(toNumberList(data.nesting_productivity_pct), nestingWeeks, 100);
    const nestingAht = ensureLength(toNumberList(data.nesting_aht_uplift_pct), nestingWeeks, 0);
    const sdaLogin = ensureLength(toNumberList(data.sda_productivity_pct), sdaWeeks, 100);
    const sdaAht = ensureLength(toNumberList(data.sda_aht_uplift_pct), sdaWeeks, 0);

    setSettings({
      intervalMinutes: Number(data.interval_minutes ?? DEFAULT_SETTINGS.intervalMinutes),
      hoursPerFte: Number(data.hours_per_fte ?? DEFAULT_SETTINGS.hoursPerFte),
      shrinkagePct: Number((data.shrinkage_pct ?? DEFAULT_SETTINGS.shrinkagePct / 100) * 100),
      targetSlPct: Number((data.target_sl ?? DEFAULT_SETTINGS.targetSlPct / 100) * 100),
      slSeconds: Number(data.sl_seconds ?? DEFAULT_SETTINGS.slSeconds),
      occupancyCapVoicePct: Number((data.occupancy_cap_voice ?? DEFAULT_SETTINGS.occupancyCapVoicePct / 100) * 100),
      utilBoPct: Number((data.util_bo ?? DEFAULT_SETTINGS.utilBoPct / 100) * 100),
      utilObPct: Number((data.util_ob ?? DEFAULT_SETTINGS.utilObPct / 100) * 100),
      chatShrinkPct: Number((data.chat_shrinkage_pct ?? DEFAULT_SETTINGS.chatShrinkPct / 100) * 100),
      obShrinkPct: Number((data.ob_shrinkage_pct ?? DEFAULT_SETTINGS.obShrinkPct / 100) * 100),
      utilChatPct: Number((data.util_chat ?? DEFAULT_SETTINGS.utilChatPct / 100) * 100),
      chatConcurrency: Number(data.chat_concurrency ?? DEFAULT_SETTINGS.chatConcurrency),
      boCapacityModel: (data.bo_capacity_model || DEFAULT_SETTINGS.boCapacityModel).toLowerCase() === "erlang" ? "erlang" : "tat",
      boTatDays: Number(data.bo_tat_days ?? DEFAULT_SETTINGS.boTatDays),
      boWorkdaysPerWeek: Number(data.bo_workdays_per_week ?? DEFAULT_SETTINGS.boWorkdaysPerWeek),
      boHoursPerDay: Number(data.bo_hours_per_day ?? DEFAULT_SETTINGS.boHoursPerDay),
      boShrinkPct: Number((data.bo_shrinkage_pct ?? DEFAULT_SETTINGS.boShrinkPct / 100) * 100),
      nestingWeeks,
      sdaWeeks,
      nestingLoginPct: nestingLogin,
      nestingAhtPct: nestingAht,
      sdaLoginPct: sdaLogin,
      sdaAhtPct: sdaAht,
      throughputTrainPct: Number(data.throughput_train_pct ?? DEFAULT_SETTINGS.throughputTrainPct),
      throughputNestPct: Number(data.throughput_nest_pct ?? DEFAULT_SETTINGS.throughputNestPct)
    });
  };

  const saveSettings = async () => {
    if (!validateScope()) return;
    const payload = {
      interval_minutes: settings.intervalMinutes,
      hours_per_fte: settings.hoursPerFte,
      shrinkage_pct: settings.shrinkagePct / 100,
      target_sl: settings.targetSlPct / 100,
      sl_seconds: settings.slSeconds,
      occupancy_cap_voice: settings.occupancyCapVoicePct / 100,
      util_bo: settings.utilBoPct / 100,
      util_ob: settings.utilObPct / 100,
      chat_shrinkage_pct: settings.chatShrinkPct / 100,
      ob_shrinkage_pct: settings.obShrinkPct / 100,
      util_chat: settings.utilChatPct / 100,
      chat_concurrency: settings.chatConcurrency,
      bo_capacity_model: settings.boCapacityModel,
      bo_tat_days: settings.boTatDays,
      bo_workdays_per_week: settings.boWorkdaysPerWeek,
      bo_hours_per_day: settings.boHoursPerDay,
      bo_shrinkage_pct: settings.boShrinkPct / 100,
      nesting_weeks: settings.nestingWeeks,
      sda_weeks: settings.sdaWeeks,
      nesting_productivity_pct: settings.nestingLoginPct,
      nesting_aht_uplift_pct: settings.nestingAhtPct,
      sda_productivity_pct: settings.sdaLoginPct,
      sda_aht_uplift_pct: settings.sdaAhtPct,
      throughput_train_pct: settings.throughputTrainPct,
      throughput_nest_pct: settings.throughputNestPct
    };

    const now = new Date();
    const day = now.getDay();
    const monday = new Date(now);
    const delta = (day + 6) % 7;
    monday.setDate(now.getDate() - delta);
    const effectiveDate = monday.toISOString().slice(0, 10);
    setLoading(true);
    try {
      await apiPost("/api/forecast/settings", {
        scope: scopePayload,
        settings: payload,
        effective_date: effectiveDate
      });
      setSettingsMessage("Settings saved.");
      notify("success", "Settings saved.");
      notifySettingsUpdated();
    } catch (error: any) {
      setSettingsMessage("Save failed.");
      notify("error", error?.message || "Save failed.");
    } finally {
      setLoading(false);
    }
  };

  const loadHolidays = async () => {
    setLoading(true);
    try {
      const query = new URLSearchParams();
      query.set("scope_type", scopeMode);
      if (scopeMode === "location" && scope.location.trim()) {
        query.set("location", scope.location.trim());
      }
      if (scopeMode === "hier") {
        if (scope.businessArea.trim()) query.set("ba", scope.businessArea.trim());
        if (scope.subBusinessArea.trim()) query.set("sba", scope.subBusinessArea.trim());
        if (scope.channel.trim()) query.set("channel", scope.channel.trim());
        if (scope.site.trim()) query.set("site", scope.site.trim());
      }
      const res = await apiGet<{ rows?: Record<string, any>[] }>(`/api/forecast/holidays?${query.toString()}`);
      setHolidaysRows(res.rows ?? []);
      setHolidaysMessage(res.rows?.length ? "Saved holidays loaded." : "");
    } catch {
      setHolidaysRows([]);
      setHolidaysMessage("");
    } finally {
      setLoading(false);
    }
  };

  const handleHolidayUpload = async (file?: File | null) => {
    if (!file) return;
    setLoading(true);
    try {
      const rows = await parseFile(file);
      setHolidaysRows(rows);
      setHolidaysMessage(rows.length ? `Loaded ${rows.length} rows from ${file.name}.` : "No rows detected.");
    } catch {
      setHolidaysRows([]);
      setHolidaysMessage("Could not parse file.");
    } finally {
      setLoading(false);
    }
  };

  const saveHolidays = async () => {
    if (!validateScope()) return;
    setLoading(true);
    try {
      const res = await apiPost<{ rows?: Record<string, any>[] }>("/api/forecast/holidays", {
        scope: scopePayload,
        rows: holidaysRows
      });
      setHolidaysRows(res.rows ?? []);
      setHolidaysMessage("Holidays saved.");
      notify("success", "Holidays saved.");
      notifySettingsUpdated();
    } catch (error: any) {
      setHolidaysMessage("Save failed.");
      notify("error", error?.message || "Save failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleHeadcountUpload = async (file?: File | null) => {
    if (!file) return;
    setLoading(true);
    try {
      const rows = await parseFile(file);
      setHeadcountRows(rows);
      setHeadcountMessage(rows.length ? `Loaded ${rows.length} rows from ${file.name}.` : "No rows detected.");
    } catch (error: any) {
      setHeadcountRows([]);
      const detail = error?.message ? ` (${error.message})` : "";
      setHeadcountMessage(`Could not parse file${detail}.`);
    } finally {
      setLoading(false);
    }
  };

  const saveHeadcount = async () => {
    setLoading(true);
    try {
      const res = await apiPost<{ rows?: Record<string, any>[] }>("/api/forecast/headcount", {
        scope: scopePayload,
        rows: headcountRows
      });
      setHeadcountRows(res.rows ?? []);
      setHeadcountMessage("Headcount saved.");
      notify("success", "Headcount updated.");
      notifySettingsUpdated();
      void loadHeadcountOptions({
        ba: scope.businessArea,
        sba: scope.subBusinessArea,
        location: scope.location
      });
    } catch (error: any) {
      setHeadcountMessage("Save failed.");
      notify("error", error?.message || "Save failed.");
    } finally {
      setLoading(false);
    }
  };

  const scopeNote = useMemo(() => {
    if (scopeMode === "location") {
      return scope.location.trim()
        ? `Editing settings for location: ${scope.location.trim()}.`
        : "Select a location to edit location-level settings.";
    }
    if (scopeMode === "hier") {
      const parts = [scope.businessArea, scope.subBusinessArea, scope.channel, scope.site].filter(Boolean);
      return parts.length
        ? `Editing settings for: ${parts.join(" > ")}.`
        : "Select Business Area, Sub BA, Channel, and Site for hierarchical settings.";
    }
    return "Editing settings for Global scope.";
  }, [scopeMode, scope]);

  const headerNote = scopeLocked
    ? "Uploads are saved to the plan scope below."
    : "Uploads are saved to the selected scope. Choose scope before uploading.";

  return (
    <div className={embedded ? "settings-embedded" : undefined}>
      <section className="section gauravi">
        <h2>{scopeLocked ? "Plan Scope (Locked)" : "Default Settings - Scope"}</h2>
        <div className="grid grid-2">
          <label className="label">
            <input
              type="radio"
              name="scope"
              checked={scopeMode === "global"}
              onChange={() => setScopeMode("global")}
              disabled={scopeLocked}
            />{" "}
            Global
          </label>
          <label className="label">
            <input
              type="radio"
              name="scope"
              checked={scopeMode === "location"}
              onChange={() => setScopeMode("location")}
              disabled={scopeLocked}
            />{" "}
            Location (Country)
          </label>
          <label className="label">
            <input
              type="radio"
              name="scope"
              checked={scopeMode === "hier"}
              onChange={() => setScopeMode("hier")}
              disabled={scopeLocked}
            />{" "}
            Business Area ▶️ Sub Business Area ▶️ Channel ▶️ Site
          </label>
        </div>

        {scopeMode === "location" ? (
          <div className="grid grid-2" style={{ marginTop: 12 }}>
            <div>
              <div className="label">Location / Country</div>
              <select
                className="select"
                value={scope.location}
                onChange={(event) => setScope({ ...scope, location: event.target.value })}
                disabled={scopeLocked}
              >
                <option value="">Select Country (from headcount)</option>
                {options.locations.map((loc) => (
                  <option key={loc} value={loc}>
                    {loc}
                  </option>
                ))}
              </select>
            </div>
            <div className="forecast-muted">Location options load from the latest headcount file.</div>
          </div>
        ) : null}

        {scopeMode === "hier" ? (
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", marginTop: 12 }}>
            <div>
              <div className="label">Business Area</div>
              <select
                className="select"
                value={scope.businessArea}
                onChange={(event) =>
                  setScope({
                    ...scope,
                    businessArea: event.target.value,
                    subBusinessArea: "",
                    site: ""
                  })
                }
                disabled={scopeLocked}
              >
                <option value="">Business Area</option>
                {options.businessAreas.map((ba) => (
                  <option key={ba} value={ba}>
                    {ba}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="label">Sub Business Area</div>
              <select
                className="select"
                value={scope.subBusinessArea}
                onChange={(event) => setScope({ ...scope, subBusinessArea: event.target.value })}
                disabled={scopeLocked}
              >
                <option value="">Sub Business Area</option>
                {options.subBusinessAreas.map((sba) => (
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
                value={scope.channel}
                onChange={(event) => setScope({ ...scope, channel: event.target.value })}
                disabled={scopeLocked}
              >
                {options.channels.map((channel) => (
                  <option key={channel} value={channel}>
                    {channel}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="label">Site (Building)</div>
              <select
                className="select"
                value={scope.site}
                onChange={(event) => setScope({ ...scope, site: event.target.value })}
                disabled={scopeLocked}
              >
                <option value="">Site</option>
                {options.sites.map((site) => (
                  <option key={site} value={site}>
                    {site}
                  </option>
                ))}
              </select>
            </div>
          </div>
        ) : null}

        <div className="forecast-muted" style={{ marginTop: 12 }}>
          {scopeNote}
        </div>
      </section>

      <section className="section gauravi">
        <h2>Parameters</h2>
        <div className="grid grid-2">
          <div>
            <div className="label">Interval Minutes (Voice)</div>
            <input
              className="input"
              type="number"
              min={5}
              max={120}
              step={5}
              value={settings.intervalMinutes}
              onChange={(event) =>
                setSettings({ ...settings, intervalMinutes: Number(event.target.value) || 0 })
              }
            />

            <div className="label" style={{ marginTop: 12 }}>
              Work Hours per FTE / Day
            </div>
            <input
              className="input"
              type="number"
              min={1}
              max={12}
              step={0.25}
              value={settings.hoursPerFte}
              onChange={(event) => setSettings({ ...settings, hoursPerFte: Number(event.target.value) || 0 })}
            />

            <div className="label" style={{ marginTop: 12 }}>
              Shrinkage % (0-100)
            </div>
            <input
              className="input"
              type="number"
              min={0}
              max={100}
              step={0.5}
              value={settings.shrinkagePct}
              onChange={(event) => setSettings({ ...settings, shrinkagePct: Number(event.target.value) || 0 })}
            />
          </div>

          <div>
            <div className="label">Service Level % (0-100)</div>
            <input
              className="input"
              type="number"
              min={50}
              max={99}
              step={1}
              value={settings.targetSlPct}
              onChange={(event) => setSettings({ ...settings, targetSlPct: Number(event.target.value) || 0 })}
            />

            <div className="label" style={{ marginTop: 12 }}>
              Service Level T (seconds)
            </div>
            <input
              className="input"
              type="number"
              min={1}
              max={120}
              step={1}
              value={settings.slSeconds}
              onChange={(event) => setSettings({ ...settings, slSeconds: Number(event.target.value) || 0 })}
            />

            <div className="label" style={{ marginTop: 12 }}>
              Max Occupancy % (Voice)
            </div>
            <input
              className="input"
              type="number"
              min={60}
              max={100}
              step={1}
              value={settings.occupancyCapVoicePct}
              onChange={(event) =>
                setSettings({ ...settings, occupancyCapVoicePct: Number(event.target.value) || 0 })
              }
            />
          </div>
        </div>

        <div style={{ marginTop: 18 }} className="forecast-section-title">
          Back Office (TAT / Capacity)
        </div>
        <div className="grid grid-2" style={{ marginTop: 8 }}>
          <div>
            <div className="label">Capacity Model</div>
            <select
              className="select"
              value={settings.boCapacityModel}
              onChange={(event) =>
                setSettings({
                  ...settings,
                  boCapacityModel: event.target.value === "erlang" ? "erlang" : "tat"
                })
              }
            >
              <option value="tat">TAT (Within X days)</option>
              <option value="erlang">Erlang (queueing)</option>
            </select>
          </div>
          <div>
            <div className="label">TAT (days)</div>
            <input
              className="input"
              type="number"
              min={1}
              max={30}
              step={1}
              value={settings.boTatDays}
              onChange={(event) => setSettings({ ...settings, boTatDays: Number(event.target.value) || 0 })}
            />
          </div>
          <div>
            <div className="label">Workdays per Week</div>
            <input
              className="input"
              type="number"
              min={1}
              max={7}
              step={1}
              value={settings.boWorkdaysPerWeek}
              onChange={(event) =>
                setSettings({ ...settings, boWorkdaysPerWeek: Number(event.target.value) || 0 })
              }
            />
          </div>
          <div>
            <div className="label">Work Hours / Day (BO)</div>
            <input
              className="input"
              type="number"
              min={1}
              max={12}
              step={0.25}
              value={settings.boHoursPerDay}
              onChange={(event) => setSettings({ ...settings, boHoursPerDay: Number(event.target.value) || 0 })}
            />
          </div>
        </div>

        <div style={{ marginTop: 18 }} className="forecast-section-title">
          Productivity Targets
        </div>
        <div className="grid grid-2" style={{ marginTop: 8 }}>
          <div>
            <div className="label">Utilization % (Back Office)</div>
            <input
              className="input"
              type="number"
              min={50}
              max={100}
              step={1}
              value={settings.utilBoPct}
              onChange={(event) => setSettings({ ...settings, utilBoPct: Number(event.target.value) || 0 })}
            />
          </div>
          <div>
            <div className="label">Utilization % (Outbound)</div>
            <input
              className="input"
              type="number"
              min={50}
              max={100}
              step={1}
              value={settings.utilObPct}
              onChange={(event) => setSettings({ ...settings, utilObPct: Number(event.target.value) || 0 })}
            />
          </div>
          <div>
            <div className="label">Chat Concurrency</div>
            <input
              className="input"
              type="number"
              min={1}
              max={10}
              step={0.1}
              value={settings.chatConcurrency}
              onChange={(event) => setSettings({ ...settings, chatConcurrency: Number(event.target.value) || 0 })}
            />
          </div>
          <div>
            <div className="label">Utilization % (Chat)</div>
            <input
              className="input"
              type="number"
              min={50}
              max={100}
              step={1}
              value={settings.utilChatPct}
              onChange={(event) => setSettings({ ...settings, utilChatPct: Number(event.target.value) || 0 })}
            />
          </div>
        </div>

        <div style={{ marginTop: 18 }} className="forecast-section-title">
          Shrinkage Targets
        </div>
        <div className="grid grid-2" style={{ marginTop: 8 }}>
          <div>
            <div className="label">BO Shrinkage % (0-100)</div>
            <input
              className="input"
              type="number"
              min={0}
              max={100}
              step={0.5}
              value={settings.boShrinkPct}
              onChange={(event) => setSettings({ ...settings, boShrinkPct: Number(event.target.value) || 0 })}
            />
          </div>
          <div>
            <div className="label">Shrinkage % (Chat)</div>
            <input
              className="input"
              type="number"
              min={0}
              max={100}
              step={0.5}
              value={settings.chatShrinkPct}
              onChange={(event) => setSettings({ ...settings, chatShrinkPct: Number(event.target.value) || 0 })}
            />
          </div>
          <div>
            <div className="label">Shrinkage % (Outbound)</div>
            <input
              className="input"
              type="number"
              min={0}
              max={100}
              step={0.5}
              value={settings.obShrinkPct}
              onChange={(event) => setSettings({ ...settings, obShrinkPct: Number(event.target.value) || 0 })}
            />
          </div>
        </div>

        <div style={{ marginTop: 18 }} className="forecast-section-title">
          Learning Curve & SDA
        </div>
        <div className="grid grid-2" style={{ marginTop: 8 }}>
          <div>
            <div className="label">Nesting (OJT) Weeks</div>
            <input
              className="input"
              type="number"
              min={0}
              step={1}
              value={settings.nestingWeeks}
              onChange={(event) => setSettings({ ...settings, nestingWeeks: Number(event.target.value) || 0 })}
            />
            {settings.nestingLoginPct.map((val, idx) => (
              <div key={`nest-login-${idx}`} style={{ marginTop: 8 }}>
                <div className="label">Week {idx + 1} Login %</div>
                <input
                  className="input"
                  type="number"
                  step={1}
                  value={val}
                  onChange={(event) => {
                    const next = [...settings.nestingLoginPct];
                    next[idx] = Number(event.target.value) || 0;
                    setSettings({ ...settings, nestingLoginPct: next });
                  }}
                />
              </div>
            ))}
            {settings.nestingAhtPct.map((val, idx) => (
              <div key={`nest-aht-${idx}`} style={{ marginTop: 8 }}>
                <div className="label">Week {idx + 1} AHT Uplift %</div>
                <input
                  className="input"
                  type="number"
                  step={1}
                  value={val}
                  onChange={(event) => {
                    const next = [...settings.nestingAhtPct];
                    next[idx] = Number(event.target.value) || 0;
                    setSettings({ ...settings, nestingAhtPct: next });
                  }}
                />
              </div>
            ))}
          </div>
          <div>
            <div className="label">SDA (Phase 2) Weeks</div>
            <input
              className="input"
              type="number"
              min={0}
              step={1}
              value={settings.sdaWeeks}
              onChange={(event) => setSettings({ ...settings, sdaWeeks: Number(event.target.value) || 0 })}
            />
            {settings.sdaLoginPct.map((val, idx) => (
              <div key={`sda-login-${idx}`} style={{ marginTop: 8 }}>
                <div className="label">Week {idx + 1} Login %</div>
                <input
                  className="input"
                  type="number"
                  step={1}
                  value={val}
                  onChange={(event) => {
                    const next = [...settings.sdaLoginPct];
                    next[idx] = Number(event.target.value) || 0;
                    setSettings({ ...settings, sdaLoginPct: next });
                  }}
                />
              </div>
            ))}
            {settings.sdaAhtPct.map((val, idx) => (
              <div key={`sda-aht-${idx}`} style={{ marginTop: 8 }}>
                <div className="label">Week {idx + 1} AHT Uplift %</div>
                <input
                  className="input"
                  type="number"
                  step={1}
                  value={val}
                  onChange={(event) => {
                    const next = [...settings.sdaAhtPct];
                    next[idx] = Number(event.target.value) || 0;
                    setSettings({ ...settings, sdaAhtPct: next });
                  }}
                />
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-2" style={{ marginTop: 12 }}>
          <div>
            <div className="label">Throughput Train %</div>
            <input
              className="input"
              type="number"
              step={1}
              value={settings.throughputTrainPct}
              onChange={(event) =>
                setSettings({ ...settings, throughputTrainPct: Number(event.target.value) || 0 })
              }
            />
          </div>
          <div>
            <div className="label">Throughput Nest %</div>
            <input
              className="input"
              type="number"
              step={1}
              value={settings.throughputNestPct}
              onChange={(event) =>
                setSettings({ ...settings, throughputNestPct: Number(event.target.value) || 0 })
              }
            />
          </div>
        </div>

        <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 12 }}>
          <button type="button" className="btn btn-primary" onClick={saveSettings}>
            Save Settings
          </button>
          {settingsMessage ? <span className="forecast-muted">{settingsMessage}</span> : null}
        </div>
      </section>

      <section className="section gauravi">
        <h2>Upload Volume & AHT/SUT (by scope)</h2>
        <div className="forecast-muted">Voice uses 30-min intervals; Back Office uses daily totals.</div>
        <div className="forecast-muted" style={{ marginTop: 6 }}>
          {headerNote}
        </div>
        <div className="tabs" style={{ marginTop: 12 }}>
          {FORECAST_TABS.map((tab) => (
            <div
              key={tab.key}
              className={`tab ${activeTab === tab.key ? "active" : ""}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </div>
          ))}
        </div>

        {activeTab === "voice" ? (
          <>
            <UploadCard
              title="Voice Forecast"
              state={uploads.voiceForecast}
              onFile={(file) => handleFile("voiceForecast", file)}
              onSave={() => saveTimeseries("voiceForecast", "voice_forecast")}
              onTemplate={() =>
                downloadCsv(
                  "voice_forecast_template.csv",
                  [
                    {
                      Date: new Date().toISOString().slice(0, 10),
                      Interval: "09:00",
                      "Forecast Volume": 120,
                      "Forecast AHT": 300,
                      "Business Area": "Retail Banking",
                      "Sub Business Area": "Cards",
                      Channel: "Voice"
                    }
                  ]
                )
              }
            />
            <UploadCard
              title="Voice Actual"
              state={uploads.voiceActual}
              onFile={(file) => handleFile("voiceActual", file)}
              onSave={() => saveTimeseries("voiceActual", "voice_actual")}
              onTemplate={() =>
                downloadCsv(
                  "voice_actual_template.csv",
                  [
                    {
                      Date: new Date().toISOString().slice(0, 10),
                      Interval: "09:00",
                      "Actual Volume": 115,
                      "Actual AHT": 310,
                      "Business Area": "Retail Banking",
                      "Sub Business Area": "Cards",
                      Channel: "Voice"
                    }
                  ]
                )
              }
            />
          </>
        ) : null}

        {activeTab === "bo" ? (
          <>
            <UploadCard
              title="Back Office Forecast"
              state={uploads.boForecast}
              onFile={(file) => handleFile("boForecast", file)}
              onSave={() => saveTimeseries("boForecast", "bo_forecast")}
              onTemplate={() =>
                downloadCsv(
                  "backoffice_forecast_template.csv",
                  [
                    {
                      Date: new Date().toISOString().slice(0, 10),
                      "Forecast Volume": 550,
                      "Forecast SUT": 600,
                      "Business Area": "Retail Banking",
                      "Sub Business Area": "Cards",
                      Channel: "Back Office"
                    }
                  ]
                )
              }
            />
            <UploadCard
              title="Back Office Actual"
              state={uploads.boActual}
              onFile={(file) => handleFile("boActual", file)}
              onSave={() => saveTimeseries("boActual", "bo_actual")}
              onTemplate={() =>
                downloadCsv(
                  "backoffice_actual_template.csv",
                  [
                    {
                      Date: new Date().toISOString().slice(0, 10),
                      "Actual Volume": 520,
                      "Actual SUT": 610,
                      "Business Area": "Retail Banking",
                      "Sub Business Area": "Cards",
                      Channel: "Back Office"
                    }
                  ]
                )
              }
            />
          </>
        ) : null}

        {activeTab === "chat" ? (
          <>
            <UploadCard
              title="Chat Forecast"
              state={uploads.chatForecast}
              onFile={(file) => handleFile("chatForecast", file)}
              onSave={() => saveTimeseries("chatForecast", "chat_forecast")}
              onTemplate={() =>
                downloadCsv(
                  "chat_template.csv",
                  [],
                  ["date", "items", "aht_sec"]
                )
              }
            />
            <UploadCard
              title="Chat Actual"
              state={uploads.chatActual}
              onFile={(file) => handleFile("chatActual", file)}
              onSave={() => saveTimeseries("chatActual", "chat_actual")}
              onTemplate={() =>
                downloadCsv(
                  "chat_actual_template.csv",
                  [],
                  ["date", "items", "aht_sec"]
                )
              }
            />
          </>
        ) : null}

        {activeTab === "ob" ? (
          <>
            <UploadCard
              title="Outbound Forecast"
              state={uploads.obForecast}
              onFile={(file) => handleFile("obForecast", file)}
              onSave={() => saveTimeseries("obForecast", "ob_forecast")}
              onTemplate={() =>
                downloadCsv(
                  "outbound_template.csv",
                  [],
                  ["date", "opc", "connect_rate", "rpc_rate", "aht_sec"]
                )
              }
            />
            <UploadCard
              title="Outbound Actual"
              state={uploads.obActual}
              onFile={(file) => handleFile("obActual", file)}
              onSave={() => saveTimeseries("obActual", "ob_actual")}
              onTemplate={() =>
                downloadCsv(
                  "outbound_actual_template.csv",
                  [],
                  ["date", "opc", "connect_rate", "rpc_rate", "aht_sec"]
                )
              }
            />
          </>
        ) : null}
      </section>

      <section className="section gauravi">
        <h2>Upload Tactical Volume (by scope)</h2>
        <div className="forecast-muted">Voice uses 30-min intervals; Back Office uses daily totals.</div>
        <div className="forecast-muted" style={{ marginTop: 6 }}>
          {headerNote}
        </div>
        <div className="tabs" style={{ marginTop: 12 }}>
          {TACTICAL_TABS.map((tab) => (
            <div
              key={tab.key}
              className={`tab ${activeTacticalTab === tab.key ? "active" : ""}`}
              onClick={() => setActiveTacticalTab(tab.key)}
            >
              {tab.label}
            </div>
          ))}
        </div>

        {activeTacticalTab === "voice" ? (
          <UploadCard
            title="Voice Tactical Forecast"
            state={uploads.voiceTactical}
            onFile={(file) => handleFile("voiceTactical", file)}
            onSave={() => saveTimeseries("voiceTactical", "voice_tactical")}
            onTemplate={() =>
              downloadCsv(
                "voice_tactical_template.csv",
                [
                  {
                    date: new Date().toISOString().slice(0, 10),
                    interval: "09:00",
                    volume: 120,
                    aht_sec: 360
                  }
                ]
              )
            }
          />
        ) : null}

        {activeTacticalTab === "bo" ? (
          <UploadCard
            title="Back Office Tactical Forecast"
            state={uploads.boTactical}
            onFile={(file) => handleFile("boTactical", file)}
            onSave={() => saveTimeseries("boTactical", "bo_tactical")}
            onTemplate={() =>
              downloadCsv(
                "bo_tactical_template.csv",
                [
                  {
                    date: new Date().toISOString().slice(0, 10),
                    items: 5000,
                    sut_sec: 540
                  }
                ]
              )
            }
          />
        ) : null}

        {activeTacticalTab === "chat" ? (
          <UploadCard
            title="Chat Tactical Forecast"
            state={uploads.chatTactical}
            onFile={(file) => handleFile("chatTactical", file)}
            onSave={() => saveTimeseries("chatTactical", "chat_tactical")}
            onTemplate={() => downloadCsv("chat_tactical_template.csv", [], ["date", "items", "aht_sec"])}
          />
        ) : null}

        {activeTacticalTab === "ob" ? (
          <UploadCard
            title="Outbound Tactical Forecast"
            state={uploads.obTactical}
            onFile={(file) => handleFile("obTactical", file)}
            onSave={() => saveTimeseries("obTactical", "ob_tactical")}
            onTemplate={() =>
              downloadCsv(
                "outbound_tactical_template.csv",
                [],
                ["date", "opc", "connect_rate", "rpc_rate", "aht_sec"]
              )
            }
          />
        ) : null}
      </section>

      <section className="section gauravi">
        <h2>Holiday Calendar</h2>
        <div className="forecast-muted">Upload site/location specific holidays to exclude from working days.</div>
        <div className="upload-box" style={{ marginTop: 12 }}>
          <input type="file" accept=".csv,.xlsx,.xls" onChange={(event) => handleHolidayUpload(event.target.files?.[0])} />
          <div style={{ display: "flex", gap: 8 }}>
            <button type="button" className="btn btn-primary" onClick={saveHolidays}>
              Save Holidays
            </button>
            <button
              type="button"
              className="btn btn-outline"
              onClick={() =>
                downloadCsv(
                  "holiday_template.csv",
                  [
                    {
                      date: new Date().toISOString().slice(0, 10),
                      name: "Holiday 1"
                    }
                  ]
                )
              }
            >
              Download Template
            </button>
          </div>
        </div>
        {holidaysMessage ? <div className="forecast-muted" style={{ marginTop: 8 }}>{holidaysMessage}</div> : null}
        <DataTable data={holidaysRows} maxRows={10} />
      </section>

      <section className="section gauravi">
        <h2>Headcount Update - BRID Mapping</h2>
        <div className="forecast-muted">Upload the latest headcount file to keep BRID mappings in sync.</div>
        <div className="upload-box" style={{ marginTop: 12 }}>
          <input type="file" accept=".csv,.xlsx,.xls" onChange={(event) => handleHeadcountUpload(event.target.files?.[0])} />
          <div style={{ display: "flex", gap: 8 }}>
            <button type="button" className="btn btn-primary" onClick={saveHeadcount}>
              Save Headcount
            </button>
            <button
              type="button"
              className="btn btn-outline"
              onClick={() =>
                downloadCsv(
                  "headcount_template.csv",
                  [
                    {
                      "Level 0": "BUK",
                      "Level 1": "COO",
                      "Level 2": "Business Services",
                      "Level 3": "BFA",
                      "Level 4": "Refers",
                      "Level 5": "",
                      "Level 6": "",
                      BRID: "IN0001",
                      "Full Name": "Asha Rao",
                      "Position Description": "Agent",
                      "Headcount Operational Status Description": "Active",
                      "Employee Group Description": "FT",
                      "Corporate Grade Description": "BA4",
                      "Line Manager BRID": "IN9999",
                      "Line Manager Full Name": "Priyanka Menon",
                      "Current Organisation Unit": "Ops|BFA|Refers",
                      "Current Organisation Unit Description": "Ops BFA Refers",
                      "Position Location Country": "India",
                      "Position Location City": "Chennai",
                      "Position Location Building Description": "DLF IT Park",
                      CCID: "12345",
                      "CC Name": "Complaints",
                      Journey: "Onboarding",
                      "Position Group": "Back Office"
                    }
                  ]
                )
              }
            >
              Download Template
            </button>
          </div>
        </div>
        {headcountMessage ? <div className="forecast-muted" style={{ marginTop: 8 }}>{headcountMessage}</div> : null}
        <DataTable data={headcountRows} maxRows={8} />
      </section>

      {saveUploadModalOpen ? (
        <div className="ws-modal-backdrop">
          <div className="ws-modal ws-modal-sm">
            <div className="ws-modal-header" style={{ background: "#2f3747", color: "white" }}>
              <h3>Save Upload</h3>
              <button
                type="button"
                className="btn btn-light closeOptions"
                onClick={() => {
                  setSaveUploadModalOpen(false);
                  setSaveUploadPending(null);
                }}
              >
                <svg width="16" height="16" viewBox="0 0 16 16">
                  <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                  <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                </svg>
              </button>
            </div>
            <div className="ws-modal-body">
              <p style={{ margin: 0 }}>
                Save upload for scope <strong>{saveUploadScopeLabel}</strong>: replace existing rows for matching
                date/interval, or append this upload on top?
              </p>
            </div>
            <div className="ws-modal-footer">
              <button type="button" className="btn btn-primary" onClick={() => void handleSaveUploadChoice("replace")}>
                Replace Existing
              </button>
              <button type="button" className="btn btn-light" onClick={() => void handleSaveUploadChoice("append")}>
                Append to Existing
              </button>
              <button
                type="button"
                className="btn btn-light"
                onClick={() => {
                  setSaveUploadModalOpen(false);
                  setSaveUploadPending(null);
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function UploadCard({
  title,
  state,
  onFile,
  onSave,
  onTemplate
}: {
  title: string;
  state: UploadState;
  onFile: (file?: File | null) => void;
  onSave: () => void;
  onTemplate: () => void;
}) {
  return (
    <div className="section" style={{ marginTop: 16 }}>
      <h2>{title}</h2>
      <div className="upload-box">
        <input type="file" accept=".csv,.xlsx,.xls" onChange={(event) => onFile(event.target.files?.[0])} />
        <div style={{ display: "flex", gap: 8 }}>
          <button type="button" className="btn btn-primary" onClick={onSave}>
            Save
          </button>
          <button type="button" className="btn btn-outline" onClick={onTemplate}>
            Download Template
          </button>
        </div>
      </div>
      {state.message ? <div className="forecast-muted" style={{ marginTop: 8 }}>{state.message}</div> : null}
      <DataTable data={state.rows} maxRows={10} />
    </div>
  );
}

async function parseFile(file: File): Promise<Record<string, any>[]> {
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (ext === "xlsx" || ext === "xls") {
    return parseExcelFile(file);
  }
  const text = await file.text();
  return parseCsv(text);
}

function parseCsv(text: string): Record<string, any>[] {
  const firstLine = text.split(/\r?\n/).find((line) => line.trim() !== "") || "";
  const tabCount = (firstLine.match(/\t/g) || []).length;
  const commaCount = (firstLine.match(/,/g) || []).length;
  const delimiter = tabCount > commaCount ? "\t" : ",";

  const rows: string[][] = [];
  let current = "";
  let row: string[] = [];
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === "\"") {
      if (inQuotes && text[i + 1] === "\"") {
        current += "\"";
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === delimiter && !inQuotes) {
      row.push(current);
      current = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && text[i + 1] === "\n") {
        i += 1;
      }
      row.push(current);
      if (row.some((cell) => cell.trim() !== "")) {
        rows.push(row);
      }
      row = [];
      current = "";
    } else {
      current += char;
    }
  }
  if (current.length || row.length) {
    row.push(current);
    if (row.some((cell) => cell.trim() !== "")) {
      rows.push(row);
    }
  }

  if (!rows.length) return [];
  const headers = rows[0].map((h) => h.trim());
  return rows.slice(1).map((values) => {
    const rowObj: Record<string, any> = {};
    headers.forEach((header, idx) => {
      rowObj[header] = (values[idx] ?? "").trim();
    });
    return rowObj;
  });
}
