"use client";

import { useEffect, useMemo, useState } from "react";
import AppShell from "../_components/AppShell";
import DataTable from "../_components/DataTable";
import EditableTable from "../_components/EditableTable";
import { useGlobalLoader } from "../_components/GlobalLoader";
import { useToast } from "../_components/ToastProvider";
import { apiGet, apiPost } from "../../lib/api";
import { parseExcelFile } from "../../lib/excel";

const CHANNELS = ["Voice", "Back Office", "Outbound", "Blended", "Chat", "MessageUs"];

type BudgetTabKey = "voice" | "bo" | "chat" | "ob";

type BudgetTabState = {
  rows: Array<Record<string, any>>;
  message: string;
  startWeek: string;
  weeks: number;
};

type HeadcountOptions = {
  businessAreas: string[];
  subBusinessAreas: string[];
  sites: string[];
  channels: string[];
};

const TAB_CONFIG: Record<BudgetTabKey, { label: string; channel: string; sampleBase: number; metric: "aht" | "sut" }>
= {
  voice: { label: "Voice budget", channel: "Voice", sampleBase: 25, metric: "aht" },
  bo: { label: "Back Office budget", channel: "Back Office", sampleBase: 30, metric: "sut" },
  chat: { label: "Chat budget", channel: "Chat", sampleBase: 20, metric: "aht" },
  ob: { label: "Outbound budget", channel: "Outbound", sampleBase: 15, metric: "aht" }
};

const DEFAULT_TAB_STATE: BudgetTabState = {
  rows: [],
  message: "",
  startWeek: "",
  weeks: 8
};

function normalizeChannel(channel: string) {
  const raw = channel.trim().toLowerCase();
  if (["back office", "backoffice", "bo"].includes(raw)) return "Back Office";
  if (["voice", "call", "telephony"].includes(raw)) return "Voice";
  if (["chat", "messageus", "message us", "messaging"].includes(raw)) return "Chat";
  if (["outbound", "ob", "out bound"].includes(raw)) return "Outbound";
  return channel.trim();
}

function parseDateValue(value: any): Date | null {
  if (value === null || value === undefined || value === "") return null;
  if (value instanceof Date) return value;
  if (typeof value === "number" && Number.isFinite(value)) {
    const excelEpoch = new Date(Date.UTC(1899, 11, 30));
    const excelDate = new Date(excelEpoch.getTime() + value * 86400000);
    if (!Number.isNaN(excelDate.valueOf())) return excelDate;
  }
  const parsed = new Date(String(value));
  if (!Number.isNaN(parsed.valueOf())) return parsed;
  return null;
}

function weekMonday(value: any): string | null {
  const date = parseDateValue(value);
  if (!date) return null;
  const normalized = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const day = normalized.getUTCDay();
  const diff = (day + 6) % 7;
  normalized.setUTCDate(normalized.getUTCDate() - diff);
  return normalized.toISOString().slice(0, 10);
}

function buildTemplate(startWeek: string, weeks: number, base: number, metric: "aht" | "sut") {
  const startDate = weekMonday(startWeek || new Date()) || weekMonday(new Date());
  if (!startDate) return [];
  const start = new Date(`${startDate}T00:00:00Z`);
  const rows: Record<string, any>[] = [];
  const safeWeeks = Math.max(1, Number(weeks) || 1);
  for (let idx = 0; idx < safeWeeks; idx += 1) {
    const next = new Date(start);
    next.setUTCDate(start.getUTCDate() + idx * 7);
    const row: Record<string, any> = {
      week: next.toISOString().slice(0, 10),
      budget_headcount: base + (idx % 3) * (base >= 25 ? 5 : 4)
    };
    if (metric === "sut") {
      row.budget_sut_sec = 600;
    } else {
      row.budget_aht_sec = 300;
    }
    rows.push(row);
  }
  return rows;
}

function normalizeRows(rows: Array<Record<string, any>>, channel: string) {
  if (!rows.length) return [];
  const lowerCols = Object.keys(rows[0] || {}).map((c) => c.toLowerCase());
  const weekKey = lowerCols.find((c) => ["week", "start_week", "monday"].includes(c)) || lowerCols[0];
  const hcKey = lowerCols.find((c) => c === "budget_headcount" || c === "headcount");
  const ahtKey = lowerCols.find((c) => ["budget_aht_sec", "aht_sec", "aht"].includes(c));
  const sutKey = lowerCols.find((c) => ["budget_sut_sec", "sut_sec", "sut"].includes(c));

  const out = rows
    .map((row) => {
      const normalized: Record<string, any> = {};
      Object.keys(row || {}).forEach((key) => {
        const lower = key.toLowerCase();
        if (lower === weekKey) normalized.week = row[key];
        if (hcKey && lower === hcKey) normalized.budget_headcount = row[key];
        if (ahtKey && lower === ahtKey) normalized.budget_aht_sec = row[key];
        if (sutKey && lower === sutKey) normalized.budget_sut_sec = row[key];
      });
      normalized.week = weekMonday(normalized.week || row[Object.keys(row || {})[0]]) || "";
      return normalized;
    })
    .filter((row) => row.week);

  const canon = normalizeChannel(channel);
  if (canon === "Back Office") {
    return out.map((row) => ({
      week: row.week,
      budget_headcount: row.budget_headcount ?? "",
      budget_sut_sec: row.budget_sut_sec ?? ""
    }));
  }
  return out.map((row) => ({
    week: row.week,
    budget_headcount: row.budget_headcount ?? "",
    budget_aht_sec: row.budget_aht_sec ?? ""
  }));
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
  const headers = rows[0].map((header) => header.trim());
  return rows.slice(1).map((values) => {
    const rowObj: Record<string, any> = {};
    headers.forEach((header, idx) => {
      rowObj[header] = (values[idx] ?? "").trim();
    });
    return rowObj;
  });
}

function toCsv(rows: Array<Record<string, any>>) {
  if (!rows.length) return "";
  const columnsSet = rows.reduce<Set<string>>((set, row) => {
    Object.keys(row || {}).forEach((key) => set.add(key));
    return set;
  }, new Set<string>());
  const columns = Array.from(columnsSet);
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
  if (!csv) return;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export default function BudgetPage() {
  const { notify } = useToast();
  const { setLoading } = useGlobalLoader();

  const [businessArea, setBusinessArea] = useState("");
  const [subBusinessArea, setSubBusinessArea] = useState("");
  const [channel, setChannel] = useState("Voice");
  const [site, setSite] = useState("");
  const [options, setOptions] = useState<HeadcountOptions>({
    businessAreas: [],
    subBusinessAreas: [],
    sites: [],
    channels: CHANNELS
  });

  const [activeTab, setActiveTab] = useState<BudgetTabKey>("voice");
  const [tabs, setTabs] = useState<Record<BudgetTabKey, BudgetTabState>>({
    voice: { ...DEFAULT_TAB_STATE },
    bo: { ...DEFAULT_TAB_STATE },
    chat: { ...DEFAULT_TAB_STATE },
    ob: { ...DEFAULT_TAB_STATE }
  });

  const scopeReady = businessArea && subBusinessArea && channel && site;

  const scopePayload = useMemo(
    () => ({
      business_area: businessArea,
      sub_business_area: subBusinessArea,
      channel: channel,
      site: site
    }),
    [businessArea, subBusinessArea, channel, site]
  );

  const loadOptions = async (ba?: string) => {
    const params = new URLSearchParams();
    if (ba) params.set("ba", ba);
    const res = await apiGet<{
      business_areas?: string[];
      sub_business_areas?: string[];
      sites?: string[];
      channels?: string[];
    }>(`/api/forecast/headcount/options${params.toString() ? `?${params.toString()}` : ""}`);
    setOptions((prev) => ({
      businessAreas: res.business_areas ?? prev.businessAreas,
      subBusinessAreas: res.sub_business_areas ?? (ba ? [] : prev.subBusinessAreas),
      sites: res.sites ?? (ba ? [] : prev.sites),
      channels: res.channels ?? prev.channels
    }));
  };

  useEffect(() => {
    void loadOptions();
  }, []);

  useEffect(() => {
    if (!businessArea) return;
    void loadOptions(businessArea);
  }, [businessArea]);

  useEffect(() => {
    if (!options.channels.length) return;
    if (options.channels.includes(channel)) return;
    setChannel(options.channels[0]);
  }, [options.channels, channel]);

  useEffect(() => {
    if (!options.businessAreas.length || businessArea) return;
    setBusinessArea(options.businessAreas[0] || "");
  }, [options.businessAreas, businessArea]);

  useEffect(() => {
    if (!options.subBusinessAreas.length || subBusinessArea) return;
    setSubBusinessArea(options.subBusinessAreas[0] || "");
  }, [options.subBusinessAreas, subBusinessArea]);

  useEffect(() => {
    if (!options.sites.length || site) return;
    setSite(options.sites[0] || "");
  }, [options.sites, site]);

  const updateTab = (key: BudgetTabKey, patch: Partial<BudgetTabState>) => {
    setTabs((prev) => ({
      ...prev,
      [key]: { ...prev[key], ...patch }
    }));
  };

  const loadBudgetRows = async (key: BudgetTabKey) => {
    if (!scopeReady) {
      updateTab(key, { rows: [] });
      return;
    }
    const tabChannel = TAB_CONFIG[key].channel;
    if (normalizeChannel(channel) !== tabChannel) {
      updateTab(key, { rows: [] });
      return;
    }
    setLoading(true);
    try {
      const params = new URLSearchParams({
        ba: businessArea,
        subba: subBusinessArea,
        channel: tabChannel,
        site: site
      });
      const res = await apiGet<{ rows?: Array<Record<string, any>> }>(`/api/forecast/budget?${params.toString()}`);
      updateTab(key, { rows: res.rows ?? [] });
    } catch (error: any) {
      notify("error", error?.message || "Failed to load budget.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadBudgetRows(activeTab);
  }, [businessArea, subBusinessArea, channel, site, activeTab]);

  const handleUpload = async (key: BudgetTabKey, file?: File | null) => {
    if (!file) return;
    setLoading(true);
    try {
      const parsed = await parseFile(file);
      const normalized = normalizeRows(parsed, TAB_CONFIG[key].channel);
      updateTab(key, {
        rows: normalized,
        message: normalized.length ? `Loaded ${normalized.length} rows from ${file.name}.` : "No valid rows found."
      });
    } catch (error: any) {
      updateTab(key, { rows: [], message: "Could not parse file." });
    } finally {
      setLoading(false);
    }
  };

  const handleTemplateDownload = (key: BudgetTabKey) => {
    const tab = TAB_CONFIG[key];
    const state = tabs[key];
    const rows = buildTemplate(state.startWeek, state.weeks, tab.sampleBase, tab.metric);
    downloadCsv(`${tab.channel.toLowerCase().replace(" ", "_")}_budget_template.csv`, rows);
  };

  const handleSave = async (key: BudgetTabKey) => {
    if (!scopeReady) {
      updateTab(key, { message: "Pick BA, Sub BA, Channel and Site first." });
      return;
    }
    if (normalizeChannel(channel) !== TAB_CONFIG[key].channel) {
      updateTab(key, { message: `Select Channel = ${TAB_CONFIG[key].channel} to save.` });
      return;
    }
    const rows = tabs[key].rows;
    if (!rows.length) {
      updateTab(key, { message: "Nothing to save." });
      return;
    }
    setLoading(true);
    try {
      const res = await apiPost<{ rows?: number }>("/api/forecast/budget", {
        scope: scopePayload,
        rows
      });
      updateTab(key, { message: `Saved ${res.rows ?? rows.length} rows.` });
      notify("success", "Budget saved.");
    } catch (error: any) {
      updateTab(key, { message: "Save failed." });
      notify("error", error?.message || "Save failed.");
    } finally {
      setLoading(false);
    }
  };

  const renderTab = (key: BudgetTabKey) => {
    const tab = TAB_CONFIG[key];
    const state = tabs[key];
    const data = state.rows;
    const columns = data.length ? Object.keys(data[0]) : [];
    return (
      <div className="budget-tab">
        <div className="budget-row">
          <div>
            <div className="label">Start week (Monday)</div>
            <input
              id={`bud-${key}-start`}
              className="input"
              type="date"
              value={state.startWeek}
              onChange={(event) => updateTab(key, { startWeek: event.target.value })}
            />
          </div>
          <div>
            <div className="label">Weeks</div>
            <input
              className="input"
              type="number"
              min={1}
              step={1}
              value={state.weeks}
              onChange={(event) => updateTab(key, { weeks: Number(event.target.value) || 1 })}
            />
          </div>
          <div>
            <button
              id={`btn-bud-${key}-tmpl`}
              className="btn btn-outline budget-template"
              type="button"
              onClick={() => handleTemplateDownload(key)}
            >
              Download Template
            </button>
          </div>
          <div>
            <label className="upload-box budget-upload">
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={(event) => handleUpload(key, event.target.files?.[0])}
                style={{ display: "none" }}
              />
              <span>⬆️ Upload CSV/Excel</span>
            </label>
          </div>
        </div>

        <div className="budget-table">
          {data.length ? (
            <EditableTable
              data={data}
              editableColumns={columns}
              onChange={(rows) => updateTab(key, { rows })}
            />
          ) : (
            <DataTable data={[]} emptyLabel="No data" />
          )}
        </div>

        <div className="budget-actions">
          <button className="btn btn-primary" type="button" onClick={() => handleSave(key)}>
            Save {tab.label.replace(" budget", "")} Budget
          </button>
          <div className="budget-message">{state.message}</div>
        </div>
      </div>
    );
  };

  return (
    <AppShell crumbs="CAP-CONNECT / Budget">
      <div className="budget-page">
        <h4 className="ghanii">Budgets</h4>
        <section className="section budget-card">
          <div className="budget-scope-row">
            <div>
              <div className="label">Business Area</div>
              <select className="select" value={businessArea} onChange={(event) => setBusinessArea(event.target.value)}>
                <option value="">Select Business Area</option>
                {options.businessAreas.map((ba) => (
                  <option key={ba} value={ba}>
                    {ba}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="label">Sub Business Area</div>
              <select className="select" value={subBusinessArea} onChange={(event) => setSubBusinessArea(event.target.value)}>
                <option value="">Select Sub Business Area</option>
                {options.subBusinessAreas.map((sba) => (
                  <option key={sba} value={sba}>
                    {sba}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="label">Channel</div>
              <select className="select" value={channel} onChange={(event) => setChannel(event.target.value)}>
                {options.channels.map((ch) => (
                  <option key={ch} value={ch}>
                    {ch}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="label">Site</div>
              <select className="select" value={site} onChange={(event) => setSite(event.target.value)}>
                <option value="">Select Site</option>
                {options.sites.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="tabs" style={{ marginTop: 16 }}>
            {Object.entries(TAB_CONFIG).map(([key, tab]) => (
              <div
                key={key}
                className={`tab ${activeTab === key ? "active" : ""}`}
                onClick={() => setActiveTab(key as BudgetTabKey)}
              >
                {tab.label}
              </div>
            ))}
          </div>

          {renderTab(activeTab)}
        </section>
      </div>
    </AppShell>
  );
}
