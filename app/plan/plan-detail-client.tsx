"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AppShell from "../_components/AppShell";
import DataTable from "../_components/DataTable";
import EditableTable from "../_components/EditableTable";
import { useGlobalLoader } from "../_components/GlobalLoader";
import { useToast } from "../_components/ToastProvider";
import { apiGet, apiPost } from "../../lib/api";

type PlanRecord = {
  id: number;
  plan_name?: string;
  business_area?: string;
  sub_business_area?: string;
  channel?: string;
  location?: string;
  site?: string;
  plan_type?: string;
  start_week?: string;
  end_week?: string;
  status?: string;
  owner?: string;
  created_by?: string;
  updated_by?: string;
  created_at?: string;
  updated_at?: string;
};

type PlanTableConfig = {
  key: string;
  label: string;
  editable?: boolean;
  description?: string;
  persist?: boolean;
};

type PlanOption = {
  id: number;
  plan_name?: string;
  status?: string;
  is_current?: boolean;
  label?: string;
};

type CompareResult = {
  current?: Array<Record<string, any>>;
  compare?: Array<Record<string, any>>;
  delta?: Array<Record<string, any>>;
};

type WhatIfState = {
  ahtDelta: number;
  shrinkDelta: number;
  attrDelta: number;
  volDelta: number;
  backlogCarryover: boolean;
};

type PlanComputeResponse = {
  status: "ready" | "running" | "failed" | "missing";
  job?: Record<string, any>;
  data?: {
    tables?: Record<string, Array<Record<string, any>>>;
    upper?: Array<Record<string, any>>;
  };
};

const TABLES: PlanTableConfig[] = [
  { key: "fw", label: "Forecast & Workload"},
  { key: "hc", label: "Headcount" },
  { key: "attr", label: "Attrition", editable: true },
  { key: "shr", label: "Shrinkage" },
  { key: "train", label: "Training Lifecycle"},
  { key: "ratio", label: "Ratios", editable: true },
  { key: "seat", label: "Seat Utilization", editable: true },
  { key: "bva", label: "Budget vs Actual" },
  { key: "nh", label: "New Hire"},
  { key: "emp", label: "Employee Roster"},
  { key: "notes", label: "Notes" },
];

const EDITABLE_TABLES = new Set(TABLES.filter((t) => t.editable).map((t) => t.key));
const PERSIST_TABLE_KEYS = [
  "fw",
  "hc",
  "attr",
  "shr",
  "train",
  "ratio",
  "seat",
  "bva",
  "nh",
  "emp",
  "bulk_files",
  "notes"
];

const DATE_KEY_RE = /^\d{4}-\d{2}-\d{2}$/;
const FORECAST_BASE =
  process.env.NEXT_PUBLIC_FORECAST_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8080";

const ROSTER_COLUMNS = [
  { id: "brid", label: "BRID" },
  { id: "name", label: "Name" },
  { id: "class_ref", label: "Class Reference" },
  { id: "work_status", label: "Work Status" },
  { id: "role", label: "Role" },
  { id: "ftpt_status", label: "FT/PT Status" },
  { id: "ftpt_hours", label: "FT/PT Hours" },
  { id: "current_status", label: "Current Status" },
  { id: "training_start", label: "Training Start" },
  { id: "training_end", label: "Training End" },
  { id: "nesting_start", label: "Nesting Start" },
  { id: "nesting_end", label: "Nesting End" },
  { id: "production_start", label: "Production Start" },
  { id: "terminate_date", label: "Terminate Date" },
  { id: "team_leader", label: "Team Leader" },
  { id: "avp", label: "AVP" },
  { id: "biz_area", label: "Business Area" },
  { id: "sub_biz_area", label: "Sub Business Area" },
  { id: "lob", label: "LOB" },
  { id: "loa_date", label: "LOA Date" },
  { id: "back_from_loa_date", label: "Back From LOA Date" },
  { id: "site", label: "Site" }
];

const NH_FORM_DEFAULT = {
  emp_type: "full-time",
  status: "tentative",
  class_type: "ramp-up",
  class_level: "new-agent",
  grads_needed: 0,
  billable_hc: 0,
  training_weeks: 2,
  nesting_weeks: 1,
  induction_start: "",
  training_start: "",
  training_end: "",
  nesting_start: "",
  nesting_end: "",
  production_start: ""
};

const WHATIF_DEFAULT: WhatIfState = {
  ahtDelta: 0,
  shrinkDelta: 0,
  attrDelta: 0,
  volDelta: 0,
  backlogCarryover: true
};

function formatMetaRow(label: string, value?: string) {
  return (
    <div className="plan-meta-row">
      <span className="plan-meta-label">{label}</span>
      <span className="plan-meta-value">{value || "—"}</span>
    </div>
  );
}

function formatDate(value?: string) {
  if (!value) return "—";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return value;
  return dt.toISOString().slice(0, 10);
}

function formatDateTime(value?: string) {
  if (!value) return "—";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return value;
  return dt.toISOString().replace("T", " ").slice(0, 19);
}

function isDateKey(key: string) {
  return DATE_KEY_RE.test(key);
}

function toDate(value?: string) {
  if (!value) return null;
  const dt = new Date(`${value}T00:00:00`);
  if (Number.isNaN(dt.getTime())) return null;
  return dt;
}

function filterRowsByDateRange(
  rows: Array<Record<string, any>>,
  from?: string,
  to?: string
): Array<Record<string, any>> {
  if (!from || !to) return rows;
  const fromDate = toDate(from);
  const toDateValue = toDate(to);
  if (!fromDate || !toDateValue) return rows;
  return rows.map((row) => {
    const next: Record<string, any> = {};
    Object.keys(row || {}).forEach((key) => {
      if (!isDateKey(key)) {
        next[key] = row[key];
        return;
      }
      const keyDate = toDate(key);
      if (!keyDate) {
        next[key] = row[key];
        return;
      }
      if (keyDate >= fromDate && keyDate <= toDateValue) {
        next[key] = row[key];
      }
    });
    return next;
  });
}

function mergeFilteredRows(
  baseRows: Array<Record<string, any>>,
  updatedRows: Array<Record<string, any>>
) {
  return baseRows.map((row, idx) => {
    const updated = updatedRows[idx];
    if (!updated) return row;
    const next = { ...row };
    Object.keys(updated).forEach((key) => {
      if (key in row || key === "metric") {
        next[key] = updated[key];
      }
    });
    return next;
  });
}

function normalizeKey(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function toRosterRows(rows: Array<Record<string, any>>) {
  return rows.map((row) => {
    const out: Record<string, any> = {};
    const normalizedKeys = new Map(
      Object.keys(row || {}).map((key) => [normalizeKey(key), key] as const)
    );
    ROSTER_COLUMNS.forEach((col) => {
      const normalized = normalizeKey(col.id);
      const normalizedLabel = normalizeKey(col.label);
      const originalKey =
        normalizedKeys.get(normalized) ||
        normalizedKeys.get(normalizedLabel) ||
        normalizedKeys.get(normalizeKey(col.label.replace(" ", "")));
      out[col.id] = originalKey ? row[originalKey] : "";
    });
    return out;
  });
}

function buildIntervalOptions(startWeek?: string, endWeek?: string) {
  const start = toDate(startWeek);
  const end = toDate(endWeek);
  if (!start || !end) return [];
  const dates: string[] = [];
  const cursor = new Date(start);
  while (cursor <= end) {
    for (let i = 0; i < 7; i += 1) {
      const day = new Date(cursor);
      day.setDate(day.getDate() + i);
      dates.push(day.toISOString().slice(0, 10));
    }
    cursor.setDate(cursor.getDate() + 7);
  }
  const seen = new Set<string>();
  return dates
    .filter((value) => {
      if (seen.has(value)) return false;
      seen.add(value);
      return true;
    })
    .map((value) => ({
      value,
      label: new Date(`${value}T00:00:00`).toDateString()
    }));
}

async function parseCsv(text: string): Promise<Record<string, any>[]> {
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

async function parseFile(file: File): Promise<Record<string, any>[]> {
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (ext === "xlsx" || ext === "xls") {
    const buf = await file.arrayBuffer();
    const XLSX = await import("xlsx");
    const workbook = XLSX.read(buf, { type: "array" });
    const sheetName = workbook.SheetNames[0];
    const sheet = workbook.Sheets[sheetName];
    return XLSX.utils.sheet_to_json(sheet, { defval: "" }) as Record<string, any>[];
  }
  const text = await file.text();
  return parseCsv(text);
}

function toCsv(rows: Array<Record<string, any>>) {
  if (!rows.length) return "";
  const columns = Object.keys(rows[0] ?? {});
  const escape = (val: any) => {
    if (val === null || val === undefined) return "";
    const str = String(val).replace(/"/g, "\"\"");
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

type PlanDetailClientProps = {
  planId?: number;
  rollupBa?: string;
};

export default function PlanDetailClient({ planId, rollupBa }: PlanDetailClientProps) {
  const router = useRouter();
  const { setLoading } = useGlobalLoader();
  const { notify } = useToast();
  const [planMeta, setPlanMeta] = useState<PlanRecord | null>(null);
  const [tables, setTables] = useState<Record<string, Array<Record<string, any>>>>({});
  const [upperRows, setUpperRows] = useState<Array<Record<string, any>>>([]);
  const [activeTab, setActiveTab] = useState("fw");
  const [message, setMessage] = useState("");
  const pollRef = useRef<number | null>(null);
  const [grain, setGrain] = useState("week");
  const [intervalDate, setIntervalDate] = useState("");
  const [upperCollapsed, setUpperCollapsed] = useState(false);
  const [optionsOpen, setOptionsOpen] = useState(false);
  const [viewModalOpen, setViewModalOpen] = useState(false);
  const [viewFrom, setViewFrom] = useState("");
  const [viewTo, setViewTo] = useState("");
  const [viewDraftFrom, setViewDraftFrom] = useState("");
  const [viewDraftTo, setViewDraftTo] = useState("");
  const [saveAsOpen, setSaveAsOpen] = useState(false);
  const [saveAsName, setSaveAsName] = useState("");
  const [saveAsMessage, setSaveAsMessage] = useState("");
  const [extendOpen, setExtendOpen] = useState(false);
  const [extendWeeks, setExtendWeeks] = useState(4);
  const [extendMessage, setExtendMessage] = useState("");
  const [switchOpen, setSwitchOpen] = useState(false);
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareResultOpen, setCompareResultOpen] = useState(false);
  const [scopeOptions, setScopeOptions] = useState<PlanOption[]>([]);
  const [switchPlanId, setSwitchPlanId] = useState<number | "">("");
  const [comparePlanId, setComparePlanId] = useState<number | "">("");
  const [compareWarning, setCompareWarning] = useState("");
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [whatIf, setWhatIf] = useState<WhatIfState>(WHATIF_DEFAULT);
  const [noteText, setNoteText] = useState("");
  const [rosterTab, setRosterTab] = useState<"roster" | "bulk">("roster");
  const rosterSnapshotRef = useRef<Array<Record<string, any>> | null>(null);

  const [nhClasses, setNhClasses] = useState<Array<Record<string, any>>>([]);
  const [nhClassOptions, setNhClassOptions] = useState<{
    class_types?: Array<{ label: string; value: string }>;
    class_levels?: Array<{ label: string; value: string }>;
  }>({});
  const [nhModalOpen, setNhModalOpen] = useState(false);
  const [nhDetailsOpen, setNhDetailsOpen] = useState(false);
  const [nhMessage, setNhMessage] = useState("");
  const [nhForm, setNhForm] = useState<Record<string, any>>(NH_FORM_DEFAULT);
  const [rosterModalOpen, setRosterModalOpen] = useState(false);
  const [rosterForm, setRosterForm] = useState<Record<string, any>>(
    Object.fromEntries(ROSTER_COLUMNS.map((col) => [col.id, ""]))
  );

  const isRollup = Boolean(rollupBa);
  const activeConfig = useMemo(() => TABLES.find((t) => t.key === activeTab) ?? TABLES[0], [activeTab]);
  const activeRows = tables[activeConfig.key] ?? [];
  const filteredRows = useMemo(
    () => filterRowsByDateRange(activeRows, viewFrom, viewTo),
    [activeRows, viewFrom, viewTo]
  );
  const canEdit = EDITABLE_TABLES.has(activeConfig.key) && grain === "week" && !isRollup;
  const editableColumns = useMemo(() => {
    const cols = new Set<string>();
    filteredRows.forEach((row) => {
      Object.keys(row || {}).forEach((key) => cols.add(key));
    });
    return Array.from(cols).filter((key) => key.toLowerCase() !== "metric");
  }, [filteredRows]);

  const planName = planMeta?.plan_name || (planId ? `Plan ${planId}` : "Plan Detail");
  const planStatus = planMeta?.status || (isRollup ? "rollup" : "draft");
  const planChannel = String(planMeta?.channel || "").split(",")[0].trim().toLowerCase();
  const isBackOffice = ["back office", "backoffice", "bo"].includes(planChannel);
  const intervalOptions = useMemo(
    () => buildIntervalOptions(planMeta?.start_week, planMeta?.end_week),
    [planMeta?.start_week, planMeta?.end_week]
  );

  const loadPlan = useCallback(async () => {
    if (!planId) return;
    const res = await apiGet<{ plan?: PlanRecord }>(`/api/planning/plan?plan_id=${planId}`);
    setPlanMeta(res.plan ?? null);
  }, [planId]);

  const loadTables = useCallback(async () => {
    if (!planId) return;
    const entries = await Promise.all(
      PERSIST_TABLE_KEYS.map(async (table) => {
        try {
          const res = await apiGet<{ rows?: Array<Record<string, any>> }>(
            `/api/planning/plan/table?plan_id=${planId}&name=${table}`
          );
          return [table, res.rows ?? []] as const;
        } catch {
          return [table, []] as const;
        }
      })
    );
    const next: Record<string, Array<Record<string, any>>> = {};
    entries.forEach(([key, rows]) => {
      next[key] = [...rows];
    });
    setTables((prev) => ({ ...prev, ...next }));
  }, [planId]);

  const computePlanTables = useCallback(async () => {
    if (!planId) return;
    const payload: Record<string, any> = {
      plan_id: planId,
      grain,
      persist: grain === "week"
    };
    if (grain === "interval" && intervalDate) {
      payload.interval_date = intervalDate;
    }
    for (let attempt = 0; attempt < 60; attempt += 1) {
      const res = await apiPost<PlanComputeResponse>("/api/planning/plan/detail/compute", payload);
      if (res.status === "ready") {
        if (res.data?.tables) {
          setTables((prev) => ({ ...prev, ...res.data?.tables }));
        }
        if (res.data?.upper) {
          setUpperRows(res.data.upper);
        }
        return;
      }
      if (res.status === "missing") {
        throw new Error("Plan not found.");
      }
      if (res.status === "failed") {
        const detail = res.job?.error ? ` ${res.job.error}` : "";
        throw new Error(`Plan detail calculations failed.${detail}`);
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1200));
    }
    throw new Error("Plan detail calculations timed out.");
  }, [grain, intervalDate, planId]);

  const loadRollupTables = useCallback(async () => {
    if (!rollupBa) return;
    try {
      const res = await apiPost<{
        data?: { tables?: Record<string, Array<Record<string, any>>>; upper?: Array<Record<string, any>> };
      }>("/api/planning/rollup", { business_area: rollupBa, grain: "week" });
      setTables(res.data?.tables ?? {});
      setUpperRows(res.data?.upper ?? []);
    } catch (error: any) {
      notify("error", error?.message || "Could not load rollup data.");
    }
  }, [notify, rollupBa]);

  const loadNewHireClasses = useCallback(async () => {
    if (!planId || isRollup) return;
    try {
      const res = await apiGet<{
        rows?: Array<Record<string, any>>;
        class_types?: Array<{ label: string; value: string }>;
        class_levels?: Array<{ label: string; value: string }>;
      }>(`/api/planning/plan/new-hire/classes?plan_id=${planId}`);
      setNhClasses(res.rows ?? []);
      setNhClassOptions({
        class_types: res.class_types ?? [],
        class_levels: res.class_levels ?? []
      });
    } catch (error: any) {
      notify("error", error?.message || "Could not load new hire classes.");
    }
  }, [isRollup, notify, planId]);

  const loadWhatIf = useCallback(async () => {
    if (!planId || isRollup) return;
    try {
      const res = await apiGet<{ overrides?: Record<string, any> }>(
        `/api/planning/plan/whatif?plan_id=${planId}`
      );
      const overrides = res.overrides ?? {};
      setWhatIf({
        ahtDelta: Number(overrides.aht_delta ?? 0),
        shrinkDelta: Number(overrides.shrink_delta ?? 0),
        attrDelta: Number(overrides.attr_delta ?? 0),
        volDelta: Number(overrides.vol_delta ?? 0),
        backlogCarryover: Boolean(overrides.backlog_carryover ?? true)
      });
    } catch {
      setWhatIf(WHATIF_DEFAULT);
    }
  }, [isRollup, planId]);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    try {
      if (rollupBa) {
        await loadRollupTables();
      } else {
        await computePlanTables();
        await loadPlan();
        if (grain === "week") {
          await loadTables();
        }
      }
      setMessage("Refreshed.");
    } catch (error: any) {
      notify("error", error?.message || "Could not refresh plan.");
    } finally {
      setLoading(false);
    }
  }, [computePlanTables, grain, loadPlan, loadRollupTables, loadTables, notify, rollupBa, setLoading]);

  const saveAll = useCallback(async () => {
    if (!planId || isRollup) return;
    setLoading(true);
    try {
      await Promise.all(
        PERSIST_TABLE_KEYS.map((table) =>
          apiPost("/api/planning/plan/table", {
            plan_id: planId,
            name: table,
            rows: tables[table] ?? []
          })
        )
      );
      setMessage("Saved.");
      notify("success", "Plan saved.");
    } catch (error: any) {
      notify("error", error?.message || "Could not save plan.");
    } finally {
      setLoading(false);
    }
  }, [isRollup, notify, planId, setLoading, tables]);

  const loadCapacityRollups = useCallback(
    async (meta: PlanRecord | null) => {
      if (!meta) return;
      if (!meta.business_area || !meta.channel) return;
      if (pollRef.current) {
        window.clearTimeout(pollRef.current);
      }

      const channel = String(meta.channel || "")
        .split(",")
        .map((val) => val.trim())
        .filter(Boolean)[0];
      const scope = {
        business_area: meta.business_area || "",
        sub_business_area: meta.sub_business_area || "",
        channel: channel || "",
        location: meta.location || "",
        site: meta.site || ""
      };
      const planKey = [scope.business_area, scope.sub_business_area, scope.channel, scope.site || scope.location]
        .map((val) => String(val || "").trim())
        .filter(Boolean)
        .join("|");

      const poll = async () => {
        try {
          const res = await apiPost<{
            status?: string;
            data?: Record<string, string>;
          }>("/api/planning/calc/consolidated", {
            scope,
            plan_key: planKey
          });
          if (res.status === "running") {
            pollRef.current = window.setTimeout(poll, 1200);
            return;
          }
          if (res.status !== "ready" || !res.data) {
            notify("error", "Capacity rollup calculations failed.");
            return;
          }

          const parseRows = (key: string) => {
            try {
              const payload = res.data?.[key];
              if (!payload) return [];
              const parsed = JSON.parse(payload);
              return Array.isArray(parsed?.data) ? parsed.data : [];
            } catch {
              return [];
            }
          };

          const weekly: Array<Record<string, any>> = [];
          const monthly: Array<Record<string, any>> = [];
          const addRows = (
            rows: Array<Record<string, any>>,
            channelLabel: string,
            periodKey: "week" | "month",
            target: Array<Record<string, any>>
          ) => {
            rows.forEach((row) => {
              target.push({
                period: row?.[periodKey],
                channel: channelLabel,
                program: row?.program ?? channelLabel,
                fte_req: row?.fte_req ?? 0,
                phc: row?.phc ?? 0,
                service_level: row?.service_level ?? 0
              });
            });
          };

          addRows(parseRows("voice_week"), "Voice", "week", weekly);
          addRows(parseRows("chat_week"), "Chat", "week", weekly);
          addRows(parseRows("ob_week"), "Outbound", "week", weekly);
          addRows(parseRows("bo_week"), "Back Office", "week", weekly);

          addRows(parseRows("voice_month"), "Voice", "month", monthly);
          addRows(parseRows("chat_month"), "Chat", "month", monthly);
          addRows(parseRows("ob_month"), "Outbound", "month", monthly);
          addRows(parseRows("bo_month"), "Back Office", "month", monthly);

          setTables((prev) => ({
            ...prev,
            capacity_weekly: weekly,
            capacity_monthly: monthly
          }));
        } catch (error: any) {
          notify("error", error?.message || "Capacity rollup calculations failed.");
        }
      };

      await poll();
    },
    [notify]
  );

  const loadScopeOptions = useCallback(async () => {
    if (!planId) return;
    try {
      const res = await apiGet<{ options?: PlanOption[] }>(`/api/planning/plan/scope-options?plan_id=${planId}`);
      setScopeOptions(res.options ?? []);
      if (res.options?.length) {
        setSwitchPlanId(res.options[0].id);
        setComparePlanId(res.options[0].id);
      } else {
        setSwitchPlanId("");
        setComparePlanId("");
      }
    } catch (error: any) {
      notify("error", error?.message || "Could not load plan options.");
    }
  }, [notify, planId]);

  useEffect(() => {
    if (isRollup) return;
    void refreshAll();
  }, [refreshAll, isRollup]);

  useEffect(() => {
    if (isRollup) return;
    if (!planId) return;
    void computePlanTables().catch((error: any) => {
      notify("error", error?.message || "Plan detail calculations failed.");
    });
  }, [computePlanTables, isRollup, notify, planId]);

  useEffect(() => {
    if (isRollup) return;
    void loadCapacityRollups(planMeta);
  }, [loadCapacityRollups, planMeta, isRollup]);

  useEffect(() => {
    if (!isRollup) return;
    setPlanMeta({
      id: 0,
      plan_name: `${rollupBa} (Roll-up)`,
      business_area: rollupBa,
      status: "rollup"
    });
    void loadRollupTables();
  }, [loadRollupTables, isRollup, rollupBa]);

  useEffect(() => {
    if (isRollup) return;
    void loadNewHireClasses();
    void loadWhatIf();
  }, [isRollup, loadNewHireClasses, loadWhatIf]);

  useEffect(() => {
    if (!intervalOptions.length || grain !== "interval") return;
    if (!intervalDate) {
      setIntervalDate(intervalOptions[0].value);
    }
  }, [grain, intervalDate, intervalOptions]);

  useEffect(() => {
    if (grain === "day" || grain === "interval") {
      setOptionsOpen(false);
    }
  }, [grain]);

  useEffect(
    () => () => {
      if (pollRef.current) {
        window.clearTimeout(pollRef.current);
      }
    },
    []
  );

  useEffect(() => {
    if (!tables.emp?.length || rosterSnapshotRef.current) return;
    rosterSnapshotRef.current = tables.emp.map((row) => ({ ...row }));
  }, [tables.emp]);

  useEffect(() => {
    rosterSnapshotRef.current = null;
  }, [planId]);

  useEffect(() => {
    // Only schedule a timeout when there is a message to display
    if (message) {
      const timerId = window.setTimeout(() => {
        setMessage("");  // Clear the message after 5 seconds
      }, 5000);
      // Clear the timer if the message changes or the component unmounts
      return () => clearTimeout(timerId);
    }
  }, [message, setMessage]);

  useEffect(() => {
    if (saveAsMessage) {
      const id = setTimeout(() => setSaveAsMessage(""), 5000);
      return () => clearTimeout(id);
    }
  }, [saveAsMessage]);

  useEffect(() => {
    if (extendMessage) {
      const id = setTimeout(() => setExtendMessage(""), 5000);
      return () => clearTimeout(id);
    }
  }, [extendMessage]);

  useEffect(() => {
    if (nhMessage) {
      const id = setTimeout(() => setNhMessage(""), 5000);
      return () => clearTimeout(id);
    }
  }, [nhMessage]);

  const handleFilteredChange = (rows: Array<Record<string, any>>) => {
    if (!viewFrom || !viewTo) {
      setTables((prev) => ({ ...prev, [activeConfig.key]: rows }));
      return;
    }
    setTables((prev) => ({
      ...prev,
      [activeConfig.key]: mergeFilteredRows(prev[activeConfig.key] ?? [], rows)
    }));
  };

  const handleApplyWhatIf = async () => {
    if (!planId || isRollup) return;
    setLoading(true);
    try {
      await apiPost("/api/planning/plan/whatif", {
        plan_id: planId,
        overrides: {
          aht_delta: whatIf.ahtDelta,
          shrink_delta: whatIf.shrinkDelta,
          attr_delta: whatIf.attrDelta,
          vol_delta: whatIf.volDelta,
          backlog_carryover: whatIf.backlogCarryover
        },
        action: "apply"
      });
      await refreshAll();
      setMessage("What-if applied.");
    } catch (error: any) {
      notify("error", error?.message || "Could not apply what-if.");
    } finally {
      setLoading(false);
    }
  };

  const handleClearWhatIf = async () => {
    if (!planId || isRollup) return;
    setLoading(true);
    try {
      await apiPost("/api/planning/plan/whatif", {
        plan_id: planId,
        action: "clear"
      });
      setWhatIf(WHATIF_DEFAULT);
      await refreshAll();
      setMessage("What-if cleared.");
    } catch (error: any) {
      notify("error", error?.message || "Could not clear what-if.");
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (!planId) return;
    setLoading(true);
    try {
      const res = await fetch(`${FORECAST_BASE}/api/planning/plan/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan_id: planId })
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Export failed.");
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `plan_${planId}_export.xlsx`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (error: any) {
      notify("error", error?.message || "Could not export plan.");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveAs = async () => {
    if (!planId || !saveAsName.trim()) return;
    setLoading(true);
    try {
      const res = await apiPost<{ id?: number }>("/api/planning/plan/save-as", {
        plan_id: planId,
        name: saveAsName.trim()
      });
      setSaveAsMessage(`Saved as "${saveAsName}" (id ${res.id ?? "new"}).`);
      await loadScopeOptions();
    } catch (error: any) {
      setSaveAsMessage(error?.message || "Save as failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleExtendPlan = async () => {
    if (!planId) return;
    setLoading(true);
    try {
      await apiPost("/api/planning/plan/extend", {
        plan_id: planId,
        weeks: extendWeeks
      });
      setExtendMessage(`Extended by ${extendWeeks} week(s).`);
      await loadPlan();
    } catch (error: any) {
      setExtendMessage(error?.message || "Extend failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleDeletePlan = async () => {
    if (!planId) return;
    setLoading(true);
    try {
      await apiPost("/api/planning/plan/delete", { plan_id: planId });
      notify("success", "Plan deleted.");
      router.push("/planning");
    } catch (error: any) {
      notify("error", error?.message || "Could not delete plan.");
    } finally {
      setLoading(false);
    }
  };

  const handleSwitchPlan = () => {
    if (!switchPlanId) return;
    setSwitchOpen(false);
    router.push(`/plan/${switchPlanId}`);
  };

  const handleCompare = async () => {
    if (!planId || !comparePlanId) return;
    setLoading(true);
    try {
      const res = await apiPost<CompareResult>("/api/planning/plan/compare", {
        plan_id: planId,
        compare_id: comparePlanId
      });
      setCompareResult(res);
      setCompareWarning("");
      setCompareResultOpen(true);
      setCompareOpen(false);
    } catch (error: any) {
      setCompareWarning(error?.message || "Compare failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleRosterAdd = () => {
    setTables((prev) => ({
      ...prev,
      emp: [...(prev.emp ?? []), { ...rosterForm }]
    }));
    setRosterForm(Object.fromEntries(ROSTER_COLUMNS.map((col) => [col.id, ""])));
    setRosterModalOpen(false);
  };

  const handleRosterUndo = () => {
    if (!rosterSnapshotRef.current) return;
    setTables((prev) => ({ ...prev, emp: rosterSnapshotRef.current ?? [] }));
  };

  const handleRosterUpload = async (file: File) => {
    if (!file) return;
    setLoading(true);
    try {
      const rawRows = await parseFile(file);
      const rosterRows = toRosterRows(rawRows);
      setTables((prev) => ({
        ...prev,
        emp: [...(prev.emp ?? []), ...rosterRows],
        bulk_files: [
          ...(prev.bulk_files ?? []),
          {
            file_name: file.name,
            ext: file.name.split(".").pop() || "",
            size_kb: Math.round(file.size / 1024),
            is_valid: "Yes",
            status: "Uploaded"
          }
        ]
      }));
      notify("success", "Roster rows added.");
    } catch (error: any) {
      notify("error", error?.message || "Roster upload failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveNote = async () => {
    if (!noteText.trim()) return;
    const nextNote = {
      when: new Date().toISOString().slice(0, 10),
      user: planMeta?.updated_by || planMeta?.owner || "local",
      note: noteText.trim()
    };
    const nextNotes = [...(tables.notes ?? []), nextNote];
    setTables((prev) => ({ ...prev, notes: nextNotes }));
    setNoteText("");
    if (!planId) return;
    try {
      await apiPost("/api/planning/plan/table", {
        plan_id: planId,
        name: "notes",
        rows: nextNotes
      });
    } catch (error: any) {
      notify("error", error?.message || "Could not save note.");
    }
  };

  const handleNewHireSave = async () => {
    if (!planId) return;
    setLoading(true);
    try {
      const res = await apiPost<{ rows?: Array<Record<string, any>> }>("/api/planning/plan/new-hire/class", {
        plan_id: planId,
        data: nhForm
      });
      setNhClasses(res.rows ?? []);
      setNhForm(NH_FORM_DEFAULT);
      setNhModalOpen(false);
      setNhMessage("New hire class added.");
    } catch (error: any) {
      setNhMessage(error?.message || "Could not add class.");
    } finally {
      setLoading(false);
    }
  };

  const renderRoster = () => {
    const rosterRows = tables.emp ?? [];
    const bulkRows = tables.bulk_files ?? [];

    return (
      <div className="plan-roster">
        <div className="plan-subtabs">
          <button
            type="button"
            className={`plan-subtab ${rosterTab === "roster" ? "active" : ""}`}
            onClick={() => setRosterTab("roster")}
          >
            Roster
          </button>
          <button
            type="button"
            className={`plan-subtab ${rosterTab === "bulk" ? "active" : ""}`}
            onClick={() => setRosterTab("bulk")}
          >
            Bulk Upload
          </button>
        </div>

        {rosterTab === "roster" ? (
          <>
            <div className="plan-toolbar">
              <button type="button" className="btn btn-primary" onClick={() => setRosterModalOpen(true)}>
                Add Employee
              </button>
              <button type="button" className="btn btn-light" onClick={handleRosterUndo}>
                Undo
              </button>
              <button
                type="button"
                className="btn btn-light"
                onClick={() => downloadCsv(`plan_${planId}_roster_template.csv`, [Object.fromEntries(ROSTER_COLUMNS.map((c) => [c.id, ""]))])}
              >
                Download Template
              </button>
            </div>
            {canEdit ? (
              <EditableTable
                data={rosterRows}
                editableColumns={editableColumns}
                onChange={(rows) => setTables((prev) => ({ ...prev, emp: rows }))}
              />
            ) : (
              <DataTable data={rosterRows} emptyLabel="No roster entries yet." />
            )}
          </>
        ) : (
          <>
            <div className="plan-toolbar">
              <label className="upload-box">
                Upload CSV/XLSX
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void handleRosterUpload(file);
                    event.currentTarget.value = "";
                  }}
                />
              </label>
              <button
                type="button"
                className="btn btn-light"
                onClick={() =>
                  downloadCsv(
                    `plan_${planId}_bulk_files.csv`,
                    bulkRows.length ? bulkRows : [{ file_name: "", ext: "", size_kb: "", is_valid: "", status: "" }]
                  )
                }
              >
                Download Table
              </button>
            </div>
            <DataTable data={bulkRows} emptyLabel="No bulk uploads yet." />
          </>
        )}
      </div>
    );
  };

  const renderNewHire = () => {
    const nhRows = tables.nh ?? [];
    const recentClasses = nhClasses.slice(0, 10);
    const nhDisplayRows = filterRowsByDateRange(nhRows, viewFrom, viewTo);

    return (
      <div className="plan-nh">
        <div className="plan-toolbar">
          <button type="button" className="btn btn-primary" onClick={() => setNhModalOpen(true)}>
            Add New
          </button>
          <button type="button" className="btn btn-light" onClick={() => setNhDetailsOpen(true)}>
            Training Details
          </button>
          <button type="button" className="btn btn-light" onClick={() => downloadCsv(`plan_${planId}_nh_classes.csv`, nhClasses)}>
            Download Dataset
          </button>
          {nhMessage ? <span className="plan-message">{nhMessage}</span> : null}
        </div>
        <div className="plan-nh-grid">
          <div className="plan-nh-card">
            <h4>Recent New Hire Classes</h4>
            <DataTable data={recentClasses} emptyLabel="No classes yet." />
          </div>
          <div className="plan-nh-card">
            <h4>Weekly New Hire Plan/Actual</h4>
            {canEdit ? (
              <EditableTable
                data={nhDisplayRows}
                editableColumns={editableColumns}
                onChange={(rows) => {
                  if (!viewFrom || !viewTo) {
                    setTables((prev) => ({ ...prev, nh: rows }));
                    return;
                  }
                  setTables((prev) => ({
                    ...prev,
                    nh: mergeFilteredRows(prev.nh ?? [], rows)
                  }));
                }}
              />
            ) : (
              <DataTable data={nhDisplayRows} emptyLabel="No new hire data available." />
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderNotes = () => {
    const notes = tables.notes ?? [];
    return (
      <div className="plan-notes">
        <div className="plan-notes-input">
          <textarea
            className="textarea"
            placeholder="Write a note and click Save"
            value={noteText}
            onChange={(event) => setNoteText(event.target.value)}
          />
          <button type="button" className="btn btn-primary" onClick={handleSaveNote}>
            Save Note
          </button>
        </div>
        <DataTable data={notes} emptyLabel="No notes yet." />
      </div>
    );
  };

  return (
    <AppShell crumbs="CAP-CONNECT / Planning Workspace">
      <div className="plan-detail">
        <div className="plan-header">
          <div className="plan-header-left">
            <Link className="btn btn-light back" href="/planning">
              🢀
            </Link>
            <div>
              <div className="plan-title">{planName}</div>
              <div className="plan-subtitle">{planStatus}</div>
            </div>
          </div>
          <div className="plan-header-actions">
            <button type="button" className="btn btn-light save" onClick={saveAll} disabled={isRollup}>
              💾
            </button>
            <button type="button" className="btn btn-light refresh" onClick={refreshAll}>
              🔄
            </button>
            <label className="plan-toggle">
              <input
                type="checkbox"
                checked={grain === "month"}
                onChange={(event) => setGrain(event.target.checked ? "month" : "week")}
                disabled={isRollup}
              />
              <span>Monthly</span>
            </label>
            {message ? <span className="plan-message">{message}</span> : null}
          </div>
          <div className="plan-header-actions">
            <button type="button" className="btn btn-light expcoll" onClick={() => setUpperCollapsed((prev) => !prev)}>
              {upperCollapsed ? "▲" : "▼"}
            </button>
          </div>
        </div>

        {grain === "interval" && intervalOptions.length ? (
          <div className="plan-upper-card plan-controls">
            <div className="plan-controls-row">
              <div>
                <label className="label">Interval Date</label>
                <select
                  className="select"
                  value={intervalDate}
                  onChange={(event) => setIntervalDate(event.target.value)}
                >
                  {intervalOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        ) : null}

        {!upperCollapsed ? (
          <>
            {/* <div className="plan-upper-card">
              <div className="plan-meta">
                {formatMetaRow("Business Area", planMeta?.business_area)}
                {formatMetaRow("Sub Business Area", planMeta?.sub_business_area)}
                {formatMetaRow("Channel", planMeta?.channel)}
                {formatMetaRow("Location", planMeta?.location)}
                {formatMetaRow("Site", planMeta?.site)}
                {formatMetaRow("Plan Type", planMeta?.plan_type)}
                {formatMetaRow("Start Week", formatDate(planMeta?.start_week))}
                {formatMetaRow("End Week", formatDate(planMeta?.end_week))}
              </div>
            </div> */}

            <div className="plan-upper-card">
              <div className="plan-tab-title">
                <h3>Upper Summary</h3>
              </div>
              <DataTable data={upperRows} emptyLabel="No upper summary yet." />
            </div>
          </>
        ) : null}

        <div className="plan-tabs">
          {TABLES.map((table) => (
            <button
              key={table.key}
              type="button"
              className={`plan-tab ${activeTab === table.key ? "active" : ""}`}
              onClick={() => setActiveTab(table.key)}
            >
              {table.label}
            </button>
          ))}
        </div>

        <div className="plan-tab-body">
          <div className="plan-tab-title">
            <h3>{activeConfig.label}</h3>
            {activeConfig.description ? <p>{activeConfig.description}</p> : null}
          </div>
          {isRollup ? (
            <div className="plan-rollup">
              <DataTable data={filteredRows} emptyLabel="No roll-up data available yet." />
            </div>
          ) : activeConfig.key === "emp" ? (
            renderRoster()
          ) : activeConfig.key === "nh" ? (
            renderNewHire()
          ) : activeConfig.key === "notes" ? (
            renderNotes()
          ) : canEdit ? (
            <EditableTable data={filteredRows} editableColumns={editableColumns} onChange={handleFilteredChange} />
          ) : (
            <DataTable data={filteredRows} emptyLabel="No data available." />
          )}
        </div>
      </div>

      {!isRollup ? (
        <>
          <button type="button" className="plan-options-toggle" onClick={() => setOptionsOpen((prev) => !prev)}>
            ⚙ Options
          </button>
          <div className={`plan-options-panel ${optionsOpen ? "open" : ""}`}>
            <div className="plan-options-header">
              <h4>Options</h4>
              <button type="button" className="btn btn-light closeOptions" onClick={() => setOptionsOpen(false)}>
                <svg width="16" height="16" viewBox="0 0 16 16">
                  <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                  <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                </svg>
              </button>
            </div>
            <div className="plan-options-meta">
              <div>
                <span>Plan Type</span>
                <strong>{planMeta?.plan_type || "Volume Based"}</strong>
              </div>
              <div>
                <span>Start Week</span>
                <strong>{formatDate(planMeta?.start_week)}</strong>
              </div>
              <div>
                <span>End Week</span>
                <strong>{formatDate(planMeta?.end_week)}</strong>
              </div>
              <div>
                <span>Created By</span>
                <strong>{planMeta?.created_by || "—"}</strong>
              </div>
              <div>
                <span>Created On</span>
                <strong>{formatDateTime(planMeta?.created_at)}</strong>
              </div>
              <div>
                <span>Last Updated By</span>
                <strong>{planMeta?.updated_by || "—"}</strong>
              </div>
              <div>
                <span>Last Updated On</span>
                <strong>{formatDateTime(planMeta?.updated_at)}</strong>
              </div>
            </div>

            <div className="plan-options-actions">
              <button
                type="button"
                className="btn btn-light"
                onClick={() => {
                  setSwitchOpen(true);
                  void loadScopeOptions();
                }}
              >
                ⇄
              </button>
              <button
                type="button"
                className="btn btn-light"
                onClick={() => {
                  setCompareOpen(true);
                  void loadScopeOptions();
                }}
              >
                ⚖️
              </button>
              <button
                type="button"
                className="btn btn-light"
                onClick={() => {
                  setViewDraftFrom(viewFrom);
                  setViewDraftTo(viewTo);
                  setViewModalOpen(true);
                }}
              >
                📆
              </button>
              <button type="button" className="btn btn-light" onClick={() => setSaveAsOpen(true)}>
                💾
              </button>
              <button type="button" className="btn btn-light" onClick={handleExport}>
                📤
              </button>
              <button type="button" className="btn btn-light" onClick={() => setExtendOpen(true)}>
                ➕
              </button>
              <button type="button" className="btn btn-danger" onClick={handleDeletePlan}>
                Delete Plan
              </button>
            </div>

            <div className="plan-options-grain">
              <span>View Grain</span>
              <div>
                <button type="button" className="btn btn-light day" onClick={() => setGrain("day")}>
                  Day
                </button>
                <button
                  type="button"
                  className="btn btn-light interval"
                  onClick={() => setGrain("interval")}
                  disabled={isBackOffice}
                >
                  Interval
                </button>
              </div>
            </div>

            <div className="plan-options-whatif">
              <h5>Live What-If</h5>
              <div className="plan-whatif-grid">
                <label>
                  AHT/SUT Delta (%)
                  <input
                    className="input"
                    type="number"
                    value={whatIf.ahtDelta}
                    step={5}
                    onChange={(event) => setWhatIf((prev) => ({ ...prev, ahtDelta: Number(event.target.value) }))}
                  />
                </label>
                <label>
                  Shrink Delta (%)
                  <input
                    className="input"
                    type="number"
                    value={whatIf.shrinkDelta}
                    step={1}
                    onChange={(event) => setWhatIf((prev) => ({ ...prev, shrinkDelta: Number(event.target.value) }))}
                  />
                </label>
                <label>
                  Attrition Delta (HC)
                  <input
                    className="input"
                    type="number"
                    value={whatIf.attrDelta}
                    step={1}
                    onChange={(event) => setWhatIf((prev) => ({ ...prev, attrDelta: Number(event.target.value) }))}
                  />
                </label>
                <label>
                  Forecast Delta (%)
                  <input
                    className="input"
                    type="number"
                    value={whatIf.volDelta}
                    step={1}
                    onChange={(event) => setWhatIf((prev) => ({ ...prev, volDelta: Number(event.target.value) }))}
                  />
                </label>
              </div>
              <label className="plan-whatif-checkbox">
                <input
                  type="checkbox"
                  checked={whatIf.backlogCarryover}
                  onChange={(event) => setWhatIf((prev) => ({ ...prev, backlogCarryover: event.target.checked }))}
                />
                Apply backlog to next week (Back Office)
              </label>
              <div className="plan-whatif-actions">
                <button type="button" className="btn btn-primary" onClick={handleApplyWhatIf}>
                  Apply What-If
                </button>
                <button type="button" className="btn btn-light" onClick={handleClearWhatIf}>
                  Clear
                </button>
              </div>
            </div>
          </div>

          {viewModalOpen ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal ws-modal-sm">
                <div className="ws-modal-header">
                  <h3>View Plan Between Weeks</h3>
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setViewModalOpen(false)}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
                <div className="ws-modal-body">
                  <div className="plan-modal-grid">
                    <label>
                      From
                      <input
                        className="input"
                        type="date"
                        value={viewDraftFrom}
                        onChange={(event) => setViewDraftFrom(event.target.value)}
                      />
                    </label>
                    <label>
                      To
                      <input
                        className="input"
                        type="date"
                        value={viewDraftTo}
                        onChange={(event) => setViewDraftTo(event.target.value)}
                      />
                    </label>
                  </div>
                </div>
                <div className="ws-modal-footer">
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => {
                      setViewModalOpen(false);
                      setViewFrom(viewDraftFrom);
                      setViewTo(viewDraftTo);
                      setMessage("View range applied.");
                    }}
                  >
                    Save
                  </button>
                  <button type="button" className="btn btn-light" onClick={() => setViewModalOpen(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {saveAsOpen ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal ws-modal-sm">
                <div className="ws-modal-header">
                  <h3>Save Plan As</h3>
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setSaveAsOpen(false)}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
                <div className="ws-modal-body">
                  <label>
                    Plan Name
                    <input
                      className="input"
                      value={saveAsName}
                      onChange={(event) => setSaveAsName(event.target.value)}
                    />
                  </label>
                  {saveAsMessage ? <div className="plan-message">{saveAsMessage}</div> : null}
                </div>
                <div className="ws-modal-footer">
                  <button type="button" className="btn btn-primary" onClick={handleSaveAs}>
                    Save
                  </button>
                  <button type="button" className="btn btn-light" onClick={() => setSaveAsOpen(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {extendOpen ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal ws-modal-sm">
                <div className="ws-modal-header">
                  <h3>Extend Plan</h3>
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setExtendOpen(false)}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
                <div className="ws-modal-body">
                  <label>
                    Add Weeks
                    <input
                      className="input"
                      type="number"
                      min={1}
                      value={extendWeeks}
                      onChange={(event) => setExtendWeeks(Number(event.target.value))}
                    />
                  </label>
                  {extendMessage ? <div className="plan-message">{extendMessage}</div> : null}
                </div>
                <div className="ws-modal-footer">
                  <button type="button" className="btn btn-primary" onClick={handleExtendPlan}>
                    Extend
                  </button>
                  <button type="button" className="btn btn-light" onClick={() => setExtendOpen(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {switchOpen ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal ws-modal-sm">
                <div className="ws-modal-header">
                  <h3>Switch Plan</h3>
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setSwitchOpen(false)}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
                <div className="ws-modal-body">
                  <label>
                    Select Plan
                    <select
                      className="select"
                      value={switchPlanId}
                      onChange={(event) => {
                        const value = event.target.value;
                        setSwitchPlanId(value ? Number(value) : "");
                      }}
                      disabled={!scopeOptions.length}
                    >
                      {scopeOptions.map((option) => (
                        <option key={option.id} value={option.id}>
                          {option.label || `${option.plan_name || "Plan"} (id ${option.id})`}
                        </option>
                      ))}
                    </select>
                  </label>
                  {!scopeOptions.length ? <div className="plan-message">No other plans in scope.</div> : null}
                </div>
                <div className="ws-modal-footer">
                  <button type="button" className="btn btn-primary" onClick={handleSwitchPlan} disabled={!switchPlanId}>
                    Switch
                  </button>
                  <button type="button" className="btn btn-light" onClick={() => setSwitchOpen(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {compareOpen ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal ws-modal-sm">
                <div className="ws-modal-header">
                  <h3>Compare Plans</h3>
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setCompareOpen(false)}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
                <div className="ws-modal-body">
                  <label>
                    Select Plan
                    <select
                      className="select"
                      value={comparePlanId}
                      onChange={(event) => {
                        const value = event.target.value;
                        setComparePlanId(value ? Number(value) : "");
                      }}
                      disabled={!scopeOptions.length}
                    >
                      {scopeOptions.map((option) => (
                        <option key={option.id} value={option.id}>
                          {option.label || `${option.plan_name || "Plan"} (id ${option.id})`}
                        </option>
                      ))}
                    </select>
                  </label>
                  {!scopeOptions.length ? <div className="plan-message">No other plans in scope.</div> : null}
                  {compareWarning ? <div className="plan-message">{compareWarning}</div> : null}
                </div>
                <div className="ws-modal-footer">
                  <button type="button" className="btn btn-primary" onClick={handleCompare} disabled={!comparePlanId}>
                    Compare
                  </button>
                  <button type="button" className="btn btn-light" onClick={() => setCompareOpen(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {compareResultOpen && compareResult ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal">
                <div className="ws-modal-header">
                  <h3>Comparison Result</h3>
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setCompareResultOpen(false)}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
                <div className="ws-modal-body plan-compare">
                  <div>
                    <h4>Current Plan</h4>
                    <DataTable data={compareResult.current} emptyLabel="No data." />
                  </div>
                  <div>
                    <h4>Comparison Plan</h4>
                    <DataTable data={compareResult.compare} emptyLabel="No data." />
                  </div>
                  <div>
                    <h4>Delta (Current - Comparison)</h4>
                    <DataTable data={compareResult.delta} emptyLabel="No data." />
                  </div>
                </div>
                <div className="ws-modal-footer">
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setCompareResultOpen(false)}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {nhModalOpen ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal">
                <div className="ws-modal-header">
                  <h3>Add New Hire Class</h3>
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setNhModalOpen(false)}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
                <div className="ws-modal-body plan-modal-form">
                  <div className="plan-modal-grid">
                    <label>
                      Emp Type
                      <select
                        className="select"
                        value={nhForm.emp_type}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, emp_type: event.target.value }))}
                      >
                        <option value="full-time">Full-time</option>
                        <option value="part-time">Part-time</option>
                      </select>
                    </label>
                    <label>
                      Status
                      <select
                        className="select"
                        value={nhForm.status}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, status: event.target.value }))}
                      >
                        <option value="tentative">Tentative</option>
                        <option value="confirmed">Confirmed</option>
                      </select>
                    </label>
                    <label>
                      Class Type
                      <select
                        className="select"
                        value={nhForm.class_type}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, class_type: event.target.value }))}
                      >
                        {(nhClassOptions.class_types?.length
                          ? nhClassOptions.class_types
                          : [
                              { label: "Ramp-Up", value: "ramp-up" },
                              { label: "Backfill", value: "backfill" }
                            ]
                        ).map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Class Level
                      <select
                        className="select"
                        value={nhForm.class_level}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, class_level: event.target.value }))}
                      >
                        {(nhClassOptions.class_levels?.length
                          ? nhClassOptions.class_levels
                          : [
                              { label: "Trainee", value: "trainee" },
                              { label: "New Agent", value: "new-agent" },
                              { label: "Tenured Agent", value: "tenured" },
                              { label: "Senior Agent", value: "senior-agent" },
                              { label: "SME", value: "sme" },
                              { label: "Cross-Skill", value: "cross-skill" }
                            ]
                        ).map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Grads Needed
                      <input
                        className="input"
                        type="number"
                        value={nhForm.grads_needed}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, grads_needed: Number(event.target.value) }))}
                      />
                    </label>
                    <label>
                      Billable HC
                      <input
                        className="input"
                        type="number"
                        value={nhForm.billable_hc}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, billable_hc: Number(event.target.value) }))}
                      />
                    </label>
                    <label>
                      Training Weeks
                      <input
                        className="input"
                        type="number"
                        value={nhForm.training_weeks}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, training_weeks: Number(event.target.value) }))}
                      />
                    </label>
                    <label>
                      Nesting Weeks
                      <input
                        className="input"
                        type="number"
                        value={nhForm.nesting_weeks}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, nesting_weeks: Number(event.target.value) }))}
                      />
                    </label>
                    <label>
                      Induction Start
                      <input
                        className="input"
                        type="date"
                        value={nhForm.induction_start}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, induction_start: event.target.value }))}
                      />
                    </label>
                    <label>
                      Training Start
                      <input
                        className="input"
                        type="date"
                        value={nhForm.training_start}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, training_start: event.target.value }))}
                      />
                    </label>
                    <label>
                      Training End
                      <input
                        className="input"
                        type="date"
                        value={nhForm.training_end}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, training_end: event.target.value }))}
                      />
                    </label>
                    <label>
                      Nesting Start
                      <input
                        className="input"
                        type="date"
                        value={nhForm.nesting_start}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, nesting_start: event.target.value }))}
                      />
                    </label>
                    <label>
                      Nesting End
                      <input
                        className="input"
                        type="date"
                        value={nhForm.nesting_end}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, nesting_end: event.target.value }))}
                      />
                    </label>
                    <label>
                      Production Start
                      <input
                        className="input"
                        type="date"
                        value={nhForm.production_start}
                        onChange={(event) => setNhForm((prev) => ({ ...prev, production_start: event.target.value }))}
                      />
                    </label>
                  </div>
                </div>
                <div className="ws-modal-footer">
                  <button type="button" className="btn btn-primary" onClick={handleNewHireSave}>
                    Save
                  </button>
                  <button type="button" className="btn btn-light" onClick={() => setNhModalOpen(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {nhDetailsOpen ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal">
                <div className="ws-modal-header">
                  <h3>Training Details</h3>
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setNhDetailsOpen(false)}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
                <div className="ws-modal-body">
                  <DataTable data={nhClasses} emptyLabel="No classes yet." />
                </div>
                <div className="ws-modal-footer">
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setNhDetailsOpen(false)}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {rosterModalOpen ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal">
                <div className="ws-modal-header">
                  <h3>Add Employee</h3>
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setRosterModalOpen(false)}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
                <div className="ws-modal-body plan-modal-form">
                  <div className="plan-modal-grid">
                    {ROSTER_COLUMNS.map((col) => (
                      <label key={col.id}>
                        {col.label}
                        <input
                          className="input"
                          value={rosterForm[col.id] ?? ""}
                          onChange={(event) =>
                            setRosterForm((prev) => ({
                              ...prev,
                              [col.id]: event.target.value
                            }))
                          }
                        />
                      </label>
                    ))}
                  </div>
                </div>
                <div className="ws-modal-footer">
                  <button type="button" className="btn btn-primary" onClick={handleRosterAdd}>
                    Save
                  </button>
                  <button type="button" className="btn btn-light" onClick={() => setRosterModalOpen(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : null}
        </>
      ) : null}
    </AppShell>
  );
}
