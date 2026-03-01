"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AppShell from "../_components/AppShell";
import DataTable from "../_components/DataTable";
import EditableTable from "../_components/EditableTable";
import MultiSelect from "../_components/MultiSelect";
import { useGlobalLoader } from "../_components/GlobalLoader";
import { useToast } from "../_components/ToastProvider";
import { apiGet, apiPost } from "../../lib/api";
import { parseExcelFile } from "../../lib/excel";

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
  rebalanceEnabled: boolean;
  xskillEfficiencyPct: number;
  xskillMaxLendPct: number;
  lockCriticalTeams: string[];
};

type SelectOption = {
  label: string;
  value: string;
};

type WorkforcePreview = {
  org_hiring?: {
    required_fte?: number;
    supply_fte?: number;
    estimated_shortfall_fte?: number;
    estimated_surplus_fte?: number;
    potential_hiring_saved_fte?: number;
    net_hiring_fte?: number;
    post_rebalance_shortfall_fte?: number;
    cross_skill_efficiency_pct?: number;
    max_lend_pct?: number;
    locked_critical_scope_count?: number;
  };
  scope_balance?: Array<Record<string, any>>;
  rebalancing?: Array<Record<string, any>>;
};

type PlanComputeResponse = {
  status: "ready" | "running" | "failed" | "missing";
  job?: Record<string, any>;
  data?: {
    grain?: string;
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
  { key: "xskill", label: "Borrow / Lend" },
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
const NON_GRAIN_TABLES = new Set(["emp", "bulk_files", "notes"]);

const DATE_KEY_RE = /^\d{4}-\d{2}-\d{2}$/;
const DATE_PREFIX_RE = /^\d{4}-\d{2}-\d{2}/;
const FORECAST_BASE = (() => {
  if (typeof window !== "undefined") {
    return process.env.NEXT_PUBLIC_BROWSER_FORECAST_URL || "";
  }
  return (
    process.env.NEXT_PUBLIC_FORECAST_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8080"
  );
})();

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

const ROSTER_BULK_TEMPLATE_HEADERS = [
  "BRID",
  "Name",
  "Class Ref",
  "Work Stat",
  "Role",
  "FT/PT Stat",
  "FT/PT Hou",
  "Current St",
  "Training St",
  "Training E",
  "Nesting St",
  "Nesting Er",
  "Production",
  "Terminate",
  "Team Lead",
  "AVP",
  "Business A",
  "Sub Busin",
  "LOB",
  "LOA Date",
  "Back from",
  "Site"
];

const ROSTER_BULK_TEMPLATE_ROWS = [
  {
    BRID: "IN0001",
    Name: "Asha Rao",
    "Class Ref": "NH-2024-1",
    "Work Stat": "Production",
    Role: "Agent",
    "FT/PT Stat": "Full-time",
    "FT/PT Hou": 40,
    "Current St": "Production",
    "Training St": "2024-01-08",
    "Training E": "2024-01-22",
    "Nesting St": "2024-01-23",
    "Nesting Er": "2024-01-29",
    Production: "2024-01-30",
    Terminate: "",
    "Team Lead": "Priyanka",
    AVP: "Anil Sharma",
    "Business A": "Bereavement",
    "Sub Busin": "Bereavement",
    LOB: "Back Office",
    "LOA Date": "",
    "Back from": "",
    Site: "Candor Techspace, Noida"
  },
  {
    BRID: "UK0002",
    Name: "Alex Doe",
    "Class Ref": "",
    "Work Stat": "Production",
    Role: "CSA",
    "FT/PT Stat": "Part-time",
    "FT/PT Hou": 20,
    "Current St": "Production",
    "Training St": "",
    "Training E": "",
    "Nesting St": "",
    "Nesting Er": "",
    Production: "",
    Terminate: "",
    "Team Lead": "Chris Lee",
    AVP: "Samantha",
    "Business A": "Retail",
    "Sub Busin": "Cards",
    LOB: "Voice",
    "LOA Date": "",
    "Back from": "",
    Site: "Manchester"
  }
];

const ROSTER_HEADER_ALIASES: Record<string, string> = {
  brid: "brid",
  name: "name",
  classref: "class_ref",
  classrefe: "class_ref",
  workstat: "work_status",
  workstatus: "work_status",
  role: "role",
  ftptstat: "ftpt_status",
  ftptstatus: "ftpt_status",
  ftpthou: "ftpt_hours",
  ftpthours: "ftpt_hours",
  currentst: "current_status",
  currentstatus: "current_status",
  trainingst: "training_start",
  trainingstart: "training_start",
  traininge: "training_end",
  trainingend: "training_end",
  nestingst: "nesting_start",
  nestingstart: "nesting_start",
  nestinger: "nesting_end",
  nestingend: "nesting_end",
  production: "production_start",
  productionstart: "production_start",
  terminate: "terminate_date",
  terminatedate: "terminate_date",
  teamlead: "team_leader",
  teamleader: "team_leader",
  businessa: "biz_area",
  businessarea: "biz_area",
  subbusin: "sub_biz_area",
  subbusinessarea: "sub_biz_area",
  lob: "lob",
  loadate: "loa_date",
  backfrom: "back_from_loa_date",
  backfromloa: "back_from_loa_date",
  site: "site"
};

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
  backlogCarryover: true,
  rebalanceEnabled: true,
  xskillEfficiencyPct: 85,
  xskillMaxLendPct: 60,
  lockCriticalTeams: []
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
  return DATE_KEY_RE.test(key) || DATE_PREFIX_RE.test(String(key || ""));
}

function toDate(value?: string) {
  if (!value) return null;
  const raw = String(value).trim();
  if (!raw) return null;
  if (DATE_PREFIX_RE.test(raw)) {
    const dt = new Date(`${raw.slice(0, 10)}T00:00:00`);
    return Number.isNaN(dt.getTime()) ? null : dt;
  }
  const dt = new Date(raw);
  if (Number.isNaN(dt.getTime())) return null;
  return dt;
}

function addWeeksToDate(value: string, weeks: number) {
  const base = toDate(value);
  if (!base) return value;
  const next = new Date(base);
  next.setDate(next.getDate() + Math.max(0, Number(weeks || 0)) * 7);
  return next.toISOString().slice(0, 10);
}

function tableSuffix(grain: string, intervalDate?: string) {
  if (grain === "week") return "";
  if (grain === "interval") {
    const stamp = intervalDate || "";
    return stamp ? `_interval_${stamp}` : "_interval";
  }
  return `_${grain}`;
}

function toIsoDate(value?: Date | null) {
  if (!value) return "";
  const iso = value.toISOString();
  return iso.slice(0, 10);
}

function getYesterdayDate() {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  d.setHours(0, 0, 0, 0);
  return d;
}

function startOfWeekMonday(value: Date) {
  const d = new Date(value);
  const day = d.getDay(); // 0=Sun..6=Sat
  const offset = (day + 6) % 7;
  d.setDate(d.getDate() - offset);
  d.setHours(0, 0, 0, 0);
  return d;
}

function targetDateForGrain(grain: string) {
  const yesterday = getYesterdayDate();
  if (grain === "week") return startOfWeekMonday(yesterday);
  if (grain === "month") return new Date(yesterday.getFullYear(), yesterday.getMonth(), 1);
  return yesterday;
}

function collectDateKeys(rows: Array<Record<string, any>>) {
  const keys = new Map<string, Date>();
  rows.forEach((row) => {
    Object.keys(row || {}).forEach((key) => {
      if (!isDateKey(key)) return;
      const dt = toDate(key);
      if (!dt) return;
      if (!keys.has(key)) keys.set(key, dt);
    });
  });
  return Array.from(keys.entries())
    .sort((a, b) => a[1].getTime() - b[1].getTime())
    .map(([key]) => key);
}

function closestDateKey(keys: string[], target: Date) {
  if (!keys.length) return null;
  let bestKey = keys[0];
  let bestDiff = Number.POSITIVE_INFINITY;
  keys.forEach((key) => {
    const dt = toDate(key);
    if (!dt) return;
    const diff = Math.abs(dt.getTime() - target.getTime());
    if (diff < bestDiff) {
      bestDiff = diff;
      bestKey = key;
    }
  });
  return bestKey;
}

function escapeAttrValue(value: string) {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
    return CSS.escape(value);
  }
  return value.replace(/["\\]/g, "\\$&");
}

function scrollTableToKey(wrapper: HTMLDivElement | null, key?: string | null) {
  if (!wrapper || !key) return;
  const selector = `th[data-col="${escapeAttrValue(key)}"]`;
  const header = wrapper.querySelector<HTMLElement>(selector);
  if (!header) return;
  const left = Math.max(0, header.offsetLeft - 60);
  wrapper.scrollTo({ left, behavior: "smooth" });
}

function applyAutoDates(input: Record<string, any>) {
  const next = { ...input };
  const ts = toDate(next.training_start);
  const ns = toDate(next.nesting_start);
  const ps = toDate(next.production_start);
  const tw = Number(next.training_weeks || 0);
  const nw = Number(next.nesting_weeks || 0);

  let trainingEnd = toDate(next.training_end);
  let nestingStart = ns;
  let nestingEnd = toDate(next.nesting_end);
  let productionStart = ps;

  if (ts && tw > 0 && !trainingEnd) {
    const te = new Date(ts);
    te.setDate(te.getDate() + tw * 7 - 1);
    trainingEnd = te;
  }
  if (!nestingStart && trainingEnd) {
    const nsNext = new Date(trainingEnd);
    nsNext.setDate(nsNext.getDate() + 1);
    nestingStart = nsNext;
  }
  if (nestingStart && nw > 0 && !nestingEnd) {
    const ne = new Date(nestingStart);
    ne.setDate(ne.getDate() + nw * 7 - 1);
    nestingEnd = ne;
  }
  if (!productionStart && nestingEnd) {
    const psNext = new Date(nestingEnd);
    psNext.setDate(psNext.getDate() + 1);
    productionStart = psNext;
  }

  next.training_end = next.training_end || toIsoDate(trainingEnd);
  next.nesting_start = next.nesting_start || toIsoDate(nestingStart);
  next.nesting_end = next.nesting_end || toIsoDate(nestingEnd);
  next.production_start = next.production_start || toIsoDate(productionStart);
  return next;
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
      const aliasKeys = Object.keys(ROSTER_HEADER_ALIASES).filter(
        (key) => ROSTER_HEADER_ALIASES[key] === col.id
      );
      const originalKey =
        normalizedKeys.get(normalized) ||
        normalizedKeys.get(normalizedLabel) ||
        normalizedKeys.get(normalizeKey(col.label.replace(" ", ""))) ||
        aliasKeys.map((key) => normalizedKeys.get(key)).find(Boolean);
      out[col.id] = originalKey ? row[originalKey] : "";
    });
    return out;
  });
}

function hasAnyValue(value: any) {
  return String(value ?? "").trim() !== "";
}

function normalizeRosterUploadRows(rows: Array<Record<string, any>>) {
  const mapped = toRosterRows(rows);
  return mapped.filter((row) => Object.values(row || {}).some((value) => hasAnyValue(value)));
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
    return parseExcelFile(file);
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
  const [computeStatus, setComputeStatus] = useState<"idle" | "running">("idle");
  const pollRef = useRef<number | null>(null);
  const computePollRef = useRef<number | null>(null);
  const computeQueueRef = useRef<Promise<void>>(Promise.resolve());
  const tablesReqRef = useRef(0);
  const [grain, setGrain] = useState("week");
  const [intervalDate, setIntervalDate] = useState("");
  const [upperCollapsed, setUpperCollapsed] = useState(false);
  const [optionsOpen, setOptionsOpen] = useState(false);
  const [debugOpen, setDebugOpen] = useState(false);
  const [debugInfo, setDebugInfo] = useState<any>(null);
  const [debugError, setDebugError] = useState("");
  const [lastRecomputeCause, setLastRecomputeCause] = useState<string[]>([]);
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
  const [xskillPreview, setXskillPreview] = useState<WorkforcePreview | null>(null);
  const [xskillLoading, setXskillLoading] = useState(false);
  const [criticalTeamOptions, setCriticalTeamOptions] = useState<SelectOption[]>([]);
  const [noteText, setNoteText] = useState("");
  const [rosterTab, setRosterTab] = useState<"roster" | "bulk">("roster");
  const rosterSnapshotRef = useRef<Array<Record<string, any>> | null>(null);
  const upperTableRef = useRef<HTMLDivElement | null>(null);
  const lowerTableRef = useRef<HTMLDivElement | null>(null);
  const prevGrainRef = useRef<string>("week");
  const lastAutoIntervalRef = useRef<string>("");

  const [nhClasses, setNhClasses] = useState<Array<Record<string, any>>>([]);
  const [nhClassOptions, setNhClassOptions] = useState<{
    class_types?: Array<{ label: string; value: string }>;
    class_levels?: Array<{ label: string; value: string }>;
  }>({});
  const [nhModalOpen, setNhModalOpen] = useState(false);
  const [nhDetailsOpen, setNhDetailsOpen] = useState(false);
  const [nhDetailsSelection, setNhDetailsSelection] = useState<Set<string>>(new Set());
  const [nhMessage, setNhMessage] = useState("");
  const [nhForm, setNhForm] = useState<Record<string, any>>(NH_FORM_DEFAULT);
  const [rosterModalOpen, setRosterModalOpen] = useState(false);
  const [rosterSelected, setRosterSelected] = useState<Set<number>>(new Set());
  const [rosterAction, setRosterAction] = useState<
    "" | "class" | "ftpt" | "loa" | "back" | "term" | "remove" | "tp"
  >("");
  const [rosterActionValue, setRosterActionValue] = useState<Record<string, any>>({});
  const rosterSaveTimeoutRef = useRef<number | null>(null);
  const pendingRosterSaveRef = useRef<Array<Record<string, any>> | null>(null);
  const [rosterSaveTick, setRosterSaveTick] = useState(0);
  const lastRosterSaveAtRef = useRef<number | null>(null);
  const rosterSavingRef = useRef(false);
  const lastRosterSavedRowsRef = useRef<Array<Record<string, any>>>([]);
  const lastBulkSaveAtRef = useRef<number | null>(null);
  const bulkSavingRef = useRef(false);
  const lastBulkSavedRowsRef = useRef<Array<Record<string, any>>>([]);
  const [tpTab, setTpTab] = useState<"tp-transfer" | "tp-promo" | "tp-both">("tp-transfer");
  const [tpForm, setTpForm] = useState<Record<string, any>>({
    biz_area: "",
    sub_biz_area: "",
    lob: "",
    site: "",
    transfer_type: "perm",
    new_class: false,
    class_ref: "",
    date_from: "",
    date_to: ""
  });
  const [promoForm, setPromoForm] = useState<Record<string, any>>({
    promo_type: "perm",
    role: "",
    date_from: "",
    date_to: ""
  });
  const [twpForm, setTwpForm] = useState<Record<string, any>>({
    biz_area: "",
    sub_biz_area: "",
    lob: "",
    site: "",
    transfer_type: "perm",
    new_class: false,
    class_ref: "",
    role: "",
    date_from: "",
    date_to: ""
  });
  const [tpOptions, setTpOptions] = useState<{
    business_areas: string[];
    sub_business_areas: string[];
    channels: string[];
    sites: string[];
  }>({ business_areas: [], sub_business_areas: [], channels: [], sites: [] });
  const [rosterForm, setRosterForm] = useState<Record<string, any>>({
    brid: "",
    name: "",
    ftpt_status: "Full-time",
    role: "Agent",
    production_start: "",
    team_leader: "",
    avp: "",
    ftpt_hours: "",
    work_status: "Production",
    current_status: "Production",
    terminate_date: "",
    class_ref: "",
    training_start: "",
    training_end: "",
    nesting_start: "",
    nesting_end: "",
    site: "",
    biz_area: "",
    sub_biz_area: "",
    lob: "",
    loa_date: "",
    back_from_loa_date: ""
  });

  const isRollup = Boolean(rollupBa);
  const planChannel = String(planMeta?.channel || "").split(",")[0].trim().toLowerCase();
  const isBackOffice = ["back office", "backoffice", "bo"].includes(planChannel);
  const isLocked = String(planMeta?.status || "").toLowerCase() === "history";
  const activeConfig = useMemo(() => TABLES.find((t) => t.key === activeTab) ?? TABLES[0], [activeTab]);
  const activeRows = tables[activeConfig.key] ?? [];
  const filteredRows = useMemo(
    () => filterRowsByDateRange(activeRows, viewFrom, viewTo),
    [activeRows, viewFrom, viewTo]
  );
  const filteredUpperRows = useMemo(
    () => filterRowsByDateRange(upperRows, viewFrom, viewTo),
    [upperRows, viewFrom, viewTo]
  );
  const rebalanceRows = useMemo(() => xskillPreview?.rebalancing ?? [], [xskillPreview?.rebalancing]);
  const scopeBalanceRows = useMemo(() => xskillPreview?.scope_balance ?? [], [xskillPreview?.scope_balance]);
  const orgHiring = xskillPreview?.org_hiring ?? {};
  const lockCriticalTeamChoices = useMemo(() => {
    const merged = new Map<string, string>();
    criticalTeamOptions.forEach((option) => merged.set(option.value, option.label));
    (whatIf.lockCriticalTeams ?? []).forEach((value) => {
      const key = String(value || "").trim();
      if (key && !merged.has(key)) merged.set(key, key);
    });
    return Array.from(merged.entries()).map(([value, label]) => ({ value, label }));
  }, [criticalTeamOptions, whatIf.lockCriticalTeams]);
  const canEdit = EDITABLE_TABLES.has(activeConfig.key) && grain === "week" && !isRollup && !isLocked;
  const editableColumns = useMemo(() => {
    const cols = new Set<string>();
    filteredRows.forEach((row) => {
      Object.keys(row || {}).forEach((key) => cols.add(key));
    });
    return Array.from(cols).filter((key) => key.toLowerCase() !== "metric");
  }, [filteredRows]);

  const planName = planMeta?.plan_name || (planId ? `Plan ${planId}` : "Plan Detail");
  const planStatus = planMeta?.status || (isRollup ? "rollup" : "draft");
  const breadcrumbBa = planMeta?.business_area || rollupBa || "";
  const breadcrumbText = breadcrumbBa
    ? `CAP-CONNECT / Planning Workspace / ${breadcrumbBa}`
    : "CAP-CONNECT / Planning Workspace";
  const breadcrumbLinks = breadcrumbBa
    ? { [breadcrumbBa]: `/plan/ba/${encodeURIComponent(breadcrumbBa)}` }
    : undefined;
  const intervalOptions = useMemo(
    () => buildIntervalOptions(planMeta?.start_week, planMeta?.end_week),
    [planMeta?.start_week, planMeta?.end_week]
  );
  const classRefOptions = useMemo(() => {
    const scopeLabel = [
      planMeta?.business_area,
      planMeta?.sub_business_area,
      planMeta?.channel,
      planMeta?.site
    ]
      .filter(Boolean)
      .join(" > ");
    return (nhClasses || [])
      .map((row) => {
        const cref = String(row?.class_reference ?? row?.class_ref ?? "").trim();
        if (!cref) return null;
        const labelScope = [
          row?.business_area,
          row?.sub_business_area,
          row?.channel,
          row?.site
        ]
          .filter(Boolean)
          .join(" > ");
        const label = labelScope ? `${cref} — ${labelScope}` : `${cref}${scopeLabel ? ` — ${scopeLabel}` : ""}`;
        return { label, value: cref };
      })
      .filter(Boolean) as Array<{ label: string; value: string }>;
  }, [nhClasses, planMeta?.business_area, planMeta?.channel, planMeta?.site, planMeta?.sub_business_area]);
  const classRefScopeMap = useMemo(() => {
    const map = new Map<string, string>();
    (nhClasses || []).forEach((row) => {
      const cref = String(row?.class_reference ?? row?.class_ref ?? "").trim();
      if (!cref) return;
      const scope = [
        row?.business_area,
        row?.sub_business_area,
        row?.channel,
        row?.site
      ]
        .filter(Boolean)
        .join(" > ");
      if (scope) map.set(cref, scope);
    });
    return map;
  }, [nhClasses]);

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
    try {
      const res = await apiGet<{ rows?: Array<Record<string, any>> }>(
        `/api/planning/plan/table?plan_id=${planId}&name=upper`
      );
      setUpperRows(res.rows ?? []);
    } catch {
      setUpperRows([]);
    }
  }, [planId]);

  const loadTablesForGrain = useCallback(
    async (targetGrain: string, targetInterval?: string) => {
      if (!planId) return;
      const reqId = tablesReqRef.current + 1;
      tablesReqRef.current = reqId;
      const suffix = tableSuffix(targetGrain, targetInterval);
      const entries = await Promise.all(
        PERSIST_TABLE_KEYS.map(async (table) => {
          try {
            const effectiveName = NON_GRAIN_TABLES.has(table) ? table : `${table}${suffix}`;
            const res = await apiGet<{ rows?: Array<Record<string, any>> }>(
              `/api/planning/plan/table?plan_id=${planId}&name=${effectiveName}`
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
      if (tablesReqRef.current === reqId) {
        setTables((prev) => {
          const merged = { ...next };
          const lastSaveAt = lastRosterSaveAtRef.current ?? 0;
          const keepRecentRoster = Date.now() - lastSaveAt < 7000 || rosterSavingRef.current;
          if (keepRecentRoster && (next.emp?.length ?? 0) === 0) {
            if ((prev?.emp?.length ?? 0) > 0) {
              merged.emp = prev.emp ?? [];
            } else if (lastRosterSavedRowsRef.current.length > 0) {
              merged.emp = lastRosterSavedRowsRef.current;
            }
          }
          const lastBulkSaveAt = lastBulkSaveAtRef.current ?? 0;
          const keepRecentBulk = Date.now() - lastBulkSaveAt < 7000 || bulkSavingRef.current;
          if (keepRecentBulk && (next.bulk_files?.length ?? 0) === 0) {
            if ((prev?.bulk_files?.length ?? 0) > 0) {
              merged.bulk_files = prev.bulk_files ?? [];
            } else if (lastBulkSavedRowsRef.current.length > 0) {
              merged.bulk_files = lastBulkSavedRowsRef.current;
            }
          }
          return merged;
        });
      }
      try {
        const res = await apiGet<{ rows?: Array<Record<string, any>> }>(
          `/api/planning/plan/table?plan_id=${planId}&name=upper${suffix}`
        );
        if (tablesReqRef.current === reqId) {
          setUpperRows(res.rows ?? []);
        }
      } catch {
        if (tablesReqRef.current === reqId) {
          setUpperRows([]);
        }
      }
    },
    [planId]
  );

  const loadTablesWithLoader = useCallback(async () => {
    if (!planId || isRollup) return;
    setLoading(true);
    try {
      await loadTablesForGrain(grain, intervalDate);
    } finally {
      setLoading(false);
    }
  }, [grain, intervalDate, isRollup, loadTablesForGrain, planId, setLoading]);

  const computePlanTables = useCallback(
    async (options?: { wait?: boolean; silent?: boolean }): Promise<void> => {
    if (!planId) return;
    const silent = Boolean(options?.silent);
    if (!silent) {
      setLoading(true);
    }
    const waitForReady = Boolean(options?.wait);
    const payload: Record<string, any> = {
      plan_id: planId,
      grain,
      persist: true,
      prefetch: false
    };
    if (grain === "interval" && intervalDate) {
      payload.interval_date = intervalDate;
    }

    const run = async (): Promise<void> => {
      const res = await apiPost<PlanComputeResponse>("/api/planning/plan/detail/compute", payload);
        if (res.status === "ready") {
          const resGrain = String(res.data?.grain || "").toLowerCase();
          if (resGrain && resGrain !== String(grain).toLowerCase()) {
            return;
          }
          const depChanged = Array.isArray(res.job?.dep_changed) ? res.job.dep_changed : [];
          if (depChanged.length) {
            setLastRecomputeCause(depChanged.map((val: any) => String(val)));
          }
          if (res.data?.tables) {
            setTables((prev) => {
              const next = { ...prev, ...(res.data?.tables ?? {}) };
              if ((prev?.emp?.length ?? 0) > 0 && (next?.emp?.length ?? 0) === 0) {
                next.emp = prev.emp ?? [];
              }
              return next;
            });
          }
        setUpperRows(res.data?.upper ?? []);
        setComputeStatus("idle");
        if (!silent) {
          setLoading(false);
        }
        return;
      }
      if (res.status === "missing") {
        setComputeStatus("idle");
        if (!silent) {
          setLoading(false);
        }
        throw new Error("Plan not found.");
      }
      if (res.status === "failed") {
        setComputeStatus("idle");
        if (!silent) {
          setLoading(false);
        }
        const detail = res.job?.error ? ` ${res.job.error}` : "";
        throw new Error(`Plan detail calculations failed.${detail}`);
      }
      if (res.status === "running") {
        setComputeStatus("running");
        if (!silent) {
          setLoading(true);
        }
        if (waitForReady) {
          await new Promise((resolve) => setTimeout(resolve, 1200));
          return computePlanTables({ wait: true, silent });
        }
        if (computePollRef.current) {
          window.clearTimeout(computePollRef.current);
        }
        computePollRef.current = window.setTimeout(() => {
          void computePlanTables(options);
        }, 1200);
      }
    };

    const task: Promise<void> = computeQueueRef.current.then(run);
    computeQueueRef.current = task.catch(() => {});
    return task;
  }, [grain, intervalDate, planId, setLoading]);

  const warmPlanCompute = useCallback(async () => {
    if (!planId || isRollup) return;
    try {
      await computePlanTables();
    } catch (error: any) {
      notify("error", error?.message || "Plan detail calculations failed.");
    }
  }, [computePlanTables, isRollup, notify, planId]);

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

  const loadTpOptions = useCallback(
    async (ba?: string, sba?: string) => {
      try {
        const query = new URLSearchParams();
        if (ba) query.set("ba", ba);
        if (sba) query.set("sba", sba);
        const res = await apiGet<{
          business_areas?: string[];
          sub_business_areas?: string[];
          channels?: string[];
          sites?: string[];
        }>(`/api/forecast/headcount/options?${query.toString()}`);
        setTpOptions({
          business_areas: res.business_areas ?? [],
          sub_business_areas: res.sub_business_areas ?? [],
          channels: res.channels ?? [],
          sites: res.sites ?? []
        });
      } catch {
        setTpOptions({ business_areas: [], sub_business_areas: [], channels: [], sites: [] });
      }
    },
    []
  );

  const loadCriticalTeamOptions = useCallback(async () => {
    const params = new URLSearchParams();
    if (planMeta?.business_area) params.set("ba", String(planMeta.business_area));
    if (planMeta?.sub_business_area) params.set("sba", String(planMeta.sub_business_area));
    if (planMeta?.channel) {
      const ch = String(planMeta.channel)
        .split(",")
        .map((v) => v.trim())
        .filter(Boolean)[0];
      if (ch) params.set("ch", ch);
    }
    if (planMeta?.site) params.set("site", String(planMeta.site));
    if (planMeta?.location) params.set("location", String(planMeta.location));
    try {
      const res = await apiGet<{ options?: string[] }>(
        `/api/forecast/settings/critical-team-options${params.toString() ? `?${params.toString()}` : ""}`
      );
      const next = (res.options ?? [])
        .map((value) => String(value || "").trim())
        .filter(Boolean)
        .map((value) => ({ label: value, value }));
      setCriticalTeamOptions(next);
    } catch {
      setCriticalTeamOptions([]);
    }
  }, [planMeta?.business_area, planMeta?.channel, planMeta?.location, planMeta?.site, planMeta?.sub_business_area]);

  const loadRebalancePreview = useCallback(
    async (policy?: Partial<WhatIfState>) => {
      if (!planId && !rollupBa) return;
      setXskillLoading(true);
      try {
        const eff = Number(policy?.xskillEfficiencyPct ?? whatIf.xskillEfficiencyPct);
        const lend = Number(policy?.xskillMaxLendPct ?? whatIf.xskillMaxLendPct);
        const lockTeams = Array.isArray(policy?.lockCriticalTeams)
          ? policy?.lockCriticalTeams ?? []
          : whatIf.lockCriticalTeams;
        const enabled =
          policy?.rebalanceEnabled !== undefined ? Boolean(policy.rebalanceEnabled) : Boolean(whatIf.rebalanceEnabled);
        const grainMap: Record<string, string> = {
          week: "W",
          day: "D",
          month: "M",
          interval: "D"
        };
        const payload: Record<string, any> = {
          grain: grainMap[String(grain || "week")] || "D",
          start_date: planMeta?.start_week,
          end_date: planMeta?.end_week,
          policy: {
            cross_skill_efficiency_pct: (Number.isFinite(eff) ? eff : WHATIF_DEFAULT.xskillEfficiencyPct) / 100,
            max_lend_pct: enabled
              ? (Number.isFinite(lend) ? lend : WHATIF_DEFAULT.xskillMaxLendPct) / 100
              : 0,
            lock_critical_teams: (lockTeams ?? []).map((v) => String(v || "").trim()).filter(Boolean)
          }
        };
        if (planId) {
          payload.plan_id = planId;
        } else if (rollupBa) {
          payload.rollup_ba = rollupBa;
        }
        const res = await apiPost<{ status?: string; workforce?: WorkforcePreview }>(
          "/api/planning/plan/rebalancing",
          payload
        );
        const workforce = res.workforce ?? {};
        setXskillPreview(workforce);
        setTables((prev) => ({
          ...prev,
          xskill: workforce.rebalancing ?? [],
          xskill_balance: workforce.scope_balance ?? []
        }));
      } catch (error: any) {
        notify("error", error?.message || "Could not load rebalancing preview.");
      } finally {
        setXskillLoading(false);
      }
    },
    [grain, notify, planId, planMeta?.end_week, planMeta?.start_week, rollupBa, whatIf.lockCriticalTeams, whatIf.rebalanceEnabled, whatIf.xskillEfficiencyPct, whatIf.xskillMaxLendPct]
  );

  const loadWhatIf = useCallback(async () => {
    if (!planId || isRollup) return;
    try {
      const res = await apiGet<{ overrides?: Record<string, any> }>(
        `/api/planning/plan/whatif?plan_id=${planId}`
      );
      const overrides = res.overrides ?? {};
      const toPct = (value: any, fallback: number) => {
        const num = Number(value);
        if (!Number.isFinite(num)) return fallback;
        return num <= 1 ? num * 100 : num;
      };
      setWhatIf({
        ahtDelta: Number(overrides.aht_delta ?? 0),
        shrinkDelta: Number(overrides.shrink_delta ?? 0),
        attrDelta: Number(overrides.attr_delta ?? 0),
        volDelta: Number(overrides.vol_delta ?? 0),
        backlogCarryover: Boolean(overrides.backlog_carryover ?? true),
        rebalanceEnabled: Boolean(overrides.rebalance_enabled ?? true),
        xskillEfficiencyPct: toPct(
          overrides.cross_skill_efficiency_pct ?? overrides.xskill_efficiency_pct,
          WHATIF_DEFAULT.xskillEfficiencyPct
        ),
        xskillMaxLendPct: toPct(
          overrides.max_lend_pct ?? overrides.xskill_max_lend_pct,
          WHATIF_DEFAULT.xskillMaxLendPct
        ),
        lockCriticalTeams: Array.isArray(overrides.lock_critical_teams)
          ? overrides.lock_critical_teams.map((v: any) => String(v || "").trim()).filter(Boolean)
          : []
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
        await loadTablesForGrain(grain, intervalDate);
        await computePlanTables();
        await loadPlan();
      }
      setMessage("Refreshed.");
    } catch (error: any) {
      notify("error", error?.message || "Could not refresh plan.");
    } finally {
      setLoading(false);
    }
  }, [computePlanTables, grain, intervalDate, loadPlan, loadRollupTables, loadTablesForGrain, notify, rollupBa, setLoading]);

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

  const loadDebugInfo = useCallback(async () => {
    if (!planId || isRollup) return;
    setDebugError("");
    try {
      const res = await apiGet<any>(`/api/planning/plan/debug?plan_id=${planId}`);
      setDebugInfo(res);
      const depChanged = Array.isArray(res?.dep_changed) ? res.dep_changed : [];
      if (depChanged.length) {
        setLastRecomputeCause(depChanged.map((val: any) => String(val)));
      }
    } catch (error: any) {
      setDebugInfo(null);
      setDebugError(error?.message || "Could not load debug info.");
    }
  }, [isRollup, planId]);

  useEffect(() => {
    if (isRollup || !planId) return;
    void loadPlan();
  }, [isRollup, loadPlan, planId]);

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
    if (isRollup) return;
    if (!planMeta) return;
    void loadCriticalTeamOptions();
  }, [isRollup, loadCriticalTeamOptions, planMeta]);

  useEffect(() => {
    if (!planMeta && !isRollup) return;
    void loadRebalancePreview();
  }, [isRollup, loadRebalancePreview, planMeta, planId, rollupBa]);

  useEffect(() => {
    if (rosterAction !== "tp") return;
    const ba = planMeta?.business_area || "";
    const sba = planMeta?.sub_business_area || "";
    const lob = planMeta?.channel || "";
    const site = planMeta?.site || "";
    setTpForm((prev) => ({
      ...prev,
      biz_area: prev.biz_area || ba,
      sub_biz_area: prev.sub_biz_area || sba,
      lob: prev.lob || lob,
      site: prev.site || site
    }));
    setTwpForm((prev) => ({
      ...prev,
      biz_area: prev.biz_area || ba,
      sub_biz_area: prev.sub_biz_area || sba,
      lob: prev.lob || lob,
      site: prev.site || site
    }));
    void loadTpOptions(ba || undefined, sba || undefined);
  }, [loadTpOptions, planMeta?.business_area, planMeta?.channel, planMeta?.site, planMeta?.sub_business_area, rosterAction]);

  useEffect(() => {
    if (!intervalOptions.length || grain !== "interval") {
      prevGrainRef.current = grain;
      return;
    }
    const yesterday = toIsoDate(getYesterdayDate());
    const hasYesterday = intervalOptions.some((option) => option.value === yesterday);
    const isValid = intervalOptions.some((option) => option.value === intervalDate);
    const enteringInterval = prevGrainRef.current !== "interval";
    const shouldAuto =
      enteringInterval || !intervalDate || !isValid || intervalDate === lastAutoIntervalRef.current;
    if (hasYesterday && shouldAuto && intervalDate !== yesterday) {
      lastAutoIntervalRef.current = yesterday;
      setIntervalDate(yesterday);
      prevGrainRef.current = grain;
      return;
    }
    if (!intervalDate || !isValid) {
      const fallback = intervalOptions[0]?.value || "";
      if (fallback) {
        lastAutoIntervalRef.current = fallback;
        setIntervalDate(fallback);
      }
    }
    prevGrainRef.current = grain;
  }, [grain, intervalDate, intervalOptions]);

  useEffect(() => {
    if (!planId || isRollup) return;
    if (grain === "interval" && !intervalDate) return;
    void loadTablesWithLoader();
    void warmPlanCompute();
  }, [grain, intervalDate, isRollup, loadTablesWithLoader, warmPlanCompute, planId]);

  useEffect(() => {
    if (grain === "day" || grain === "interval") {
      setOptionsOpen(false);
    }
  }, [grain]);

  useEffect(() => {
    if (grain === "interval" && !intervalDate) return;
    const targetDate = targetDateForGrain(grain);
    const upperKeys = collectDateKeys(filteredUpperRows);
    const lowerKeys = collectDateKeys(filteredRows);
    const upperKey = closestDateKey(upperKeys, targetDate);
    const lowerKey = closestDateKey(lowerKeys, targetDate);
    if (!upperKey && !lowerKey) return;
    const frame = window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        if (!upperCollapsed && upperKey) {
          scrollTableToKey(upperTableRef.current, upperKey);
        }
        if (lowerKey) {
          scrollTableToKey(lowerTableRef.current, lowerKey);
        }
      });
    });
    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [
    grain,
    intervalDate,
    activeTab,
    viewFrom,
    viewTo,
    filteredUpperRows,
    filteredRows,
    upperCollapsed
  ]);

  useEffect(
    () => () => {
      if (pollRef.current) {
        window.clearTimeout(pollRef.current);
      }
      if (computePollRef.current) {
        window.clearTimeout(computePollRef.current);
      }
    },
    []
  );

  useEffect(() => {
    if (!pendingRosterSaveRef.current) return;
    const rows = pendingRosterSaveRef.current;
    pendingRosterSaveRef.current = null;
    void saveRoster(rows);
  }, [rosterSaveTick]);

  useEffect(
    () => () => {
      if (rosterSaveTimeoutRef.current) {
        window.clearTimeout(rosterSaveTimeoutRef.current);
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
    lastBulkSavedRowsRef.current = [];
    lastBulkSaveAtRef.current = null;
  }, [planId]);

  useEffect(() => {
    setDebugInfo(null);
    setDebugError("");
    setDebugOpen(false);
  }, [planId, isRollup]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = () => {
      void computePlanTables();
      void loadRebalancePreview();
    };
    window.addEventListener("settingsUpdated", handler);
    return () => {
      window.removeEventListener("settingsUpdated", handler);
    };
  }, [computePlanTables, loadRebalancePreview]);

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
    if (!planId || isRollup || isLocked) {
      if (isLocked) notify("warning", "Plan is locked (history).");
      return;
    }
    setLoading(true);
    try {
      await apiPost("/api/planning/plan/whatif", {
        plan_id: planId,
        overrides: {
          aht_delta: whatIf.ahtDelta,
          shrink_delta: whatIf.shrinkDelta,
          attr_delta: whatIf.attrDelta,
          vol_delta: whatIf.volDelta,
          backlog_carryover: whatIf.backlogCarryover,
          rebalance_enabled: whatIf.rebalanceEnabled,
          cross_skill_efficiency_pct: (Number(whatIf.xskillEfficiencyPct) || WHATIF_DEFAULT.xskillEfficiencyPct) / 100,
          max_lend_pct: (Number(whatIf.xskillMaxLendPct) || WHATIF_DEFAULT.xskillMaxLendPct) / 100,
          lock_critical_teams: (whatIf.lockCriticalTeams || []).map((v) => String(v || "").trim()).filter(Boolean)
        },
        action: "apply"
      });
      await refreshAll();
      await loadRebalancePreview();
      setMessage("What-if applied.");
    } catch (error: any) {
      notify("error", error?.message || "Could not apply what-if.");
    } finally {
      setLoading(false);
    }
  };

  const handleClearWhatIf = async () => {
    if (!planId || isRollup || isLocked) {
      if (isLocked) notify("warning", "Plan is locked (history).");
      return;
    }
    setLoading(true);
    try {
      await apiPost("/api/planning/plan/whatif", {
        plan_id: planId,
        action: "clear"
      });
      setWhatIf(WHATIF_DEFAULT);
      await refreshAll();
      await loadRebalancePreview(WHATIF_DEFAULT);
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
    if (isLocked) {
      setExtendMessage("Plan is locked (history).");
      return;
    }
    setLoading(true);
    try {
      await apiPost("/api/planning/plan/extend", {
        plan_id: planId,
        weeks: extendWeeks
      });
      if (planMeta?.end_week) {
        const optimisticEnd = addWeeksToDate(planMeta.end_week, extendWeeks);
        setPlanMeta((prev) => (prev ? { ...prev, end_week: optimisticEnd } : prev));
      }
      setExtendMessage(`Extended by ${extendWeeks} week(s).`);
      if (setLoading) {
        setLoading(false);
      }
      void computePlanTables({ wait: true, silent: true })
        .then(() => loadTablesForGrain(grain, intervalDate))
        .then(() => loadPlan())
        .catch(() => {});
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
    const nextRows = [
      ...(tables.emp ?? []),
      {
        ...rosterForm,
        biz_area: rosterForm.biz_area || planMeta?.business_area || "",
        sub_biz_area: rosterForm.sub_biz_area || planMeta?.sub_business_area || "",
        lob: rosterForm.lob || planMeta?.channel || "",
        site: rosterForm.site || planMeta?.site || ""
      }
    ];
    setTables((prev) => ({ ...prev, emp: nextRows }));
    pendingRosterSaveRef.current = nextRows;
    lastRosterSaveAtRef.current = Date.now();
    setRosterSaveTick((v) => v + 1);
    setRosterForm({
      brid: "",
      name: "",
      ftpt_status: "Full-time",
      role: "Agent",
      production_start: "",
      team_leader: "",
      avp: "",
      ftpt_hours: "",
      work_status: "Production",
      current_status: "Production",
      terminate_date: "",
      class_ref: "",
      training_start: "",
      training_end: "",
      nesting_start: "",
      nesting_end: "",
      site: "",
      biz_area: "",
      sub_biz_area: "",
      lob: "",
      loa_date: "",
      back_from_loa_date: ""
    });
    setRosterModalOpen(false);
  };

  const saveRoster = async (rowsOverride?: Array<Record<string, any>>) => {
    if (!planId) return;
    setLoading(true);
    rosterSavingRef.current = true;
    const rowsToSave = rowsOverride ?? tables.emp ?? [];
    try {
      await apiPost("/api/planning/plan/table", {
        plan_id: planId,
        name: "emp",
        rows: rowsToSave
      });
      let savedRows = rowsToSave;
      try {
        const res = await apiGet<{ rows?: Array<Record<string, any>> }>(
          `/api/planning/plan/table?plan_id=${planId}&name=emp`
        );
        if (Array.isArray(res.rows) && res.rows.length) {
          savedRows = res.rows;
        }
      } catch {
        // Keep optimistic rows if fetch fails
      }
      lastRosterSavedRowsRef.current = savedRows;
      setTables((prev) => ({ ...prev, emp: savedRows }));
      lastRosterSaveAtRef.current = Date.now();
      setMessage("Roster saved.");
      void computePlanTables();
      window.setTimeout(() => {
        void loadTablesForGrain("week");
      }, 800);
    } catch (error: any) {
      notify("error", error?.message || "Could not save roster.");
    } finally {
      rosterSavingRef.current = false;
      setLoading(false);
    }
  };

  const handleRosterUndo = () => {
    if (!rosterSnapshotRef.current) return;
    const nextRows = rosterSnapshotRef.current ?? [];
    setTables((prev) => ({ ...prev, emp: nextRows }));
    if (rosterSaveTimeoutRef.current) {
      window.clearTimeout(rosterSaveTimeoutRef.current);
    }
    rosterSaveTimeoutRef.current = window.setTimeout(() => {
      pendingRosterSaveRef.current = nextRows;
      setRosterSaveTick((v) => v + 1);
    }, 400);
  };

  const handleRosterUpload = async (file: File) => {
    if (!file) return;
    setLoading(true);
    try {
      const rawRows = await parseFile(file);
      const rosterRows = normalizeRosterUploadRows(rawRows);
      const ext = (file.name.split(".").pop() || "").toLowerCase();
      const validExt = ext === "csv" || ext === "xlsx" || ext === "xls";
      const hasBrid = rosterRows.some((row) => hasAnyValue(row?.brid));
      const validUpload = validExt && rosterRows.length > 0 && hasBrid;
      const invalidStatus = !validExt
        ? "Invalid format (supported: csv/xlsx/xls)"
        : rosterRows.length === 0
          ? "Invalid format (no valid roster rows found)"
          : "Invalid format (BRID column/value missing)";
      const bulkEntry = {
        file_name: file.name,
        ext,
        size_kb: Math.round(file.size / 1024),
        is_valid: validUpload ? "Yes" : "No",
        status: validUpload ? `Uploaded (${rosterRows.length} rows)` : invalidStatus
      };
      const nextBulkRows = [...(tables.bulk_files ?? []), bulkEntry];
      lastBulkSavedRowsRef.current = nextBulkRows;
      lastBulkSaveAtRef.current = Date.now();
      let bulkSaveFailed = false;

      // Persist bulk upload audit rows immediately.
      if (planId) {
        bulkSavingRef.current = true;
        try {
          await apiPost("/api/planning/plan/table", {
            plan_id: planId,
            name: "bulk_files",
            rows: nextBulkRows
          });
          lastBulkSavedRowsRef.current = nextBulkRows;
          lastBulkSaveAtRef.current = Date.now();
        } catch {
          bulkSaveFailed = true;
        } finally {
          bulkSavingRef.current = false;
        }
      }
      if (bulkSaveFailed) {
        notify("warning", "Bulk upload audit could not be persisted. Please save the plan.");
      }

      if (!validUpload) {
        setTables((prev) => ({ ...prev, bulk_files: nextBulkRows }));
        notify("error", "Upload failed. Please upload the correct roster template format.");
        return;
      }

      const nextRows = [...(tables.emp ?? []), ...rosterRows];
      setTables((prev) => ({
        ...prev,
        emp: nextRows,
        bulk_files: nextBulkRows
      }));
      pendingRosterSaveRef.current = nextRows;
      setRosterSaveTick((v) => v + 1);
      notify("success", `Roster rows added (${rosterRows.length}).`);
    } catch (error: any) {
      const detail = error?.message ? ` (${error.message})` : "";
      notify("error", `Upload failed. Please upload the correct roster template format.${detail}`);
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
    if (!nhForm.training_start) {
      setNhMessage("Training Start is required.");
      return;
    }
    setLoading(true);
    try {
      const payload = applyAutoDates(nhForm);
      const res = await apiPost<{ rows?: Array<Record<string, any>> }>("/api/planning/plan/new-hire/class", {
        plan_id: planId,
        data: payload
      });
      setNhClasses(res.rows ?? []);
      setNhForm(NH_FORM_DEFAULT);
      setNhModalOpen(false);
      setNhMessage("New hire class added.");
      await computePlanTables();
      void loadTablesForGrain("week");
    } catch (error: any) {
      setNhMessage(error?.message || "Could not add class.");
    } finally {
      setLoading(false);
    }
  };

  const handleConfirmNhClasses = async () => {
    if (!planId || !nhDetailsSelection.size) return;
    setLoading(true);
    try {
      const class_refs = Array.from(nhDetailsSelection);
      const res = await apiPost<{ rows?: Array<Record<string, any>> }>("/api/planning/plan/new-hire/confirm", {
        plan_id: planId,
        class_refs
      });
      setNhClasses(res.rows ?? []);
      setNhDetailsSelection(new Set());
      setNhDetailsOpen(false);
      setNhMessage("Selected classes confirmed.");
      await computePlanTables();
      void loadTablesForGrain("week");
    } catch (error: any) {
      setNhMessage(error?.message || "Could not confirm classes.");
    } finally {
      setLoading(false);
    }
  };

  const handleRosterCellChange = (rowIdx: number, colId: string, value: any) => {
    const rows = [...(tables.emp ?? [])];
    if (!rows[rowIdx]) return;
    rows[rowIdx] = { ...rows[rowIdx], [colId]: value };
    setTables((prev) => ({ ...prev, emp: rows }));
    if (rosterSaveTimeoutRef.current) {
      window.clearTimeout(rosterSaveTimeoutRef.current);
    }
    rosterSaveTimeoutRef.current = window.setTimeout(() => {
      pendingRosterSaveRef.current = rows;
      setRosterSaveTick((v) => v + 1);
    }, 600);
  };

  const applyRosterAction = () => {
    if (!rosterSelected.size) return;
    const rows = [...(tables.emp ?? [])];
    rosterSelected.forEach((idx) => {
      if (!rows[idx]) return;
      const row = { ...rows[idx] };
      if (rosterAction === "class") {
        row.class_ref = rosterActionValue.class_ref || row.class_ref;
      } else if (rosterAction === "ftpt") {
        row.ftpt_status = rosterActionValue.ftpt_status || row.ftpt_status;
      } else if (rosterAction === "loa") {
        row.work_status = "LOA";
        row.current_status = "LOA";
        row.loa_date = rosterActionValue.loa_date || row.loa_date;
      } else if (rosterAction === "back") {
        row.work_status = "Production";
        row.current_status = "Production";
        row.back_from_loa_date = rosterActionValue.back_from_loa_date || row.back_from_loa_date;
      } else if (rosterAction === "term") {
        row.current_status = "Terminated";
        row.terminate_date = rosterActionValue.terminate_date || row.terminate_date;
      } else if (rosterAction === "tp") {
        if (tpTab === "tp-transfer") {
          row.biz_area = tpForm.biz_area || row.biz_area;
          row.sub_biz_area = tpForm.sub_biz_area || row.sub_biz_area;
          row.lob = tpForm.lob || row.lob;
          row.site = tpForm.site || row.site;
          if (tpForm.new_class) row.class_ref = tpForm.class_ref || row.class_ref;
        } else if (tpTab === "tp-promo") {
          row.role = promoForm.role || row.role;
        } else if (tpTab === "tp-both") {
          row.biz_area = twpForm.biz_area || row.biz_area;
          row.sub_biz_area = twpForm.sub_biz_area || row.sub_biz_area;
          row.lob = twpForm.lob || row.lob;
          row.site = twpForm.site || row.site;
          row.role = twpForm.role || row.role;
          if (twpForm.new_class) row.class_ref = twpForm.class_ref || row.class_ref;
        }
      }
      rows[idx] = row;
    });
    setTables((prev) => ({ ...prev, emp: rows }));
    pendingRosterSaveRef.current = rows;
    setRosterSaveTick((v) => v + 1);
    setRosterActionValue({});
    setRosterAction("");
  };

  const removeSelectedRoster = () => {
    if (!rosterSelected.size) return;
    const rows = (tables.emp ?? []).filter((_row, idx) => !rosterSelected.has(idx));
    setTables((prev) => ({ ...prev, emp: rows }));
    pendingRosterSaveRef.current = rows;
    setRosterSaveTick((v) => v + 1);
    setRosterSelected(new Set());
    setRosterAction("");
  };

  const renderRoster = () => {
    const rosterRows = tables.emp ?? [];
    const bulkRows = tables.bulk_files ?? [];
    const rosterHasSelection = rosterRows.length > 0 && rosterSelected.size > 0;

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
              <button
                type="button"
                className="btn btn-light"
                disabled={!rosterHasSelection}
                onClick={() => setRosterAction("tp")}
              >
                Transfer/Promotion
              </button>
              <button
                type="button"
                className="btn btn-light"
                disabled={!rosterHasSelection}
                onClick={() => setRosterAction("class")}
              >
                Change Class
              </button>
              <button
                type="button"
                className="btn btn-light"
                disabled={!rosterHasSelection}
                onClick={() => setRosterAction("ftpt")}
              >
                FT/PT
              </button>
              <button
                type="button"
                className="btn btn-light"
                disabled={!rosterHasSelection}
                onClick={() => setRosterAction("loa")}
              >
                LOA
              </button>
              <button
                type="button"
                className="btn btn-light"
                disabled={!rosterHasSelection}
                onClick={() => setRosterAction("back")}
              >
                Back from LOA
              </button>
              <button
                type="button"
                className="btn btn-light"
                disabled={!rosterHasSelection}
                onClick={() => setRosterAction("term")}
              >
                Terminate
              </button>
              <button
                type="button"
                className="btn btn-light"
                disabled={!rosterHasSelection}
                onClick={() => setRosterAction("remove")}
              >
                Remove
              </button>
              <button
                type="button"
                className="btn btn-light"
                disabled={!rosterHasSelection}
                onClick={handleRosterUndo}
              >
                Undo
              </button>
              <div className="plan-toolbar-spacer" />
              <span className="plan-toolbar-meta">
                Total: {String(rosterRows.length).padStart(2, "0")} Records
              </span>
              <button
                type="button"
                className="btn btn-workstatus"
                onClick={() => downloadCsv(`plan_${planId}_workstatus_dataset.csv`, rosterRows)}
              >
                Workstatus Dataset
              </button>
            </div>
            <div className="table-wrap">
              {!rosterRows.length ? (
                <div className="forecast-muted">No roster entries yet.</div>
              ) : (
                <table className="table">
                  <thead>
                    <tr>
                      <th>
                        <input
                          type="checkbox"
                          checked={rosterSelected.size === rosterRows.length && rosterRows.length > 0}
                          onChange={(event) => {
                            if (event.target.checked) {
                              setRosterSelected(new Set(rosterRows.map((_r, idx) => idx)));
                            } else {
                              setRosterSelected(new Set());
                            }
                          }}
                        />
                      </th>
                      {ROSTER_COLUMNS.map((col) => (
                        <th key={col.id}>{col.label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rosterRows.map((row, idx) => (
                      <tr key={`${idx}-${row?.brid || "row"}`}>
                        <td>
                          <input
                            type="checkbox"
                            checked={rosterSelected.has(idx)}
                            onChange={(event) => {
                              setRosterSelected((prev) => {
                                const next = new Set(prev);
                                if (event.target.checked) next.add(idx);
                                else next.delete(idx);
                                return next;
                              });
                            }}
                          />
                        </td>
                        {ROSTER_COLUMNS.map((col) => (
                          <td key={col.id}>
                            <input
                              className="table-input"
                              value={row?.[col.id] ?? ""}
                              onChange={(event) => handleRosterCellChange(idx, col.id, event.target.value)}
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </>
        ) : (
          <>
            <div className="plan-toolbar">
              <label className="upload-box">
                ⬆️ Upload CSV/Excel
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
                  downloadCsv(`plan_${planId}_roster_bulk_template.csv`, ROSTER_BULK_TEMPLATE_ROWS)
                }
              >
                Download Template
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
    <AppShell crumbs={breadcrumbText} crumbLinks={breadcrumbLinks}>
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
                aria-label="Toggle monthly view"
              />
              <span className="plan-toggle-track" aria-hidden="true">
                <span className="plan-toggle-thumb" />
              </span>
            </label>
            {message ? <span className="plan-message">{message}</span> : null}
          </div>
          <div className="plan-header-actions">
            <button
              type="button"
              className="btn btn-light expcoll"
              title="Collapse/Expand"
              onClick={() => setUpperCollapsed((prev) => !prev)}
            >
              ▼
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

        {isLocked ? (
          <div className="plan-message">Plan is locked (history). Editing and extensions are disabled.</div>
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
              <DataTable
                data={filteredUpperRows}
                emptyLabel="No upper summary yet."
                dateMode={grain === "month" ? "month" : "day"}
                wrapperRef={upperTableRef}
              />
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
          {activeConfig.key === "xskill" ? (
            <div className="plan-rollup">
              <div className="plan-meta" style={{ marginBottom: 12 }}>
                {formatMetaRow("Required FTE", String(orgHiring.required_fte ?? 0))}
                {formatMetaRow("Supply FTE", String(orgHiring.supply_fte ?? 0))}
                {formatMetaRow("Pre-Rebalance Shortfall", String(orgHiring.estimated_shortfall_fte ?? 0))}
                {formatMetaRow("Hiring Saved", String(orgHiring.potential_hiring_saved_fte ?? 0))}
                {formatMetaRow("Net Hiring", String(orgHiring.net_hiring_fte ?? 0))}
                {formatMetaRow("Post-Rebalance Shortfall", String(orgHiring.post_rebalance_shortfall_fte ?? 0))}
                {formatMetaRow("Efficiency %", String(orgHiring.cross_skill_efficiency_pct ?? 0))}
                {formatMetaRow("Max Lend %", String(orgHiring.max_lend_pct ?? 0))}
                {formatMetaRow("Locked Scopes", String(orgHiring.locked_critical_scope_count ?? 0))}
              </div>
              {xskillLoading ? <div className="plan-message">Loading rebalancing preview…</div> : null}
              <div style={{ marginBottom: 12 }}>
                <h4 style={{ margin: "8px 0" }}>Borrow / Lend Moves</h4>
                <DataTable data={rebalanceRows} emptyLabel="No borrowing/lending moves for this scope." />
              </div>
              <div>
                <h4 style={{ margin: "8px 0" }}>Scope Balance</h4>
                <DataTable data={scopeBalanceRows} emptyLabel="No scope balance rows available." />
              </div>
            </div>
          ) : isRollup ? (
            <div className="plan-rollup">
              <DataTable
                data={filteredRows}
                emptyLabel="No roll-up data available yet."
                dateMode={grain === "month" ? "month" : "day"}
                wrapperRef={lowerTableRef}
              />
            </div>
          ) : activeConfig.key === "emp" ? (
            renderRoster()
          ) : activeConfig.key === "nh" ? (
            renderNewHire()
          ) : activeConfig.key === "notes" ? (
            renderNotes()
          ) : canEdit ? (
            <EditableTable
              data={filteredRows}
              editableColumns={editableColumns}
              onChange={handleFilteredChange}
              wrapperRef={lowerTableRef}
            />
          ) : (
            <DataTable
              data={filteredRows}
              emptyLabel="No data available."
              dateMode={grain === "month" ? "month" : "day"}
              wrapperRef={lowerTableRef}
            />
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
              <button
                type="button"
                className="btn btn-light"
                onClick={() => setExtendOpen(true)}
                disabled={isLocked}
                title={isLocked ? "Plan is locked (history)." : ""}
              >
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
                  onClick={() => {
                    if (intervalOptions.length && !intervalDate) {
                      setIntervalDate(intervalOptions[0].value);
                    }
                    setGrain("interval");
                  }}
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
                <label>
                  Cross-Skill Efficiency (%)
                  <input
                    className="input"
                    type="number"
                    min={0}
                    max={100}
                    value={whatIf.xskillEfficiencyPct}
                    step={1}
                    onChange={(event) =>
                      setWhatIf((prev) => ({ ...prev, xskillEfficiencyPct: Number(event.target.value) }))
                    }
                  />
                </label>
                <label>
                  Max Lend (%)
                  <input
                    className="input"
                    type="number"
                    min={0}
                    max={100}
                    value={whatIf.xskillMaxLendPct}
                    step={1}
                    onChange={(event) =>
                      setWhatIf((prev) => ({ ...prev, xskillMaxLendPct: Number(event.target.value) }))
                    }
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
              <label className="plan-whatif-checkbox">
                <input
                  type="checkbox"
                  checked={whatIf.rebalanceEnabled}
                  onChange={(event) => setWhatIf((prev) => ({ ...prev, rebalanceEnabled: event.target.checked }))}
                />
                Enable cross-skill borrowing/lending
              </label>
              <div style={{ marginTop: 8 }}>
                <div className="label">Lock Critical Teams</div>
                <MultiSelect
                  options={lockCriticalTeamChoices}
                  values={whatIf.lockCriticalTeams}
                  onChange={(next) => setWhatIf((prev) => ({ ...prev, lockCriticalTeams: next }))}
                  placeholder="Select protected teams"
                />
              </div>
              <div className="plan-whatif-actions">
                <button
                  type="button"
                  className="btn btn-light"
                  onClick={() => void loadRebalancePreview()}
                >
                  Preview Rebalance
                </button>
                <button type="button" className="btn btn-primary" onClick={handleApplyWhatIf}>
                  Apply What-If
                </button>
                <button type="button" className="btn btn-light" onClick={handleClearWhatIf}>
                  Clear
                </button>
              </div>
            </div>

            <div className="plan-options-debug">
              <div className="plan-options-debug-head">
                <h5>Dataflow Debug</h5>
                <div className="plan-options-debug-actions">
                  <button
                    type="button"
                    className="btn btn-light"
                    onClick={() => {
                      setDebugOpen((prev) => {
                        const next = !prev;
                        if (next) {
                          void loadDebugInfo();
                        }
                        return next;
                      });
                    }}
                  >
                    {debugOpen ? "Hide" : "Show"}
                  </button>
                  <button
                    type="button"
                    className="btn btn-light"
                    onClick={() => void loadDebugInfo()}
                    disabled={!debugOpen}
                  >
                    Refresh
                  </button>
                </div>
              </div>
              {debugOpen ? (
                <>
                  {debugError ? <div className="plan-message">{debugError}</div> : null}
                  {lastRecomputeCause.length ? (
                    <div className="plan-message">
                      Last recompute cause: {lastRecomputeCause.join(", ")}
                    </div>
                  ) : null}
                  <pre className="plan-debug">
                    {debugInfo ? JSON.stringify(debugInfo, null, 2) : "Loading..."}
                  </pre>
                </>
              ) : null}
            </div>
          </div>

          {viewModalOpen ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal ws-modal-sm">
                <div className="ws-modal-header" style={{ background: "#2f3747", color: "white" }}>
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
                  {!nhClasses.length ? (
                    <div className="forecast-muted">No classes yet.</div>
                  ) : (
                    <div className="table-wrap">
                      <table className="table">
                        <thead>
                          <tr>
                            <th />
                            {Object.keys(nhClasses[0] || {}).map((col) => (
                              <th key={col}>{col.replace(/_/g, " ")}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {nhClasses.map((row, idx) => {
                            const cref = String(row?.class_reference ?? idx);
                            const selected = nhDetailsSelection.has(cref);
                            return (
                              <tr key={cref}>
                                <td>
                                  <input
                                    type="checkbox"
                                    checked={selected}
                                    onChange={(event) => {
                                      setNhDetailsSelection((prev) => {
                                        const next = new Set(prev);
                                        if (event.target.checked) next.add(cref);
                                        else next.delete(cref);
                                        return next;
                                      });
                                    }}
                                  />
                                </td>
                                {Object.keys(nhClasses[0] || {}).map((col) => (
                                  <td key={col}>{String(row?.[col] ?? "")}</td>
                                ))}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
                <div className="ws-modal-footer">
                  <button type="button" className="btn btn-primary" onClick={handleConfirmNhClasses} disabled={!nhDetailsSelection.size}>
                    Confirm Selected
                  </button>
                  <button type="button" className="btn btn-light" onClick={() => setNhDetailsOpen(false)}>
                    Close
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {rosterAction === "tp" ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal">
                <div className="ws-modal-header">
                  <h3>Transfer & Promotion</h3>
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setRosterAction("")}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
                <div className="ws-modal-body">
                  <div className="plan-subtabs">
                    <button
                      type="button"
                      className={`plan-subtab ${tpTab === "tp-transfer" ? "active" : ""}`}
                      onClick={() => setTpTab("tp-transfer")}
                    >
                      Transfer
                    </button>
                    <button
                      type="button"
                      className={`plan-subtab ${tpTab === "tp-promo" ? "active" : ""}`}
                      onClick={() => setTpTab("tp-promo")}
                    >
                      Promotion
                    </button>
                    <button
                      type="button"
                      className={`plan-subtab ${tpTab === "tp-both" ? "active" : ""}`}
                      onClick={() => setTpTab("tp-both")}
                    >
                      Transfer with Promotion
                    </button>
                  </div>

                  {tpTab === "tp-transfer" ? (
                    <div className="plan-modal-grid">
                      <label>
                        Business Area
                        <select
                          className="select"
                          value={tpForm.biz_area ?? ""}
                          onChange={(event) => {
                            const val = event.target.value;
                            setTpForm((prev) => ({ ...prev, biz_area: val }));
                            void loadTpOptions(val);
                          }}
                        >
                          <option value="">Select</option>
                          {tpOptions.business_areas.map((ba) => (
                            <option key={ba} value={ba}>{ba}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Sub Business Area
                        <select
                          className="select"
                          value={tpForm.sub_biz_area ?? ""}
                          onChange={(event) => {
                            const val = event.target.value;
                            setTpForm((prev) => ({ ...prev, sub_biz_area: val }));
                            void loadTpOptions(tpForm.biz_area, val);
                          }}
                        >
                          <option value="">Select</option>
                          {tpOptions.sub_business_areas.map((sba) => (
                            <option key={sba} value={sba}>{sba}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Channel
                        <select
                          className="select"
                          value={tpForm.lob ?? ""}
                          onChange={(event) => setTpForm((prev) => ({ ...prev, lob: event.target.value }))}
                        >
                          <option value="">Select</option>
                          {tpOptions.channels.map((ch) => (
                            <option key={ch} value={ch}>{ch}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Site
                        <select
                          className="select"
                          value={tpForm.site ?? ""}
                          onChange={(event) => setTpForm((prev) => ({ ...prev, site: event.target.value }))}
                        >
                          <option value="">Select</option>
                          {tpOptions.sites.map((site) => (
                            <option key={site} value={site}>{site}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Transfer Type
                        <select
                          className="select"
                          value={tpForm.transfer_type ?? "perm"}
                          onChange={(event) => setTpForm((prev) => ({ ...prev, transfer_type: event.target.value }))}
                        >
                          <option value="perm">Permanent</option>
                          <option value="interim">Interim</option>
                        </select>
                      </label>
                      <label>
                        Effective Date
                        <input
                          className="input"
                          type="date"
                          value={tpForm.date_from ?? ""}
                          onChange={(event) => setTpForm((prev) => ({ ...prev, date_from: event.target.value }))}
                        />
                      </label>
                      <label>
                        Return Date (Interim only)
                        <input
                          className="input"
                          type="date"
                          value={tpForm.date_to ?? ""}
                          onChange={(event) => setTpForm((prev) => ({ ...prev, date_to: event.target.value }))}
                        />
                      </label>
                      <label>
                        Transfer with new class
                        <select
                          className="select"
                          value={tpForm.new_class ? "yes" : "no"}
                          onChange={(event) => setTpForm((prev) => ({ ...prev, new_class: event.target.value === "yes" }))}
                        >
                          <option value="no">No</option>
                          <option value="yes">Yes</option>
                        </select>
                      </label>
                      <label>
                        Class Reference (if new class)
                        <select
                          className="select"
                          value={tpForm.class_ref ?? ""}
                          onChange={(event) => setTpForm((prev) => ({ ...prev, class_ref: event.target.value }))}
                        >
                          <option value="">Select</option>
                          {classRefOptions.map((opt) => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ))}
                        </select>
                      </label>
                    </div>
                  ) : null}

                  {tpTab === "tp-promo" ? (
                    <div className="plan-modal-grid">
                      <label>
                        Promotion Type
                        <select
                          className="select"
                          value={promoForm.promo_type ?? "perm"}
                          onChange={(event) => setPromoForm((prev) => ({ ...prev, promo_type: event.target.value }))}
                        >
                          <option value="perm">Permanent</option>
                          <option value="interim">Temporary</option>
                        </select>
                      </label>
                      <label>
                        Effective Date
                        <input
                          className="input"
                          type="date"
                          value={promoForm.date_from ?? ""}
                          onChange={(event) => setPromoForm((prev) => ({ ...prev, date_from: event.target.value }))}
                        />
                      </label>
                      <label>
                        Stop Date (Temporary only)
                        <input
                          className="input"
                          type="date"
                          value={promoForm.date_to ?? ""}
                          onChange={(event) => setPromoForm((prev) => ({ ...prev, date_to: event.target.value }))}
                        />
                      </label>
                      <label>
                        Role
                        <select
                          className="select"
                          value={promoForm.role ?? ""}
                          onChange={(event) => setPromoForm((prev) => ({ ...prev, role: event.target.value }))}
                        >
                          <option value="">Select</option>
                          {["Agent", "SME", "Trainer", "Team Leader", "QA", "HR", "WFM", "AVP", "VP"].map((role) => (
                            <option key={role} value={role}>{role}</option>
                          ))}
                        </select>
                      </label>
                    </div>
                  ) : null}

                  {tpTab === "tp-both" ? (
                    <div className="plan-modal-grid">
                      <label>
                        Business Area
                        <select
                          className="select"
                          value={twpForm.biz_area ?? ""}
                          onChange={(event) => {
                            const val = event.target.value;
                            setTwpForm((prev) => ({ ...prev, biz_area: val }));
                            void loadTpOptions(val);
                          }}
                        >
                          <option value="">Select</option>
                          {tpOptions.business_areas.map((ba) => (
                            <option key={ba} value={ba}>{ba}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Sub Business Area
                        <select
                          className="select"
                          value={twpForm.sub_biz_area ?? ""}
                          onChange={(event) => {
                            const val = event.target.value;
                            setTwpForm((prev) => ({ ...prev, sub_biz_area: val }));
                            void loadTpOptions(twpForm.biz_area, val);
                          }}
                        >
                          <option value="">Select</option>
                          {tpOptions.sub_business_areas.map((sba) => (
                            <option key={sba} value={sba}>{sba}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Channel
                        <select
                          className="select"
                          value={twpForm.lob ?? ""}
                          onChange={(event) => setTwpForm((prev) => ({ ...prev, lob: event.target.value }))}
                        >
                          <option value="">Select</option>
                          {tpOptions.channels.map((ch) => (
                            <option key={ch} value={ch}>{ch}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Site
                        <select
                          className="select"
                          value={twpForm.site ?? ""}
                          onChange={(event) => setTwpForm((prev) => ({ ...prev, site: event.target.value }))}
                        >
                          <option value="">Select</option>
                          {tpOptions.sites.map((site) => (
                            <option key={site} value={site}>{site}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Transfer Type
                        <select
                          className="select"
                          value={twpForm.transfer_type ?? "perm"}
                          onChange={(event) => setTwpForm((prev) => ({ ...prev, transfer_type: event.target.value }))}
                        >
                          <option value="perm">Permanent</option>
                          <option value="interim">Temporary</option>
                        </select>
                      </label>
                      <label>
                        Transfer with new class
                        <select
                          className="select"
                          value={twpForm.new_class ? "yes" : "no"}
                          onChange={(event) => setTwpForm((prev) => ({ ...prev, new_class: event.target.value === "yes" }))}
                        >
                          <option value="no">No</option>
                          <option value="yes">Yes</option>
                        </select>
                      </label>
                      <label>
                        Effective Date
                        <input
                          className="input"
                          type="date"
                          value={twpForm.date_from ?? ""}
                          onChange={(event) => setTwpForm((prev) => ({ ...prev, date_from: event.target.value }))}
                        />
                      </label>
                      <label>
                        Stop Date (Temporary only)
                        <input
                          className="input"
                          type="date"
                          value={twpForm.date_to ?? ""}
                          onChange={(event) => setTwpForm((prev) => ({ ...prev, date_to: event.target.value }))}
                        />
                      </label>
                      <label>
                        Class Reference (if new class)
                        <select
                          className="select"
                          value={twpForm.class_ref ?? ""}
                          onChange={(event) => setTwpForm((prev) => ({ ...prev, class_ref: event.target.value }))}
                        >
                          <option value="">Select</option>
                          {classRefOptions.map((opt) => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Role
                        <select
                          className="select"
                          value={twpForm.role ?? ""}
                          onChange={(event) => setTwpForm((prev) => ({ ...prev, role: event.target.value }))}
                        >
                          <option value="">Select</option>
                          {["Agent", "SME", "Trainer", "Team Leader", "QA", "HR", "WFM", "AVP", "VP"].map((role) => (
                            <option key={role} value={role}>{role}</option>
                          ))}
                        </select>
                      </label>
                    </div>
                  ) : null}
                  <div className="plan-toolbar-meta">Selected: {rosterSelected.size} employee(s)</div>
                </div>
                <div className="ws-modal-footer">
                  <button type="button" className="btn btn-primary" onClick={applyRosterAction}>
                    Save
                  </button>
                  <button type="button" className="btn btn-light" onClick={() => setRosterAction("")}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {rosterAction && rosterAction !== "tp" ? (
            <div className="ws-modal-backdrop">
              <div className="ws-modal ws-modal-sm">
                <div className="ws-modal-header">
                  <h3>
                    {rosterAction === "class" && "Change Class"}
                    {rosterAction === "ftpt" && "Change FT/PT"}
                    {rosterAction === "loa" && "Set LOA"}
                    {rosterAction === "back" && "Back from LOA"}
                    {rosterAction === "term" && "Terminate"}
                    {rosterAction === "remove" && "Remove Employees"}
                  </h3>
                  <button type="button" className="btn btn-light closeOptions" onClick={() => setRosterAction("")}>
                    <svg width="16" height="16" viewBox="0 0 16 16">
                      <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                      <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
                <div className="ws-modal-body">
                  {rosterAction === "class" ? (
                    <label>
                      Class Reference
                      <select
                        className="select"
                        value={rosterActionValue.class_ref ?? ""}
                        onChange={(event) => setRosterActionValue({ class_ref: event.target.value })}
                      >
                        <option value="">Select</option>
                        {classRefOptions.map((opt) => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </select>
                      <div className="plan-modal-hint">
                        {rosterActionValue.class_ref
                          ? classRefScopeMap.get(rosterActionValue.class_ref) ||
                            [planMeta?.business_area, planMeta?.sub_business_area, planMeta?.channel, planMeta?.site]
                              .filter(Boolean)
                              .join(" > ")
                          : ""}
                      </div>
                    </label>
                  ) : null}
                  {rosterAction === "ftpt" ? (
                    <label>
                      FT/PT Status
                      <select
                        className="select"
                        value={rosterActionValue.ftpt_status ?? "Full-time"}
                        onChange={(event) => setRosterActionValue({ ftpt_status: event.target.value })}
                      >
                        <option value="Full-time">Full-time</option>
                        <option value="Part-time">Part-time</option>
                      </select>
                    </label>
                  ) : null}
                  {rosterAction === "loa" ? (
                    <label>
                      LOA Date
                      <input
                        className="input"
                        type="date"
                        value={rosterActionValue.loa_date ?? ""}
                        onChange={(event) => setRosterActionValue({ loa_date: event.target.value })}
                      />
                    </label>
                  ) : null}
                  {rosterAction === "back" ? (
                    <label>
                      Back From LOA Date
                      <input
                        className="input"
                        type="date"
                        value={rosterActionValue.back_from_loa_date ?? ""}
                        onChange={(event) => setRosterActionValue({ back_from_loa_date: event.target.value })}
                      />
                    </label>
                  ) : null}
                  {rosterAction === "term" ? (
                    <label>
                      Terminate Date
                      <input
                        className="input"
                        type="date"
                        value={rosterActionValue.terminate_date ?? ""}
                        onChange={(event) => setRosterActionValue({ terminate_date: event.target.value })}
                      />
                    </label>
                  ) : null}
                  {rosterAction === "remove" ? (
                    <div>Remove the selected employees from the roster?</div>
                  ) : null}
                </div>
                <div className="ws-modal-footer">
                  {rosterAction === "remove" ? (
                    <button type="button" className="btn btn-primary" onClick={removeSelectedRoster}>
                      Remove
                    </button>
                  ) : (
                    <button type="button" className="btn btn-primary" onClick={applyRosterAction}>
                      Save
                    </button>
                  )}
                  <button type="button" className="btn btn-light" onClick={() => setRosterAction("")}>
                    Cancel
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
                    <label>
                      BRID
                      <input
                        className="input"
                        value={rosterForm.brid ?? ""}
                        onChange={(event) => setRosterForm((prev) => ({ ...prev, brid: event.target.value }))}
                      />
                    </label>
                    <label>
                      Employee Name
                      <input
                        className="input"
                        value={rosterForm.name ?? ""}
                        onChange={(event) => setRosterForm((prev) => ({ ...prev, name: event.target.value }))}
                      />
                    </label>
                    <label>
                      FT/PT Status
                      <select
                        className="select"
                        value={rosterForm.ftpt_status ?? "Full-time"}
                        onChange={(event) => setRosterForm((prev) => ({ ...prev, ftpt_status: event.target.value }))}
                      >
                        <option value="Full-time">Full-time</option>
                        <option value="Part-time">Part-time</option>
                      </select>
                    </label>
                    <label>
                      Role
                      <select
                        className="select"
                        value={rosterForm.role ?? "Agent"}
                        onChange={(event) => setRosterForm((prev) => ({ ...prev, role: event.target.value }))}
                      >
                        {["Agent", "SME", "Trainer", "Team Leader", "QA", "HR", "WFM", "AVP", "VP"].map((role) => (
                          <option key={role} value={role}>
                            {role}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Production Date
                      <input
                        className="input"
                        type="date"
                        value={rosterForm.production_start ?? ""}
                        onChange={(event) =>
                          setRosterForm((prev) => ({ ...prev, production_start: event.target.value }))
                        }
                      />
                    </label>
                    <label>
                      Team Leader
                      <input
                        className="input"
                        value={rosterForm.team_leader ?? ""}
                        onChange={(event) => setRosterForm((prev) => ({ ...prev, team_leader: event.target.value }))}
                      />
                    </label>
                    <label>
                      AVP
                      <input
                        className="input"
                        value={rosterForm.avp ?? ""}
                        onChange={(event) => setRosterForm((prev) => ({ ...prev, avp: event.target.value }))}
                      />
                    </label>
                    <label>
                      FT/PT Hours
                      <input
                        className="input"
                        type="number"
                        value={rosterForm.ftpt_hours ?? ""}
                        onChange={(event) => setRosterForm((prev) => ({ ...prev, ftpt_hours: event.target.value }))}
                      />
                    </label>
                    <label>
                      Work Status
                      <select
                        className="select"
                        value={rosterForm.work_status ?? "Production"}
                        onChange={(event) => setRosterForm((prev) => ({ ...prev, work_status: event.target.value }))}
                      >
                        {["Production", "Training", "Nesting", "LOA"].map((status) => (
                          <option key={status} value={status}>
                            {status}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Current Status
                      <select
                        className="select"
                        value={rosterForm.current_status ?? "Production"}
                        onChange={(event) => setRosterForm((prev) => ({ ...prev, current_status: event.target.value }))}
                      >
                        {["Production", "LOA", "Terminated"].map((status) => (
                          <option key={status} value={status}>
                            {status}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Terminate Date
                      <input
                        className="input"
                        type="date"
                        value={rosterForm.terminate_date ?? ""}
                        onChange={(event) =>
                          setRosterForm((prev) => ({ ...prev, terminate_date: event.target.value }))
                        }
                      />
                    </label>
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
