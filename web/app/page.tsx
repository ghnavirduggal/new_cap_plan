"use client";

import Link from "next/link";
import React from "react";
import { useEffect, useMemo, useState } from "react";
import AppShell from "./_components/AppShell";
import { apiGet, apiPost } from "../lib/api";

type PlanRecord = {
  id?: number;
  plan_name?: string;
  business_area?: string;
  sub_business_area?: string;
  channel?: string;
  plan_type?: string;
  status?: string;
  start_week?: string;
  end_week?: string;
  owner?: string;
  created_by?: string;
  updated_at?: string;
};


const kpiCards = [
  { title: "Staffing Gap", status: "Watch", value: "-", suffix: "FTE", delta: "-", meta: "Last Week - plan detail", tone: "blue" },
  { title: "Hiring", status: "Watch", value: "-", suffix: "starts", delta: "-", meta: "Last Week - plan detail", tone: "mint" },
  { title: "Shrinkage", status: "Watch", value: "-", delta: "-", meta: "Last Week - plan detail", tone: "indigo" },
  { title: "Attrition", status: "Watch", value: "-", delta: "-", meta: "Last Week - plan detail", tone: "peach" },
  { title: "Service Level", status: "Watch", value: "-", delta: "-", meta: "Last Week - plan detail", tone: "teal" },
  { title: "Handling Capacity", status: "Watch", value: "-", suffix: "Calls", delta: "-", meta: "Last Week - plan detail", tone: "sun" }
];

const opsTabs = [
  { id: "critical", label: "Critical (P1)" },
  { id: "warning", label: "Warnings (P3)" },
  { id: "info", label: "Info" }
];

type OpsItem = {
  id: string;
  severity: "critical" | "warning" | "info";
  title: string;
  meta: string;
  cta: string;
  href?: string;
};

type DriverItem = {
  title: string;
  metric: string;
  value: string;
  suffix: string;
};

type ActivityItem = {
  id: string;
  user: string;
  photoUrl: string;
  action: string;
  time: string;
  metrics: Array<{ label: string; value: string; suffix: string }>;
  plan: string;
  range: string;
};

type ActivityRow = {
  id?: number;
  plan_id?: number;
  actor?: string;
  action?: string;
  entity_type?: string;
  entity_id?: string;
  created_at?: string;
  plan_name?: string;
  business_area?: string;
  sub_business_area?: string;
  channel?: string;
  site?: string;
  start_week?: string;
  end_week?: string;
};

function pickPrimaryPlan(plans: PlanRecord[], user?: UserInfo | null) {
  if (!plans.length) return null;
  const userKey = String(user?.name || user?.email || "").trim().toLowerCase();
  if (userKey) {
    const matched = plans.find((plan) => {
      const owner = String(plan.owner || plan.created_by || "").trim().toLowerCase();
      return owner && owner.includes(userKey);
    });
    if (matched) return matched;
  }
  return plans[0];
}

type UserInfo = {
  name?: string;
  email?: string;
  photo_url?: string;
};


function computeStatus(name: string, delta: number | null): string {
  if (delta === null) return "Watch";
  switch (name) {
    case "Staffing Gap":
      return delta < 0 ? "Risk" : "Met";
    case "Hiring":
      return delta < 0 ? "Risk" : "Watch";
    case "Shrinkage":
      // Increases in shrinkage are bad
      return delta > 0 ? "Risk" : "Met";
    case "Attrition":
      // Increases in attrition are bad
      return delta > 0 ? "Risk" : "Met";
    case "Service Level":
      // Decreases in service level are bad
      return delta < 0 ? "Risk" : "Met";
    case "Handling Capacity":
      return delta < 0 ? "Risk" : "Met";
    default:
      return "Watch";
  }
}


const statusStyles: { [key: string]: React.CSSProperties } = {
  risk: {
    backgroundColor: "#fee2e2",
    color: "#b91c1c",
    borderRadius: "4px",
    padding: "0px 4px",
    fontSize: "0.65rem",
    fontWeight: 500,
    marginLeft: "0.5rem"
  },
  watch: {
    backgroundColor: "#fef9c3",
    color: "#854d0e",
    borderRadius: "4px",
    padding: "0px 4px",
    fontSize: "0.65rem",
    fontWeight: 500,
    marginLeft: "0.5rem"
  },
  met: {
    backgroundColor: "#d1fae5",
    color: "#065f46",
    borderRadius: "4px",
    padding: "0px 4px",
    fontSize: "0.65rem",
    fontWeight: 500,
    marginLeft: "0.5rem"
  }
};

function timeAgo(value?: string) {
  if (!value) return "-";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return value;
  const diff = Math.max(0, Date.now() - dt.getTime());
  const minutes = Math.floor(diff / 60000);
  if (minutes < 60) return `${minutes} minutes ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hours ago`;
  const days = Math.floor(hours / 24);
  return `${days} days ago`;
}

function formatRange(start?: string, end?: string) {
  if (!start && !end) return "-";
  const safeStart = start ? start.slice(0, 10) : "-";
  const safeEnd = end ? end.slice(0, 10) : "-";
  return `${safeStart} to ${safeEnd}`;
}

function toNumber(value: any) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const text = String(value).replace(/[%+,]/g, "").trim();
  if (!text) return null;
  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

function extractDateColumns(rows: Array<Record<string, any>>) {
  const set = new Set<string>();
  rows.forEach((row) => {
    Object.keys(row || {}).forEach((key) => {
      const trimmed = String(key).trim();
      if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) {
        set.add(trimmed);
      }
    });
  });
  return Array.from(set).sort();
}

function parseDateKey(key?: string) {
  if (!key) return null;
  const dt = new Date(`${key.trim()}T00:00:00`);
  return Number.isNaN(dt.getTime()) ? null : dt;
}

function filterDateColumnsByRange(columns: string[], from?: string, to?: string) {
  if (!from && !to) return columns;
  const fromDate = from ? new Date(`${from}T00:00:00`) : null;
  const toDate = to ? new Date(`${to}T00:00:00`) : null;
  return columns.filter((col) => {
    const dt = parseDateKey(col);
    if (!dt) return false;
    if (fromDate && dt < fromDate) return false;
    if (toDate && dt > toDate) return false;
    return true;
  });
}

function pickActualColumns(columns: string[]) {
  if (!columns.length) return { current: null as string | null, previous: null as string | null };
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const dated = columns
    .map((col) => ({ col, date: parseDateKey(col) }))
    .filter((item) => item.date)
    .sort((a, b) => (a.date!.getTime() - b.date!.getTime()));
  if (!dated.length) {
    const current = columns[columns.length - 1] || null;
    const previous = columns.length > 1 ? columns[columns.length - 2] : null;
    return { current, previous };
  }
  const actuals = dated.filter((item) => item.date!.getTime() <= today.getTime());
  const usable = actuals.length ? actuals : dated;
  const current = actuals.length ? usable[usable.length - 1]?.col ?? null : null;
  const previous = actuals.length && usable.length > 1 ? usable[usable.length - 2]?.col ?? null : null;
  return { current, previous };
}

function formatDateKey(date: Date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function startOfWeekMonday(date: Date) {
  const day = date.getDay();
  const diff = (day + 6) % 7;
  const monday = new Date(date);
  monday.setDate(date.getDate() - diff);
  monday.setHours(0, 0, 0, 0);
  return monday;
}

function pickMetricRow(rows: Array<Record<string, any>>, keys: string[]) {
  const lowerKeys = keys.map((key) => key.toLowerCase());
  return rows.find((row) => {
    const metric = String(row?.metric ?? "").toLowerCase();
    return lowerKeys.some((key) => key === metric);
  });
}

function metricValue(
  rows: Array<Record<string, any>>,
  metricKeys: string[],
  column?: string
) {
  if (!rows.length) return null;
  const row = pickMetricRow(rows, metricKeys);
  if (!row) return null;
  const columns = extractDateColumns([row]);
  const selected = column || columns[columns.length - 1];
  return selected ? row[selected] : null;
}

function metricDelta(
  rows: Array<Record<string, any>>,
  metricKeys: string[],
  currentColumn?: string | null,
  previousColumn?: string | null
) {
  if (!rows.length) return null;
  const row = pickMetricRow(rows, metricKeys);
  if (!row) return null;
  const columns = extractDateColumns([row]);
  const currentKey = currentColumn || columns[columns.length - 1];
  const previousKey = previousColumn || columns[columns.length - 2];
  if (!currentKey || !previousKey) return null;
  const current = toNumber(row[currentKey]);
  const previous = toNumber(row[previousKey]);
  if (current === null || previous === null) return null;
  return current - previous;
}

function formatDelta(value: number | null, suffix?: string) {
  if (value === null) return "-";
  const sign = value >= 0 ? "+" : "";
  const rounded = Math.abs(value) >= 1 ? value.toFixed(1) : value.toFixed(2);
  return `${sign}${rounded}${suffix || ""}`;
}

function formatValue(value: any, suffix?: string) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "string") return value;
  if (typeof value === "number") return `${value}${suffix || ""}`;
  return `${value}${suffix || ""}`;
}

function resolveHandlingCapacitySuffix(channel?: string) {
  const raw = String(channel || "").trim().toLowerCase();
  if (!raw) return "Calls";
  const callTokens = ["voice", "call", "calls", "phone", "telephony", "inbound", "outbound", "ob"];
  if (callTokens.some((token) => raw.includes(token))) return "Calls";
  return "Items";
}

function applyHandlingSuffix(cards: typeof kpiCards, suffix: string) {
  return cards.map((card) =>
    card.title === "Handling Capacity" ? { ...card, suffix } : card
  );
}

export default function HomePage() {
  const [plans, setPlans] = useState<PlanRecord[]>([]);
  const [plansLoading, setPlansLoading] = useState(true);
  const [opsTab, setOpsTab] = useState("critical");
  const [kpiLoading, setKpiLoading] = useState(true);
  const [kpis, setKpis] = useState(() => applyHandlingSuffix(kpiCards, "Calls"));
  const [opsItems, setOpsItems] = useState<OpsItem[]>([]);
  const [topDrivers, setTopDrivers] = useState<DriverItem[]>([]);
  const [insightSummary, setInsightSummary] = useState("");
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [currentUser, setCurrentUser] = useState<UserInfo | null>(null);
  const [selectedBa, setSelectedBa] = useState("");
  const [selectedSba, setSelectedSba] = useState("");
  const [selectedChannel, setSelectedChannel] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  // Controls visibility of the filter dropdown triggered by the three‑dots button.
  const [showFilters, setShowFilters] = useState(false);
  const toggleFilters = () => setShowFilters((v) => !v);

  const filteredPlans = useMemo(() => {
    const fromDate = dateFrom ? new Date(`${dateFrom}T00:00:00`) : null;
    const toDate = dateTo ? new Date(`${dateTo}T00:00:00`) : null;
    return plans.filter((plan) => {
      if (selectedBa && plan.business_area !== selectedBa) return false;
      if (selectedSba && plan.sub_business_area !== selectedSba) return false;
      if (selectedChannel) {
        const channels = String(plan.channel || "")
          .split(",")
          .map((val) => val.trim())
          .filter(Boolean);
        if (!channels.includes(selectedChannel)) return false;
      }
      if (fromDate || toDate) {
        const start = plan.start_week ? new Date(`${plan.start_week}T00:00:00`) : null;
        const end = plan.end_week ? new Date(`${plan.end_week}T00:00:00`) : null;
        if (fromDate && end && end < fromDate) return false;
        if (toDate && start && start > toDate) return false;
      }
      return true;
    });
  }, [dateFrom, dateTo, plans, selectedBa, selectedChannel, selectedSba]);

  const primaryPlan = useMemo(() => pickPrimaryPlan(filteredPlans, currentUser), [filteredPlans, currentUser]);
  const handlingCapacitySuffix = useMemo(() => {
    const rawChannel = selectedChannel || primaryPlan?.channel || "";
    const firstChannel = String(rawChannel).split(",")[0].trim();
    return resolveHandlingCapacitySuffix(firstChannel);
  }, [primaryPlan?.channel, selectedChannel]);

  useEffect(() => {
    setKpis((prev) => applyHandlingSuffix(prev, handlingCapacitySuffix));
  }, [handlingCapacitySuffix]);

  useEffect(() => {
    let active = true;
    apiGet<UserInfo>("/api/user")
      .then((data) => {
        if (!active) return;
        setCurrentUser(data ?? null);
      })
      .catch(() => {
        if (!active) return;
        setCurrentUser(null);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    apiGet<{ rows?: ActivityRow[] }>("/api/planning/activity?limit=8")
      .then((res) => {
        if (!active) return;
        const rows = res.rows ?? [];
        const items: ActivityItem[] = rows.map((row) => {
          const actor = row.actor || "system";
          const avatar = `https://ui-avatars.com/api/?name=${encodeURIComponent(actor || "User")}`;
          const planLabel =
            row.business_area ||
            row.plan_name ||
            row.channel ||
            "Plan";
          const range = formatRange(row.start_week, row.end_week);
          return {
            id: String(row.id ?? `${actor}-${row.action}-${row.created_at}`),
            user: actor,
            photoUrl: avatar,
            action: row.action || "updated",
            time: timeAgo(row.created_at),
            metrics: [],
            plan: planLabel,
            range
          };
        });
        setActivities(items);
      })
      .catch(() => {
        if (!active) return;
        setActivities([]);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    apiGet<{ plans?: PlanRecord[] }>("/api/planning/plan?status=current")
      .then((res) => {
        if (!active) return;
        setPlans(res.plans ?? []);
      })
      .catch(() => {
        if (!active) return;
        setPlans([]);
      })
      .finally(() => {
        if (!active) return;
        setPlansLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    const primary = primaryPlan;
    if (!primary?.id) {
      setKpiLoading(false);
      setKpis(applyHandlingSuffix(kpiCards, handlingCapacitySuffix));
      setOpsItems([
        {
          id: "info-noplan",
          severity: "info",
          title: "No active plan selected",
          meta: "Create or open a plan to see Inbox items.",
          cta: "View Plans",
          href: "/planning"
        }
      ]);
      setTopDrivers([]);
      setInsightSummary("");
      return () => {
        active = false;
      };
    }
    setKpiLoading(true);
    const loadKpis = async () => {
      const tableNames = ["upper", "fw", "shr", "attr", "nh", "emp", "seat", "ratio"] as const;
      const tableEntries = await Promise.all(
        tableNames.map(async (name) => {
          try {
            const res = await apiGet<{ rows?: Array<Record<string, any>> }>(
              `/api/planning/plan/table?plan_id=${primary.id}&name=${name}`
            );
            return [name, (res.rows ?? []) as Array<Record<string, any>>] as const;
          } catch {
            return [name, [] as Array<Record<string, any>>] as const;
          }
        })
      );
      const tables = tableEntries.reduce<Record<string, Array<Record<string, any>>>>((acc, [key, rows]) => {
        acc[key] = rows;
        return acc;
      }, {});

      const [shrinkageUploadRows, attritionUploadRows] = await Promise.all([
        apiGet<{ rows?: Array<Record<string, any>> }>("/api/forecast/shrinkage")
          .then((res) => res.rows ?? [])
          .catch(() => []),
        apiGet<{ rows?: Array<Record<string, any>> }>("/api/forecast/attrition")
          .then((res) => res.rows ?? [])
          .catch(() => [])
      ]);

      const upper = tables.upper ?? [];
      const shr = tables.shr ?? [];
      const attr = tables.attr ?? [];
      const nh = tables.nh ?? [];
      const fw = tables.fw ?? [];
      const seat = tables.seat ?? [];
      const ratio = tables.ratio ?? [];

          const actualColumnsAll = extractDateColumns([...fw, ...upper, ...shr, ...attr, ...nh]);
          const actualColumns = filterDateColumnsByRange(actualColumnsAll, dateFrom, dateTo);
          let { current: lastActualCol, previous: prevActualCol } = pickActualColumns(actualColumns);
          if (!lastActualCol) {
            const today = new Date();
            const thisWeek = startOfWeekMonday(today);
            const prevWeek = new Date(thisWeek);
            prevWeek.setDate(thisWeek.getDate() - 7);
            const todayKey = formatDateKey(thisWeek);
            const prevKey = formatDateKey(prevWeek);
            const available = new Set(actualColumns);
            if (available.has(todayKey)) lastActualCol = todayKey;
            if (available.has(prevKey)) prevActualCol = prevKey;
            if (!lastActualCol && actualColumns.length) {
              const sorted = actualColumns.slice().sort();
              const past = sorted.filter((col) => {
                const dt = parseDateKey(col);
                return dt ? dt.getTime() <= today.getTime() : false;
              });
              lastActualCol = past[past.length - 1] || sorted[0] || null;
              const idx = lastActualCol ? sorted.indexOf(lastActualCol) : -1;
              prevActualCol = idx > 0 ? sorted[idx - 1] : null;
            }
          }
          const kpiMeta =
            lastActualCol && prevActualCol
              ? `${lastActualCol} Actual vs ${prevActualCol}`
              : lastActualCol
                ? `${lastActualCol} Actual`
                : dateFrom || dateTo
                  ? "No data in selected range"
                  : "Plan detail";

          const staffingValue = metricValue(
            upper,
            [
            "FTE Over/Under MTP Vs Actual",
            "FTE Over/Under (#)",
            "FTE Over/Under Budgeted Vs Actual"
            ],
            lastActualCol || undefined
          );
          const staffingDelta = metricDelta(
            upper,
            [
            "FTE Over/Under MTP Vs Actual",
            "FTE Over/Under (#)",
            "FTE Over/Under Budgeted Vs Actual"
            ],
            lastActualCol,
            prevActualCol
          );
          const hiringValue = metricValue(
            nh,
            ["Planned New Hire HC (#)", "Actual New Hire HC (#)"],
            lastActualCol || undefined
          );
          const hiringDelta = metricDelta(
            nh,
            ["Planned New Hire HC (#)", "Actual New Hire HC (#)"],
            lastActualCol,
            prevActualCol
          );
          const shrinkValue = metricValue(
            shr,
            ["Overall Shrinkage %", "Planned Shrinkage %"],
            lastActualCol || undefined
          );
          const shrinkDelta = metricDelta(
            shr,
            ["Overall Shrinkage %", "Planned Shrinkage %"],
            lastActualCol,
            prevActualCol
          );
          const attrValue = metricValue(
            attr,
            ["Actual Attrition %", "Planned Attrition %"],
            lastActualCol || undefined
          );
          const attrDelta = metricDelta(
            attr,
            ["Actual Attrition %", "Planned Attrition %"],
            lastActualCol,
            prevActualCol
          );
          const slValue = metricValue(upper, ["Projected Service Level"], lastActualCol || undefined);
          const slDelta = metricDelta(upper, ["Projected Service Level"], lastActualCol, prevActualCol);
          const capValue = metricValue(upper, ["Projected Handling Capacity (#)"], lastActualCol || undefined);
          const capDelta = metricDelta(upper, ["Projected Handling Capacity (#)"], lastActualCol, prevActualCol);

          setKpis([
            {
              title: "Staffing Gap",
              value: formatValue(staffingValue),
              suffix: "FTE",
              delta: formatDelta(staffingDelta),
              meta: kpiMeta,
              tone: "blue",
              status: computeStatus("Staffing Gap", staffingDelta)
            },
            {
              title: "Hiring",
              value: formatValue(hiringValue),
              suffix: "starts",
              delta: formatDelta(hiringDelta),
              meta: kpiMeta,
              tone: "mint",
              status: computeStatus("Hiring", hiringDelta)
            },
            {
              title: "Shrinkage",
              value: formatValue(shrinkValue),
              delta: formatDelta(shrinkDelta, "%"),
              meta: kpiMeta,
              tone: "indigo",
              status: computeStatus("Shrinkage", shrinkDelta)
            },
            {
              title: "Attrition",
              value: formatValue(attrValue),
              delta: formatDelta(attrDelta, "%"),
              meta: kpiMeta,
              tone: "peach",
              status: computeStatus("Attrition", attrDelta)
            },
            {
              title: "Service Level",
              value: formatValue(slValue),
              delta: formatDelta(slDelta, "%"),
              meta: kpiMeta,
              tone: "teal",
              status: computeStatus("Service Level", slDelta)
            },
            {
              title: "Handling Capacity",
              value: formatValue(capValue),
              suffix: handlingCapacitySuffix,
              delta: formatDelta(capDelta),
              meta: kpiMeta,
              tone: "sun",
              status: computeStatus("Handling Capacity", capDelta)
            }
          ]);

          const missingShrink = !shrinkageUploadRows.length;
          const missingAttr = !attritionUploadRows.length;
          const missingRoster = !(tables.emp ?? []).length;
          const missingFw = !fw.length;
          const missingSeat = !seat.length;
          const missingRatio = !ratio.length;
          const ops: OpsItem[] = [];
          if (missingShrink) {
            ops.push({
              id: "critical-shrink",
              severity: "critical",
              title: "Update shrinkage data",
              meta: "Shrinkage is missing for this scope. Please upload the latest file.",
              cta: "Upload Now",
              href: "/shrinkage"
            });
          }
          if (missingAttr) {
            ops.push({
              id: "critical-attr",
              severity: "critical",
              title: "Update attrition data",
              meta: "Attrition is missing for this scope. Please upload the latest file.",
              cta: "Upload",
              href: "/shrinkage"
            });
          }
          if (missingSeat) {
            ops.push({
              id: "warning-seat",
              severity: "warning",
              title: "Update seat utilization",
              meta: "Seat utilization is missing for this scope. Please update it.",
              cta: "Update Seats",
              href: primary.id ? `/plan/${primary.id}` : "/planning"
            });
          }
          if (missingRatio) {
            ops.push({
              id: "warning-ratio",
              severity: "warning",
              title: "Update ratios",
              meta: "Ratios are missing for this scope. Please update them.",
              cta: "Update Ratios",
              href: primary.id ? `/plan/${primary.id}` : "/planning"
            });
          }
          if (missingRoster) {
            ops.push({
              id: "info-roster",
              severity: "info",
              title: "Employee roster missing",
              meta: "Roster is missing for this plan. Upload it to enable roster views.",
              cta: "Open Plan",
              href: primary.id ? `/plan/${primary.id}` : "/planning"
            });
          }
          if (missingFw) {
            ops.push({
              id: "info-fw",
              severity: "info",
              title: "Forecast & workload empty",
              meta: "Forecast/Workload is empty for this plan. Recompute or upload.",
              cta: "Open Plan",
              href: primary.id ? `/plan/${primary.id}` : "/planning"
            });
          }
          if (!ops.length) {
            ops.push({
              id: "info-ok",
              severity: "info",
              title: "All key datasets present",
              meta: "No blocking issues detected.",
              cta: "View",
              href: primary.id ? `/plan/${primary.id}` : "/planning"
            });
          }
          const hasCritical = ops.some((item) => item.severity === "critical");
          const hasWarning = ops.some((item) => item.severity === "warning");
          if (!hasCritical) {
            ops.push({
              id: "critical-none",
              severity: "critical",
              title: "No critical issues",
              meta: "All critical datasets look healthy for this scope.",
              cta: "OK"
            });
          }
          if (!hasWarning) {
            ops.push({
              id: "warning-none",
              severity: "warning",
              title: "No warnings",
              meta: "No warnings detected for this scope.",
              cta: "OK"
            });
          }
          setOpsItems(ops);

          const dateCols = extractDateColumns(fw);
          const lastCol = lastActualCol || dateCols[dateCols.length - 1];

          const metricChange = (
            rows: Array<Record<string, any>>,
            keys: string[],
            currentCol?: string | null,
            previousCol?: string | null
          ) => {
            const current = toNumber(metricValue(rows, keys, currentCol || undefined));
            const previous = toNumber(metricValue(rows, keys, previousCol || undefined));
            if (current === null || previous === null) return null;
            return current - previous;
          };

          const driversSpec = [
            {
              title: "Shrinkage",
              rows: shr,
              keys: ["Overall Shrinkage %", "Planned Shrinkage %"],
              worseIf: "increase" as const,
              suffix: "%"
            },
            {
              title: "Attrition",
              rows: attr,
              keys: ["Actual Attrition %", "Planned Attrition %"],
              worseIf: "increase" as const,
              suffix: "%"
            },
            {
              title: "Volume",
              rows: fw,
              keys: ["Forecast", "Actual Volume", "Backlog (Items)"],
              worseIf: "increase" as const,
              suffix: "calls"
            },
            {
              title: "AHT/SUT",
              rows: fw,
              keys: ["AHT", "Average Handle Time", "AHT (sec)", "SUT", "SUT (sec)", "Service Time"],
              worseIf: "increase" as const,
              suffix: "sec"
            },
            {
              title: "Service Level",
              rows: upper,
              keys: ["Projected Service Level"],
              worseIf: "decrease" as const,
              suffix: "%"
            }
          ];

          const driverChanges = driversSpec
            .map((spec) => {
              const delta = metricChange(spec.rows, spec.keys, lastActualCol, prevActualCol);
              if (delta === null) return null;
              const isWorse =
                spec.worseIf === "increase" ? delta > 0 : spec.worseIf === "decrease" ? delta < 0 : false;
              return {
                title: spec.title,
                delta,
                isWorse,
                suffix: spec.suffix
              };
            })
            .filter(Boolean) as Array<{ title: string; delta: number; isWorse: boolean; suffix: string }>;

          const worstDrivers = driverChanges
            .filter((item) => item.isWorse)
            .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
            .slice(0, 3);

          if (staffingDelta !== null && staffingDelta < 0 && worstDrivers.length) {
            const summary = worstDrivers
              .map((item) => `${item.title} ${formatDelta(item.delta, item.suffix)}`)
              .join(", ");
            setInsightSummary(`Staffing gap worsened last week. Drivers: ${summary}.`);
          } else if (staffingDelta !== null && staffingDelta < 0) {
            setInsightSummary("Staffing gap worsened last week. Driver data unavailable for this scope.");
          } else if (staffingDelta !== null && staffingDelta >= 0) {
            setInsightSummary("Staffing gap improved last week. No adverse drivers detected.");
          } else {
            setInsightSummary("Staffing gap trend unavailable for the latest actual week.");
          }

          const driverCandidates = [
            "Shrinkage",
            "Training",
            "Backlog (Items)",
            "Forecast",
            "Actual Volume",
            "Billable FTE Required",
            "Billable Hours",
            "Billable Transactions"
          ];
          const drivers = fw
            .filter((row) =>
              driverCandidates.some((key) => String(row?.metric || "").toLowerCase() === key.toLowerCase())
            )
            .map((row) => {
              const raw = lastCol ? toNumber(row?.[lastCol]) : null;
              const metricName = String(row?.metric || "");
              let suffix = "";
              if (metricName.toLowerCase().includes("%")) suffix = "%";
              else if (metricName.toLowerCase().includes("fte")) suffix = "FTE";
              else if (metricName.toLowerCase().includes("volume") || metricName.toLowerCase().includes("forecast")) suffix = "calls";
              return {
                title: metricName,
                metric: lastCol ? `Week ${lastCol} Actual` : "Latest",
                value: raw === null ? "-" : `${raw >= 0 ? "+" : ""}${raw.toFixed(2)}`,
                suffix
              };
            })
            .sort((a, b) => Math.abs(toNumber(b.value) ?? 0) - Math.abs(toNumber(a.value) ?? 0))
            .slice(0, 3);
          const insightDrivers = worstDrivers.length
            ? worstDrivers.map((item) => ({
                title: item.title,
                metric: prevActualCol ? `Δ vs ${prevActualCol}` : "Δ vs prior week",
                value: formatDelta(item.delta),
                suffix: item.suffix
              }))
            : drivers;
          setTopDrivers(insightDrivers);

    };

    loadKpis()
      .catch(() => {
        if (!active) return;
        setKpis(applyHandlingSuffix(kpiCards, handlingCapacitySuffix));
        setOpsItems([
          {
            id: "info-error",
            severity: "info",
            title: "Unable to load Inbox data",
            meta: "We could not retrieve plan tables right now.",
            cta: "Retry",
            href: primary.id ? `/plan/${primary.id}` : "/planning"
          }
        ]);
      })
      .finally(() => {
        if (!active) return;
        setKpiLoading(false);
      });
    return () => {
      active = false;
    };
  }, [primaryPlan, dateFrom, dateTo, selectedChannel]);

  const visiblePlans = filteredPlans.slice(0, 4);
  const filteredOps = useMemo(
    () => opsItems.filter((item) => item.severity === opsTab),
    [opsItems, opsTab]
  );

  // Keep tab selection user-controlled; empty tabs should still be selectable.

  const baOptions = useMemo(() => {
    const set = new Set<string>();
    plans.forEach((plan) => {
      const ba = String(plan.business_area || "").trim();
      if (ba) set.add(ba);
    });
    return Array.from(set).sort();
  }, [plans]);

  const sbaOptions = useMemo(() => {
    const set = new Set<string>();
    plans.forEach((plan) => {
      if (selectedBa && plan.business_area !== selectedBa) return;
      const sba = String(plan.sub_business_area || "").trim();
      if (sba) set.add(sba);
    });
    return Array.from(set).sort();
  }, [plans, selectedBa]);

  const channelOptions = useMemo(() => {
    const set = new Set<string>();
    plans.forEach((plan) => {
      const channels = String(plan.channel || "")
        .split(",")
        .map((val) => val.trim())
        .filter(Boolean);
      channels.forEach((ch) => set.add(ch));
    });
    return Array.from(set).sort();
  }, [plans]);

  useEffect(() => {
    if (selectedSba && !sbaOptions.includes(selectedSba)) {
      setSelectedSba("");
    }
  }, [selectedSba, sbaOptions]);

  const heroMeta = primaryPlan
    ? `${primaryPlan.plan_name || primaryPlan.business_area || "Capacity Plan"} · ${formatRange(
        primaryPlan.start_week,
        primaryPlan.end_week
      )}`
    : "No active plan selected";

  return (
    <AppShell crumbs="Home" crumbIcon="🏠">
      <div className="home-dashboard">
        {/* Filter fields have been moved into a dropdown triggered by the three‑dots button */}

        <div className="home-grid">
          <section className="home-card home-hero">
            <div className="home-hero-header">
              <div>
                <div className="home-hero-title">Capacity Connect</div>
                <div className="home-hero-meta">{heroMeta}</div>
              </div>
                  <div className="home-hero-actions" style={{ position: "relative" }}>
                    <button type="button" className="btn btn-light" onClick={toggleFilters}>
                      ...
                    </button>
                    {showFilters && (
                      <div
                        style={{
                          position: "absolute",
                          right: 0,
                          top: "calc(100% + 0.5rem)",
                          zIndex: 1000,
                          background: "#fff",
                          border: "1px solid #ddd",
                          borderRadius: "8px",
                          padding: "1rem",
                          width: "320px"
                        }}
                      >
                        <div className="home-filter-fields">
                          <label className="home-filter-field">
                            <span className="label">Business Area</span>
                            <select
                              className="select"
                              value={selectedBa}
                              onChange={(event) => setSelectedBa(event.target.value)}
                            >
                              <option value="">All</option>
                              {baOptions.map((ba) => (
                                <option key={ba} value={ba}>
                                  {ba}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label className="home-filter-field">
                            <span className="label">Sub Business Area</span>
                            <select
                              className="select"
                              value={selectedSba}
                              onChange={(event) => setSelectedSba(event.target.value)}
                            >
                              <option value="">All</option>
                              {sbaOptions.map((sba) => (
                                <option key={sba} value={sba}>
                                  {sba}
                                </option>
                              ))}
                            </select>
                          </label>
                          <div className="home-filter-field home-filter-range">
                            <span className="label">Date Range</span>
                            <div className="home-filter-range-inputs">
                              <input
                                className="input"
                                type="date"
                                value={dateFrom}
                                onChange={(event) => setDateFrom(event.target.value)}
                              />
                              <input
                                className="input"
                                type="date"
                                value={dateTo}
                                onChange={(event) => setDateTo(event.target.value)}
                              />
                            </div>
                          </div>
                          <label className="home-filter-field">
                            <span className="label">Channel</span>
                            <select
                              className="select"
                              value={selectedChannel}
                              onChange={(event) => setSelectedChannel(event.target.value)}
                            >
                              <option value="">All</option>
                              {channelOptions.map((channel) => (
                                <option key={channel} value={channel}>
                                  {channel}
                                </option>
                              ))}
                            </select>
                          </label>
                        </div>
                      </div>
                    )}
                  </div>
            </div>
            <div className="home-kpi-grid">
              {kpis.map((card) => (
                <div key={card.title} className={`kpi-card kpi-card--${card.tone}`}>
                  {/* KPI header with optional status chip */}
                  <div className="kpi-title">
                    <span className={`kpi-dot kpi-dot--${String(card.status || "watch").toLowerCase()}`} />
                    {card.title}
                    {card.status ? (
                      <span
                        className={`kpi-chip kpi-chip--${String(card.status).toLowerCase()}`}
                        style={statusStyles[String(card.status).toLowerCase()]}
                      >
                        {card.status}
                      </span>
                    ) : null}
                  </div>
                  {/* Value and delta appear on the same row to better match the reference design */}
                  <div className="kpi-value">
                    {kpiLoading ? "Loading" : card.value}
                    {card.suffix ? <span> {card.suffix}</span> : null}
                    {!kpiLoading && card.delta ? (
                      <span className="kpi-pill">{card.delta}</span>
                    ) : null}
                  </div>
                  {/* Meta text remains unchanged */}
                  <div className="kpi-meta">{card.meta}</div>
                  {/* Sparkline placeholder */}
                  <div className={`kpi-spark kpi-spark--${String(card.status || "watch").toLowerCase()}`} />
                </div>
              ))}
            </div>
          </section>

          <section className="home-card home-ops">
            <div className="home-ops-header">
              <div className="home-ops-title">Inbox</div>
              {/* <div className="home-ops-nav">
                <button type="button" className="btn btn-light">
                  {"<"}
                </button>
                <button type="button" className="btn btn-light">
                  {">"}
                </button>
              </div> */}
            </div>
            <div className="home-ops-tabs">
                {opsTabs.map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    className={`home-ops-tab home-ops-tab--${tab.id} ${opsTab === tab.id ? "active" : ""}`}
                    onClick={() => setOpsTab(tab.id)}
                  >
                    {tab.label}
                  </button>
              ))}
            </div>
            <div className="home-ops-list">
              {filteredOps.map((item) => (
                <div key={item.id} className={`home-ops-item home-ops-item--${item.severity}`}>
                  <div className="home-ops-item-title">{item.title}</div>
                  <div className="home-ops-item-meta">{item.meta}</div>
                  {item.href ? (
                    <Link className="btn btn-primary" href={item.href}>
                      {item.cta}
                    </Link>
                  ) : (
                    <button type="button" className="btn btn-primary" disabled>
                      {item.cta}
                    </button>
                  )}
                </div>
              ))}
              {!filteredOps.length ? <div className="home-empty">No items right now.</div> : null}
            </div>
          </section>
        </div>

        {/* Bottom grid with two columns: Active Plans (including insight) and Recent Activity. */}
        <div
          className="home-grid home-grid--bottom">
          <div className="left-pane"> 
            <section className="home-card home-plans">
              <div className="home-card-title-row">
                <div className="home-card-title">Active Plans</div>
              </div>
              {plansLoading ? (
                <div>Loading plans...</div>
              ) : visiblePlans.length ? (
                <>
                  <table className="table plans-table">
                    <thead>
                      <tr>
                        <th>Business Area</th>
                        <th>Last Updated</th>
                        <th>Coverage Dates</th>
                        <th>Scenario</th>
                        <th>Owner</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {visiblePlans.map((plan) => (
                        <tr key={plan.id ?? plan.plan_name}>
                          <td>
                            <div className="plan-title">{plan.business_area || plan.plan_name || "Plan"}</div>
                            <div className="plan-subtitle">{plan.sub_business_area || plan.channel || ""}</div>
                          </td>
                          <td>{timeAgo(plan.updated_at)}</td>
                          <td>{formatRange(plan.start_week, plan.end_week)}</td>
                          <td>{plan.plan_type || plan.status || "Forecast"}</td>
                          <td>{plan.owner || plan.created_by || "-"}</td>
                          <td>
                            {plan.id ? (
                              <Link className="btn btn-primary" href={`/plan/${plan.id}`}>
                                Open
                              </Link>
                            ) : (
                              <span className="home-muted">-</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : (
                <div>No plans match the current filters.</div>
              )}
            </section>
            <section className="home-card home-insight"
              /* Use gridColumn to ensure this card spans the first column of the bottom grid */
              style={{ gridColumn: "1 / span 2" }}
            >
              <div className="home-card-title-row">
                <div className="home-card-title">Insight&nbsp;|&nbsp;Top Drivers of Staffing Gap</div>
              </div>
              <div style={{ marginTop: "0.5rem", fontSize: "0.8rem", color: "#555" }}>
                {insightSummary || "No insight available yet."}
              </div>
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: "0.5rem",
                  marginTop: "0.5rem"
                }}
              >
                {topDrivers.map((driver) => (
                  <div
                    key={driver.title}
                    style={{
                      border: "1px solid #dcdcdc",
                      borderRadius: "8px",
                      padding: "0.75rem",
                      minWidth: "150px",
                      fontSize: "0.75rem"
                    }}
                  >
                    <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>{driver.title}</div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span>{driver.metric}</span>
                      <span>
                        {driver.value}&nbsp;
                        <span style={{ fontSize: "0.7rem" }}>{driver.suffix}</span>
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </div>
          <div className="right-pane">
            <section className="home-card home-activity">
              <div className="home-card-title-row">
                <div className="home-card-title">Recent Activity</div>
              </div>
              <div
                className="home-activity-list"
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.75rem",
                  marginTop: "0.5rem"
                }}
              >
                {activities.map((act) => (
                  <div
                    key={act.id}
                    style={{
                      border: "1px solid #e6e6e6",
                      borderRadius: "8px",
                      padding: "0.75rem",
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.25rem"
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      {/* User photo */}
                      <img
                        src={act.photoUrl}
                        alt={act.user}
                        style={{
                          width: "32px",
                          height: "32px",
                          borderRadius: "50%",
                          objectFit: "cover"
                        }}
                      />
                      <div>
                        <div style={{ fontWeight: 600 }}>
                          {act.user} <span style={{ fontWeight: 400 }}>{act.action}</span>
                        </div>
                        <div style={{ fontSize: "0.75rem", color: "#666" }}>{act.time}</div>
                      </div>
                    </div>
                    {act.metrics.length ? (
                      <div style={{ fontSize: "0.75rem", marginLeft: "3rem" }}>
                        {act.metrics.map((m, idx) => (
                          <span key={idx} style={{ marginRight: "0.5rem" }}>
                            {m.label} {m.value} {m.suffix}
                          </span>
                        ))}
                      </div>
                    ) : null}
                    <div
                      style={{
                        fontSize: "0.75rem",
                        color: "#666",
                        marginLeft: "3rem"
                      }}
                    >
                      {act.plan} · {act.range}
                    </div>
                  </div>
                ))}
                {!activities.length ? (
                  <div className="home-empty">No recent activity yet.</div>
                ) : null}
              </div>
            </section>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
