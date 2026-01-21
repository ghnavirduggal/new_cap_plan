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
  { title: "Handling Capacity", status: "Watch", value: "-", suffix: "FTE", delta: "-", meta: "Last Week - plan detail", tone: "sun" }
];

const opsItems = [
  {
    id: "critical-1",
    severity: "critical",
    title: "Missing Shrinkage Upload",
    meta: "Alert Inc. Scotland - % past due",
    cta: "Upload Now"
  },
  {
    id: "critical-2",
    severity: "critical",
    title: "Employee Roster is 9 days old",
    meta: "Alert Inc. Scotland - 43,533 - 96.7%",
    cta: "Refresh Roster"
  },
  {
    id: "warning-1",
    severity: "warning",
    title: "Back-office volume spike",
    meta: "Risk: +12% week-over-week",
    cta: "Review"
  },
  {
    id: "info-1",
    severity: "info",
    title: "Forecast saved",
    meta: "Scenario: Forecast - 3 minutes ago",
    cta: "View"
  }
];

const opsTabs = [
  { id: "critical", label: "Critical (P1)" },
  { id: "warning", label: "Warnings (P3)" },
  { id: "info", label: "Info" }
];

const topDrivers = [
  { title: "C. Serv - Premier", metric: "Head 7", value: "+0.54", suffix: "FTE" },
  { title: "Loan Support", metric: "Head 9", value: "+1.8", suffix: "FTE" },
  { title: "Scheduling Support", metric: "Head 3", value: "+0.3", suffix: "FTE" }
];


const activities = [
  {
    id: "act-1",
    user: "Nat Adams",
    photoUrl: "https://ui-avatars.com/api/?name=Nat+Adams",
    action: "upload Shrinkage data",
    time: "11 minutes ago",
    metrics: [
      { label: "Staffing", value: "+6", suffix: "FTE" },
      { label: "Shrinkage", value: "+0.53", suffix: "%" }
    ],
    plan: "Barclays Financial Assistance",
    range: "Apr 1 – Jun 28"
  },
  {
    id: "act-2",
    user: "Nat Adams",
    photoUrl: "https://ui-avatars.com/api/?name=Nat+Adams",
    action: "update Forecast",
    time: "8 minutes ago",
    metrics: [
      { label: "Staffing", value: "+3", suffix: "FTE" },
      { label: "Service Level", value: "+2.5", suffix: "%" }
    ],
    plan: "Inc Financial Services",
    range: "Mar 1 – Jul 31"
  },
  {
    id: "act-3",
    user: "Nat Adams",
    photoUrl: "https://ui-avatars.com/api/?name=Nat+Adams",
    action: "upload Attrition data",
    time: "2 hours ago",
    metrics: [
      { label: "Attrition", value: "+0.12", suffix: "%" },
      { label: "Shrinkage", value: "-0.2", suffix: "%" }
    ],
    plan: "General Operations",
    range: "Apr 1 – Jul 15"
  }
];


function computeStatus(name: string, delta: number | null): string {
  if (delta === null) return "Watch";
  switch (name) {
    case "Staffing Gap":
      return delta < 0 ? "Risk" : "Free";
    case "Hiring":
      return delta < 0 ? "Risk" : "Watch";
    case "Shrinkage":
      // Increases in shrinkage are bad
      return delta > 0 ? "Risk" : "Free";
    case "Attrition":
      // Increases in attrition are bad
      return delta > 0 ? "Risk" : "Free";
    case "Service Level":
      // Decreases in service level are bad
      return delta < 0 ? "Risk" : "Watch";
    case "Handling Capacity":
      return delta < 0 ? "Risk" : "Free";
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
  free: {
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
      if (/^\d{4}-\d{2}-\d{2}$/.test(key)) {
        set.add(key);
      }
    });
  });
  return Array.from(set).sort();
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
  metricKeys: string[]
) {
  if (!rows.length) return null;
  const row = pickMetricRow(rows, metricKeys);
  if (!row) return null;
  const columns = extractDateColumns([row]);
  if (columns.length < 2) return null;
  const current = toNumber(row[columns[columns.length - 1]]);
  const previous = toNumber(row[columns[columns.length - 2]]);
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

export default function HomePage() {
  const [plans, setPlans] = useState<PlanRecord[]>([]);
  const [plansLoading, setPlansLoading] = useState(true);
  const [opsTab, setOpsTab] = useState("critical");
  const [kpiLoading, setKpiLoading] = useState(true);
  const [kpis, setKpis] = useState(kpiCards);
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
    const primary = filteredPlans[0];
    if (!primary?.id) {
      setKpiLoading(false);
      setKpis(kpiCards);
      return () => {
        active = false;
      };
    }
    setKpiLoading(true);
    const loadKpis = async () => {
      const payload = { plan_id: primary.id, grain: "week", persist: false };
      for (let attempt = 0; attempt < 10; attempt += 1) {
        const res = await apiPost<{
          status?: string;
          data?: { tables?: Record<string, Array<Record<string, any>>>; upper?: Array<Record<string, any>> };
          job?: Record<string, any>;
        }>("/api/planning/plan/detail/compute", payload);
        if (res.status === "failed") {
          throw new Error(res.job?.error || "Plan detail calculations failed.");
        }
        if (res.status === "missing") {
          throw new Error("Plan not found.");
        }
        if (res.status === "ready") {
          const upper = res.data?.upper ?? [];
          const tables = res.data?.tables ?? {};
          const shr = tables.shr ?? [];
          const attr = tables.attr ?? [];
          const nh = tables.nh ?? [];

          const staffingValue = metricValue(upper, [
            "FTE Over/Under MTP Vs Actual",
            "FTE Over/Under (#)",
            "FTE Over/Under Budgeted Vs Actual"
          ]);
          const staffingDelta = metricDelta(upper, [
            "FTE Over/Under MTP Vs Actual",
            "FTE Over/Under (#)",
            "FTE Over/Under Budgeted Vs Actual"
          ]);
          const hiringValue = metricValue(nh, ["Planned New Hire HC (#)", "Actual New Hire HC (#)"]);
          const hiringDelta = metricDelta(nh, ["Planned New Hire HC (#)", "Actual New Hire HC (#)"]);
          const shrinkValue = metricValue(shr, ["Overall Shrinkage %", "Planned Shrinkage %"]);
          const shrinkDelta = metricDelta(shr, ["Overall Shrinkage %", "Planned Shrinkage %"]);
          const attrValue = metricValue(attr, ["Actual Attrition %", "Planned Attrition %"]);
          const attrDelta = metricDelta(attr, ["Actual Attrition %", "Planned Attrition %"]);
          const slValue = metricValue(upper, ["Projected Service Level"]);
          const slDelta = metricDelta(upper, ["Projected Service Level"]);
          const capValue = metricValue(upper, ["Projected Handling Capacity (#)"]);
          const capDelta = metricDelta(upper, ["Projected Handling Capacity (#)"]);

          setKpis([
            {
              title: "Staffing Gap",
              value: formatValue(staffingValue),
              suffix: "FTE",
              delta: formatDelta(staffingDelta),
              meta: "Last Week - plan detail",
              tone: "blue",
              status: computeStatus("Staffing Gap", staffingDelta)
            },
            {
              title: "Hiring",
              value: formatValue(hiringValue),
              suffix: "starts",
              delta: formatDelta(hiringDelta),
              meta: "Last Week - plan detail",
              tone: "mint",
              status: computeStatus("Hiring", hiringDelta)
            },
            {
              title: "Shrinkage",
              value: formatValue(shrinkValue),
              delta: formatDelta(shrinkDelta, "%"),
              meta: "Last Week - plan detail",
              tone: "indigo",
              status: computeStatus("Shrinkage", shrinkDelta)
            },
            {
              title: "Attrition",
              value: formatValue(attrValue),
              delta: formatDelta(attrDelta, "%"),
              meta: "Last Week - plan detail",
              tone: "peach",
              status: computeStatus("Attrition", attrDelta)
            },
            {
              title: "Service Level",
              value: formatValue(slValue),
              delta: formatDelta(slDelta, "%"),
              meta: "Last Week - plan detail",
              tone: "teal",
              status: computeStatus("Service Level", slDelta)
            },
            {
              title: "Handling Capacity",
              value: formatValue(capValue),
              suffix: "FTE",
              delta: formatDelta(capDelta),
              meta: "Last Week - plan detail",
              tone: "sun",
              status: computeStatus("Handling Capacity", capDelta)
            }
          ]);
          return;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 900));
      }
      throw new Error("Plan detail calculations timed out.");
    };

    loadKpis()
      .catch(() => {
        if (!active) return;
        setKpis(kpiCards);
      })
      .finally(() => {
        if (!active) return;
        setKpiLoading(false);
      });
    return () => {
      active = false;
    };
  }, [filteredPlans]);

  const visiblePlans = filteredPlans.slice(0, 4);
  const filteredOps = useMemo(
    () => opsItems.filter((item) => item.severity === opsTab),
    [opsTab]
  );

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

  return (
    <AppShell crumbs="Home" crumbIcon="🏠">
      <div className="home-dashboard">
        {/* Filter fields have been moved into a dropdown triggered by the three‑dots button */}

        <div className="home-grid">
          <section className="home-card home-hero">
            <div className="home-hero-header">
              <div>
                <div className="home-hero-title">Capacity Connect</div>
                <div className="home-hero-meta">Forecast - default - version A</div>
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
                    <span className="kpi-dot" />
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
                  <div className="kpi-spark" />
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
                  <button type="button" className="btn btn-primary">
                    {item.cta}
                  </button>
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
                <div className="home-card-actions">
                  <button type="button" className="btn btn-light">
                    Grid
                  </button>
                  <button type="button" className="btn btn-light">
                    Export
                  </button>
                </div>
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
                    <div style={{ fontSize: "0.75rem", marginLeft: "3rem" }}>
                      {act.metrics.map((m, idx) => (
                        <span key={idx} style={{ marginRight: "0.5rem" }}>
                          {m.label} {m.value} {m.suffix}
                        </span>
                      ))}
                    </div>
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
              </div>
            </section>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
