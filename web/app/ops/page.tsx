"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import AppShell from "../_components/AppShell";
import BarChart from "../_components/BarChart";
import DataTable from "../_components/DataTable";
import LineChart from "../_components/LineChart";
import MultiSelect from "../_components/MultiSelect";
import PieChart from "../_components/PieChart";
import WaterfallChart from "../_components/WaterfallChart";
import { useToast } from "../_components/ToastProvider";
import { apiGet, apiPost } from "../../lib/api";

type OpsOptions = {
  business_areas?: string[];
  sub_business_areas?: string[];
  channels?: string[];
  locations?: string[];
  sites?: string[];
};

type OpsSummary = {
  kpis?: { required_fte: number; supply_fte: number; gap_fte: number };
  insights?: {
    coverage_pct?: number;
    total_calls?: number;
    total_items?: number;
    avg_aht_sec?: number;
    peak_required?: { date?: string; value?: number };
    peak_supply?: { date?: string; value?: number };
    worst_gap?: { date?: string; value?: number };
    best_gap?: { date?: string; value?: number };
    top_shortfalls?: Array<{ date: string; required_fte: number; supply_fte: number; gap_fte: number }>;
    top_volume_days?: Array<{ date: string; volume: number }>;
  };
  line?: { x: string[]; series: Array<{ name: string; points: Array<{ x: string; y: number }> }> };
  bar?: { labels: string[]; series: Array<{ name: string; values: number[] }> };
  pie?: { labels: string[]; values: number[] };
  site?: { labels: string[]; values: number[] };
  waterfall?: { labels: string[]; values: number[]; measure?: Array<"relative" | "total"> };
  summary?: Array<Record<string, any>>;
};

const GRAIN_OPTIONS = [
  { label: "Interval", value: "interval" },
  { label: "Daily", value: "D" },
  { label: "Weekly", value: "W" },
  { label: "Monthly", value: "M" },
  { label: "Quarterly", value: "Q" },
  { label: "Yearly", value: "Y" }
];

function todayIso(daysAgo = 0) {
  const dt = new Date();
  dt.setDate(dt.getDate() - daysAgo);
  return dt.toISOString().slice(0, 10);
}

function formatNumber(value: number) {
  if (!Number.isFinite(value)) return "0";
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

function formatCount(value: number) {
  if (!Number.isFinite(value)) return "0";
  return Math.round(value).toLocaleString();
}

function formatSeconds(value?: number) {
  if (!Number.isFinite(value ?? NaN)) return "--";
  const total = Math.max(0, Math.round(value ?? 0));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export default function OpsPage() {
  const { notify } = useToast();

  const [options, setOptions] = useState<Required<OpsOptions>>({
    business_areas: [],
    sub_business_areas: [],
    channels: ["Voice", "Back Office", "Outbound", "Blended", "Chat", "MessageUs"],
    locations: [],
    sites: []
  });
  const [filters, setFilters] = useState({
    startDate: todayIso(28),
    endDate: todayIso(0),
    grain: "D",
    ba: [] as string[],
    sba: [] as string[],
    ch: [] as string[],
    loc: [] as string[],
    site: [] as string[]
  });
  const [summary, setSummary] = useState<OpsSummary | null>(null);
  const [partStatus, setPartStatus] = useState<Record<string, "idle" | "loading" | "refreshing" | "ready" | "error">>({});
  const pollRef = useRef<Record<string, number | null>>({});

  const filterKey = useMemo(
    () =>
      [
        filters.startDate,
        filters.endDate,
        filters.grain,
        filters.ba.join("|"),
        filters.sba.join("|"),
        filters.ch.join("|"),
        filters.loc.join("|"),
        filters.site.join("|")
      ].join("::"),
    [filters]
  );
  const parts = ["kpis", "line", "bar", "pie", "site", "waterfall", "summary"];

  const loadOptions = async () => {
    const params = new URLSearchParams();
    if (filters.ba.length) params.set("ba", filters.ba.join(","));
    if (filters.sba.length) params.set("sba", filters.sba.join(","));
    if (filters.ch.length) params.set("ch", filters.ch.join(","));
    if (filters.loc.length) params.set("loc", filters.loc.join(","));
    try {
      const res = await apiGet<OpsOptions>(`/api/forecast/ops/options${params.toString() ? `?${params.toString()}` : ""}`);
      const next = {
        business_areas: res.business_areas ?? [],
        sub_business_areas: res.sub_business_areas ?? [],
        channels: res.channels ?? options.channels,
        locations: res.locations ?? [],
        sites: res.sites ?? []
      };
      setOptions(next);
      setFilters((prev) => ({
        ...prev,
        sba: prev.sba.filter((val) => next.sub_business_areas.includes(val)),
        ch: prev.ch.filter((val) => next.channels.includes(val)),
        loc: prev.loc.filter((val) => next.locations.includes(val)),
        site: prev.site.filter((val) => next.sites.includes(val))
      }));
    } catch (error: any) {
      notify("error", error?.message || "Could not load options.");
    }
  };

  const hasPartData = (part: string) => {
    if (!summary) return false;
    switch (part) {
      case "kpis":
        return Boolean(summary.kpis);
      case "line":
        return Boolean(summary.line);
      case "bar":
        return Boolean(summary.bar);
      case "pie":
        return Boolean(summary.pie);
      case "site":
        return Boolean(summary.site);
      case "waterfall":
        return Boolean(summary.waterfall);
      case "summary":
        return Boolean(summary.summary);
      default:
        return false;
    }
  };

  const isPartLoading = (part: string) => {
    const status = partStatus[part] ?? "idle";
    return status === "loading" || status === "refreshing";
  };

  const showPartSkeleton = (part: string) => isPartLoading(part) && !hasPartData(part);

  const clearPoll = (part?: string) => {
    if (!part) {
      Object.values(pollRef.current).forEach((id) => {
        if (id) window.clearTimeout(id);
      });
      pollRef.current = {};
      return;
    }
    const id = pollRef.current[part];
    if (id) window.clearTimeout(id);
    pollRef.current[part] = null;
  };

  const loadPart = async (part: string, allowPoll = true) => {
    clearPoll(part);
    setPartStatus((prev) => ({
      ...prev,
      [part]: hasPartData(part) ? "refreshing" : "loading"
    }));
    try {
      const res = await apiPost<any>("/api/forecast/ops/summary/part", {
        part,
        start_date: filters.startDate,
        end_date: filters.endDate,
        grain: filters.grain,
        ba: filters.ba,
        sba: filters.sba,
        ch: filters.ch,
        loc: filters.loc,
        site: filters.site
      });
      if (res && typeof res === "object" && "status" in res) {
        const status = res.status as string;
        const data = res.data as OpsSummary | null | undefined;
        if (data) {
          setSummary((prev) => ({ ...(prev ?? {}), ...data }));
        }
        if (status === "ready") {
          setPartStatus((prev) => ({ ...prev, [part]: "ready" }));
        } else if (status === "error") {
          setPartStatus((prev) => ({ ...prev, [part]: "error" }));
        } else {
          setPartStatus((prev) => ({ ...prev, [part]: "refreshing" }));
          if (allowPoll) {
            pollRef.current[part] = window.setTimeout(() => {
              void loadPart(part, true);
            }, 1500);
          }
        }
      } else {
        setSummary((prev) => ({ ...(prev ?? {}), ...(res as OpsSummary) }));
        setPartStatus((prev) => ({ ...prev, [part]: "ready" }));
      }
    } catch (error: any) {
      notify("error", error?.message || "Could not load operational metrics.");
      setPartStatus((prev) => ({ ...prev, [part]: "error" }));
    }
  };

  useEffect(() => {
    void loadOptions();
  }, [filters.ba.join("|"), filters.sba.join("|"), filters.ch.join("|"), filters.loc.join("|")]);

  useEffect(() => {
    parts.forEach((part) => void loadPart(part));
    return () => clearPoll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterKey]);

  const resetFilters = () => {
    setFilters((prev) => ({ ...prev, ba: [], sba: [], ch: [], loc: [], site: [] }));
  };

  const kpis = summary?.kpis ?? { required_fte: 0, supply_fte: 0, gap_fte: 0 };
  const insights = summary?.insights ?? {};
  const kpisLoading = showPartSkeleton("kpis");
  const kpisUpdating = isPartLoading("kpis");
  const lineLoading = showPartSkeleton("line");
  const barLoading = showPartSkeleton("bar");
  const pieLoading = showPartSkeleton("pie");
  const siteLoading = showPartSkeleton("site");
  const waterfallLoading = showPartSkeleton("waterfall");
  const summaryLoading = showPartSkeleton("summary");
  const anyLoading = parts.some((part) => isPartLoading(part));
  const coverage = kpis.required_fte > 0 ? (kpis.supply_fte / kpis.required_fte) * 100 : 0;
  const gapIsSurplus = kpis.gap_fte <= 0;
  const gapLabel = gapIsSurplus ? "Surplus" : "Shortfall";
  const absGap = Math.abs(kpis.gap_fte);
  const grainLabel = GRAIN_OPTIONS.find((opt) => opt.value === filters.grain)?.label ?? filters.grain;
  const grainUnitLabel = (() => {
    switch (filters.grain) {
      case "interval":
        return "Intervals";
      case "W":
        return "Weeks";
      case "M":
        return "Months";
      case "Q":
        return "Quarters";
      case "Y":
        return "Years";
      case "D":
      default:
        return "Days";
    }
  })();
  const channelTag = (() => {
    if (!filters.ch.length) return "All Channels";
    if (filters.ch.length === 1) return filters.ch[0];
    return filters.ch.join(", ");
  })();
  const quickRanges = [
    { label: "Last 7 Days", days: 7 },
    { label: "Last 28 Days", days: 28 },
    { label: "Last 90 Days", days: 90 }
  ];

  const channelMix = useMemo(() => {
    const labels = summary?.pie?.labels ?? [];
    const values = summary?.pie?.values ?? [];
    const total = values.reduce((sum, val) => sum + (Number.isFinite(val) ? val : 0), 0);
    const rows = labels.map((label, idx) => {
      const value = values[idx] ?? 0;
      const pct = total > 0 ? (value / total) * 100 : 0;
      return { label, value, pct };
    });
    return rows.sort((a, b) => b.value - a.value).slice(0, 4);
  }, [summary?.pie?.labels, summary?.pie?.values]);

  const topShortfalls = insights.top_shortfalls ?? [];
  const topVolumeDays = insights.top_volume_days ?? [];
  const summaryRows = useMemo(() => {
    const rows = summary?.summary ?? [];
    return rows.map((row) => {
      const next = { ...row };
      if (typeof next.volume === "number") next.volume = Math.round(next.volume);
      if (typeof next.items === "number") next.items = Math.round(next.items);
      return next;
    });
  }, [summary?.summary]);

  const activeFilters = [
    { label: "Business Area", values: filters.ba },
    { label: "Sub Business Area", values: filters.sba },
    { label: "Channel", values: filters.ch },
    { label: "Location", values: filters.loc },
    { label: "Site", values: filters.site }
  ].filter((item) => item.values.length);

  const LoadingPill = ({ label }: { label?: string }) => (
    <span className="ops-loading-pill">
      <span className="ops-loading-spinner" />
      {label ?? "Updating"}
    </span>
  );

  return (
    <AppShell crumbs="CAP-CONNECT / Ops">
      <div className="ops-page">
        <section className="ops-hero">
          <div className="ops-hero__content">
            <div className="ops-hero__tag">Ops Command Center</div>
            <h1>Operational Intelligence</h1>
            <p>Track staffing coverage, workload mix, and capacity health with live slicers and instant rollups.</p>
            <div className="ops-hero__meta">
              <span>{filters.startDate}</span>
              <span>→</span>
              <span>{filters.endDate}</span>
              <span className="ops-hero__dot" />
              <span>{grainLabel}</span>
              {anyLoading ? <LoadingPill label="Refreshing" /> : null}
            </div>
            {activeFilters.length ? (
              <div className="ops-chip-row">
                {activeFilters.map((item) => (
                  <span key={item.label} className="ops-chip">
                    <strong>{item.label}:</strong> {item.values.join(", ")}
                  </span>
                ))}
              </div>
            ) : (
              <div className="ops-chip-row">
                <span className="ops-chip ops-chip--muted">All scopes included</span>
              </div>
            )}
          </div>
          <div className="ops-hero__kpis">
            <div className="stat-card stat-card--teal">
              <div className="stat-header">Required FTE</div>
              <div className="ops-kpi-value">{kpisLoading ? "—" : formatNumber(kpis.required_fte)}</div>
              <div className="ops-kpi-sub">Demanded workforce</div>
              {kpisUpdating ? <div className="ops-kpi-loading">Updating…</div> : null}
            </div>
            <div className="stat-card stat-card--blue">
              <div className="stat-header">Supply FTE</div>
              <div className="ops-kpi-value">{kpisLoading ? "—" : formatNumber(kpis.supply_fte)}</div>
              <div className="ops-kpi-sub">Roster + hiring supply</div>
              {kpisUpdating ? <div className="ops-kpi-loading">Updating…</div> : null}
            </div>
            <div className={`stat-card ${gapIsSurplus ? "stat-card--green" : "stat-card--red"}`}>
              <div className="stat-header">{gapLabel}</div>
              <div className="ops-kpi-value">{kpisLoading ? "—" : formatNumber(absGap)}</div>
              <div className="ops-kpi-sub">Gap vs required</div>
              {kpisUpdating ? <div className="ops-kpi-loading">Updating…</div> : null}
            </div>
          </div>
        </section>

        <section className="ops-panel ops-panel--filters">
          <div className="ops-panel__header">
            <div>
              <h2>Slicers &amp; View Controls</h2>
              <p>Slice by BA, channel, or location. Combine with time grain to explore patterns.</p>
            </div>
            <div className="ops-panel__actions">
              {quickRanges.map((range) => (
                <button
                  key={range.label}
                  type="button"
                  className="btn btn-ghost"
                  onClick={() =>
                    setFilters((prev) => ({
                      ...prev,
                      startDate: todayIso(range.days),
                      endDate: todayIso(0)
                    }))
                  }
                >
                  {range.label}
                </button>
              ))}
              <button type="button" className="btn btn-secondary" onClick={resetFilters}>
                Reset Filters
              </button>
            </div>
          </div>
          <div className="ops-filter-grid">
            <div>
              <div className="label">Date Range</div>
              <div className="ops-date-row">
                <input
                  type="date"
                  className="input"
                  value={filters.startDate}
                  onChange={(event) => setFilters((prev) => ({ ...prev, startDate: event.target.value }))}
                />
                <input
                  type="date"
                  className="input"
                  value={filters.endDate}
                  onChange={(event) => setFilters((prev) => ({ ...prev, endDate: event.target.value }))}
                />
              </div>
            </div>
            <div>
              <div className="label">Grain</div>
              <select
                className="select"
                value={filters.grain}
                onChange={(event) => setFilters((prev) => ({ ...prev, grain: event.target.value }))}
              >
                {GRAIN_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="label">Business Area</div>
              <MultiSelect
                options={options.business_areas.map((ba) => ({ label: ba, value: ba }))}
                values={filters.ba}
                placeholder="Select Business Area"
                onChange={(values) => setFilters((prev) => ({ ...prev, ba: values, sba: [], site: [] }))}
              />
            </div>
            <div>
              <div className="label">Sub Business Area</div>
              <MultiSelect
                options={options.sub_business_areas.map((sba) => ({ label: sba, value: sba }))}
                values={filters.sba}
                placeholder="Select Sub Business Area"
                onChange={(values) => setFilters((prev) => ({ ...prev, sba: values }))}
              />
            </div>
            <div>
              <div className="label">Channel</div>
              <MultiSelect
                options={options.channels.map((channel) => ({ label: channel, value: channel }))}
                values={filters.ch}
                placeholder="Select Channel"
                onChange={(values) => setFilters((prev) => ({ ...prev, ch: values, site: [] }))}
              />
            </div>
            <div>
              <div className="label">Location</div>
              <MultiSelect
                options={options.locations.map((loc) => ({ label: loc, value: loc }))}
                values={filters.loc}
                placeholder="Select Location"
                onChange={(values) => setFilters((prev) => ({ ...prev, loc: values, site: [] }))}
              />
            </div>
            <div>
              <div className="label">Site</div>
              <MultiSelect
                options={options.sites.map((site) => ({ label: site, value: site }))}
                values={filters.site}
                placeholder="Select Site"
                onChange={(values) => setFilters((prev) => ({ ...prev, site: values }))}
              />
            </div>
          </div>
        </section>

        <section className="ops-panel ops-panel--insights">
          <div className="ops-insight-grid">
            <div className="ops-insight-card">
              <div className="ops-insight-label">Coverage</div>
              <div className="ops-insight-value">{kpisLoading ? "—" : `${formatNumber(insights.coverage_pct ?? coverage)}%`}</div>
              <div className="ops-insight-sub">Supply ÷ Required</div>
            </div>
            <div className="ops-insight-card">
              <div className="ops-insight-label">Net Gap</div>
              <div className="ops-insight-value">{kpisLoading ? "—" : `${gapIsSurplus ? "+" : "-"}${formatNumber(absGap)}`}</div>
              <div className="ops-insight-sub">{gapIsSurplus ? "Capacity buffer available" : "Additional hiring needed"}</div>
            </div>
            <div className="ops-insight-card">
              <div className="ops-insight-label">Focus Window</div>
              <div className="ops-insight-value">{grainLabel}</div>
              <div className="ops-insight-sub">Trend granularity</div>
            </div>
            <div className="ops-insight-card">
              <div className="ops-insight-label">Avg AHT</div>
              <div className="ops-insight-value">{kpisLoading ? "—" : formatSeconds(insights.avg_aht_sec)}</div>
              <div className="ops-insight-sub">Weighted by volume</div>
            </div>
          </div>
        </section>

        <section className="ops-panel ops-panel--signals">
          <div className="ops-panel__header">
            <div>
              <h2>Operational Signals</h2>
              <p>Quick reads on demand, staffing risk, and workload concentration.</p>
            </div>
            {anyLoading ? <LoadingPill /> : null}
          </div>
          <div className="ops-signal-grid">
            <div className="ops-signal-card">
              <h4>Demand &amp; Workload</h4>
              <div className="ops-signal-metrics">
                <div>
                  <span>Total Voice Calls</span>
                  <strong>{kpisLoading ? "—" : formatCount(insights.total_calls ?? 0)}</strong>
                </div>
                <div>
                  <span>Total BO Items</span>
                  <strong>{kpisLoading ? "—" : formatCount(insights.total_items ?? 0)}</strong>
                </div>
                <div>
                  <span>Peak Required</span>
                  <strong>{kpisLoading ? "—" : formatNumber(insights.peak_required?.value ?? 0)}</strong>
                  <em>{kpisLoading ? "--" : (insights.peak_required?.date ?? "--")}</em>
                </div>
              </div>
            </div>
            <div className="ops-signal-card">
              <h4>Staffing Extremes</h4>
              <div className="ops-signal-metrics">
                <div>
                  <span>Peak Supply</span>
                  <strong>{kpisLoading ? "—" : formatNumber(insights.peak_supply?.value ?? 0)}</strong>
                  <em>{kpisLoading ? "--" : (insights.peak_supply?.date ?? "--")}</em>
                </div>
                <div>
                  <span>Worst Shortfall</span>
                  <strong>{kpisLoading ? "—" : formatNumber(insights.worst_gap?.value ?? 0)}</strong>
                  <em>{kpisLoading ? "--" : (insights.worst_gap?.date ?? "--")}</em>
                </div>
                <div>
                  <span>Best Surplus</span>
                  <strong>{kpisLoading ? "—" : formatNumber(Math.abs(insights.best_gap?.value ?? 0))}</strong>
                  <em>{kpisLoading ? "--" : (insights.best_gap?.date ?? "--")}</em>
                </div>
              </div>
            </div>
            <div className="ops-signal-card">
              <h4>Channel Mix</h4>
              <div className="ops-signal-list">
                {pieLoading ? (
                  <div className="ops-signal-empty">Loading channel mix…</div>
                ) : channelMix.length ? (
                  channelMix.map((item) => (
                    <div key={item.label} className="ops-signal-row">
                      <span>{item.label}</span>
                      <strong>{formatCount(item.value)} ({item.pct.toFixed(1)}%)</strong>
                    </div>
                  ))
                ) : (
                  <div className="ops-signal-empty">No channel mix data.</div>
                )}
              </div>
            </div>
            <div className="ops-signal-card">
              <h4>Top Shortfall {grainUnitLabel}</h4>
              <div className="ops-signal-list">
                {kpisLoading ? (
                  <div className="ops-signal-empty">Loading shortfalls…</div>
                ) : topShortfalls.length ? (
                  topShortfalls.map((row) => (
                    <div key={row.date} className="ops-signal-row">
                      <span>{row.date}</span>
                      <strong>{formatNumber(row.gap_fte)}</strong>
                    </div>
                  ))
                ) : (
                  <div className="ops-signal-empty">No shortfalls detected.</div>
                )}
              </div>
            </div>
            <div className="ops-signal-card">
              <h4>Peak Volume {grainUnitLabel}</h4>
              <div className="ops-signal-list">
                {kpisLoading ? (
                  <div className="ops-signal-empty">Loading volume days…</div>
                ) : topVolumeDays.length ? (
                  topVolumeDays.map((row) => (
                    <div key={row.date} className="ops-signal-row">
                      <span>{row.date}</span>
                      <strong>{formatCount(row.volume)}</strong>
                    </div>
                  ))
                ) : (
                  <div className="ops-signal-empty">No volume data.</div>
                )}
              </div>
            </div>
          </div>
        </section>

        <section className="ops-chart-layout">
          <div className="ops-chart-card ops-chart-card--main">
            <div className="ops-chart-head">
              <h3>Requirements vs Supply</h3>
              <span className="ops-chart-tag">{grainLabel} View</span>
            </div>
            {lineLoading ? <div className="ops-chart-loading">Loading chart…</div> : <LineChart data={summary?.line ?? null} height={320} />}
          </div>
          <div className="ops-chart-card">
            <div className="ops-chart-head">
              <h3>Workload by Time</h3>
              <span className="ops-chart-tag">{channelTag}</span>
            </div>
            {barLoading ? (
              <div className="ops-chart-loading">Loading chart…</div>
            ) : (
              <BarChart
                data={summary?.bar ?? null}
                stacked
                height={260}
                valueFormatter={(value) => formatCount(value)}
              />
            )}
          </div>
          <div className="ops-chart-card">
            <div className="ops-chart-head">
              <h3>Workload Share</h3>
              <span className="ops-chart-tag">By Channel</span>
            </div>
            {pieLoading ? (
              <div className="ops-chart-loading">Loading chart…</div>
            ) : (
              <PieChart data={summary?.pie ?? null} height={260} valueFormatter={(value) => formatCount(value)} />
            )}
          </div>
          <div className="ops-chart-card">
            <div className="ops-chart-head">
              <h3>Site Load</h3>
              <span className="ops-chart-tag">Top Sites</span>
            </div>
            {siteLoading ? (
              <div className="ops-chart-loading">Loading chart…</div>
            ) : (
              <BarChart
                data={summary?.site ? { labels: summary.site.labels, series: [{ name: "Workload", values: summary.site.values }] } : null}
                height={260}
                valueFormatter={(value) => formatCount(value)}
              />
            )}
          </div>
          <div className="ops-chart-card">
            <div className="ops-chart-head">
              <h3>Net Gap Waterfall</h3>
              <span className="ops-chart-tag">Required vs Supply</span>
            </div>
            {waterfallLoading ? <div className="ops-chart-loading">Loading chart…</div> : <WaterfallChart data={summary?.waterfall ?? null} height={260} />}
          </div>
        </section>

        <section className="ops-panel">
          <div className="ops-panel__header">
            <div>
              <h2>Scope Summary</h2>
              <p>Aggregate totals for each BA, channel, site, and location combination.</p>
            </div>
            {summaryLoading || anyLoading ? <LoadingPill /> : null}
          </div>
          {summaryLoading ? (
            <div className="ops-table-loading">Loading summary…</div>
          ) : (
            <DataTable data={summaryRows} />
          )}
        </section>
      </div>
    </AppShell>
  );
}
