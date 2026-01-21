"use client";

import { useEffect, useMemo, useState } from "react";
import AppShell from "../_components/AppShell";
import BarChart from "../_components/BarChart";
import DataTable from "../_components/DataTable";
import LineChart from "../_components/LineChart";
import MultiSelect from "../_components/MultiSelect";
import PieChart from "../_components/PieChart";
import WaterfallChart from "../_components/WaterfallChart";
import { useGlobalLoader } from "../_components/GlobalLoader";
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

export default function OpsPage() {
  const { notify } = useToast();
  const { setLoading } = useGlobalLoader();

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

  const loadSummary = async () => {
    setLoading(true);
    try {
      const res = await apiPost<OpsSummary>("/api/forecast/ops/summary", {
        start_date: filters.startDate,
        end_date: filters.endDate,
        grain: filters.grain,
        ba: filters.ba,
        sba: filters.sba,
        ch: filters.ch,
        loc: filters.loc,
        site: filters.site
      });
      setSummary(res);
    } catch (error: any) {
      notify("error", error?.message || "Could not load operational metrics.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadOptions();
  }, [filters.ba.join("|"), filters.sba.join("|"), filters.ch.join("|"), filters.loc.join("|")]);

  useEffect(() => {
    void loadSummary();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterKey]);

  const resetFilters = () => {
    setFilters((prev) => ({ ...prev, ba: [], sba: [], ch: [], loc: [], site: [] }));
  };

  const kpis = summary?.kpis ?? { required_fte: 0, supply_fte: 0, gap_fte: 0 };

  return (
    <AppShell crumbs="CAP-CONNECT / Ops">
      <div className="ops-page">
        <section className="section">
          <h2>Operational Dashboard</h2>
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
            <div className="ops-filter-action">
              <button type="button" className="btn btn-secondary bhaiya" onClick={resetFilters}>
                Reset Filters
              </button>
            </div>
          </div>
        </section>

        <section className="section">
          <div className="ops-kpi-grid">
            <div className="stat-card stat-card--teal">
              <div className="stat-header">Required FTE</div>
              <div className="ops-kpi-value">{formatNumber(kpis.required_fte)}</div>
            </div>
            <div className="stat-card stat-card--blue">
              <div className="stat-header">Supply FTE</div>
              <div className="ops-kpi-value">{formatNumber(kpis.supply_fte)}</div>
            </div>
            <div className="stat-card stat-card--red">
              <div className="stat-header">Gap (Req - Sup)</div>
              <div className="ops-kpi-value">{formatNumber(kpis.gap_fte)}</div>
            </div>
          </div>
        </section>

        <section className="section">
          <div className="ops-chart-grid">
            <div className="ops-chart-card">
              <h3>Requirements vs Supply</h3>
              <LineChart data={summary?.line ?? null} height={280} />
            </div>
            <div className="ops-chart-card">
              <h3>Workload by Time</h3>
              <BarChart data={summary?.bar ?? null} stacked height={280} />
            </div>
          </div>
          <div className="ops-chart-grid-3">
            <div className="ops-chart-card">
              <h3>Workload Share by Channel</h3>
              <PieChart data={summary?.pie ?? null} height={280} />
            </div>
            <div className="ops-chart-card">
              <h3>Workload by Site</h3>
              <BarChart
                data={summary?.site ? { labels: summary.site.labels, series: [{ name: "Workload", values: summary.site.values }] } : null}
                height={280}
              />
            </div>
            <div className="ops-chart-card">
              <h3>Requirement vs Supply</h3>
              <WaterfallChart data={summary?.waterfall ?? null} height={280} />
            </div>
          </div>
        </section>

        <section className="section">
          <h2>Summary Table</h2>
          <DataTable data={summary?.summary ?? []} />
        </section>
      </div>
    </AppShell>
  );
}
