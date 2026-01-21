"use client";

import { useEffect, useMemo, useState } from "react";
import AppShell from "../_components/AppShell";
import DataTable from "../_components/DataTable";
import LineChart from "../_components/LineChart";
import MultiSelect from "../_components/MultiSelect";
import { useGlobalLoader } from "../_components/GlobalLoader";
import { useToast } from "../_components/ToastProvider";
import { apiGet, apiPost } from "../../lib/api";

type HeadcountOptions = {
  business_areas?: string[];
  sub_business_areas?: string[];
  channels?: string[];
  locations?: string[];
  sites?: string[];
};

type DatasetResponse = {
  rows?: Array<Record<string, any>>;
  chart?: Array<Record<string, any>>;
};

const SERIES_OPTIONS = [
  { label: "Auto", value: "auto" },
  { label: "Actual", value: "actual" },
  { label: "Forecast", value: "forecast" }
];

function todayIso(daysAgo = 0) {
  const dt = new Date();
  dt.setDate(dt.getDate() - daysAgo);
  return dt.toISOString().slice(0, 10);
}

function buildLineData(chartRows: Array<Record<string, any>>) {
  if (!chartRows.length) return null;
  const x = chartRows.map((row) => String(row.date ?? row.Date ?? ""));
  const required = chartRows.map((row) => Number(row.total_req_fte ?? 0));
  const supply = chartRows.map((row) => Number(row.supply_fte ?? 0));
  return {
    x,
    series: [
      { name: "Required FTE", points: x.map((label, idx) => ({ x: label, y: required[idx] })) },
      { name: "Supply FTE", points: x.map((label, idx) => ({ x: label, y: supply[idx] })) }
    ]
  };
}

export default function DatasetPage() {
  const { setLoading } = useGlobalLoader();
  const { notify } = useToast();
  const [options, setOptions] = useState<Required<HeadcountOptions>>({
    business_areas: [],
    sub_business_areas: [],
    channels: ["Voice", "Back Office", "Outbound", "Blended", "Chat", "MessageUs"],
    locations: [],
    sites: []
  });
  const [filters, setFilters] = useState({
    startDate: todayIso(56),
    endDate: todayIso(0),
    series: "auto",
    ba: [] as string[],
    sba: [] as string[],
    ch: [] as string[],
    loc: [] as string[],
    site: [] as string[]
  });
  const [rows, setRows] = useState<Array<Record<string, any>>>([]);
  const [chartRows, setChartRows] = useState<Array<Record<string, any>>>([]);
  const [channelsInitialized, setChannelsInitialized] = useState(false);

  const filterKey = useMemo(
    () =>
      [
        filters.startDate,
        filters.endDate,
        filters.series,
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
    if (filters.ba.length === 1) params.set("ba", filters.ba[0]);
    if (filters.sba.length === 1) params.set("sba", filters.sba[0]);
    if (filters.loc.length === 1) params.set("location", filters.loc[0]);
    try {
      const res = await apiGet<HeadcountOptions>(
        `/api/forecast/headcount/options${params.toString() ? `?${params.toString()}` : ""}`
      );
      const next = {
        business_areas: res.business_areas ?? [],
        sub_business_areas: res.sub_business_areas ?? [],
        channels: res.channels ?? options.channels,
        locations: res.locations ?? [],
        sites: res.sites ?? []
      };
      setOptions(next);
      setFilters((prev) => {
        const nextCh = channelsInitialized
          ? prev.ch.filter((val) => next.channels.includes(val))
          : (prev.ch.length ? prev.ch.filter((val) => next.channels.includes(val)) : next.channels);
        return {
          ...prev,
          sba: prev.sba.filter((val) => next.sub_business_areas.includes(val)),
          ch: nextCh,
          loc: prev.loc.filter((val) => next.locations.includes(val)),
          site: prev.site.filter((val) => next.sites.includes(val))
        };
      });
      if (!channelsInitialized) {
        setChannelsInitialized(true);
      }
    } catch (error: any) {
      notify("error", error?.message || "Could not load dataset options.");
    }
  };

  const loadSnapshot = async () => {
    setLoading(true);
    try {
      const res = await apiPost<DatasetResponse>("/api/forecast/dataset", {
        start_date: filters.startDate,
        end_date: filters.endDate,
        series: filters.series,
        ba: filters.ba,
        sba: filters.sba,
        ch: filters.ch,
        loc: filters.loc,
        site: filters.site
      });
      setRows(res.rows ?? []);
      setChartRows(res.chart ?? []);
    } catch (error: any) {
      notify("error", error?.message || "Could not load dataset snapshot.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadOptions();
  }, [filters.ba.join("|"), filters.sba.join("|"), filters.loc.join("|")]);

  useEffect(() => {
    void loadSnapshot();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterKey]);

  const chartData = useMemo(() => buildLineData(chartRows), [chartRows]);

  return (
    <AppShell crumbs="CAP-CONNECT / Planner Dataset">
      <div className="dataset-page">
        <section className="section">
          <h2>Planner Dataset — Inputs Snapshot</h2>
          <div className="dataset-filter-grid">
            <div>
              <div className="label">Business Area</div>
              <MultiSelect
                options={options.business_areas.map((val) => ({ label: val, value: val }))}
                values={filters.ba}
                placeholder="Select Business Area"
                onChange={(values) => setFilters((prev) => ({ ...prev, ba: values, sba: [], site: [] }))}
              />
            </div>
            <div>
              <div className="label">Sub Business Area</div>
              <MultiSelect
                options={options.sub_business_areas.map((val) => ({ label: val, value: val }))}
                values={filters.sba}
                placeholder="Select Sub Business Area"
                onChange={(values) => setFilters((prev) => ({ ...prev, sba: values }))}
              />
            </div>
            <div>
              <div className="label">Channel</div>
              <MultiSelect
                options={options.channels.map((val) => ({ label: val, value: val }))}
                values={filters.ch}
                placeholder="Select Channel"
                onChange={(values) => setFilters((prev) => ({ ...prev, ch: values, site: [] }))}
              />
            </div>
            <div>
              <div className="label">Series</div>
              <select
                className="select"
                value={filters.series}
                onChange={(event) => setFilters((prev) => ({ ...prev, series: event.target.value }))}
              >
                {SERIES_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="label">Location</div>
              <MultiSelect
                options={options.locations.map((val) => ({ label: val, value: val }))}
                values={filters.loc}
                placeholder="Select Location"
                onChange={(values) => setFilters((prev) => ({ ...prev, loc: values, site: [] }))}
              />
            </div>
            <div>
              <div className="label">Site</div>
              <MultiSelect
                options={options.sites.map((val) => ({ label: val, value: val }))}
                values={filters.site}
                placeholder="Select Site"
                onChange={(values) => setFilters((prev) => ({ ...prev, site: values }))}
              />
            </div>
            <div>
              <div className="label">Date Range</div>
              <div className="dataset-date-row">
                <input
                  className="input"
                  type="date"
                  value={filters.startDate}
                  onChange={(event) => setFilters((prev) => ({ ...prev, startDate: event.target.value }))}
                />
                <input
                  className="input"
                  type="date"
                  value={filters.endDate}
                  onChange={(event) => setFilters((prev) => ({ ...prev, endDate: event.target.value }))}
                />
              </div>
            </div>
          </div>
        </section>

        <section className="section">
          <DataTable data={rows} maxRows={12} />
          <LineChart data={chartData} />
        </section>
      </div>
    </AppShell>
  );
}
