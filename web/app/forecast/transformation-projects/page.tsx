"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import AppShell from "../../_components/AppShell";
import DataTable from "../../_components/DataTable";
import EditableTable from "../../_components/EditableTable";
import { useGlobalLoader } from "../../_components/GlobalLoader";
import { useToast } from "../../_components/ToastProvider";
import { apiGet, apiPost } from "../../../lib/api";
import { loadForecastStore, saveForecastStore } from "../forecast-store";

const TRANSFORMATION_COLUMNS = [
  "Transformation 1",
  "Remarks_Tr 1",
  "Transformation 2",
  "Remarks_Tr 2",
  "Transformation 3",
  "Remarks_Tr 3",
  "IA 1",
  "Remarks_IA 1",
  "IA 2",
  "Remarks_IA 2",
  "IA 3",
  "Remarks_IA 3",
  "Marketing Campaign 1",
  "Remarks_Mkt 1",
  "Marketing Campaign 2",
  "Remarks_Mkt 2",
  "Marketing Campaign 3",
  "Remarks_Mkt 3"
];

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

function ensureMonthYear(row: Record<string, any>) {
  if (row.Month_Year) return row.Month_Year;
  const year = row.Year ?? row.year;
  const month = row.Month ?? row.month;
  if (!year || !month) return "";
  const monthStr = String(month).slice(0, 3);
  return `${monthStr}-${String(year).slice(-2)}`;
}

function buildTransformRows(rows: Array<Record<string, any>>) {
  return rows.map((row) => {
    const base = { ...row };
    base.Month_Year = ensureMonthYear(base);
    if (!base.Base_Forecast_for_Forecast_Group) {
      base.Base_Forecast_for_Forecast_Group =
        base.Base_Forecast_Category ?? base.Final_Forecast ?? base.Forecast ?? "";
    }
    TRANSFORMATION_COLUMNS.forEach((col) => {
      if (base[col] === undefined) base[col] = "";
    });
    return base;
  });
}

export default function TransformationProjectsPage() {
  const { notify } = useToast();
  const { setLoading: setGlobalLoading } = useGlobalLoader();
  const [phase2Result, setPhase2Result] = useState<any>(null);
  const [phase2Adjusted, setPhase2Adjusted] = useState<any>(null);
  const [transformResult, setTransformResult] = useState<any>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [savedRuns, setSavedRuns] = useState<Array<Record<string, any>>>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [loadedRows, setLoadedRows] = useState<Array<Record<string, any>>>([]);
  const [dataLoaded, setDataLoaded] = useState(false);
  const [selectionApplied, setSelectionApplied] = useState(false);
  const [transformApplied, setTransformApplied] = useState(false);
  const [savedToDisk, setSavedToDisk] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedYear, setSelectedYear] = useState<string>("");
  const [filteredData, setFilteredData] = useState<Array<Record<string, any>>>([]);
  const [transformRows, setTransformRows] = useState<Array<Record<string, any>>>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const store = loadForecastStore();
    if (store.phase2) setPhase2Result(store.phase2);
    if (store.phase2Adjusted) setPhase2Adjusted(store.phase2Adjusted);
    if (store.transformations) setTransformResult(store.transformations);
    if (store.metadata?.lastUpdated) setLastUpdated(store.metadata.lastUpdated);
  }, []);

  useEffect(() => {
    if (!phase2Adjusted?.results?.adjusted_raw?.length) return;
    setLoadedRows(phase2Adjusted.results.adjusted_raw);
    setDataLoaded(true);
    setSelectedRun("__latest__");
  }, [phase2Adjusted]);

  useEffect(() => {
    apiGet<{ runs?: Array<Record<string, any>> }>("/api/forecast/saved-runs")
      .then((res) => setSavedRuns(res.runs || []))
      .catch(() => setSavedRuns([]));
  }, []);

  useEffect(() => {
    setGlobalLoading(loading);
    return () => setGlobalLoading(false);
  }, [loading, setGlobalLoading]);

  const rawData = useMemo(() => {
    return dataLoaded ? loadedRows : [];
  }, [dataLoaded, loadedRows]);

  const runOptions = useMemo(() => {
    const options: Array<Record<string, any>> = [];
    if (phase2Adjusted?.results?.adjusted_raw?.length) {
      options.push({ name: "__latest__", label: "Latest from Volume Summary (Adjusted)" });
    } else if (phase2Result?.results?.base_raw?.length || phase2Result?.results?.base?.length) {
      options.push({ name: "__latest__", label: "Latest from Volume Summary" });
    }
    savedRuns.forEach((run) => {
      const name = String(run.name || run.filename || "");
      if (name) options.push({ name, label: run.label || name });
    });
    return options;
  }, [phase2Adjusted, phase2Result, savedRuns]);

  const groupOptions = useMemo(() => {
    if (!rawData.length) return [];
    const cols = ["forecast_group", "Forecast_Group", "category", "Category"];
    const col = cols.find((c) => rawData[0]?.[c] !== undefined);
    if (!col) return [];
    return Array.from(new Set(rawData.map((row) => String(row[col] ?? "").trim()).filter(Boolean)));
  }, [rawData]);

  const modelOptions = useMemo(() => {
    if (!rawData.length) return [];
    return Array.from(new Set(rawData.map((row) => String(row.Model ?? "").trim()).filter(Boolean)));
  }, [rawData]);

  const yearOptions = useMemo(() => {
    if (!rawData.length) return [];
    return Array.from(new Set(rawData.map((row) => String(row.Year ?? "").trim()).filter(Boolean)));
  }, [rawData]);

  useEffect(() => {
    if (!selectedModel && modelOptions.length) setSelectedModel(modelOptions[0]);
    if (!selectedYear && yearOptions.length) setSelectedYear(yearOptions[0]);
    if (!selectedGroup && groupOptions.length) setSelectedGroup(groupOptions[0]);
  }, [modelOptions, yearOptions, groupOptions, selectedModel, selectedYear, selectedGroup]);

  useEffect(() => {
    if (modelOptions.length && !modelOptions.includes(selectedModel)) {
      setSelectedModel(modelOptions[0]);
    }
  }, [modelOptions, selectedModel]);

  useEffect(() => {
    if (yearOptions.length && !yearOptions.includes(selectedYear)) {
      setSelectedYear(yearOptions[0]);
    }
  }, [yearOptions, selectedYear]);

  useEffect(() => {
    if (groupOptions.length && !groupOptions.includes(selectedGroup)) {
      setSelectedGroup(groupOptions[0]);
    }
  }, [groupOptions, selectedGroup]);

  useEffect(() => {
    if (!selectedRun && runOptions.length) setSelectedRun(runOptions[0].name);
  }, [selectedRun, runOptions]);

  const handleLoadRun = async () => {
    if (!selectedRun) {
      notify("warning", "Select a saved forecast first.");
      return;
    }
    setLoading(true);
    try {
      if (selectedRun === "__latest__") {
        const rows =
          phase2Adjusted?.results?.adjusted_raw ||
          phase2Result?.results?.base_raw ||
          phase2Result?.results?.base ||
          [];
        if (!rows.length) {
          notify("warning", "No Volume Summary forecast data found.");
          return;
        }
        setLoadedRows(rows);
      } else {
        const res = await apiGet<{ rows?: Array<Record<string, any>> }>(
          `/api/forecast/saved-runs/${encodeURIComponent(selectedRun)}`
        );
        setLoadedRows(res.rows || []);
      }
      setDataLoaded(true);
      setSelectionApplied(false);
      setTransformApplied(false);
      setFilteredData([]);
      setTransformRows([]);
      setTransformResult(null);
      setSavedToDisk(false);
      notify("success", "Forecast data loaded.");
    } catch (err: any) {
      notify("error", err?.message || "Failed to load saved forecast.");
    } finally {
      setLoading(false);
    }
  };

  const applySelection = () => {
    if (!rawData.length) {
      notify("warning", "Load forecast data first.");
      return;
    }
    let next = [...rawData];
    if (selectedModel) {
      next = next.filter((row) => String(row.Model ?? "") === selectedModel);
    }
    if (selectedYear) {
      next = next.filter((row) => String(row.Year ?? "") === selectedYear);
    }
    if (selectedGroup) {
      const cols = ["forecast_group", "Forecast_Group", "category", "Category"];
      const col = cols.find((c) => rawData[0]?.[c] !== undefined);
      if (col) {
        next = next.filter((row) => String(row[col] ?? "") === selectedGroup);
      }
    }
    setFilteredData(next);
    setTransformRows(buildTransformRows(next));
    setSelectionApplied(true);
    notify("success", `Loaded ${next.length} rows for transformation.`);
  };

  const applyTransformations = async () => {
    const source = transformRows.length
      ? transformRows
      : buildTransformRows(filteredData.length ? filteredData : rawData);
    if (!source.length) {
      notify("warning", "Select and load data before applying transformations.");
      return;
    }
    setSavedToDisk(false);
    setLoading(true);
    try {
      const res = await apiPost<any>("/api/forecast/transformations/apply", { data: source });
      setTransformResult(res);
      setTransformApplied(true);
      saveForecastStore({ transformations: res });
      if (!res.results || !res.results.final?.length) {
        notify("warning", res.status || "Transformations returned no rows.");
      } else {
        notify("success", "Transformations applied.");
      }
    } catch (err: any) {
      notify("error", err?.message || "Transformations failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFilteredData([]);
    setTransformRows([]);
    setTransformResult(null);
    setSelectionApplied(false);
    setTransformApplied(false);
    setSavedToDisk(false);
    saveForecastStore({ transformations: null });
    notify("success", "Transformations reset.");
  };

  const handleSaveToDisk = async () => {
    const rows = transformResult?.results?.final || [];
    if (!rows.length) {
      notify("warning", "Apply transformations first.");
      return;
    }
    setLoading(true);
    try {
      const res = await apiPost<any>("/api/forecast/save/transformations", { data: rows });
      const path = res?.paths?.[0] ? `Saved to ${res.paths[0]}` : "Saved to disk.";
      notify("success", path);
      setSavedToDisk(true);
    } catch (err: any) {
      notify("error", err?.message || "Save to disk failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AppShell crumbs="CAP-CONNECT / Forecast / Transformation Projects">
      <div className="forecast-page forecast-volume-page">
        <div className="forecast-page-header">
          <div>
            <h1 className="forecast-page-title">Transformation Projects</h1>
            <p className="forecast-page-subtitle">Follow the steps to keep moving forward.</p>
          </div>
          <Link className="forecast-back-link" href="/forecast/volume-summary">
            Back to models
          </Link>
        </div>

        <div className="forecast-stepper">
          <div className="forecast-step-pill">📊 Volume Summary</div>
          <span className="forecast-step-arrow">→</span>
          <div className="forecast-step-pill active">⚙️ Transformation Projects</div>
          <span className="forecast-step-arrow">→</span>
          <div className="forecast-step-pill">⏱️ Daily Interval Forecast</div>
        </div>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Load forecast data</div>
          <div className="forecast-form-row">
            <div>
              <div className="label">Saved forecast run</div>
              <select
                className="select"
                value={selectedRun}
                onChange={(event) => setSelectedRun(event.target.value)}
              >
                {runOptions.length ? (
                  runOptions.map((run) => (
                    <option key={run.name} value={run.name}>
                      {run.label}
                    </option>
                  ))
                ) : (
                  <option value="">No saved forecasts found</option>
                )}
              </select>
            </div>
            <div>
              <div className="label">Status</div>
              <input
                className="input"
                type="text"
                value={
                  dataLoaded
                    ? "Loaded"
                    : lastUpdated
                      ? `Last update: ${new Date(lastUpdated).toLocaleString()}`
                      : "Not loaded"
                }
                readOnly
              />
            </div>
          </div>
          <div className="forecast-actions-row">
            <button className="btn btn-primary" type="button" disabled={loading} onClick={handleLoadRun}>
              Load forecast
            </button>
          </div>
        </section>

        {dataLoaded ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Selection criteria</div>
          <div className="forecast-form-row">
            <div>
              <div className="label">Forecast group</div>
              <select
                className="select"
                value={selectedGroup}
                onChange={(event) => setSelectedGroup(event.target.value)}
              >
                {groupOptions.length ? (
                  groupOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))
                ) : (
                  <option value="">No forecast groups</option>
                )}
              </select>
            </div>
            <div>
              <div className="label">Model</div>
              <select
                className="select"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
              >
                {modelOptions.length ? (
                  modelOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))
                ) : (
                  <option value="">No models</option>
                )}
              </select>
            </div>
            <div>
              <div className="label">Year</div>
              <select
                className="select"
                value={selectedYear}
                onChange={(event) => setSelectedYear(event.target.value)}
              >
                {yearOptions.length ? (
                  yearOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))
                ) : (
                  <option value="">No years</option>
                )}
              </select>
            </div>
          </div>
          <div className="forecast-actions-row">
            <button className="btn btn-primary" type="button" onClick={applySelection}>
              Apply selection &amp; load data
            </button>
            {selectionApplied ? (
              <button className="btn btn-success" type="button" disabled={loading} onClick={applyTransformations}>
                Apply transformations
              </button>
            ) : null}
          </div>
        </section>
        ) : null}

        {dataLoaded ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Raw data</div>
          {rawData.length ? <DataTable data={rawData} /> : <div className="forecast-placeholder-bar" />}
        </section>
        ) : null}

        {selectionApplied ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Filtered data</div>
          {filteredData.length ? (
            <DataTable data={filteredData} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>
        ) : null}

        {selectionApplied ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Transformation editor</div>
          <div className="forecast-section-note">
            Enter % change values (e.g., 10 for +10%). Base columns are locked.
          </div>
          {transformRows.length ? (
            <EditableTable
              data={transformRows}
              editableColumns={TRANSFORMATION_COLUMNS}
              onChange={setTransformRows}
            />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>
        ) : null}

        {transformApplied && transformResult ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Results</div>
          {transformResult?.results?.summary?.length ? (
            <DataTable data={transformResult.results.summary} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
          {transformResult?.results?.transposed?.length ? (
            <DataTable data={transformResult.results.transposed} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
          {transformResult?.results?.final?.length ? (
            <DataTable data={transformResult.results.final} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>
        ) : null}

        {transformApplied && transformResult ? (
          <section className="forecast-section-block">
          <div className="forecast-section-title">Exports</div>
          <div className="forecast-actions-row">
            <button
              className="btn btn-outline"
              type="button"
              onClick={() =>
                transformResult?.results?.final
                  ? downloadCsv("final-forecast.csv", transformResult.results.final)
                  : notify("warning", "Apply transformations first.")
              }
            >
              Download final forecast
            </button>
            <button
              className="btn btn-outline"
              type="button"
              onClick={() =>
                transformResult?.results?.processed
                  ? downloadCsv("forecast-detail.csv", transformResult.results.processed)
                  : notify("warning", "Apply transformations first.")
              }
            >
              Download full detail
            </button>
            <button className="btn btn-outline" type="button" disabled={loading} onClick={handleSaveToDisk}>
              Save to disk
            </button>
            <button className="btn btn-outline" type="button" onClick={handleReset}>
              Reset
            </button>
            {savedToDisk ? (
              <Link className="btn btn-primary" href="/forecast/daily-interval">
                Continue
              </Link>
            ) : null}
          </div>
        </section>
        ) : null}
      </div>
    </AppShell>
  );
}
