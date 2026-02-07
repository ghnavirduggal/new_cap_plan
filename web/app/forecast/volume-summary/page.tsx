"use client";

import Link from "next/link";
import type { ChangeEvent, DragEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import AppShell from "../../_components/AppShell";
import DataTable from "../../_components/DataTable";
import EditableTable from "../../_components/EditableTable";
import { useGlobalLoader } from "../../_components/GlobalLoader";
import LineChart from "../../_components/LineChart";
import { useToast } from "../../_components/ToastProvider";
import { apiPost, apiPostForm } from "../../../lib/api";
import { loadForecastStore, saveForecastStore } from "../forecast-store";

type VolumeSummaryResponse = {
  message?: string;
  summary?: Array<Record<string, any>>;
  normalized?: Array<Record<string, any>>;
  categories?: string[];
  pivot?: Array<Record<string, any>>;
  volume_split?: Array<Record<string, any>>;
  holidays?: { mapping?: Record<string, string> };
  iq_summary?: Record<
    string,
    {
      IQ?: Array<Record<string, any>>;
      Volume?: Array<Record<string, any>>;
      Contact_Ratio?: Array<Record<string, any>>;
    }
  >;
  auto_smoothing?: any;
  auto_meta?: any;
};

function parseNumber(value: any) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const cleaned = value.replace(/[%,$]/g, "").trim();
  const num = Number(cleaned);
  return Number.isFinite(num) ? num : null;
}


function normalizePhaseRows(rows: Array<Record<string, any>>) {
  return rows.map((row) => {
    const ds = row.ds ?? row.date ?? row.Month_Year ?? row.month_year;
    const value =
      row.Final_Smoothed_Value ??
      row.smoothed ??
      row.y ??
      row.value ??
      row.volume;
    return {
      ...row,
      ds,
      Final_Smoothed_Value: value
    };
  });
}

function buildNorm2Pivot(rows: Array<Record<string, any>>) {
  if (!rows.length) return [];
  const monthOrder = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const buckets: Record<string, Record<string, number[]>> = {};
  rows.forEach((row) => {
    const year = row.Year ?? row.year;
    const month = row.Month ?? row.month;
    const value = parseNumber(row.Final_Smoothed_Value ?? row.final_smoothed_value);
    if (!year || !month || value === null) return;
    const yearKey = String(year);
    const monthKey = String(month).slice(0, 3);
    buckets[yearKey] = buckets[yearKey] || {};
    buckets[yearKey][monthKey] = buckets[yearKey][monthKey] || [];
    buckets[yearKey][monthKey].push(value);
  });
  return Object.keys(buckets)
    .sort()
    .map((year) => {
      const row: Record<string, any> = { Year: year };
      let avgSum = 0;
      let avgCount = 0;
      monthOrder.forEach((month) => {
        const values = buckets[year][month];
        if (!values?.length) return;
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        row[month] = Number(mean.toFixed(1));
        avgSum += mean;
        avgCount += 1;
      });
      row.Avg = avgCount ? Number((avgSum / avgCount).toFixed(1)) : null;
      return row;
    });
}

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

export default function VolumeSummaryPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { notify } = useToast();
  const { setLoading: setGlobalLoading } = useGlobalLoader();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [volumeSummary, setVolumeSummary] = useState<VolumeSummaryResponse | null>(null);
  const [seasonality, setSeasonality] = useState<any>(null);
  const [prophetResult, setProphetResult] = useState<any>(null);
  const [prophetTable, setProphetTable] = useState<Array<Record<string, any>>>([]);
  const [prophetNorm2, setProphetNorm2] = useState<Array<Record<string, any>>>([]);
  const [smoothingResult, setSmoothingResult] = useState<any>(null);
  const [phase1Result, setPhase1Result] = useState<any>(null);
  const [phase2Result, setPhase2Result] = useState<any>(null);
  const [adjustedResult, setAdjustedResult] = useState<any>(null);
  const [finalSmoothed, setFinalSmoothed] = useState<Array<Record<string, any>>>([]);
  const [activeCategory, setActiveCategory] = useState<string>("");
  const [lowerCap, setLowerCap] = useState("0.8");
  const [upperCap, setUpperCap] = useState("1.15");
  const [cappedRatio, setCappedRatio] = useState<Array<Record<string, any>>>([]);
  const [customBaseVolume, setCustomBaseVolume] = useState("");
  const [baseVolumeEdited, setBaseVolumeEdited] = useState(false);
  const [phase2Start, setPhase2Start] = useState("");
  const [phase2End, setPhase2End] = useState("");
  const [phase2Basis, setPhase2Basis] = useState("iq");
  const [volumeSplitEdit, setVolumeSplitEdit] = useState<Array<Record<string, any>>>([]);
  const [volumeSplitInfo, setVolumeSplitInfo] = useState("");
  const [loading, setLoading] = useState(false);
  const [adjustedSaved, setAdjustedSaved] = useState(false);

  useEffect(() => {
    const store = loadForecastStore();
    if (store.phase2Basis) setPhase2Basis(store.phase2Basis);
  }, []);

  useEffect(() => {
    setGlobalLoading(loading);
    return () => setGlobalLoading(false);
  }, [loading, setGlobalLoading]);

  useEffect(() => {
    if (seasonality?.results?.base_volume !== undefined && !baseVolumeEdited) {
      setCustomBaseVolume(String(seasonality.results.base_volume ?? ""));
    }
  }, [seasonality, baseVolumeEdited]);

  const hasVolumeSummary = Boolean(volumeSummary);
  const hasProphetStage = Boolean(prophetResult);
  const hasPhase1Stage = Boolean(phase1Result);
  const hasPhase2Stage = Boolean(phase2Result);
  const hasSplitStage = Boolean(adjustedResult);

  const categories = useMemo(() => {
    const volumeCategories = volumeSummary?.categories ?? [];
    const iqCategories = volumeSummary?.iq_summary ? Object.keys(volumeSummary.iq_summary) : [];
    if (!volumeCategories.length) return iqCategories;
    if (volumeCategories.length === 1 && volumeCategories[0] === "All" && iqCategories.length) {
      return iqCategories;
    }
    if (!iqCategories.length) return volumeCategories;
    const merged = [...volumeCategories];
    iqCategories.forEach((cat) => {
      if (!merged.includes(cat)) merged.push(cat);
    });
    return merged;
  }, [volumeSummary]);

  const phase1ConfigRows = useMemo(() => {
    const cfg = phase1Result?.config;
    if (!cfg) return [];
    return Object.entries(cfg).flatMap(([model, params]) => {
      const entries = Object.entries(params || {});
      if (!entries.length) return [{ Model: model, Parameter: "", Value: "" }];
      return entries.map(([key, value]) => ({
        Model: model,
        Parameter: key,
        Value: typeof value === "object" ? JSON.stringify(value) : value
      }));
    });
  }, [phase1Result]);
  const phase1CombinedDisplay = useMemo(() => {
    const rows = phase1Result?.results?.combined || [];
    return rows.map((row: Record<string, any>) => {
      if (!("Forecast" in row)) return row;
      const raw = parseNumber(row.Forecast);
      if (raw === null) return row;
      return { ...row, Forecast: raw * 100 };
    });
  }, [phase1Result]);
  const phase2CombinedDisplay = useMemo(() => {
    const rows = phase2Result?.results?.combined || [];
    return rows.map((row: Record<string, any>) => {
      if (!("Forecast" in row)) return row;
      const raw = parseNumber(row.Forecast);
      if (raw === null) return row;
      return { ...row, Forecast: raw * 100 };
    });
  }, [phase2Result]);

  useEffect(() => {
    if (!activeCategory && categories.length) setActiveCategory(categories[0]);
  }, [activeCategory, categories]);

  useEffect(() => {
    setBaseVolumeEdited(false);
    setCustomBaseVolume("");
  }, [activeCategory]);

  const activeIQ = activeCategory && volumeSummary?.iq_summary
    ? volumeSummary.iq_summary[activeCategory]
    : undefined;

  useEffect(() => {
    const ratioRows = activeIQ?.Contact_Ratio ?? [];
    setProphetResult(null);
    setProphetTable([]);
    setProphetNorm2([]);
    if (!activeCategory || !ratioRows.length) {
      setSeasonality(null);
      setCappedRatio([]);
      return;
    }
    setLoading(true);
    apiPost<any>("/api/forecast/seasonality/build", { ratio_table: ratioRows })
      .then((res) => {
        setSeasonality(res);
        setCappedRatio(res?.results?.capped || []);
        if (res?.results?.base_volume && !customBaseVolume) {
          setCustomBaseVolume(String(res.results.base_volume));
        }
        saveForecastStore({ seasonality: res });
      })
      .catch((err: any) => {
        setSeasonality(null);
        setCappedRatio([]);
        notify("warning", err?.message || "Seasonality data not available.");
      })
      .finally(() => setLoading(false));
  }, [activeCategory, activeIQ]);

  useEffect(() => {
    if (prophetResult?.results?.prophet_table) {
      setProphetTable(prophetResult.results.prophet_table);
    }
    if (prophetResult?.results?.norm2) {
      setProphetNorm2(prophetResult.results.norm2);
    }
  }, [prophetResult]);

  const autoSmoothing = volumeSummary?.auto_smoothing ?? null;

  const previewRows = volumeSummary?.normalized?.slice(0, 8) ?? [];
  const cappedEditableColumns = useMemo(() => {
    if (!cappedRatio.length) return [];
    return Object.keys(cappedRatio[0]).filter((key) => key !== "Year");
  }, [cappedRatio]);
  const volumeSplitEditableColumns = useMemo(() => ["Vol_Split_Normalized"], []);

  const handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      notify("warning", "Select a CSV/Excel file first.");
      return;
    }
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const res = await apiPostForm<VolumeSummaryResponse>("/api/forecast/volume-summary", formData);
      setVolumeSummary(res);
      setSeasonality(null);
      setProphetResult(null);
      setProphetTable([]);
      setProphetNorm2([]);
      setSmoothingResult(null);
      setPhase1Result(null);
      setPhase2Result(null);
      setAdjustedResult(null);
      setVolumeSplitEdit([]);
      setFinalSmoothed([]);
      setAdjustedSaved(false);
      saveForecastStore({
        volumeSummary: res,
        seasonality: null,
        prophet: null,
        smoothing: null,
        phase1: null,
        phase2: null,
        phase2Adjusted: null,
        volumeSplitEdit: [],
        finalSmoothed: []
      });
      notify("success", res.message || "Volume summary ready.");
    } catch (err: any) {
      notify("error", err?.message || "Volume summary failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleApplyCaps = async () => {
    const cappedRows = cappedRatio.length ? cappedRatio : seasonality?.results?.capped || [];
    if (!cappedRows.length) {
      notify("warning", "No ratio data to cap yet.");
      return;
    }
    const lower = Number(lowerCap);
    const upper = Number(upperCap);
    if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
      notify("warning", "Enter valid lower/upper caps.");
      return;
    }
    setLoading(true);
    try {
      const res = await apiPost<any>("/api/forecast/seasonality/apply", {
        capped_rows: cappedRows,
        lower_cap: lower,
        upper_cap: upper,
        base_volume: parseNumber(customBaseVolume) ?? customBaseVolume
      });
      const next = {
        ...(seasonality || {}),
        results: {
          ...(seasonality?.results || {}),
          ...res.results
        }
      };
      setSeasonality(next);
      setCappedRatio(res?.results?.capped || []);
      setCustomBaseVolume(String(res?.results?.base_volume ?? customBaseVolume));
      saveForecastStore({ seasonality: next });
      notify("success", res.status || "Seasonality updated.");
    } catch (err: any) {
      notify("error", err?.message || "Seasonality update failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleProphetSmoothing = async () => {
    if (!seasonality?.results?.normalized?.length) {
      notify("warning", "Run seasonality adjustments first.");
      return;
    }
    if (!activeIQ?.IQ?.length) {
      notify("warning", "Select a category with IQ data.");
      return;
    }
    setLoading(true);
    try {
      const res = await apiPost<any>("/api/forecast/volume-summary/prophet-smoothing", {
        normalized_ratio: seasonality.results.normalized,
        ratio_table: seasonality.results.ratio,
        iq_table: activeIQ.IQ,
        holidays: volumeSummary?.holidays || null
      });
      setProphetResult(res);
      setProphetTable(res?.results?.prophet_table || []);
      setProphetNorm2(res?.results?.norm2 || []);
      saveForecastStore({ prophet: res });
      if (res?.warning) {
        notify("warning", res.warning);
      }
      if (!res.results || !res.results.prophet_table?.length) {
        notify("warning", res.status || "Prophet smoothing returned no rows.");
      } else {
        notify("success", res.status || "Prophet smoothing complete.");
      }
    } catch (err: any) {
      notify("error", err?.message || "Prophet smoothing failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSmoothed = async () => {
    const source =
      prophetTable.length
        ? prophetTable
        : smoothingResult?.smoothed || autoSmoothing?.smoothed || volumeSummary?.normalized || [];
    if (!source.length) {
      notify("warning", "No smoothing data available.");
      return;
    }
    setLoading(true);
    try {
      if (prophetTable.length) {
        const res = await apiPost<any>("/api/forecast/volume-summary/prophet-save", {
          edited: source,
          original: prophetResult?.results?.prophet_table || []
        });
        const nextTable = res?.results?.prophet_table || source;
        const nextNorm2 = res?.results?.norm2 || buildNorm2Pivot(nextTable);
        setProphetTable(nextTable);
        setProphetNorm2(nextNorm2);
        saveForecastStore({
          prophet: {
            ...(prophetResult || {}),
            status: res?.status || prophetResult?.status,
            results: {
              ...(prophetResult?.results || {}),
              ...res?.results,
              prophet_table: nextTable,
              norm2: nextNorm2
            }
          }
        });
        const normalized = normalizePhaseRows(nextTable);
        setFinalSmoothed(normalized);
        saveForecastStore({ finalSmoothed: normalized });
        notify("success", res?.status || "Changes saved. Phase 1 is ready.");
        return;
      }

      const normalized = normalizePhaseRows(source);
      setFinalSmoothed(normalized);
      saveForecastStore({ finalSmoothed: normalized });
      notify("success", "Smoothed data saved for Phase 1/2.");
    } catch (err: any) {
      notify("error", err?.message || "Save changes failed.");
    } finally {
      setLoading(false);
    }
  };

  const handlePhase1 = async () => {
    const source = finalSmoothed.length
      ? finalSmoothed
      : normalizePhaseRows(
          prophetTable.length
            ? prophetTable
            : smoothingResult?.smoothed || autoSmoothing?.smoothed || []
        );
    if (!source.length) {
      notify("warning", "Run smoothing and save it first.");
      return;
    }
    setLoading(true);
    try {
      const res = await apiPost<any>("/api/forecast/phase1", {
        data: source,
        holidays: volumeSummary?.holidays?.mapping || null
      });
      setPhase1Result(res);
      saveForecastStore({ phase1: res });
      if (!res.results || !res.results.combined?.length) {
        notify("warning", res.status || "Phase 1 returned no results.");
      } else {
        notify("success", "Phase 1 results ready.");
      }
    } catch (err: any) {
      notify("error", err?.message || "Phase 1 failed.");
    } finally {
      setLoading(false);
    }
  };

  const handlePhase2 = async () => {
    if (!phase2Start || !phase2End) {
      notify("warning", "Enter start and end dates.");
      return;
    }
    const source = finalSmoothed.length
      ? finalSmoothed
      : normalizePhaseRows(
          prophetTable.length
            ? prophetTable
            : smoothingResult?.smoothed || autoSmoothing?.smoothed || []
        );
    if (!source.length) {
      notify("warning", "Run smoothing and save it first.");
      return;
    }
    const iqTable = activeIQ?.IQ ?? null;
    setLoading(true);
    try {
      const res = await apiPost<any>("/api/forecast/phase2", {
        data: source,
        start_date: phase2Start,
        end_date: phase2End,
        iq_summary: iqTable,
        volume_summary: activeIQ?.Volume || null,
        volume_data: volumeSummary?.normalized || null,
        basis: phase2Basis,
        category: activeCategory || null
      });
      setPhase2Result(res);
      setAdjustedResult(null);
      setVolumeSplitEdit(res?.results?.volume_split_edit || []);
      setVolumeSplitInfo(res?.results?.volume_split_info || "");
      setAdjustedSaved(false);
      saveForecastStore({
        phase2: res,
        volumeSplitEdit: res?.results?.volume_split_edit || [],
        phase2Basis
      });
      if (!res.results || !res.results.combined?.length) {
        notify("warning", res.status || "Phase 2 returned no results.");
      } else {
        notify("success", "Phase 2 results ready.");
      }
    } catch (err: any) {
      notify("error", err?.message || "Phase 2 failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleApplyVolumeSplit = async () => {
    const baseRows =
      phase2Result?.results?.phase2_store?.base_df ||
      phase2Result?.results?.base_raw ||
      [];
    const fgMonthly = phase2Result?.results?.phase2_store?.forecast_group_monthly || [];
    if (!baseRows.length) {
      notify("warning", "Run Phase 2 first.");
      return;
    }
    if (!volumeSplitEdit.length) {
      notify("warning", "Volume split data missing.");
      return;
    }
    setLoading(true);
    try {
      const res = await apiPost<any>("/api/forecast/phase2/volume-split", {
        base_df: baseRows,
        split_rows: volumeSplitEdit,
        forecast_group_monthly: fgMonthly
      });
      setAdjustedResult(res);
      saveForecastStore({ phase2Adjusted: res });
      setAdjustedSaved(false);
      notify("success", res.status || "Volume split applied.");
    } catch (err: any) {
      notify("error", err?.message || "Volume split failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveAdjusted = async () => {
    const rows =
      adjustedResult?.results?.adjusted_raw ||
      phase2Result?.results?.base_raw ||
      phase2Result?.results?.base ||
      [];
    if (!rows.length) {
      notify("warning", "Run Phase 2 first.");
      return;
    }
    setLoading(true);
    try {
      const res = await apiPost<any>("/api/forecast/save/adjusted-forecast", {
        data: rows,
        group_name: activeCategory || null
      });
      const path = res?.paths?.[0] ? `Saved to ${res.paths[0]}` : "Saved to disk.";
      notify("success", path);
      setAdjustedSaved(true);
    } catch (err: any) {
      notify("error", err?.message || "Save to disk failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadConfig = () => {
    if (!phase1Result?.config) {
      notify("warning", "Run Phase 1 to generate configuration.");
      return;
    }
    const rows = Object.entries(phase1Result.config).flatMap(([model, params]) =>
      Object.entries(params || {}).map(([key, value]) => ({ Model: model, Parameter: key, Value: value }))
    );
    if (!downloadCsv("phase1-config.csv", rows)) {
      notify("warning", "No configuration rows to download.");
      return;
    }
    notify("success", "Configuration summary downloaded.");
  };

  const handleDownloadPhase1 = () => {
    if (phase1Result?.download_csv) {
      const blob = new Blob([phase1Result.download_csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "phase1-results.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      notify("success", "Phase 1 results downloaded.");
      return;
    }
    const rows = phase1Result?.results?.wide || [];
    if (!downloadCsv("phase1-results.csv", rows)) {
      notify("warning", "No Phase 1 results to download.");
      return;
    }
    notify("success", "Phase 1 results downloaded.");
  };

  return (
    <AppShell crumbs="CAP-CONNECT / Forecast / Volume Summary">
      <div className="forecast-page forecast-volume-page">
        <div className="forecast-page-header">
          <div>
            <h1 className="forecast-page-title">Volume Summary</h1>
            <p className="forecast-page-subtitle">Follow the steps to keep moving forward.</p>
          </div>
          <Link className="forecast-back-link" href="/forecast">
            Back to models
          </Link>
        </div>

        <div className="forecast-stepper">
          <div className="forecast-step-pill active">📊 Volume Summary</div>
          <span className="forecast-step-arrow">→</span>
          <div className="forecast-step-pill">⚙️ Transformation Projects</div>
          <span className="forecast-step-arrow">→</span>
          <div className="forecast-step-pill">⏱️ Daily Interval Forecast</div>
        </div>

        <div className="forecast-upload-row">
          <div className="forecast-upload-label">Upload data</div>
          <div
            className="forecast-upload-box"
            onDrop={handleDrop}
            onDragOver={(event) => event.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
          >
            Drag & drop or <strong>select a CSV/Excel file</strong>
            <input
              ref={fileInputRef}
              className="file-input"
              type="file"
              accept=".csv,.xlsx,.xls,.xlsm"
              onChange={handleFileSelect}
            />
          </div>
          <button className="btn btn-primary" type="button" disabled={loading} onClick={handleUpload}>
            Run Volume Summary
          </button>
        </div>
        <div className="forecast-muted">
          {selectedFile ? `Selected file: ${selectedFile.name}` : "No file supplied."}
        </div>

        {!hasVolumeSummary ? (
          <div className="forecast-muted">Run Volume Summary to load results.</div>
        ) : (
          <>
            <section className="forecast-section-block">
              <div className="forecast-section-title">Preview</div>
              {previewRows.length ? (
                <DataTable data={previewRows} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Summary</div>
          {volumeSummary?.pivot?.length ? (
            <DataTable data={volumeSummary.pivot} />
          ) : volumeSummary?.summary?.length ? (
            <DataTable data={volumeSummary.summary} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>

        <section className="forecast-section-block">
          <div className="forecast-section-title">IQ Summary</div>
          <div className="forecast-section-note">
            Upload an Excel file with an IQ_Data sheet to populate these tables.
          </div>
          {categories.length ? (
            <div className="forecast-form-row" style={{ marginBottom: 12 }}>
              <div>
                <div className="label">Category</div>
                <select
                  className="select"
                  value={activeCategory}
                  onChange={(event) => setActiveCategory(event.target.value)}
                >
                  {categories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          ) : null}
          {activeIQ?.IQ?.length ? (
            <DataTable data={activeIQ.IQ} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
          {activeIQ?.Volume?.length ? (
            <DataTable data={activeIQ.Volume} />
          ) : null}
          {activeIQ?.Contact_Ratio?.length ? (
            <DataTable data={activeIQ.Contact_Ratio} />
          ) : null}
        </section>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Volume Summary</div>
          {volumeSummary?.pivot?.length ? (
            <DataTable data={volumeSummary.pivot} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Contact Ratio Summary (Volume/IQ)</div>
          {activeIQ?.Contact_Ratio?.length ? (
            <DataTable data={activeIQ.Contact_Ratio} />
          ) : seasonality?.results?.ratio?.length ? (
            <DataTable data={seasonality.results.ratio} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Category</div>
          {categories.length ? (
            <DataTable data={categories.map((cat) => ({ Category: cat }))} />
          ) : (
            <>
              <div className="forecast-placeholder-bar" />
              <div className="forecast-placeholder-bar" />
            </>
          )}
        </section>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Seasonality (Based on Contact Ratio)</div>
          {seasonality?.results?.ratio_chart ? (
            <LineChart data={seasonality.results.ratio_chart} />
          ) : (
            <div className="forecast-chart-placeholder" />
          )}
          {seasonality?.results?.ratio?.length ? (
            <DataTable data={seasonality.results.ratio} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Seasonality Cap Adjustment</div>
          <div className="forecast-form-row">
            <div>
              <div className="label">Lower Cap</div>
              <input
                className="input"
                type="number"
                value={lowerCap}
                step="0.05"
                onChange={(event) => setLowerCap(event.target.value)}
              />
            </div>
            <div>
              <div className="label">Upper Cap</div>
              <input
                className="input"
                type="number"
                value={upperCap}
                step="0.05"
                onChange={(event) => setUpperCap(event.target.value)}
              />
            </div>
          </div>
          <div className="forecast-section-note">Seasonality Capped Adjustment Table</div>
          {cappedRatio.length ? (
            <EditableTable
              data={cappedRatio}
              editableColumns={cappedEditableColumns}
              onChange={(rows) => {
                setCappedRatio(rows);
                saveForecastStore({ seasonality: { ...(seasonality || {}), results: { ...(seasonality?.results || {}), capped: rows } } });
              }}
            />
          ) : seasonality?.results?.capped?.length ? (
            <DataTable data={seasonality.results.capped} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Updated Seasonality with Recalculated Avg</div>
          {seasonality?.results?.capped_chart ? (
            <LineChart data={seasonality.results.capped_chart} />
          ) : (
            <div className="forecast-chart-placeholder" />
          )}
          {seasonality?.results?.recalc?.length ? (
            <DataTable data={seasonality.results.recalc} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Custom Base Volume</div>
          <input
            className="input"
            type="text"
            value={customBaseVolume}
            onChange={(event) => {
              setCustomBaseVolume(event.target.value);
              setBaseVolumeEdited(true);
            }}
          />
        </section>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Normalized Ratio</div>
          {seasonality?.results?.normalized?.length ? (
            <DataTable data={seasonality.results.normalized} />
          ) : (
            <div className="forecast-placeholder-bar" />
          )}
        </section>

        <div className="forecast-actions-row">
          <button className="btn btn-primary" type="button" onClick={handleApplyCaps}>
            Apply Changes
          </button>
        </div>

        <section className="forecast-section-block">
          <div className="forecast-section-title">Prophet Smoothing</div>
          <button className="btn btn-outline" type="button" disabled={loading} onClick={handleProphetSmoothing}>
            Run Prophet Smoothing
          </button>
        </section>

        {hasProphetStage ? (
          <>
            <section className="forecast-section-block">
              <div className="forecast-section-title">Data After Prophet Smoothing</div>
              {prophetTable.length ? (
                <EditableTable
                  data={prophetTable}
                  editableColumns={["Final_Smoothed_Value"]}
                  onChange={(rows) => {
                    setProphetTable(rows);
                    saveForecastStore({
                      prophet: {
                        ...(prophetResult || {}),
                        results: { ...(prophetResult?.results || {}), prophet_table: rows }
                      }
                    });
                  }}
                />
              ) : prophetResult?.results?.prophet_table?.length ? (
                <DataTable data={prophetResult.results.prophet_table} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
              {prophetResult?.results?.line_chart ? (
                <LineChart data={prophetResult.results.line_chart} />
              ) : (
                <div className="forecast-chart-placeholder" />
              )}
            </section>

            <section className="forecast-section-block">
              <div className="forecast-section-title">Normalized Contact Ratio 2</div>
              {prophetResult?.results?.norm2_chart ? (
                <LineChart data={prophetResult.results.norm2_chart} />
              ) : (
                <div className="forecast-chart-placeholder" />
              )}
              {prophetNorm2.length ? (
                <DataTable data={prophetNorm2} />
              ) : prophetResult?.results?.norm2?.length ? (
                <DataTable data={prophetResult.results.norm2} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

            <div className="forecast-actions-row">
              <button className="btn btn-primary" type="button" onClick={handleSaveSmoothed}>
                Save Changes if any else skip
              </button>
            </div>

            <div className="forecast-actions-row">
              <button className="btn btn-success" type="button" disabled={loading} onClick={handlePhase1}>
                Run Phase 1
              </button>
            </div>
          </>
        ) : null}

        {hasPhase1Stage ? (
          <>
            <section className="forecast-section-block">
              <div className="forecast-section-title">Phase 1 - Forecast Results</div>
              {phase1CombinedDisplay.length ? (
                <DataTable data={phase1CombinedDisplay} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

            <section className="forecast-section-block">
              <div className="forecast-section-title">Phase 1 Accuracy Before Iterations</div>
              {phase1Result?.results?.accuracy?.length ? (
                <DataTable data={phase1Result.results.accuracy} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

            <section className="forecast-section-block">
              <div className="forecast-section-title">Output Iterative Tuning</div>
              {phase1Result?.results?.tuning?.length ? (
                <DataTable data={phase1Result.results.tuning} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

            <section className="forecast-section-block">
              <div className="forecast-section-title">Final Accuracy (After Tuning)</div>
              {phase1Result?.results?.final_accuracy?.length ? (
                <DataTable data={phase1Result.results.final_accuracy} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

            <section className="forecast-section-block">
              <div className="forecast-section-title">Current Model Configurations</div>
              {phase1ConfigRows.length ? (
                <DataTable data={phase1ConfigRows} />
              ) : (
                <div className="forecast-section-note">Run Phase 1 to view model configurations.</div>
              )}
              <div className="forecast-actions-row">
                <button className="btn btn-outline" type="button" onClick={handleDownloadConfig}>
                  Download Configuration Summary
                </button>
                <button className="btn btn-outline" type="button" onClick={handleDownloadPhase1}>
                  Download Phase 1 Results
                </button>
              </div>
            </section>

            <section className="forecast-section-block">
              <div className="forecast-section-title">Phase 2 Forecast</div>
              <div className="forecast-form-row">
                <div>
                  <div className="label">Forecast basis</div>
                  <select
                    className="select"
                    value={phase2Basis}
                    onChange={(event) => {
                      setPhase2Basis(event.target.value);
                      saveForecastStore({ phase2Basis: event.target.value });
                    }}
                  >
                    <option value="iq">IQ (contact ratio)</option>
                    <option value="volume">Volume (forecast volume directly)</option>
                  </select>
                </div>
                <input
                  className="input rashvi"
                  type="date"
                  placeholder="Start date"
                  value={phase2Start}
                  onChange={(event) => setPhase2Start(event.target.value)}
                />
                <input
                  className="input rashvi"
                  type="date"
                  placeholder="End date"
                  value={phase2End}
                  onChange={(event) => setPhase2End(event.target.value)}
                />
              </div>
              <div className="forecast-actions-row">
                <button className="btn btn-primary" type="button" disabled={loading} onClick={handlePhase2}>
                  Run Phase 2 Forecast
                </button>
                <button
                  className="btn btn-outline"
                  type="button"
                  onClick={() => {
                    setPhase2Result(null);
                    setAdjustedResult(null);
                    setVolumeSplitEdit([]);
                    saveForecastStore({ phase2: null, phase2Adjusted: null, volumeSplitEdit: [] });
                  }}
                >
                  Clear Phase 2 Cache
                </button>
              </div>
            </section>
          </>
        ) : null}

        {hasPhase2Stage ? (
          <>
            <section className="forecast-section-block">
              <div className="forecast-section-title">Phase 2 Contact Ratio Forecast</div>
              {phase2CombinedDisplay.length ? (
                <DataTable data={phase2CombinedDisplay} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

            <section className="forecast-section-block">
              <div className="forecast-section-title">Phase 2 Base Forecast with Volumes</div>
              {phase2Result?.results?.base?.length ? (
                <DataTable data={phase2Result.results.base} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

            <section className="forecast-section-block">
              <div className="forecast-section-title">Forecast Group Summary</div>
              {phase2Result?.results?.forecast_group_summary?.length ? (
                <DataTable data={phase2Result.results.forecast_group_summary} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

            <section className="forecast-section-block">
              <div className="forecast-section-title">Volume Split (%)</div>
              {phase2Result?.results?.forecast_group_split?.length ? (
                <DataTable data={phase2Result.results.forecast_group_split} />
              ) : volumeSummary?.volume_split?.length ? (
                <DataTable data={volumeSummary.volume_split} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

            <section className="forecast-section-block">
              <div className="forecast-section-title">Edit Volume Split Allocation (Latest Year Data)</div>
              {volumeSplitEdit.length ? (
                <EditableTable
                  data={volumeSplitEdit}
                  editableColumns={volumeSplitEditableColumns}
                  onChange={(rows) => {
                    setVolumeSplitEdit(rows);
                    saveForecastStore({ volumeSplitEdit: rows });
                  }}
                />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
              {volumeSplitInfo ? <div className="forecast-muted">{volumeSplitInfo}</div> : null}
              <div className="forecast-actions-row">
                <button
                  className="btn btn-primary"
                  type="button"
                  onClick={handleApplyVolumeSplit}
                >
                  Apply Volume Split to Base Forecast
                </button>
              </div>
            </section>
          </>
        ) : null}

        {hasSplitStage ? (
          <>
            <section className="forecast-section-block">
              <div className="forecast-section-title">Base Forecast Adjusted by Volume Split</div>
              {adjustedResult?.results?.adjusted?.length ? (
                <DataTable data={adjustedResult.results.adjusted} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

            <section className="forecast-section-block">
              <div className="forecast-section-title">Verification</div>
              {adjustedResult?.results?.verify?.length ? (
                <DataTable data={adjustedResult.results.verify} />
              ) : (
                <div className="forecast-placeholder-bar" />
              )}
            </section>

            <div className="forecast-actions-row">
              <button
                className="btn btn-outline"
                type="button"
                onClick={() =>
                  adjustedResult?.results?.adjusted
                    ? downloadCsv("phase2-adjusted-forecast.csv", adjustedResult.results.adjusted)
                    : notify("warning", "Apply volume split first.")
                }
              >
                Download Adjusted Forecast by Group
              </button>
              <button
                className="btn btn-success"
                type="button"
                disabled={loading}
                onClick={handleSaveAdjusted}
              >
                Save Monthly Forecast With Adjustments
              </button>
              {adjustedSaved ? (
                <Link className="btn btn-primary" href="/forecast/transformation-projects">
                  Continue
                </Link>
              ) : null}
            </div>
          </>
        ) : null}
          </>
        )}
      </div>
    </AppShell>
  );
}
