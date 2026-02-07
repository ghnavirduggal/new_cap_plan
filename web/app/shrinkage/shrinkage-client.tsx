"use client";

import { useEffect, useMemo, useState } from "react";
import DataTable from "../_components/DataTable";
import EditableTable from "../_components/EditableTable";
import LineChart from "../_components/LineChart";
import { useGlobalLoader } from "../_components/GlobalLoader";
import { useToast } from "../_components/ToastProvider";
import { apiGet, apiPost } from "../../lib/api";
import { parseExcelFile } from "../../lib/excel";

type RawState = {
  sourceRows: Array<Record<string, any>>;
  rawRows: Array<Record<string, any>>;
  dailyRows: Array<Record<string, any>>;
  weeklyRows: Array<Record<string, any>>;
  message: string;
};

const SHRINK_TABS = [
  { key: "weekly", label: "Weekly Shrink %" },
  { key: "voice", label: "Voice Shrinkage (Raw)" },
  { key: "bo", label: "Back Office Shrinkage (Raw)" },
  { key: "chat", label: "Chat Shrinkage (Raw)" },
  { key: "ob", label: "Outbound Shrinkage (Raw)" }
];

const ATTRITION_SAMPLE_COLUMNS = [
  "Reporting Full Date",
  "BRID",
  "Employee Name",
  "Operational Status",
  "Corporate Grade Description",
  "Employee Email Address",
  "Employee Position",
  "Position Description",
  "Employee Line Manager Indicator",
  "Length of Service Date",
  "Cost Centre",
  "Line Manager BRID",
  "Line Manager Name",
  "IMH L05",
  "IMH L06",
  "Employee Line Manager lvl 01",
  "Employee Line Manager lvl 02",
  "Employee Line Manager lvl 03",
  "Employee Line Manager lvl 04",
  "Employee Line Manager lvl 05",
  "Employee Line Manager lvl 06",
  "Employee Line Manager lvl 07",
  "Employee Line Manager lvl 08",
  "Employee Line Manager lvl 09",
  "City",
  "Building",
  "Gender Description",
  "Voluntary Involuntary Exit Description",
  "Resignation Date",
  "Employee Contract",
  "HC",
  "HC FTE"
];

const SHRINK_LABELS: Record<string, string> = {
  voice: "Voice",
  bo: "Back Office",
  chat: "Chat",
  ob: "Outbound"
};

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

async function parseFile(file: File): Promise<Record<string, any>[]> {
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (ext === "xlsx" || ext === "xls") {
    return parseExcelFile(file);
  }
  const text = await file.text();
  return parseCsv(text);
}

function downloadCsv(filename: string, rows: Array<Record<string, any>>, columnsOverride?: string[]) {
  const headers = columnsOverride?.length
    ? columnsOverride
    : Array.from(
        rows.reduce<Set<string>>((set, row) => {
          Object.keys(row || {}).forEach((key) => set.add(key));
          return set;
        }, new Set<string>())
      );
  if (!headers.length) return;
  const csv = [headers.join(",")]
    .concat(
      rows.map((row) =>
        headers
          .map((header) => {
            const raw = row?.[header] ?? "";
            const text = String(raw).replace(/"/g, "\"\"");
            return text.includes(",") ? `"${text}"` : text;
          })
          .join(",")
      )
    )
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function lineFromRows(rows: Array<Record<string, any>>, xKey: string, yKey: string, label: string) {
  if (!rows.length) return null;
  const x = rows.map((row) => String(row?.[xKey] ?? ""));
  return {
    x,
    series: [
      {
        name: label,
        points: x.map((val, idx) => ({
          x: val,
          y: Number(rows[idx]?.[yKey] ?? 0)
        }))
      }
    ]
  };
}

function notifySettingsUpdated() {
  if (typeof window === "undefined") return;
  const event = new CustomEvent("settingsUpdated", {
    detail: { source: "shrinkage" }
  });
  window.dispatchEvent(event);
}

export default function ShrinkageClient() {
  const { setLoading } = useGlobalLoader();
  const { notify } = useToast();
  const [activeTab, setActiveTab] = useState("shrink");
  const [activeShrinkTab, setActiveShrinkTab] = useState("weekly");

  const [weeklyRows, setWeeklyRows] = useState<Array<Record<string, any>>>([]);
  const [weeklyMessage, setWeeklyMessage] = useState("");
  const [saveRawModalOpen, setSaveRawModalOpen] = useState(false);
  const [saveRawKind, setSaveRawKind] = useState("voice");

  const [rawStates, setRawStates] = useState<Record<string, RawState>>({
    voice: { sourceRows: [], rawRows: [], dailyRows: [], weeklyRows: [], message: "" },
    bo: { sourceRows: [], rawRows: [], dailyRows: [], weeklyRows: [], message: "" },
    chat: { sourceRows: [], rawRows: [], dailyRows: [], weeklyRows: [], message: "" },
    ob: { sourceRows: [], rawRows: [], dailyRows: [], weeklyRows: [], message: "" }
  });

  const [attrRows, setAttrRows] = useState<Array<Record<string, any>>>([]);
  const [attrRawRows, setAttrRawRows] = useState<Array<Record<string, any>>>([]);
  const [attrMessage, setAttrMessage] = useState("");

  const loadShrinkage = async () => {
    setLoading(true);
    try {
      const res = await apiGet<{ rows?: Array<Record<string, any>> }>("/api/forecast/shrinkage");
      setWeeklyRows(res.rows ?? []);
    } catch (error: any) {
      notify("error", error?.message || "Could not load shrinkage data.");
    } finally {
      setLoading(false);
    }
  };

  const loadAttrition = async () => {
    setLoading(true);
    try {
      const res = await apiGet<{ rows?: Array<Record<string, any>> }>("/api/forecast/attrition");
      setAttrRows(res.rows ?? []);
    } catch (error: any) {
      notify("error", error?.message || "Could not load attrition data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadShrinkage();
    void loadAttrition();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const weeklyChart = useMemo(() => lineFromRows(weeklyRows, "week", "overall_pct", "Overall Shrink %"), [weeklyRows]);
  const attrChart = useMemo(() => lineFromRows(attrRows, "week", "attrition_pct", "Attrition %"), [attrRows]);

  const handleRawUpload = async (kind: string, file: File | null) => {
    if (!file) return;
    setLoading(true);
    try {
      const sourceRows = await parseFile(file);
      if (!sourceRows.length) {
        setRawStates((prev) => ({
          ...prev,
          [kind]: { ...prev[kind], sourceRows: [], rawRows: [], dailyRows: [], weeklyRows: [], message: "Upload produced no rows." }
        }));
        return;
      }
      const res = await apiPost<{ raw?: any[]; daily?: any[]; weekly?: any[]; combined?: any[] }>("/api/forecast/shrinkage/raw", {
        kind,
        rows: sourceRows,
        save: false
      });
      setRawStates((prev) => ({
        ...prev,
        [kind]: {
          sourceRows,
          rawRows: res.raw ?? [],
          dailyRows: res.daily ?? [],
          weeklyRows: res.weekly ?? [],
          message: `Loaded ${sourceRows.length} rows.`
        }
      }));
    } catch (error: any) {
      notify("error", error?.message || "Could not process shrinkage upload.");
    } finally {
      setLoading(false);
    }
  };

  const saveRaw = async (kind: string, mode: "replace" | "append" = "replace") => {
    const state = rawStates[kind];
    if (!state?.sourceRows.length) {
      setRawStates((prev) => ({ ...prev, [kind]: { ...prev[kind], message: "No raw rows to save." } }));
      return;
    }
    setLoading(true);
    try {
      const res = await apiPost<{ combined?: any[]; weekly?: any[] }>("/api/forecast/shrinkage/raw", {
        kind,
        rows: state.sourceRows,
        save: true,
        mode
      });
      const weeklyCount = res.weekly?.length ?? 0;
      const detailParts = [`raw rows: ${state.sourceRows.length}`];
      if (weeklyCount) detailParts.push(`weekly points: ${weeklyCount}`);
      const label = SHRINK_LABELS[kind] || kind;
      setWeeklyRows(res.combined ?? weeklyRows);
      notify("success", `Saved ${label} shrinkage (${detailParts.join(", ")})`);
      notifySettingsUpdated();
    } catch (error: any) {
      notify("error", error?.message || "Could not save shrinkage rows.");
    } finally {
      setLoading(false);
    }
  };

  const saveWeekly = async () => {
    setLoading(true);
    try {
      const res = await apiPost<{ rows?: any[] }>("/api/forecast/shrinkage", { rows: weeklyRows });
      setWeeklyRows(res.rows ?? weeklyRows);
      notify("success", "Saved weekly shrinkage.");
      notifySettingsUpdated();
    } catch (error: any) {
      notify("error", error?.message || "Could not save weekly shrinkage.");
    } finally {
      setLoading(false);
    }
  };

  const downloadTemplate = async (kind: string) => {
    setLoading(true);
    try {
      const res = await apiGet<{ rows?: Array<Record<string, any>> }>(`/api/forecast/shrinkage/template?kind=${kind}`);
      const rows = res.rows ?? [];
      if (!rows.length) return;
      const filename = kind === "voice" ? "shrinkage_voice_raw_template.csv" : "shrinkage_backoffice_raw_template.csv";
      downloadCsv(filename, rows);
    } catch (error: any) {
      notify("error", error?.message || "Could not download template.");
    } finally {
      setLoading(false);
    }
  };

  const handleAttritionUpload = async (file: File | null) => {
    if (!file) return;
    setLoading(true);
    try {
      const sourceRows = await parseFile(file);
      if (!sourceRows.length) {
        setAttrMessage("Upload produced no rows.");
        return;
      }
      const res = await apiPost<{ weekly?: any[] }>("/api/forecast/attrition/raw", { rows: sourceRows, save: false });
      setAttrRows(res.weekly ?? []);
      setAttrRawRows(sourceRows);
      setAttrMessage(`Loaded ${sourceRows.length} rows.`);
    } catch (error: any) {
      notify("error", error?.message || "Could not process attrition upload.");
    } finally {
      setLoading(false);
    }
  };

  const saveAttrition = async () => {
    setLoading(true);
    try {
      if (attrRawRows.length) {
        await apiPost("/api/forecast/attrition/raw", { rows: attrRawRows, save: true });
      }
      const res = await apiPost<{ rows?: any[] }>("/api/forecast/attrition", { rows: attrRows });
      setAttrRows(res.rows ?? attrRows);
      setAttrMessage("");
      notify("success", "Saved attrition.");
      notifySettingsUpdated();
    } catch (error: any) {
      notify("error", error?.message || "Could not save attrition rows.");
    } finally {
      setLoading(false);
    }
  };

  const downloadAttritionSample = () => {
    downloadCsv("leavers_sample.csv", [], ATTRITION_SAMPLE_COLUMNS);
  };

  const activeRaw = rawStates[activeShrinkTab] ?? rawStates.voice;

  const openSaveRawModal = (kind: string) => {
    const state = rawStates[kind];
    if (!state?.sourceRows.length) {
      setRawStates((prev) => ({ ...prev, [kind]: { ...prev[kind], message: "No raw rows to save." } }));
      return;
    }
    setSaveRawKind(kind);
    setSaveRawModalOpen(true);
  };

  const confirmSaveRaw = (mode: "replace" | "append") => {
    setSaveRawModalOpen(false);
    void saveRaw(saveRawKind, mode);
  };

  return (
    <section className="section shrinkage-page">
      <div className="tabs">
        <div className={`tab ${activeTab === "shrink" ? "active" : ""}`} onClick={() => setActiveTab("shrink")}>
          Shrinkage
        </div>
        <div className={`tab ${activeTab === "attrition" ? "active" : ""}`} onClick={() => setActiveTab("attrition")}>
          Attrition
        </div>
      </div>

      {activeTab === "shrink" ? (
        <>
          <div className="tabs">
            {SHRINK_TABS.map((tab) => (
              <div
                key={tab.key}
                className={`tab ${activeShrinkTab === tab.key ? "active" : ""}`}
                onClick={() => setActiveShrinkTab(tab.key)}
              >
                {tab.label}
              </div>
            ))}
          </div>

          {activeShrinkTab === "weekly" ? (
            <div className="shrinkage-weekly">
              <EditableTable data={weeklyRows} onChange={setWeeklyRows} maxRows={10} />
              <div className="shrinkage-action-row">
                <button type="button" className="btn btn-primary" onClick={saveWeekly}>
                  Save Weekly Shrinkage
                </button>
                {weeklyMessage ? <span className="forecast-muted">{weeklyMessage}</span> : null}
              </div>
              <LineChart data={weeklyChart} />
            </div>
          ) : (
            <div className="shrinkage-raw">
              <div className="shrinkage-raw-grid">
                <div className="upload-box">
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={(event) => handleRawUpload(activeShrinkTab, event.target.files?.[0] ?? null)}
                  />
                  <button type="button" className="btn btn-primary" onClick={() => openSaveRawModal(activeShrinkTab)}>
                    Save {activeShrinkTab.toUpperCase()}
                  </button>
                {["bo", "chat", "ob"].includes(activeShrinkTab) ? (
                    <>
                      <button type="button" className="btn btn-outline" onClick={() => downloadTemplate("voice")}>
                        Download Alvaria Template
                      </button>
                      <button type="button" className="btn btn-outline" onClick={() => downloadTemplate("bo")}>
                        Download Control IQ Template
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      className="btn btn-outline"
                      onClick={() => downloadTemplate("voice")}
                    >
                      Download Alvaria Template
                    </button>
                  )}
                </div>
                {activeRaw?.message ? <div className="forecast-muted">{activeRaw.message}</div> : null}
              </div>
              <h4>Uploaded (normalized)</h4>
              <DataTable data={activeRaw?.rawRows ?? []} maxRows={8} />
              <h4>Daily Summary (derived)</h4>
              <DataTable data={activeRaw?.dailyRows ?? []} maxRows={8} />
            </div>
          )}
        </>
      ) : null}

      {activeTab === "attrition" ? (
        <div className="shrinkage-attrition">
          <div className="upload-box">
            <input type="file" accept=".csv,.xlsx,.xls" onChange={(event) => handleAttritionUpload(event.target.files?.[0] ?? null)} />
            <button type="button" className="btn btn-primary" onClick={saveAttrition}>
              Save Attrition
            </button>
            <button type="button" className="btn btn-outline" onClick={downloadAttritionSample}>
              Download Sample
            </button>
          </div>
          {attrMessage ? <div className="forecast-muted" style={{ marginTop: 8 }}>{attrMessage}</div> : null}
          <EditableTable data={attrRows} onChange={setAttrRows} maxRows={10} />
          <LineChart data={attrChart} />
        </div>
      ) : null}

      {saveRawModalOpen ? (
        <div className="ws-modal-backdrop">
          <div className="ws-modal ws-modal-sm">
            <div className="ws-modal-header" style={{ background: "#2f3747", color: "white" }}>
              <h3>Save Shrinkage Upload</h3>
              <button type="button" className="btn btn-light closeOptions" onClick={() => setSaveRawModalOpen(false)}>
                <svg width="16" height="16" viewBox="0 0 16 16">
                  <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2"/>
                  <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2"/>
                </svg>
              </button>
            </div>
            <div className="ws-modal-body">
              <p style={{ margin: 0 }}>
                Do you want to replace existing shrinkage values for the same weeks, or append this upload on top?
              </p>
            </div>
            <div className="ws-modal-footer">
              <button type="button" className="btn btn-primary" onClick={() => confirmSaveRaw("replace")}>
                Replace Existing
              </button>
              <button type="button" className="btn btn-light" onClick={() => confirmSaveRaw("append")}>
                Append to Existing
              </button>
              <button type="button" className="btn btn-light" onClick={() => setSaveRawModalOpen(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
