"use client";

import { useEffect, useMemo, useState } from "react";
import AppShell from "../_components/AppShell";
import DataTable from "../_components/DataTable";
import EditableTable from "../_components/EditableTable";
import { useGlobalLoader } from "../_components/GlobalLoader";
import { useToast } from "../_components/ToastProvider";
import { apiGet, apiPost } from "../../lib/api";
import { parseExcelFile } from "../../lib/excel";

type RosterPayload = {
  wide?: Array<Record<string, any>>;
  long?: Array<Record<string, any>>;
};

function todayIso(offsetDays = 0) {
  const dt = new Date();
  dt.setDate(dt.getDate() + offsetDays);
  return dt.toISOString().slice(0, 10);
}

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

function downloadCsv(filename: string, rows: Array<Record<string, any>>) {
  if (!rows.length) return;
  const headerSet = rows.reduce<Set<string>>((set, row) => {
    Object.keys(row || {}).forEach((key) => set.add(key));
    return set;
  }, new Set<string>());
  const headers = Array.from(headerSet);
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

function toDate(value?: string) {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? null : parsed;
}

export default function RosterPage() {
  const { setLoading } = useGlobalLoader();
  const { notify } = useToast();
  const [templateStart, setTemplateStart] = useState(todayIso(0));
  const [templateEnd, setTemplateEnd] = useState(todayIso(13));
  const [previewStart, setPreviewStart] = useState(todayIso(0));
  const [previewEnd, setPreviewEnd] = useState(todayIso(13));
  const [bulkStart, setBulkStart] = useState(todayIso(0));
  const [bulkEnd, setBulkEnd] = useState(todayIso(13));
  const [bulkAction, setBulkAction] = useState("blank");
  const [bulkBrids, setBulkBrids] = useState<string[]>([]);

  const [wideRows, setWideRows] = useState<Array<Record<string, any>>>([]);
  const [longRows, setLongRows] = useState<Array<Record<string, any>>>([]);
  const [uploadMessage, setUploadMessage] = useState("");
  const [saveMessage, setSaveMessage] = useState("");
  const [bulkMessage, setBulkMessage] = useState("");

  const loadRoster = async () => {
    setLoading(true);
    try {
      const res = await apiGet<RosterPayload>("/api/forecast/roster");
      setWideRows(res.wide ?? []);
      setLongRows(res.long ?? []);
    } catch (error: any) {
      notify("error", error?.message || "Could not load roster.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadRoster();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const bridOptions = useMemo(() => {
    const rows = longRows ?? [];
    const key = rows.some((row) => row.BRID !== undefined) ? "BRID" : "brid";
    const set = new Set<string>();
    rows.forEach((row) => {
      const value = row?.[key];
      if (value) set.add(String(value));
    });
    return Array.from(set).sort();
  }, [longRows]);

  const previewRows = useMemo(() => {
    if (!longRows.length) return [];
    const start = toDate(previewStart);
    const end = toDate(previewEnd);
    if (!start || !end) return longRows;
    return longRows.filter((row) => {
      const dateValue = toDate(row?.date);
      if (!dateValue) return false;
      return dateValue >= start && dateValue <= end;
    });
  }, [longRows, previewStart, previewEnd]);

  const buildEmptyWideRow = () => {
    const baseCols = ["BRID", "Name", "Team Manager", "Business Area", "Sub Business Area", "LOB", "Site", "Location", "Country"];
    const columns = wideRows.length ? Object.keys(wideRows[0]) : baseCols;
    if (!wideRows.length && templateStart && templateEnd) {
      const start = toDate(templateStart);
      const end = toDate(templateEnd);
      if (start && end) {
        const dates: string[] = [];
        const current = new Date(start);
        while (current <= end) {
          dates.push(current.toISOString().slice(0, 10));
          current.setDate(current.getDate() + 1);
        }
        columns.push(...dates);
      }
    }
    const row: Record<string, any> = {};
    columns.forEach((col) => {
      row[col] = "";
    });
    return row;
  };

  const handleUpload = async (file: File | null) => {
    if (!file) return;
    setLoading(true);
    try {
      const rows = await parseFile(file);
      if (!rows.length) {
        setUploadMessage("Upload produced no rows.");
        return;
      }
      setWideRows(rows);
      const res = await apiPost<{ rows?: Array<Record<string, any>> }>("/api/forecast/roster/normalize", { rows });
      setLongRows(res.rows ?? []);
      setUploadMessage(`Loaded ${rows.length} rows. Normalized rows: ${(res.rows ?? []).length}.`);
    } catch (error: any) {
      notify("error", error?.message || "Could not parse roster file.");
    } finally {
      setLoading(false);
    }
  };

  const saveRoster = async () => {
    if (!wideRows.length && !longRows.length) {
      setSaveMessage("No roster rows to save.");
      return;
    }
    setLoading(true);
    try {
      const res = await apiPost<{ counts?: Record<string, any> }>("/api/forecast/roster", {
        wide: wideRows,
        long: longRows
      });
      setSaveMessage("Saved ✓ (wide + normalized)");
      notify("success", `Roster saved (${res.counts?.wide ?? 0} wide / ${res.counts?.long ?? 0} normalized).`);
    } catch (error: any) {
      notify("error", error?.message || "Could not save roster.");
    } finally {
      setLoading(false);
    }
  };

  const downloadTemplate = async (sample: boolean) => {
    if (!templateStart || !templateEnd) return;
    setLoading(true);
    try {
      const params = new URLSearchParams({
        start: templateStart,
        end: templateEnd,
        sample: sample ? "true" : "false"
      });
      const res = await apiGet<{ rows?: Array<Record<string, any>> }>(`/api/forecast/roster/template?${params.toString()}`);
      const rows = res.rows ?? [];
      if (!rows.length) return;
      const name = sample ? "roster_sample" : "roster_template";
      downloadCsv(`${name}_${templateStart}_${templateEnd}.csv`, rows);
    } catch (error: any) {
      notify("error", error?.message || "Could not download template.");
    } finally {
      setLoading(false);
    }
  };

  const applyBulk = () => {
    const start = toDate(bulkStart);
    const end = toDate(bulkEnd);
    if (!start || !end) {
      setBulkMessage("Select a date range first.");
      return;
    }
    if (!longRows.length) {
      setBulkMessage("No normalized rows available.");
      return;
    }
    const entryKey = longRows.some((row) => row.entry !== undefined) ? "entry" : "value";
    const bridKey = longRows.some((row) => row.BRID !== undefined) ? "BRID" : "brid";
    let edits = 0;
    const next = longRows.map((row) => {
      const dateValue = toDate(row?.date);
      if (!dateValue) return row;
      if (dateValue < start || dateValue > end) return row;
      if (bulkBrids.length && row?.[bridKey] && !bulkBrids.includes(String(row[bridKey]))) return row;
      edits += 1;
      const nextRow = { ...row };
      nextRow[entryKey] = bulkAction === "blank" ? "" : bulkAction;
      if ("is_leave" in nextRow) {
        nextRow.is_leave = ["leave", "l", "off", "pto"].includes(String(nextRow[entryKey]).toLowerCase());
      }
      return nextRow;
    });
    setLongRows(next);
    if (!edits) {
      setBulkMessage("No matching rows in range.");
    } else if (bulkAction === "blank") {
      setBulkMessage(`Cleared ${edits} cells.`);
    } else {
      setBulkMessage(`Set '${bulkAction}' on ${edits} cells.`);
    }
  };

  return (
    <AppShell crumbs="CAP-CONNECT / Employee Roster">
      <section className="section gauravi">
        <h2>Download Roster Template</h2>
        <div className="grid grid-3 rudra">
          <div>
            <div className="label">Start Date</div>
            <input className="input" type="date" value={templateStart} onChange={(event) => setTemplateStart(event.target.value)} />
          </div>
          <div>
            <div className="label">End Date</div>
            <input className="input" type="date" value={templateEnd} onChange={(event) => setTemplateEnd(event.target.value)} />
          </div>
          <div className="roster-action-row">
            <button type="button" className="btn btn-secondary" onClick={() => downloadTemplate(false)}>
              Download Empty Template (CSV)
            </button>
            <button type="button" className="btn btn-outline" onClick={() => downloadTemplate(true)}>
              Download Sample
            </button>
          </div>
        </div>
      </section>

      <section className="section gauravi">
        <h2>Upload Filled Roster</h2>
        <div className="upload-box roster-upload">
          <input type="file" accept=".csv,.xlsx,.xls" onChange={(event) => handleUpload(event.target.files?.[0] ?? null)} />
          <button type="button" className="btn btn-primary" onClick={saveRoster}>
            Save
          </button>
        </div>
        {uploadMessage ? <div className="forecast-muted" style={{ marginTop: 8 }}>{uploadMessage}</div> : null}
        {saveMessage ? <div className="badge" style={{ marginTop: 8 }}>{saveMessage}</div> : null}
        <EditableTable
          data={wideRows}
          editableColumns={wideRows.length ? Object.keys(wideRows[0]) : undefined}
          onChange={setWideRows}
          maxRows={10}
        />
        <div className="roster-action-row" style={{ marginTop: 8 }}>
          <button type="button" className="btn btn-outline" onClick={() => setWideRows((prev) => [...prev, buildEmptyWideRow()])}>
            Add Row
          </button>
          <button
            type="button"
            className="btn btn-outline"
            onClick={() => setWideRows((prev) => prev.slice(0, Math.max(0, prev.length - 1)))}
          >
            Remove Last Row
          </button>
        </div>
      </section>

      <section className="section gauravi">
        <h2>Normalized Schedule Preview</h2>
        <div className="grid grid-2">
          <div>
            <div className="label">Preview Start</div>
            <input className="input" type="date" value={previewStart} onChange={(event) => setPreviewStart(event.target.value)} />
          </div>
          <div>
            <div className="label">Preview End</div>
            <input className="input" type="date" value={previewEnd} onChange={(event) => setPreviewEnd(event.target.value)} />
          </div>
        </div>
        <DataTable data={previewRows} maxRows={12} />
      </section>

      <section className="section gauravi">
        <h2>Bulk Edit Helpers</h2>
        <div className="grid grid-2">
          <div>
            <div className="label">Date Range</div>
            <div className="roster-range-row">
              <input className="input" type="date" value={bulkStart} onChange={(event) => setBulkStart(event.target.value)} />
              <input className="input" type="date" value={bulkEnd} onChange={(event) => setBulkEnd(event.target.value)} />
            </div>
          </div>
          <div>
            <div className="label">BRIDs (optional)</div>
            <select
              className="select ops-multi-select"
              multiple
              value={bulkBrids}
              onChange={(event) =>
                setBulkBrids(Array.from(event.target.selectedOptions).map((opt) => opt.value))
              }
            >
              {bridOptions.map((brid) => (
                <option key={brid} value={brid}>
                  {brid}
                </option>
              ))}
            </select>
          </div>
          <div>
            <div className="label">Action</div>
            <select className="select" value={bulkAction} onChange={(event) => setBulkAction(event.target.value)}>
              <option value="blank">Clear</option>
              <option value="Leave">Leave</option>
              <option value="OFF">OFF</option>
            </select>
          </div>
          <div className="roster-action-row">
            <button type="button" className="btn btn-outline" onClick={applyBulk}>
              Apply to range
            </button>
          </div>
        </div>
        {bulkMessage ? <div className="forecast-muted" style={{ marginTop: 8 }}>{bulkMessage}</div> : null}
      </section>
    </AppShell>
  );
}
