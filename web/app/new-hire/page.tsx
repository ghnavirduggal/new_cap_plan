"use client";

import { useEffect, useMemo, useState } from "react";
import AppShell from "../_components/AppShell";
import BarChart from "../_components/BarChart";
import EditableTable from "../_components/EditableTable";
import { useGlobalLoader } from "../_components/GlobalLoader";
import { useToast } from "../_components/ToastProvider";
import { apiGet, apiPost } from "../../lib/api";
import { parseExcelFile } from "../../lib/excel";

const NH_LABELS: Record<string, string> = {
  business_area: "Business Area",
  class_reference: "Class Reference",
  source_system_id: "Source System ID",
  emp_type: "Emp Type",
  status: "Status",
  class_type: "Class Type",
  class_level: "Class Level",
  grads_needed: "Grads Needed",
  billable_hc: "Billable HC",
  training_weeks: "Training Weeks",
  nesting_weeks: "Nesting Weeks",
  induction_start: "Induction Start",
  training_start: "Training Start",
  training_end: "Training End",
  nesting_start: "Nesting Start",
  nesting_end: "Nesting End",
  production_start: "Production Start",
  created_by: "Created By",
  created_ts: "Created On"
};

const NH_COLS = [
  "business_area",
  "class_reference",
  "source_system_id",
  "emp_type",
  "status",
  "class_type",
  "class_level",
  "grads_needed",
  "billable_hc",
  "training_weeks",
  "nesting_weeks",
  "induction_start",
  "training_start",
  "training_end",
  "nesting_start",
  "nesting_end",
  "production_start",
  "created_by",
  "created_ts"
];

function toDisplayRows(rows: Array<Record<string, any>>) {
  return rows.map((row) => {
    const out: Record<string, any> = {};
    NH_COLS.forEach((key) => {
      const label = NH_LABELS[key] || key;
      out[label] = row?.[key] ?? "";
    });
    return out;
  });
}

function fromDisplayRows(rows: Array<Record<string, any>>) {
  return rows.map((row) => {
    const out: Record<string, any> = {};
    NH_COLS.forEach((key) => {
      const label = NH_LABELS[key] || key;
      out[key] = row?.[label] ?? row?.[key] ?? "";
    });
    return out;
  });
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

function toCsv(rows: Array<Record<string, any>>) {
  if (!rows.length) return "";
  const columns = Object.keys(rows[0] ?? {});
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
  if (!csv) return;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function effectiveCount(row: Record<string, any>) {
  const billable = Number(row.billable_hc);
  if (Number.isFinite(billable) && billable > 0) return Math.round(billable);
  const grads = Number(row.grads_needed) || 0;
  const empType = String(row.emp_type || "").toLowerCase();
  if (empType === "part-time") {
    return Math.ceil(grads / 2);
  }
  return Math.round(grads);
}

function weekLabel(dateValue: any) {
  const dt = new Date(String(dateValue || ""));
  if (Number.isNaN(dt.getTime())) return null;
  const day = (dt.getDay() + 6) % 7;
  dt.setDate(dt.getDate() - day);
  return dt.toISOString().slice(0, 10);
}

const EMPTY_ROW: Record<string, any> = {
  business_area: "",
  class_reference: "",
  source_system_id: "",
  emp_type: "full-time",
  status: "tentative",
  class_type: "ramp-up",
  class_level: "new-agent",
  grads_needed: 0,
  billable_hc: 0,
  training_weeks: 0,
  nesting_weeks: 0,
  induction_start: "",
  training_start: "",
  training_end: "",
  nesting_start: "",
  nesting_end: "",
  production_start: "",
  created_by: "",
  created_ts: ""
};

export default function NewHirePage() {
  const { notify } = useToast();
  const { setLoading } = useGlobalLoader();
  const [tableRows, setTableRows] = useState<Array<Record<string, any>>>([]);
  const [message, setMessage] = useState("");

  const displayRows = useMemo(() => toDisplayRows(tableRows), [tableRows]);

  const summaryChart = useMemo(() => {
    if (!tableRows.length) {
      return { labels: [], series: [] };
    }
    const grouped = new Map<string, number>();
    tableRows.forEach((row) => {
      const week = weekLabel(row.production_start);
      if (!week) return;
      const count = effectiveCount(row);
      grouped.set(week, (grouped.get(week) ?? 0) + count);
    });
    const labels = Array.from(grouped.keys()).sort((a, b) => new Date(a).getTime() - new Date(b).getTime());
    return {
      labels,
      series: [
        {
          name: "Planned HC",
          values: labels.map((label) => grouped.get(label) ?? 0)
        }
      ]
    };
  }, [tableRows]);

  const loadRows = async () => {
    setLoading(true);
    try {
      const res = await apiGet<{ rows?: Array<Record<string, any>> }>("/api/forecast/new-hire");
      setTableRows(res.rows ?? []);
    } catch (error: any) {
      notify("error", error?.message || "Could not load new hire data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadRows();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleUpload = async (file?: File | null) => {
    if (!file) return;
    setLoading(true);
    try {
      const rows = await parseFile(file);
      if (!rows.length) {
        setMessage("Upload failed ✗");
        return;
      }
      const res = await apiPost<{ rows?: Array<Record<string, any>> }>("/api/forecast/new-hire/ingest", { rows });
      setTableRows(res.rows ?? []);
      setMessage(`Uploaded ${rows.length} row(s) ✓`);
      notify("success", "Upload processed.");
    } catch (error: any) {
      setMessage("Upload failed ✗");
      notify("error", error?.message || "Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      await apiPost("/api/forecast/new-hire", { rows: tableRows });
      setMessage(`Saved ${tableRows.length} row(s) ✓`);
      notify("success", "New hire data saved.");
    } catch (error: any) {
      setMessage("Save failed ✗");
      notify("error", error?.message || "Save failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    setLoading(true);
    try {
      const res = await apiGet<{ rows?: Array<Record<string, any>> }>("/api/forecast/new-hire/template");
      downloadCsv("new_hire_template.csv", res.rows ?? []);
    } catch (error: any) {
      notify("error", error?.message || "Download failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleTableChange = (rows: Array<Record<string, any>>) => {
    setTableRows(fromDisplayRows(rows));
  };

  const addRow = () => {
    setTableRows((prev) => [...prev, { ...EMPTY_ROW }]);
  };

  const removeRow = () => {
    setTableRows((prev) => prev.slice(0, -1));
  };

  return (
    <AppShell crumbs="CAP-CONNECT / New Hire Summary">
      <div className="newhire-page">
        <section className="section">
          <h2>New Hire Summary</h2>
          <div className="newhire-actions">
            <label className="upload-box newhire-upload">
              <input type="file" accept=".csv,.xlsx,.xls" onChange={(event) => handleUpload(event.target.files?.[0])} />
              <span>⬆️ Upload CSV/Excel</span>
            </label>
            <button type="button" className="btn btn-primary" onClick={handleSave}>
              Save
            </button>
            <div className="newhire-message">{message}</div>
            <button type="button" className="btn btn-secondary" onClick={handleDownload}>
              Download Sample
            </button>
          </div>
          <EditableTable data={displayRows} onChange={handleTableChange} className="newhire-table" />
          <div className="newhire-row-actions">
            <button type="button" className="btn btn-outline" onClick={addRow}>
              Add Row
            </button>
            <button type="button" className="btn btn-outline" onClick={removeRow} disabled={!tableRows.length}>
              Remove Last Row
            </button>
          </div>
          <div className="newhire-chart">
            <h3>Planned New Hires by Production Week</h3>
            <BarChart data={summaryChart} />
          </div>
        </section>
      </div>
    </AppShell>
  );
}
