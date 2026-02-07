"use client";

import { useMemo, useState, type ReactNode, type Ref } from "react";

type DataTableProps = {
  data?: Array<Record<string, any>>;
  emptyLabel?: string;
  maxRows?: number;
  className?: string;
  dateMode?: "auto" | "day" | "month";
  wrapperRef?: Ref<HTMLDivElement>;
};

function isWholeNumberMetric(column?: string) {
  if (!column) return false;
  const key = column.trim().toLowerCase();
  return (
    key === "volume" ||
    key === "items" ||
    key === "calls" ||
    key.endsWith("_volume") ||
    key.endsWith("_items") ||
    key.endsWith("_calls")
  );
}

function formatCell(value: any, column?: string): ReactNode {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return value;
    if (isWholeNumberMetric(column)) return Math.round(value).toLocaleString();
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function buildColumns(rows: Array<Record<string, any>>) {
  const seen = new Set<string>();
  const columns: string[] = [];
  for (const row of rows) {
    Object.keys(row || {}).forEach((key) => {
      if (key.startsWith("__")) return;
      if (!seen.has(key)) {
        seen.add(key);
        columns.push(key);
      }
    });
  }
  const metricIndex = columns.findIndex((col) => col.trim().toLowerCase() === "metric");
  if (metricIndex > 0) {
    const [metric] = columns.splice(metricIndex, 1);
    columns.unshift(metric);
  }
  return columns;
}

function formatMonthLabel(dt: Date) {
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const month = months[dt.getMonth()] ?? "Jan";
  const year = String(dt.getFullYear()).slice(-2);
  return `${month}-${year}`;
}

function weekStartMonday(dt: Date) {
  const d = new Date(dt);
  const day = d.getDay(); // 0=Sun..6=Sat
  const offset = (day + 6) % 7; // days since Monday
  d.setDate(d.getDate() - offset);
  d.setHours(0, 0, 0, 0);
  return d;
}

function headerLabel(col: string, dateMode: "auto" | "day" | "month") {
  const trimmed = col.trim();
  if (trimmed.toLowerCase() === "metric") return "Metric";
  if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) {
    const dt = new Date(`${trimmed}T00:00:00`);
    if (!Number.isNaN(dt.getTime())) {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const weekStart = weekStartMonday(today);
      const weekEnd = new Date(weekStart);
      weekEnd.setDate(weekEnd.getDate() + 6);
      const isCurrentWeek = dt >= weekStart && dt <= weekEnd;
      const tag = dt < weekStart || isCurrentWeek ? "Actual" : "Plan";
      if (dateMode === "month") {
        return `${formatMonthLabel(dt)} ${tag}`;
      }
      return `${trimmed} ${tag}`;
    }
  }
  return trimmed;
}

export default function DataTable({
  data,
  emptyLabel = "No data",
  maxRows,
  className,
  dateMode = "auto",
  wrapperRef
}: DataTableProps) {
  const rows = data ?? [];
  if (!rows.length) {
    return <div className="forecast-muted">{emptyLabel}</div>;
  }

  const [showAll, setShowAll] = useState(false);
  const defaultMax = 500;
  const effectiveMax = maxRows === 0 ? undefined : maxRows ?? defaultMax;
  const visibleRows = !showAll && effectiveMax ? rows.slice(0, effectiveMax) : rows;
  const columns = useMemo(() => buildColumns(visibleRows), [visibleRows]);
  const rawNumberCols = new Set<string>();
  columns.forEach((col) => {
    if (col.trim().toLowerCase() === "year") rawNumberCols.add(col);
  });

  return (
    <div className={`table-wrap ${className ?? ""}`.trim()} ref={wrapperRef}>
      <table className="table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col} data-col={col}>{headerLabel(col, dateMode)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visibleRows.map((row, idx) => (
            <tr key={`${idx}-${columns[0] || "row"}`}>
              {columns.map((col) => (
                <td key={col} data-col={col} title={(row as any)?.__tooltips?.[col] ?? ""}>
                  {rawNumberCols.has(col) && typeof row?.[col] === "number"
                    ? row?.[col]
                    : formatCell(row?.[col], col)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {effectiveMax && rows.length > effectiveMax ? (
        <div className="forecast-muted table-footer">
          Showing {showAll ? rows.length : effectiveMax} of {rows.length} rows.
          <button
            type="button"
            className="btn btn-outline table-expand-btn"
            onClick={() => setShowAll((prev) => !prev)}
          >
            {showAll ? "Show fewer rows" : "Show all rows"}
          </button>
        </div>
      ) : null}
    </div>
  );
}
