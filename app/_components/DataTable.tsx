"use client";

import type { ReactNode } from "react";

type DataTableProps = {
  data?: Array<Record<string, any>>;
  emptyLabel?: string;
  maxRows?: number;
  className?: string;
};

function formatCell(value: any, column?: string): ReactNode {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString() : value;
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
      if (!seen.has(key)) {
        seen.add(key);
        columns.push(key);
      }
    });
  }
  return columns;
}

function headerLabel(col: string) {
  if (col.trim().toLowerCase() === "metric") return "Metric";
  return col;
}

export default function DataTable({ data, emptyLabel = "No data", maxRows, className }: DataTableProps) {
  const rows = data ?? [];
  if (!rows.length) {
    return <div className="forecast-muted">{emptyLabel}</div>;
  }

  const visibleRows = maxRows ? rows.slice(0, maxRows) : rows;
  const columns = buildColumns(visibleRows);
  const rawNumberCols = new Set<string>();
  columns.forEach((col) => {
    if (col.trim().toLowerCase() === "year") rawNumberCols.add(col);
  });

  return (
    <div className={`table-wrap ${className ?? ""}`.trim()}>
      <table className="table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{headerLabel(col)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visibleRows.map((row, idx) => (
            <tr key={`${idx}-${columns[0] || "row"}`}>
              {columns.map((col) => (
                <td key={col}>
                  {rawNumberCols.has(col) && typeof row?.[col] === "number"
                    ? row?.[col]
                    : formatCell(row?.[col], col)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {maxRows && rows.length > maxRows ? (
        <div className="forecast-muted">Showing {maxRows} of {rows.length} rows.</div>
      ) : null}
    </div>
  );
}
