"use client";

import type { ChangeEvent, ReactNode, Ref } from "react";

type EditableTableProps = {
  data?: Array<Record<string, any>>;
  emptyLabel?: string;
  maxRows?: number;
  className?: string;
  editableColumns?: string[];
  onChange?: (rows: Array<Record<string, any>>) => void;
  wrapperRef?: Ref<HTMLDivElement>;
};

function formatCell(value: any): ReactNode {
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


export default function EditableTable({
  data,
  emptyLabel = "No data",
  maxRows,
  className,
  editableColumns,
  onChange,
  wrapperRef
}: EditableTableProps) {
  const rows = data ?? [];
  if (!rows.length) {
    return <div className="forecast-muted">{emptyLabel}</div>;
  }

  const visibleRows = maxRows ? rows.slice(0, maxRows) : rows;
  const columns = buildColumns(visibleRows);
  const editableSet = new Set(editableColumns ?? []);

  const handleChange = (rowIdx: number, col: string, event: ChangeEvent<HTMLInputElement>) => {
    const next = rows.map((row, idx) =>
      idx === rowIdx
        ? {
            ...row,
            [col]: event.target.value
          }
        : row
    );
    onChange?.(next);
  };

  return (
    <div className={`table-wrap ${className ?? ""}`.trim()} ref={wrapperRef}>
      <table className="table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col} data-col={col}>{headerLabel(col)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visibleRows.map((row, idx) => (
            <tr key={`${idx}-${columns[0] || "row"}`}>
              {columns.map((col) => (
                <td key={col} data-col={col}>
                  {editableSet.has(col) ? (
                    <input
                      className="table-input"
                      value={row?.[col] ?? ""}
                      onChange={(event) => handleChange(idx, col, event)}
                    />
                  ) : (
                    formatCell(row?.[col])
                  )}
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
