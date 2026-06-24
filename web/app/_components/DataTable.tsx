"use client";

import { useCallback, useMemo, useState, type MouseEvent as ReactMouseEvent, type ReactNode, type Ref } from "react";

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
    // Never surface NaN/Infinity as literal text in a metric cell.
    if (!Number.isFinite(value)) return "";
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

function headerLabel(col: string, dateMode: "auto" | "day" | "month") {
  const trimmed = col.trim();
  if (trimmed.toLowerCase() === "metric") return "Metric";
  if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) {
    const dt = new Date(`${trimmed}T00:00:00`);
    if (!Number.isNaN(dt.getTime())) {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      // Tag each column "Actual" once its period has reached/passed today,
      // else "Plan". Comparing the period boundary to `today` (rather than a
      // fixed this-week window) keeps month columns from being mis-tagged and
      // stops future days within the current week from showing as "Actual".
      let tag: "Actual" | "Plan";
      if (dateMode === "month") {
        const monthStart = new Date(today.getFullYear(), today.getMonth(), 1);
        tag = dt <= monthStart ? "Actual" : "Plan";
        return `${formatMonthLabel(dt)} ${tag}`;
      }
      // day / week grain: the column id is the day (day grain) or the week's
      // Monday (week grain); either is "Actual" once it is on/before today.
      tag = dt <= today ? "Actual" : "Plan";
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
  // Hooks must run unconditionally and in the same order every render, so they
  // come before any early return (React Rules of Hooks). A conditional return
  // placed above these caused the hook count to change on empty<->data
  // transitions and crashed the table.
  const [showAll, setShowAll] = useState(false);
  // Styled, never-clipped cell tooltip. A single fixed-position element is
  // positioned from the hovered cell's rect (delegation), so it escapes the
  // table's overflow:auto wrapper that would clip a CSS pseudo-element.
  const [tip, setTip] = useState<{ text: string; x: number; y: number } | null>(null);
  const defaultMax = 500;
  const effectiveMax = maxRows === 0 ? undefined : maxRows ?? defaultMax;
  const visibleRows = !showAll && effectiveMax ? rows.slice(0, effectiveMax) : rows;
  const columns = useMemo(() => buildColumns(visibleRows), [visibleRows]);

  const onCellOver = useCallback((event: ReactMouseEvent<HTMLTableElement>) => {
    const cell = (event.target as HTMLElement)?.closest?.("td[data-tip]") as HTMLElement | null;
    const text = cell?.getAttribute("data-tip") || "";
    if (!cell || !text) {
      setTip((prev) => (prev ? null : prev));
      return;
    }
    const rect = cell.getBoundingClientRect();
    setTip({ text, x: rect.left + rect.width / 2, y: rect.top });
  }, []);

  const clearTip = useCallback(() => setTip(null), []);

  if (!rows.length) {
    return <div className="forecast-muted">{emptyLabel}</div>;
  }

  const rawNumberCols = new Set<string>();
  columns.forEach((col) => {
    if (col.trim().toLowerCase() === "year") rawNumberCols.add(col);
  });

  return (
    <div className={`table-wrap ${className ?? ""}`.trim()} ref={wrapperRef}>
      <table className="table" onMouseOver={onCellOver} onMouseLeave={clearTip}>
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
                <td key={col} data-col={col} data-tip={(row as any)?.__tooltips?.[col] ?? ""}>
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
      {tip ? (
        <div
          className="app-tooltip app-tooltip--cell"
          role="tooltip"
          style={{ left: tip.x, top: tip.y }}
        >
          {tip.text}
        </div>
      ) : null}
    </div>
  );
}
