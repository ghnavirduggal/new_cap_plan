"use client";

import { useMemo } from "react";

type PieChartData = { labels: string[]; values: number[] };

type PieChartProps = {
  data?: PieChartData | null;
  height?: number;
  className?: string;
  valueFormatter?: (value: number, label?: string) => string;
};

const COLORS = ["#2563eb", "#f97316", "#16a34a", "#ef4444", "#7c3aed", "#0ea5e9", "#0f766e", "#e11d48"];

function formatValue(value: number) {
  if (!Number.isFinite(value)) return "0";
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function polarToCartesian(cx: number, cy: number, r: number, angle: number) {
  const rad = ((angle - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const start = polarToCartesian(cx, cy, r, endAngle);
  const end = polarToCartesian(cx, cy, r, startAngle);
  const largeArc = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${cx} ${cy} L ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y} Z`;
}

export default function PieChart({ data, height = 260, className, valueFormatter }: PieChartProps) {
  const chart = useMemo(() => {
    if (!data || !data.labels?.length || !data.values?.length) return null;
    const total = data.values.reduce((sum, v) => sum + (Number.isFinite(v) ? v : 0), 0);
    if (total <= 0) return null;
    return { total };
  }, [data]);

  if (!data || !chart) {
    return <div className="forecast-chart-placeholder" />;
  }

  const width = 340;
  const radius = 90;
  const cx = width / 2;
  const cy = height / 2 - 10;
  let runningAngle = 0;
  const fmt = valueFormatter ?? ((value: number) => formatValue(value));

  return (
    <div className={`forecast-chart ${className ?? ""}`.trim()}>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
        {data.values.map((val, idx) => {
          const safeVal = Number.isFinite(val) ? val : 0;
          const slice = (safeVal / chart.total) * 360;
          const percent = chart.total > 0 ? (safeVal / chart.total) * 100 : 0;
          const startAngle = runningAngle;
          const endAngle = runningAngle + slice;
          runningAngle = endAngle;
          if (slice <= 0) return null;
          return (
            <path
              key={`${data.labels[idx]}-${idx}`}
              d={describeArc(cx, cy, radius, startAngle, endAngle)}
              fill={COLORS[idx % COLORS.length]}
            >
              <title>{`${data.labels[idx]}\nValue: ${fmt(safeVal, data.labels[idx])}\nShare: ${percent.toFixed(1)}%`}</title>
            </path>
          );
        })}
        <circle cx={cx} cy={cy} r={45} fill="#fff" />
      </svg>
      <div className="chart-legend">
        {data.labels.map((label, idx) => (
          <span key={label} className="chart-legend-item">
            <span className="chart-legend-dot" style={{ background: COLORS[idx % COLORS.length] }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
