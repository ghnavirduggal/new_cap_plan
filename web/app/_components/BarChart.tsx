"use client";

import { useMemo } from "react";

type BarSeries = { name: string; values: number[]; color?: string };
type BarChartData = { labels: string[]; series: BarSeries[] };

type BarChartProps = {
  data?: BarChartData | null;
  height?: number;
  stacked?: boolean;
  className?: string;
  valueFormatter?: (value: number, label?: string, series?: string) => string;
};

const COLORS = ["#2563eb", "#f97316", "#16a34a", "#ef4444", "#7c3aed", "#0ea5e9"];

function formatValue(value: number) {
  if (!Number.isFinite(value)) return "0";
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function maxStacked(labels: string[], series: BarSeries[]) {
  let maxVal = 0;
  labels.forEach((_label, idx) => {
    const total = series.reduce((sum, s) => sum + (s.values[idx] ?? 0), 0);
    if (total > maxVal) maxVal = total;
  });
  return maxVal;
}

export default function BarChart({ data, height = 260, stacked = false, className, valueFormatter }: BarChartProps) {
  const chart = useMemo(() => {
    if (!data || !data.labels?.length || !data.series?.length) return null;
    const maxVal = stacked
      ? maxStacked(data.labels, data.series)
      : Math.max(...data.series.flatMap((s) => s.values), 0);
    return { maxVal: maxVal || 1 };
  }, [data, stacked]);

  if (!data || !data.labels?.length || !data.series?.length || !chart) {
    return <div className="forecast-chart-placeholder" />;
  }

  const width = 760;
  const padding = 36;
  const step = data.labels.length > 0 ? (width - padding * 2) / data.labels.length : 1;
  const barWidth = Math.max(8, step * (stacked ? 0.7 : 0.8));
  const labelSkip = Math.max(1, Math.ceil(data.labels.length / 12));
  const totals = stacked
    ? data.labels.map((_label, idx) => data.series.reduce((sum, s) => sum + (s.values[idx] ?? 0), 0))
    : [];
  const fmt = valueFormatter ?? ((value: number) => formatValue(value));

  return (
    <div className={`forecast-chart ${className ?? ""}`.trim()}>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#e2e8f0" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#e2e8f0" />
        {data.labels.map((label, idx) =>
          idx % labelSkip === 0 ? (
            <text
              key={`${label}-${idx}`}
              x={padding + idx * step + step / 2}
              y={height - 10}
              fontSize="10"
              fill="#64748b"
              textAnchor="middle"
            >
              {label}
            </text>
          ) : null
        )}
        {data.labels.map((label, idx) => {
          let running = 0;
          return data.series.map((series, sIdx) => {
            const value = series.values[idx] ?? 0;
            if (!stacked && value === 0) return null;
            const color = series.color || COLORS[sIdx % COLORS.length];
            const barHeight = (value / chart.maxVal) * (height - padding * 2);
            const xBase = padding + idx * step + (step - barWidth) / 2;
            if (stacked) {
              const y = height - padding - (running + barHeight);
              running += barHeight;
              const title = `${label}\n${series.name}: ${fmt(value, label, series.name)}\nTotal: ${fmt(totals[idx] ?? 0, label, "Total")}`;
              return (
                <rect key={`${label}-${series.name}`} x={xBase} y={y} width={barWidth} height={barHeight} fill={color}>
                  <title>{title}</title>
                </rect>
              );
            }
            const seriesWidth = barWidth / data.series.length;
            const x = xBase + sIdx * seriesWidth;
            const y = height - padding - barHeight;
            const title = `${label}\n${series.name}: ${fmt(value, label, series.name)}`;
            return (
              <rect key={`${label}-${series.name}`} x={x} y={y} width={seriesWidth} height={barHeight} fill={color}>
                <title>{title}</title>
              </rect>
            );
          });
        })}
      </svg>
      <div className="chart-legend">
        {data.series.map((series, idx) => (
          <span key={series.name} className="chart-legend-item">
            <span className="chart-legend-dot" style={{ background: series.color || COLORS[idx % COLORS.length] }} />
            {series.name}
          </span>
        ))}
      </div>
    </div>
  );
}
