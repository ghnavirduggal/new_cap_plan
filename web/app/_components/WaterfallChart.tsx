"use client";

import { useMemo } from "react";

type WaterfallData = {
  labels: string[];
  values: number[];
  measure?: Array<"relative" | "total">;
};

type WaterfallChartProps = {
  data?: WaterfallData | null;
  height?: number;
  className?: string;
};

const COLORS = ["#14b8a6", "#2563eb", "#ef4444", "#7c3aed"];

function formatValue(value: number) {
  if (!Number.isFinite(value)) return "0";
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export default function WaterfallChart({ data, height = 260, className }: WaterfallChartProps) {
  const chart = useMemo(() => {
    if (!data || !data.labels?.length || !data.values?.length) return null;
    const bars: Array<{ label: string; start: number; end: number; value: number }> = [];
    let running = 0;
    data.values.forEach((value, idx) => {
      const measure = data.measure?.[idx] ?? "relative";
      if (measure === "total") {
        const end = value !== 0 ? value : running;
        bars.push({ label: data.labels[idx], start: 0, end, value: end });
        running = end;
      } else {
        const start = running;
        const end = running + value;
        bars.push({ label: data.labels[idx], start, end, value });
        running = end;
      }
    });
    const minVal = Math.min(0, ...bars.map((b) => Math.min(b.start, b.end)));
    const maxVal = Math.max(0, ...bars.map((b) => Math.max(b.start, b.end)));
    return { bars, minVal, maxVal: maxVal || 1 };
  }, [data]);

  if (!data || !chart) {
    return <div className="forecast-chart-placeholder" />;
  }

  const width = 760;
  const padding = 36;
  const step = data.labels.length > 0 ? (width - padding * 2) / data.labels.length : 1;
  const barWidth = Math.max(18, step * 0.5);
  const range = chart.maxVal - chart.minVal || 1;

  const toY = (val: number) => padding + ((chart.maxVal - val) / range) * (height - padding * 2);
  const baselineY = toY(0);
  const labelSkip = Math.max(1, Math.ceil(data.labels.length / 10));

  return (
    <div className={`forecast-chart ${className ?? ""}`.trim()}>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
        <line x1={padding} y1={baselineY} x2={width - padding} y2={baselineY} stroke="#e2e8f0" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#e2e8f0" />
        {chart.bars.map((bar, idx) => {
          const top = Math.min(bar.start, bar.end);
          const bottom = Math.max(bar.start, bar.end);
          const y = toY(bottom);
          const barHeight = Math.abs(toY(top) - toY(bottom));
          const x = padding + idx * step + (step - barWidth) / 2;
          const title = `${bar.label}\nValue: ${formatValue(bar.value)}\nStart: ${formatValue(bar.start)}\nEnd: ${formatValue(bar.end)}`;
          return (
            <rect
              key={`${bar.label}-${idx}`}
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              fill={COLORS[idx % COLORS.length]}
            >
              <title>{title}</title>
            </rect>
          );
        })}
        {data.labels.map((label, idx) =>
          idx % labelSkip === 0 ? (
            <text key={label} x={padding + idx * step + step / 2} y={height - 10} fontSize="10" fill="#64748b" textAnchor="middle">
              {label}
            </text>
          ) : null
        )}
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
