"use client";

import { useMemo, useState, type ReactNode } from "react";

type ChartPoint = { x: string; y: number | null };
type ChartSeries = { name: string; points: ChartPoint[]; color?: string };
type ChartData = { x: string[]; series: ChartSeries[] };

type LineChartProps = {
  data?: ChartData | null;
  height?: number;
  className?: string;
};

const COLORS = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed", "#0ea5e9"];

function formatValue(value: number) {
  if (!Number.isFinite(value)) return "0";
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function buildPath(points: ChartPoint[], xMap: Map<string, number>, height: number, padding: number, minY: number, maxY: number) {
  const span = maxY - minY || 1;
  let path = "";
  let started = false;
  points.forEach((pt) => {
    if (pt.y === null || pt.y === undefined || Number.isNaN(pt.y)) {
      started = false;
      return;
    }
    const x = xMap.get(pt.x);
    if (x === undefined) return;
    const y = padding + ((maxY - pt.y) / span) * (height - padding * 2);
    if (!started) {
      path += `M ${x} ${y}`;
      started = true;
    } else {
      path += ` L ${x} ${y}`;
    }
  });
  return path;
}

type Tip = { x: number; y: number; content: ReactNode } | null;

export default function LineChart({ data, height = 260, className }: LineChartProps) {
  const [tip, setTip] = useState<Tip>(null);
  const chart = useMemo(() => {
    if (!data || !data.series?.length) return null;
    const allVals = data.series.flatMap((series) =>
      series.points.map((pt) => (pt.y === null || pt.y === undefined ? null : pt.y))
    );
    const numericVals = allVals.filter((val): val is number => typeof val === "number" && Number.isFinite(val));
    const minY = numericVals.length ? Math.min(...numericVals) : 0;
    const maxY = numericVals.length ? Math.max(...numericVals) : 1;
    return { minY, maxY };
  }, [data]);

  if (!data || !data.series?.length || !chart) {
    return <div className="forecast-chart-placeholder" />;
  }

  const width = 760;
  const padding = 32;
  const xLabels = data.x?.length ? data.x : data.series[0].points.map((pt) => pt.x);
  const step = xLabels.length > 1 ? (width - padding * 2) / (xLabels.length - 1) : 1;
  const xMap = new Map<string, number>();
  xLabels.forEach((label, idx) => {
    xMap.set(label, padding + idx * step);
  });
  const labelSkip = Math.max(1, Math.ceil(xLabels.length / 8));
  const span = chart.maxY - chart.minY || 1;
  const toY = (val: number) => padding + ((chart.maxY - val) / span) * (height - padding * 2);

  return (
    <div className={`forecast-chart ${className ?? ""}`.trim()} style={{ position: "relative" }} onMouseLeave={() => setTip(null)}>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#e2e8f0" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#e2e8f0" />
        {xLabels.map((label, idx) =>
          idx % labelSkip === 0 ? (
            <text key={label} x={padding + idx * step} y={height - 8} fontSize="13" fill="#475569" textAnchor="middle">
              {label}
            </text>
          ) : null
        )}
        {data.series.map((series, idx) => {
          const color = series.color || COLORS[idx % COLORS.length];
          const path = buildPath(series.points, xMap, height, padding, chart.minY, chart.maxY);
          let lastVal: number | null = null;
          return (
            <g key={series.name}>
              <path d={path} fill="none" stroke={color} strokeWidth={2} />
              {series.points.map((pt, pIdx) => {
                if (pt.y === null || pt.y === undefined || Number.isNaN(pt.y)) {
                  lastVal = null;
                  return null;
                }
                const x = xMap.get(pt.x);
                if (x === undefined) {
                  lastVal = pt.y;
                  return null;
                }
                const y = toY(pt.y);
                const delta = lastVal === null ? null : pt.y - lastVal;
                lastVal = pt.y;
                const ptVal = pt.y;
                const ptX = pt.x;
                const content = (
                  <>
                    <div className="chart-tip__title">{ptX}</div>
                    <div className="chart-tip__row">
                      <span className="chart-tip__dot" style={{ background: color }} />
                      {series.name}: <strong>{formatValue(ptVal)}</strong>
                    </div>
                    {delta !== null ? (
                      <div className="chart-tip__delta">Δ {delta >= 0 ? "+" : ""}{formatValue(delta)}</div>
                    ) : null}
                  </>
                );
                const show = (e: React.MouseEvent) => setTip({ x: e.clientX, y: e.clientY, content });
                return (
                  <g key={`${series.name}-${pIdx}`}>
                    <circle cx={x} cy={y} r={3} fill={color} className="chart-dot" />
                    <circle
                      cx={x}
                      cy={y}
                      r={12}
                      fill="transparent"
                      stroke="transparent"
                      pointerEvents="all"
                      onMouseEnter={show}
                      onMouseMove={show}
                    />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      {tip ? (
        <div className="chart-tip" style={{ position: "fixed", left: tip.x + 14, top: tip.y + 14 }}>
          {tip.content}
        </div>
      ) : null}
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
