"use client";

type SparklineProps = {
  points: number[];
  color?: string;
  fill?: string;
  width?: number;
  height?: number;
  strokeWidth?: number;
};

// Compact inline trend line for KPI cards. Pure SVG, no deps.
export default function Sparkline({
  points,
  color = "#2563eb",
  fill,
  width = 132,
  height = 40,
  strokeWidth = 2
}: SparklineProps) {
  const vals = (points ?? []).filter((v) => Number.isFinite(v));
  if (vals.length < 2) {
    return <svg width={width} height={height} aria-hidden="true" />;
  }
  let min = vals[0];
  let max = vals[0];
  for (const v of vals) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const pad = strokeWidth + 1;
  const innerH = height - pad * 2;
  const stepX = width / (vals.length - 1);
  const coords = vals.map((v, i) => {
    const x = i * stepX;
    const y = pad + innerH - ((v - min) / range) * innerH;
    return [x, y] as const;
  });
  const line = coords.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const area = `${line} L ${width} ${height} L 0 ${height} Z`;
  const [lastX, lastY] = coords[coords.length - 1];
  const fillColor = fill ?? color;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="trend">
      <path d={area} fill={fillColor} opacity={0.12} stroke="none" />
      <path d={line} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={lastX} cy={lastY} r={strokeWidth + 0.5} fill={color} />
    </svg>
  );
}
