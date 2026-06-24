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
  // Smooth, flowing curve (Catmull-Rom -> cubic bezier) instead of a jagged
  // straight-segment polyline.
  const smoothLine = (() => {
    if (coords.length < 3) {
      return coords.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
    }
    const t = 0.18; // tension: lower = tighter to the points
    let d = `M ${coords[0][0].toFixed(1)} ${coords[0][1].toFixed(1)}`;
    for (let i = 0; i < coords.length - 1; i++) {
      const p0 = coords[i - 1] ?? coords[i];
      const p1 = coords[i];
      const p2 = coords[i + 1];
      const p3 = coords[i + 2] ?? p2;
      const c1x = p1[0] + (p2[0] - p0[0]) * t;
      const c1y = p1[1] + (p2[1] - p0[1]) * t;
      const c2x = p2[0] - (p3[0] - p1[0]) * t;
      const c2y = p2[1] - (p3[1] - p1[1]) * t;
      d += ` C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
    }
    return d;
  })();
  const line = smoothLine;
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
