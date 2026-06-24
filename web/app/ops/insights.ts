// Client-side "intelligence" layer for the Ops dashboard. Turns the raw
// summary payload (KPIs + time series + mix) into ranked, plain-language
// insights, trend deltas, and an overall capacity-health score — so the page
// interprets the numbers instead of just printing them.

export type Severity = "critical" | "warning" | "good" | "info";

export type Insight = {
  id: string;
  severity: Severity;
  icon: string;
  title: string;
  detail: string;
};

export type Trend = {
  /** % change of the second half of the window vs the first half. */
  pct: number;
  direction: "up" | "down" | "flat";
  spark: number[];
};

export type HealthScore = {
  score: number;
  label: string;
  tone: Severity;
};

type Point = { x: string; y: number };

const SEVERITY_WEIGHT: Record<Severity, number> = { critical: 0, warning: 1, good: 2, info: 3 };

function num(value: unknown): number {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : 0;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function sum(nums: number[]): number {
  return nums.reduce((acc, n) => acc + (Number.isFinite(n) ? n : 0), 0);
}

function mean(nums: number[]): number {
  return nums.length ? sum(nums) / nums.length : 0;
}

function fmt(value: number, digits = 0): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function seriesPoints(line: any, match: (name: string) => boolean): Point[] {
  const series: any[] = line?.series ?? [];
  const found = series.find((s) => match(String(s?.name ?? "").toLowerCase()));
  return (found?.points ?? []).map((p: any) => ({ x: String(p?.x ?? ""), y: num(p?.y) }));
}

function requiredPoints(line: any): Point[] {
  return seriesPoints(line, (n) => n.includes("requ") || n.includes("demand") || n.includes("req"));
}

function supplyPoints(line: any): Point[] {
  return seriesPoints(line, (n) => n.includes("suppl") || n.includes("staff") || n.includes("sup"));
}

/** Trend = mean(second half) vs mean(first half), plus a downsampled sparkline. */
export function trendOf(points: Point[]): Trend {
  const ys = points.map((p) => p.y);
  const spark = downsample(ys, 24);
  if (ys.length < 4) return { pct: 0, direction: "flat", spark };
  const mid = Math.floor(ys.length / 2);
  const first = mean(ys.slice(0, mid));
  const last = mean(ys.slice(mid));
  const pct = first === 0 ? (last > 0 ? 100 : 0) : ((last - first) / Math.abs(first)) * 100;
  const direction = Math.abs(pct) < 2 ? "flat" : pct > 0 ? "up" : "down";
  return { pct, direction, spark };
}

function downsample(values: number[], target: number): number[] {
  if (values.length <= target) return values;
  const step = values.length / target;
  const out: number[] = [];
  for (let i = 0; i < target; i++) out.push(values[Math.floor(i * step)] ?? 0);
  return out;
}

const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

function weekdayOf(dateStr: string): string | null {
  const d = new Date(dateStr);
  return Number.isNaN(d.getTime()) ? null : WEEKDAYS[d.getDay()];
}

export type KpiTrends = { required: Trend; supply: Trend; gap: Trend };

export function kpiTrends(summary: any): KpiTrends {
  const req = requiredPoints(summary?.line);
  const sup = supplyPoints(summary?.line);
  const gapPts: Point[] = req.map((p, i) => ({ x: p.x, y: p.y - (sup[i]?.y ?? 0) }));
  return { required: trendOf(req), supply: trendOf(sup), gap: trendOf(gapPts) };
}

/** Blend coverage, gap trend, and volatility into a 0-100 capacity-health score. */
export function capacityHealth(summary: any): HealthScore {
  const kpis = summary?.kpis ?? {};
  const required = num(kpis.required_fte);
  const supply = num(kpis.supply_fte);
  const coverage = required > 0 ? (supply / required) * 100 : supply > 0 ? 120 : 100;

  // Coverage: best at 100; understaffing hurts more than light overstaffing.
  const coverageScore =
    coverage >= 100 ? clamp(100 - (coverage - 100) * 0.6, 55, 100) : clamp(coverage, 0, 100);

  // Trend: a widening gap drags the score down.
  const trends = kpiTrends(summary);
  const gapTrend = trends.gap.pct;
  const trendScore = clamp(75 - gapTrend * 1.2, 0, 100);

  // Volatility: large swings in the gap relative to required demand are risky.
  const gapVals = trends.gap.spark;
  const swing = gapVals.length ? Math.max(...gapVals) - Math.min(...gapVals) : 0;
  const volScore = required > 0 ? clamp(100 - (swing / required) * 120, 0, 100) : 70;

  const score = Math.round(coverageScore * 0.6 + trendScore * 0.2 + volScore * 0.2);
  let label = "Critical";
  let tone: Severity = "critical";
  if (score >= 85) {
    label = "Healthy";
    tone = "good";
  } else if (score >= 70) {
    label = "Watch";
    tone = "info";
  } else if (score >= 50) {
    label = "At Risk";
    tone = "warning";
  }
  return { score: clamp(score, 0, 100), label, tone };
}

/** Produce ranked, plain-language insights from the loaded summary. */
export function deriveInsights(summary: any): Insight[] {
  if (!summary) return [];
  const out: Insight[] = [];
  const kpis = summary.kpis ?? {};
  const ins = summary.insights ?? {};
  const required = num(kpis.required_fte);
  const supply = num(kpis.supply_fte);
  const gap = num(kpis.gap_fte);
  const coverage = num(ins.coverage_pct) || (required > 0 ? (supply / required) * 100 : 0);
  const trends = kpiTrends(summary);

  // 1) Coverage / net position.
  if (required > 0) {
    if (coverage < 85) {
      out.push({
        id: "coverage",
        severity: "critical",
        icon: "🛑",
        title: `Coverage at ${fmt(coverage, 0)}% — understaffed`,
        detail: `Supply meets only ${fmt(coverage, 0)}% of required demand, a shortfall of ${fmt(Math.abs(gap), 0)} FTE across the selected scope.`
      });
    } else if (coverage < 98) {
      out.push({
        id: "coverage",
        severity: "warning",
        icon: "⚠️",
        title: `Coverage at ${fmt(coverage, 0)}% — slightly short`,
        detail: `A gap of ${fmt(Math.abs(gap), 0)} FTE remains. Small targeted hiring or borrowing would close it.`
      });
    } else if (coverage <= 110) {
      out.push({
        id: "coverage",
        severity: "good",
        icon: "✅",
        title: `Coverage at ${fmt(coverage, 0)}% — well matched`,
        detail: `Supply and demand are balanced for the selected scope (${fmt(Math.abs(gap), 0)} FTE ${gap <= 0 ? "buffer" : "gap"}).`
      });
    } else {
      out.push({
        id: "coverage",
        severity: "warning",
        icon: "💸",
        title: `Coverage at ${fmt(coverage, 0)}% — overstaffed`,
        detail: `Supply exceeds demand by ${fmt(Math.abs(gap), 0)} FTE. Consider lending capacity to short scopes before hiring elsewhere.`
      });
    }
  }

  // 2) Gap trend over the window.
  if (Math.abs(trends.gap.pct) >= 5 && required > 0) {
    const widening = trends.gap.pct > 0;
    out.push({
      id: "gap-trend",
      severity: widening ? "warning" : "good",
      icon: widening ? "📈" : "📉",
      title: `Staffing gap ${widening ? "widening" : "narrowing"} ${fmt(Math.abs(trends.gap.pct), 0)}%`,
      detail: widening
        ? "The gap between required and supply grew over the period — the trend is moving the wrong way."
        : "Supply is catching up to demand over the period — the gap is closing."
    });
  }

  // 3) Demand trend.
  if (Math.abs(trends.required.pct) >= 8) {
    const rising = trends.required.pct > 0;
    out.push({
      id: "demand-trend",
      severity: rising ? "warning" : "info",
      icon: rising ? "🔺" : "🔻",
      title: `Required FTE trending ${rising ? "up" : "down"} ${fmt(Math.abs(trends.required.pct), 0)}%`,
      detail: rising
        ? "Demand is rising through the window; plan capacity for the back end of the period, not the average."
        : "Demand is easing; watch for over-provisioning if supply stays flat."
    });
  }

  // 4) Shortfall concentration + day-of-week pattern.
  const shortfalls: any[] = ins.top_shortfalls ?? [];
  if (shortfalls.length) {
    const totalShort = sum(shortfalls.map((s) => Math.max(0, num(s.gap_fte))));
    const worst = shortfalls[0];
    if (totalShort > 0 && worst) {
      const worstShare = (Math.max(0, num(worst.gap_fte)) / totalShort) * 100;
      out.push({
        id: "shortfall-peak",
        severity: "warning",
        icon: "🎯",
        title: `Worst shortfall: ${worst.date} (${fmt(num(worst.gap_fte), 0)} FTE)`,
        detail: `The single worst period carries ${fmt(worstShare, 0)}% of the listed shortfall — a targeted fix there has outsized impact.`
      });
    }
    const dows = shortfalls.map((s) => weekdayOf(String(s.date))).filter(Boolean) as string[];
    const counts: Record<string, number> = {};
    dows.forEach((d) => (counts[d] = (counts[d] ?? 0) + 1));
    const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    if (dominant && dominant[1] >= 2 && dominant[1] >= Math.ceil(dows.length / 2)) {
      out.push({
        id: "shortfall-dow",
        severity: "info",
        icon: "📅",
        title: `Shortfalls cluster on ${dominant[0]}s`,
        detail: `${dominant[1]} of the top shortfall periods fall on ${dominant[0]} — a recurring weekly staffing pattern worth scheduling around.`
      });
    }
  }

  // 5) Channel concentration.
  const labels: string[] = summary.pie?.labels ?? [];
  const values: number[] = (summary.pie?.values ?? []).map(num);
  const totalMix = sum(values);
  if (totalMix > 0 && labels.length) {
    let topIdx = 0;
    values.forEach((v, i) => {
      if (v > values[topIdx]) topIdx = i;
    });
    const share = (values[topIdx] / totalMix) * 100;
    if (share >= 60) {
      out.push({
        id: "channel-concentration",
        severity: "info",
        icon: "🧩",
        title: `${labels[topIdx]} drives ${fmt(share, 0)}% of workload`,
        detail: `Workload is concentrated in ${labels[topIdx]}. A shock to that channel has the largest staffing impact — keep cross-skill cover.`
      });
    }
  }

  // 6) Volume peak vs average.
  const volDays: any[] = ins.top_volume_days ?? [];
  if (volDays.length >= 1) {
    const peak = num(volDays[0]?.volume);
    const avgVol = mean(volDays.map((d) => num(d.volume)));
    if (avgVol > 0 && peak / avgVol >= 1.4) {
      out.push({
        id: "volume-peak",
        severity: "info",
        icon: "⚡",
        title: `Peak volume is ${(peak / avgVol).toFixed(1)}× the busy-day average`,
        detail: `${volDays[0]?.date} spiked to ${fmt(peak, 0)}. Intraday and peak-day staffing matter more than the average here.`
      });
    }
  }

  // 7) Cross-skill rebalancing opportunity.
  const org = summary.workforce?.org_hiring ?? {};
  const saved = num(org.potential_hiring_saved_fte);
  if (saved >= 1) {
    out.push({
      id: "rebalance",
      severity: "good",
      icon: "🔁",
      title: `Cross-skill lending could save ${fmt(saved, 0)} FTE of hiring`,
      detail: `Borrowing across BA/SBA covers part of the shortfall before any external hiring — net hire is ${fmt(num(org.net_hiring_fte), 0)} FTE after rebalancing.`
    });
  }

  // 8) Understaffed scope count.
  const balance: any[] = summary.workforce?.scope_balance ?? [];
  if (balance.length) {
    const short = balance.filter((r) => {
      const req = num(r.required_fte ?? r.required ?? r.req);
      const sup = num(r.supply_fte ?? r.supply ?? r.sup);
      return req - sup > 0.5;
    });
    if (short.length) {
      out.push({
        id: "scope-short",
        severity: short.length >= Math.ceil(balance.length / 2) ? "warning" : "info",
        icon: "🏢",
        title: `${short.length} of ${balance.length} scopes are understaffed`,
        detail: `Shortfall is spread across ${short.length} BA/SBA/site scopes — rebalancing between them may beat hiring in each.`
      });
    }
  }

  return out.sort((a, b) => SEVERITY_WEIGHT[a.severity] - SEVERITY_WEIGHT[b.severity]);
}
