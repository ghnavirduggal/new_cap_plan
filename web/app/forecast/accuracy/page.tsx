"use client";

import { useEffect, useMemo, useState } from "react";
import AppShell from "../../_components/AppShell";
import Icon from "../../_components/Icon";
import LineChart from "../../_components/LineChart";
import { apiGet } from "../../../lib/api";

type ModelRow = {
  model: string;
  metrics: Record<string, number>;
  primary_metric?: string;
  primary_value?: number;
  rank?: number;
  is_best?: boolean;
};

type Leaderboard = {
  scope: string;
  latest?: { ts?: string; run_label?: string; actor?: string } | null;
  models: ModelRow[];
  best_model?: string;
  primary_metric?: string;
  trend?: Array<Record<string, any>>;
};

type AccuracyResponse = {
  scopes: string[];
  leaderboard: Leaderboard;
  history: Array<Record<string, any>>;
};

const METRIC_COLUMNS: Array<{ key: string; label: string; lowerBetter?: boolean }> = [
  { key: "acc5", label: "Acc ±5%" },
  { key: "acc7", label: "Acc ±7%" },
  { key: "acc10", label: "Acc ±10%" },
  { key: "mape", label: "MAPE %", lowerBetter: true },
  { key: "wape", label: "WAPE %", lowerBetter: true },
  { key: "bias", label: "Bias %", lowerBetter: true }
];

const METRIC_LABELS: Record<string, string> = {
  acc5: "Accuracy within ±5%",
  acc7: "Accuracy within ±7%",
  acc10: "Accuracy within ±10%",
  mape: "MAPE (lower is better)",
  wape: "WAPE (lower is better)",
  bias: "Bias (closer to 0 is better)"
};

const LINE_COLORS = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed", "#0ea5e9"];

function fmt(value: number | undefined | null): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 });
}

export default function ForecastAccuracyPage() {
  const [scope, setScope] = useState<string>("global");
  const [data, setData] = useState<AccuracyResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    apiGet<AccuracyResponse>(`/api/forecast/accuracy?scope=${encodeURIComponent(scope)}`)
      .then((res) => {
        if (!active) return;
        setData(res);
        setError("");
        // If we asked for "global" but only other scopes exist, snap to the first.
        if (
          scope === "global" &&
          (!res.leaderboard?.models?.length) &&
          Array.isArray(res.scopes) &&
          res.scopes.length &&
          !res.scopes.includes("global")
        ) {
          setScope(res.scopes[0]);
        }
      })
      .catch((err) => {
        if (!active) return;
        setError(err?.message || "Could not load accuracy.");
        setData(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [scope]);

  const board = data?.leaderboard;
  const models = board?.models ?? [];
  const primaryMetric = board?.primary_metric || "";

  // Which metric columns actually have data, so we don't show empty columns.
  const activeMetrics = useMemo(() => {
    return METRIC_COLUMNS.filter((col) => models.some((m) => Number.isFinite(m.metrics?.[col.key])));
  }, [models]);

  // Trend chart: primary metric per model across snapshots.
  const trendChart = useMemo(() => {
    const trend = board?.trend ?? [];
    if (trend.length < 2 || !models.length) return null;
    const x = trend.map((p, i) => String(p.run_label || (p.ts ? String(p.ts).slice(0, 10) : `Run ${i + 1}`)));
    const series = models.slice(0, 6).map((m, idx) => ({
      name: m.model,
      color: LINE_COLORS[idx % LINE_COLORS.length],
      points: trend.map((p, i) => ({
        x: x[i],
        y: Number.isFinite(Number(p[m.model])) ? Number(p[m.model]) : null
      }))
    }));
    return { x, series };
  }, [board, models]);

  const scopeOptions = useMemo(() => {
    const set = new Set<string>(["global", ...(data?.scopes ?? [])]);
    if (scope) set.add(scope);
    return Array.from(set);
  }, [data, scope]);

  return (
    <AppShell title="Forecast Accuracy" crumbs="Forecasting / Forecast Accuracy" crumbIcon={<Icon name="target" size={16} />}>
      <div className="accuracy-page">
        <div className="accuracy-head">
          <div>
            <h2>Forecast Accuracy</h2>
            <p>
              Which model is winning, and whether accuracy is improving over time. Snapshots are recorded each time a
              forecast result is saved.
            </p>
          </div>
          <label className="accuracy-scope">
            Scope
            <select className="input" value={scope} onChange={(e) => setScope(e.target.value)}>
              {scopeOptions.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
        </div>

        {loading ? (
          <div className="accuracy-empty">Loading accuracy…</div>
        ) : error ? (
          <div className="accuracy-empty accuracy-empty--error">{error}</div>
        ) : !models.length ? (
          <div className="accuracy-empty">
            No accuracy recorded for this scope yet. Run Phase 1 in the forecasting workspace and save the results —
            the model accuracy will be tracked here.
          </div>
        ) : (
          <>
            <div className="accuracy-winner">
              <span className="accuracy-winner__badge"><Icon name="trophy" size={14} /> Best model</span>
              <span className="accuracy-winner__name">{board?.best_model}</span>
              {primaryMetric ? (
                <span className="accuracy-winner__metric">
                  ranked by {METRIC_LABELS[primaryMetric] || primaryMetric}
                </span>
              ) : null}
              {board?.latest?.ts ? (
                <span className="accuracy-winner__ts">as of {String(board.latest.ts).slice(0, 10)}</span>
              ) : null}
            </div>

            <div className="accuracy-card">
              <h3>Model leaderboard</h3>
              <div className="table-wrap">
                <table className="table accuracy-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Model</th>
                      {activeMetrics.map((col) => (
                        <th key={col.key} className="num">
                          {col.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((m) => (
                      <tr key={m.model} className={m.is_best ? "accuracy-row--best" : ""}>
                        <td>{m.rank}</td>
                        <td>
                          {m.is_best ? <Icon name="trophy" size={13} style={{ marginRight: 4, verticalAlign: "-1px" }} /> : null}
                          {m.model}
                        </td>
                        {activeMetrics.map((col) => (
                          <td key={col.key} className="num">
                            {fmt(m.metrics?.[col.key])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {trendChart ? (
              <div className="accuracy-card">
                <h3>Accuracy trend ({METRIC_LABELS[primaryMetric] || primaryMetric})</h3>
                <LineChart data={trendChart} height={280} />
              </div>
            ) : (
              <div className="accuracy-card accuracy-card--muted">
                Save at least two forecast runs for this scope to see an accuracy trend.
              </div>
            )}
          </>
        )}
      </div>
    </AppShell>
  );
}
