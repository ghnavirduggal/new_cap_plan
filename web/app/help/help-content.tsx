"use client";

import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";

type Section = {
  id: string;
  title: string;
  body?: string;
  bullets?: string[];
  media?: boolean;
};

const tocItems = [
  { id: "end-to-end", label: "End-to-End Flow (Start with Forecasting)" },
  { id: "forecasting-workflow", label: "Forecasting Workflow (Start Here)" },
  { id: "forecasting-inputs", label: "Forecasting Inputs" },
  { id: "transformation-projects", label: "Transformation Projects" },
  { id: "daily-interval", label: "Daily and Interval Forecast" },
  { id: "capacity-overview", label: "Capacity Planning Overview" },
  { id: "calculations", label: "Calculations Reference" },
  { id: "data-flow", label: "Data Flow (Uploads → Pages)" },
  { id: "scheduling-staffing", label: "Scheduling and Staffing" },
  { id: "dataset-ops", label: "Dataset and Ops Views" },
  { id: "budget-shrink", label: "Budget and Shrink" },
  { id: "settings-effective", label: "Settings and Effective-Dated Config" },
  { id: "uploads-storage", label: "Uploads and Storage" },
  { id: "templates-normalizers", label: "Templates and Normalizers" },
  { id: "roles-permissions", label: "Roles and Permissions" },
  { id: "duplicate-uploads", label: "Duplicate Uploads" },
  { id: "quickstart", label: "Quickstart Video and Screenshots" },
  { id: "troubleshooting", label: "Troubleshooting" }
];

const sections: Section[] = [
  {
    id: "end-to-end",
    title: "End-to-End Flow (Start with Forecasting)",
    body: "CAP CONNECT is designed to move from forecast to plan to schedule.",
    bullets: [
      "Forecasting Workspace: build monthly forecasts and select best models.",
      "Transformation Projects: apply sequential adjustments and publish final forecast.",
      "Daily and Interval Forecast: split monthly totals into daily and interval targets.",
      "Capacity Planning: create plans, validate staffing vs demand, and track plan history.",
      "Scheduling and Staffing: manage rosters and hiring inputs for supply FTE."
    ]
  },
  {
    id: "forecasting-workflow",
    title: "Forecasting Workflow (Start Here)",
    body: "Use the forecasting workspace to move from monthly volume to actionable plans.",
    bullets: [
      "Volume Summary: upload data, review IQ/Volume summaries, and generate seasonality.",
      "Normalized Ratio 1: adjust caps and base volume, then apply changes.",
      "Prophet Smoothing: smooth the series and confirm Normalized Ratio 2.",
      "Phase 1: run multi-model forecasts and review accuracy.",
      "Forecast Accuracy: after you save forecast results, the Forecast Accuracy page tracks per-model accuracy (accuracy bands / MAPE / WAPE / bias) over time, ranks the models, and highlights the winning model per scope so you can see whether forecasts are improving.",
      "Phase 2: apply best configs and generate final forecast outputs.",
      "Transformation Projects: apply sequential adjustments and save final outputs.",
      "Daily and Interval Forecast: split monthly totals into daily and interval plans."
    ]
  },
  {
    id: "forecasting-inputs",
    title: "Forecasting Inputs",
    bullets: [
      "Volume upload: CSV/Excel with date and volume; optional category and forecast_group.",
      "IQ_Data sheet: required for IQ summary and contact ratio calculations.",
      "Holidays sheet: optional, used for Prophet regressors and seasonality.",
      "Interval history: date, interval, volume (AHT optional) for daily/interval split.",
      "Programmatic ingest (REST API): instead of manual uploads, automated systems can push volume/AHT actuals and forecasts to POST /api/ingest/v1/timeseries (X-API-Key auth). See docs/INGEST_API.md for the contract and a curl example."
    ]
  },
  {
    id: "transformation-projects",
    title: "Transformation Projects",
    body: "Applies sequential adjustments to the base forecast for a selected group, model, and year.",
    bullets: [
      "Fields: Transformation 1-3, IA 1-3, Marketing Campaign 1-3 (percent changes).",
      "Forecast columns are generated sequentially from Base_Forecast_for_Forecast_Group.",
      "Outputs: Final_Forecast_Post_Transformations and forecast tables.",
      "Writes forecast_dates.csv to drive Daily and Interval Forecast selection."
    ]
  },
  {
    id: "daily-interval",
    title: "Daily and Interval Forecast",
    bullets: [
      "Uses forecast_dates.csv, holidays_list.csv, and the final transformation output.",
      "Builds daily distribution from recent history and lets you edit the split.",
      "Generates daily totals and interval forecasts for the selected month.",
      "Saves outputs alongside transformation files for downstream planning."
    ]
  },
  {
    id: "capacity-overview",
    title: "Capacity Planning Overview",
    body: "Planning Workspace manages plans across business areas and tracks history.",
    bullets: [
      "Create or duplicate plans, then open Plan Detail.",
      "Plan Detail includes weekly tables, notes, and validation views.",
      "Hiring Plan Solver (Plan Detail options): recommends new-hire class start weeks and sizes to close the projected FTE shortfall, accounting for the training + nesting lead time (and optional weekly attrition erosion). Weeks that can't be covered in time are flagged; 'Apply as classes' writes the recommendation into the plan's new-hire classes for review.",
      "BA rollups provide summarized capacity views."
    ]
  },
  {
    id: "calculations",
    title: "Calculations Reference",
    body: "Every staffing number in CAP CONNECT comes from the formulas below. Intervals are 30 minutes and the basis is kept consistent across grains. Occupancy is honoured for both Voice and Back Office.",
    bullets: [
      "Voice (Erlang C): offered load A = calls × AHT ÷ interval_seconds (erlangs). Required agents are the smallest N meeting the Service Level target (X% answered in T seconds) AND the occupancy cap (busy time ÷ staffed time ≤ cap). Outputs per interval: agents, SL, occupancy, ASA, and PHC (calls a staffed line can handle).",
      "Chat (Erlang with concurrency): effective AHT = AHT ÷ concurrency (an agent handles ~N chats at once), then the same Erlang C solve. Required FTE, SL and PHC are all Erlang-based.",
      "Outbound (Erlang): expected contacts = OPC × connect_rate × RPC (or RPC rate), then Erlang C with the outbound SL target / occupancy. Required FTE and SL are Erlang-based (not linear).",
      "Back Office — Linear (TAT): FTE = items × SUT ÷ (hours_per_day × 3600 × (1 − shrink) × util_bo). Used when the Settings BO capacity model is 'tat'/'linear'.",
      "Back Office — Erlang: when the Settings BO capacity model is 'erlang', a daily Erlang solve gives agents, FTE, PHC and a true Erlang Service Level. Which method runs is driven entirely by the Settings value.",
      "Back Office Service Level: Erlang model → true Erlang SL at the supplied agent count; linear model → coverage proxy = min(100, supply ÷ required × 100).",
      "Interval → Daily rollup (true SUMPRODUCT): daily FTE = Σ(staff-seconds across the day) ÷ (hours_per_FTE × 3600 × (1 − shrink)). staff-seconds = agents × interval_seconds. This is the only roll-up method for the daily grain.",
      "Daily → Weekly / Monthly rollup: required FTE is a LEVEL, so the higher grain is the volume-weighted average of the daily levels (not a sum). PHC sums (throughput); Service Level is volume-weighted.",
      "Operating days: weekly/monthly conversions use the Business-Area operating days from Settings (e.g. bo_workdays_per_week); a month is 52 ÷ 12 ≈ 4.333 weeks.",
      "Shrinkage: overall shrink compounds its two parts — overall = 1 − (1 − OOO) × (1 − INO) — never a simple sum. Productive fraction = 1 − overall_shrink; all values are clamped to 0–100%.",
      "Attrition: annualized = (weekly leavers ÷ average active FTE) × 52, scoped to the plan's Business Area → Sub Business Area → Modality (Voice/Back Office/etc.) → Site.",
      "Learning curve (new hires): during Nesting and SDA, agents are partially productive (productivity %) and may handle slower (AHT uplift). Effective agents = headcount × productivity ÷ (1 + uplift). Nesting precedes production; SDA agents are already in supply, so only their shortfall vs a full head is applied.",
      "Backlog (Back Office, opt-in per plan): backlog = max(0, Actual − Forecast) per period, carried once into the next OPEN period (never stacked onto a completed period). Carried backlog is not amplified by the what-if volume dial.",
      "What-if dials: volume and AHT deltas are multiplicative (× (1 + delta/100)); shrink delta is additive percentage points (30% + 5 → 35%). Dials apply only to the active/future window, never to past/locked periods.",
      "Saved scenarios: the current What-If dials can be saved as a named, reusable case (e.g. Base, Downside, Peak season). Apply re-loads a scenario's dials as the live What-If and recomputes the plan; Compare computes a baseline (no dials) plus each saved scenario side by side (weekly-average of each upper-summary metric) without disturbing the live plan.",
      "FTE Over/Under = Projected Supply − Required @ scenario (positive = surplus, negative = shortfall). Reported vs MTP (Forecast), Tactical and Budgeted requirements.",
      "Budget vs Actual (BvA): Budgeted FTE uses budget volume/AHT (it does not move with what-if dials); Actual FTE uses the shrink-adjusted actual; Variance = Actual − Budgeted. (Note: BvA Variance is demand-vs-demand, whereas FTE Over/Under is supply-vs-demand — the two use opposite sign ordering by design.)",
      "Supply projection: future supply = prior supply − attrition + new-hire additions, floored at zero, applied forward in time; learning-curve ramp reduces the effective (capacity) supply, not the displayed headcount."
    ]
  },
  {
    id: "data-flow",
    title: "Data Flow (Uploads → Pages)",
    body: "Which upload feeds which calculation and page. Uploads are normalized and stored per scope (Business Area → Sub BA → Channel → Site), then read by the plan engine and dashboards.",
    bullets: [
      "Headcount upload → org directory & hierarchy → Profile/User, manager mapping, and the roster/attrition denominators.",
      "Voice forecast/actual/tactical (volume + AHT by date/interval) → Voice Erlang calcs → Plan Detail (interval/day/week/month), Ops, and the Capacity rollup.",
      "Back Office forecast/actual/tactical (items + SUT by date) → BO linear/Erlang calcs → Plan Detail, Ops, BvA, Capacity rollup.",
      "Chat & Outbound forecast/actual (items/contacts + AHT) → Chat/OB Erlang calcs → Plan Detail and Capacity rollup.",
      "Roster (WIDE/LONG) → normalized schedule → Supply FTE and Projected Supply HC.",
      "New Hire classes (training/nesting/production starts, grads, employee type) → learning-curve ramp & supply additions.",
      "Shrinkage & Attrition raw uploads → normalized weekly series → plan shrinkage/attrition, KPIs.",
      "Budget upload → Budgeted FTE/AHT → BvA tables and the Budgeted Over/Under.",
      "Settings (effective-dated) → SL targets, occupancy caps, shrinkage, operating days, capacity model, concurrency, pt_fte_ratio → every calculation above.",
      "Home page: KPI cards (Staffing Gap, Hiring, Shrinkage, Attrition, Service Level, Handling Capacity) and their embedded trend lines read each plan's upper summary table (Required, Supply, Over/Under, SL, PHC) rolled up across plans. The 'Top Drivers of Staffing Gap' insight ranks what is moving the gap (see Calculations) from the same plan data.",
      "Ops Dashboard: requirements-vs-supply, Voice and Back Office metrics, and trends are derived from the same per-plan calculations aggregated across the selected scope — no separate data source.",
      "BA rollup: child plan upper tables are summed/averaged (Over/Under and required by scenario) to a Business-Area capacity view."
    ]
  },
  {
    id: "scheduling-staffing",
    title: "Scheduling and Staffing",
    bullets: [
      "Roster page: download template, upload schedules, and preview normalized schedule.",
      "Bulk edits: clear ranges or mark leave/off patterns.",
      "New Hire page: manage class start dates, levels, and production starts.",
      "Roster and hiring feed supply FTE calculations."
    ]
  },
  {
    id: "dataset-ops",
    title: "Dataset and Ops Views",
    bullets: [
      "Dataset page: snapshot of inputs and scope filters for planners.",
      "Ops page: requirements vs supply, voice and back office metrics.",
      "Voice uses volume + AHT; back office uses items + SUT."
    ]
  },
  {
    id: "budget-shrink",
    title: "Budget and Shrink",
    bullets: [
      "Shrink and attrition uploads are normalized to weekly series.",
      "Budget inputs support planning scenarios and validation."
    ]
  },
  {
    id: "settings-effective",
    title: "Settings and Effective-Dated Config",
    body: "Settings store an Effective Week (Monday). Computations use the latest settings where effective_week <= target date.",
    bullets: [
      "Save today -> applies to this and future weeks.",
      "Change next week -> applies from next week onward; past weeks keep older settings."
    ]
  },
  {
    id: "uploads-storage",
    title: "Uploads and Storage",
    bullets: [
      "Headcount: upsert by BRID; provides hierarchy and manager mapping.",
      "Roster: WIDE/LONG; stored as roster_wide and roster_long (dedupe by BRID,date).",
      "Forecasts/Actuals: Voice (volume + AHT by date/interval), BO (items + SUT by date).",
      "Shrinkage and Attrition: raw -> weekly series for KPIs.",
      "Forecast outputs: saved paths in latest_forecast_full_path.txt and latest_forecast_base_dir.txt."
    ]
  },
  {
    id: "templates-normalizers",
    title: "Templates and Normalizers",
    bullets: [
      "Headcount: BRID, Full Name, Line Manager BRID/Name, Journey, Level 3, Location, Group.",
      "Voice/BO: Date (+ Interval for Voice), Volume/Items, AHT/SUT.",
      "Shrinkage/Attrition: Raw -> normalized + weekly series."
    ]
  },
  {
    id: "roles-permissions",
    title: "Roles and Permissions",
    bullets: ["Admin/Planner can save settings; Admin can delete plans.", "Viewer is read-only."]
  },
  {
    id: "duplicate-uploads",
    title: "Duplicate Uploads",
    bullets: [
      "Headcount: upsert by BRID (last wins).",
      "Timeseries: append by date/week; overlapping dates are replaced.",
      "Roster snapshots: overwrite; long dedupes by (BRID,date).",
      "Plan bulk roster: upsert per BRID within a plan.",
      "Shrinkage weekly: merged; Attrition weekly: overwrite."
    ]
  },
  { id: "quickstart", title: "Quickstart Video and Screenshots", media: true },
  {
    id: "troubleshooting",
    title: "Troubleshooting",
    bullets: [
      "If KPIs do not reflect uploads, ensure saves are complete and labels match.",
      "Effective settings: refresh and re-run for target dates to pick the correct version.",
      "Role errors on save: verify Admin/Planner permissions."
    ]
  }
];

function matches(section: Section, q: string): boolean {
  if (!q) return true;
  const hay = [section.title, section.body || "", ...(section.bullets || [])]
    .join(" ")
    .toLowerCase();
  return q
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((term) => hay.includes(term));
}

function smoothScrollTo(id: string) {
  const el = typeof document !== "undefined" ? document.getElementById(id) : null;
  if (el) {
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  } else if (typeof window !== "undefined") {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

export default function HelpContent() {
  const [query, setQuery] = useState("");
  const [showTop, setShowTop] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const onScroll = () => setShowTop(window.scrollY > 400);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Focus the search field when the magnifying-glass reveals the bar.
  useEffect(() => {
    if (!searchOpen) return;
    const t = window.setTimeout(() => inputRef.current?.focus(), 120);
    return () => window.clearTimeout(t);
  }, [searchOpen]);

  const visible = useMemo(() => sections.filter((s) => matches(s, query)), [query]);
  const visibleIds = useMemo(() => new Set(visible.map((s) => s.id)), [visible]);
  const visibleToc = useMemo(() => tocItems.filter((t) => visibleIds.has(t.id)), [visibleIds]);

  const onNav = (e: MouseEvent, id: string) => {
    e.preventDefault();
    smoothScrollTo(id);
  };

  return (
    <div id="top" className="help-root">
      <div className="help-header">
        <h1>Help &amp; Documentation</h1>
        <div className={`help-search-wrap ${searchOpen ? "open" : ""}`}>
          <button
            type="button"
            className="help-search-btn"
            aria-label={searchOpen ? "Hide search" : "Search help"}
            aria-expanded={searchOpen}
            onClick={() =>
              setSearchOpen((prev) => {
                const next = !prev;
                if (!next) setQuery("");
                return next;
              })
            }
          >
            🔍
          </button>
          <div className="help-search-panel">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search help… (e.g. Erlang, backlog, shrinkage)"
              aria-label="Search help"
            />
          </div>
        </div>
      </div>

      <section className="help-card">
        <h2>Table of Contents</h2>
        {visibleToc.length ? (
          <ul className="help-toc">
            {visibleToc.map((item) => (
              <li key={item.id}>
                <a href={`#${item.id}`} onClick={(e) => onNav(e, item.id)}>
                  {item.label}
                </a>
              </li>
            ))}
          </ul>
        ) : (
          <p>No topics match “{query}”.</p>
        )}
      </section>

      {visible.map((section) => (
        <section key={section.id} id={section.id} className="help-card">
          <h2>{section.title}</h2>
          {section.body ? <p>{section.body}</p> : null}
          {section.media ? (
            <div className="help-media">
              <video controls width="100%">
                <source src="/assets/help/quickstart.mp4" type="video/mp4" />
              </video>
              <div className="help-media-note">
                Place /assets/help/quickstart.mp4 and /assets/help/screen-*.png to enable media.
              </div>
            </div>
          ) : null}
          {section.bullets ? (
            <ul>
              {section.bullets.map((bullet) => (
                <li key={bullet}>{bullet}</li>
              ))}
            </ul>
          ) : null}
        </section>
      ))}

      {showTop ? (
        <button
          type="button"
          className="help-fab"
          aria-label="Back to top"
          onClick={() => smoothScrollTo("top")}
        >
          ↑ Top
        </button>
      ) : null}
    </div>
  );
}
