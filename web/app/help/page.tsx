import AppShell from "../_components/AppShell";
import HelpSearch from "./search-client";

const tocItems = [
  { id: "end-to-end", label: "End-to-End Flow (Start with Forecasting)" },
  { id: "forecasting-workflow", label: "Forecasting Workflow (Start Here)" },
  { id: "forecasting-inputs", label: "Forecasting Inputs" },
  { id: "transformation-projects", label: "Transformation Projects" },
  { id: "daily-interval", label: "Daily and Interval Forecast" },
  { id: "capacity-overview", label: "Capacity Planning Overview" },
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

const sections = [
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
      "Interval history: date, interval, volume (AHT optional) for daily/interval split."
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
      "BA rollups provide summarized capacity views."
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
    bullets: [
      "Admin/Planner can save settings; Admin can delete plans.",
      "Viewer is read-only."
    ]
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
  {
    id: "quickstart",
    title: "Quickstart Video and Screenshots",
    media: true
  },
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

export default function HelpPage() {
  return (
    <AppShell crumbs="Home" crumbIcon="🏠">
      <div id="top" className="help-root">
        <div className="help-header">
          <h1>Help & Documentation</h1>
          <HelpSearch />
        </div>

        <section className="help-card">
          <h2>Table of Contents</h2>
          <ul className="help-toc">
            {tocItems.map((item) => (
              <li key={item.id}>
                <a href={`#${item.id}`}>{item.label}</a>
              </li>
            ))}
          </ul>
        </section>

        {sections.map((section) => (
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
            <a className="help-back" href="#top">
              Back to top
            </a>
          </section>
        ))}
      </div>
    </AppShell>
  );
}
