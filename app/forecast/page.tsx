import Link from "next/link";
import AppShell from "../_components/AppShell";
import { forecastModels } from "./forecast-data";

export default function ForecastPage() {
  return (
    <AppShell crumbs="CAP-CONNECT / Forecast">
      <div className="forecast-page">
        <section className="forecast-header-row">
          <div>
            <h1 className="forecast-heading">
              <span className="forecast-heading-icon">🔮</span>
              Power of 9 Models: A complete suite for Forecasting
            </h1>
            <p className="forecast-subtitle">Your guided path through forecasting.</p>
          </div>
          <Link className="btn btn-primary forecast-cta" href="/forecast/volume-summary">
            → Volume Summary
          </Link>
        </section>

        <section className="forecast-grid">
          {forecastModels.map((model) => (
            <article key={model.title} className="forecast-card">
              <div className="forecast-card-title">
                <span className="forecast-card-icon">{model.icon}</span>
                <span>{model.title}</span>
              </div>
              <div className="forecast-card-body">{model.description}</div>
              <div className="forecast-equation">
                <div className="forecast-equation-label">Equation:</div>
                {model.equations.map((line) => (
                  <div key={`${model.title}-${line}`} className="forecast-equation-line">
                    {line}
                  </div>
                ))}
              </div>
            </article>
          ))}
        </section>
      </div>
    </AppShell>
  );
}
