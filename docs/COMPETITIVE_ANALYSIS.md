# Competitive Gap Analysis — Forecasting & Capacity Planning

**Scope:** This document compares CAP-CONNECT against **Anaplan** and **Datanitiv**
on **forecasting and capacity-planning capabilities only**. Generic SaaS concerns
(auth, infra, UI polish) are deliberately excluded — they are covered in the
security review. The goal is to identify which forecasting/capacity features we are
missing relative to these two products and to prioritise what is worth building.

Findings about *our* platform are sourced from the code (`forecast-service/app/pipeline/*`,
`web/app/*`, `web/app/help/help-content.tsx`). Competitor capabilities are described
at the level relevant to a contact-centre / workforce capacity-planning use case.

---

## 1. What CAP-CONNECT does today (baseline)

CAP-CONNECT is a **purpose-built contact-centre capacity-planning tool** with a
genuinely strong calculation core. Notable strengths that already match or exceed
generic planning tools:

| Area | Capability (today) |
|------|--------------------|
| **Demand forecasting** | Multi-model: Prophet, SARIMAX, Random Forest, XGBoost, VAR, plus auto-tuned EWMA smoothing with anomaly detection (`forecasting/contact_ratio_dash.py`, `pipeline/volume_summary.py`) |
| **Granularity** | Monthly → daily (historical weekday distribution) → 30-min interval (`pipeline/daily_interval.py`) |
| **Multi-channel** | Voice, Chat (concurrency), Outbound (OPC×connect×RPC), Back Office, Blended |
| **Capacity engine** | Erlang C (voice), Erlang A-style concurrency (chat), Erlang for outbound, BO linear-TAT *and* BO Erlang, occupancy caps, ASA/SL/PHC (`pipeline/capacity_core.py`) |
| **Shrinkage** | Compounded OOO×INO, scoped (BA/SBA/Channel/Site), effective-dated |
| **Attrition** | Annualised, per-scope denominator |
| **New-hire ramp** | Training/nesting/SDA phases, productivity %, AHT uplift |
| **Supply** | Roster ingestion (wide/long), projected supply = prior − attrition + new-hire |
| **Scenarios** | What-if dials (volume / AHT / shrink), Budget-vs-Actual, **effective-dated settings** across all four grains |
| **Rollups** | BA→SBA→Channel→Site hierarchy, true SUMPRODUCT interval→daily, volume-weighted level rollups |
| **Rebalancing** | Basic org-hiring + **suggested borrow/lend moves** across BA/SBA (Ops dashboard) |
| **Transparency** | Every number is an explainable formula (documented in `/help`) — a real differentiator vs black-box tools |

This is a focused, transparent, formula-driven engine. The gaps below are about
**breadth of planning workflow, statistical rigour, and integration** — not about
the core math, which is solid.

---

## 2. Anaplan — comparison

**What Anaplan is:** a general-purpose **connected-planning** platform built on the
Hyperblock in-memory calculation engine. It is not contact-centre-specific; it is a
modelling environment in which workforce/capacity models are *built*. Its strength is
enterprise planning workflow, not domain-specific queueing math.

### Where Anaplan is ahead (our gaps)

| Gap | Anaplan has | We have | Impact |
|-----|-------------|---------|--------|
| **Saved scenario versions / branching** | First-class versions, side-by-side compare, "what-if" branches that persist | What-if dials are *transient* (overwrite a single record); only a 2-plan compare endpoint | **High** — planners can't keep "optimistic/base/downside" as durable artefacts |
| **Driver-based multi-dimensional modelling** | Arbitrary dimensions, formulas across cubes, top-down + bottom-up reconciliation | Fixed BA→SBA→Channel→Site hierarchy, fixed metrics | **High** — can't re-slice by tenure, language, LOB, customer tier without code |
| **Planning workflow & approvals** | Submission, review, lock, approval chains, audit of every cell change | Plan status flags (draft/current), partial activity log; no approval workflow | **Medium** — no governed plan sign-off |
| **Real-time multi-user collaboration** | Concurrent editing, cell-level audit, comments | Single-writer; no locking/conflict resolution | **Medium** |
| **Data integration** | Native connectors (Workday, SAP, Salesforce), Anaplan Data Orchestrator, REST/ODBC | Manual CSV/XLSX upload only; no inbound API/WFM/ACD connectors | **High** — every refresh is a manual upload |
| **Top-down target allocation** | Spread a target HC/budget down the hierarchy with driver weights | Bottom-up rollup only | **Medium** |
| **Audit trail granularity** | Every change attributed and reversible | created_by/updated_at metadata; no cell-level history or rollback | **Medium** |

### Where we are ahead of Anaplan

- **Out-of-the-box Erlang/queueing math.** Anaplan has no native Erlang C/A; you must
  hand-build approximations in formulas. Our SL/ASA/occupancy/PHC engine is built-in and correct.
- **Domain models for free** (shrinkage compounding, new-hire ramp, attrition annualisation,
  interval seasonality) — in Anaplan these are months of model-building.
- **Cost & time-to-value** — Anaplan is a heavy, expensive licence + implementation.

**Net:** Anaplan wins on *planning platform breadth* (versions, dimensions, workflow,
integrations); we win on *domain depth* (queueing, contact-centre primitives).

---

## 3. Datanitiv — comparison

**What Datanitiv is:** a **contact-centre / workforce analytics and capacity-planning**
SaaS — a much closer competitor than Anaplan because it targets the same domain
(omnichannel forecasting, long-term capacity planning, WFM-adjacent analytics).

### Where Datanitiv is typically ahead (our gaps)

| Gap | Datanitiv-class capability | We have | Impact |
|-----|---------------------------|---------|--------|
| **WFM / ACD / data-source integrations** | Connectors to WFM (NICE, Verint, Genesys), ACD, and BI layers for automated actuals | Manual upload only | **High** — automated actuals are table-stakes for a planning product |
| **Forecast accuracy tracking / backtesting surfaced** | MAPE/WAPE/bias tracked over time per model, accuracy dashboards driving model selection | Backtest logic exists in `daily_interval.py` but accuracy is **not surfaced** as an ongoing KPI | **High** — we run 5 models but don't *show* which is winning |
| **Confidence intervals / prediction bands** | Forecast ranges (P10/P50/P90) feeding risk-based staffing | Models produce point forecasts; CIs are not extracted/shown | **Medium-High** |
| **Long-range scenario planning & hiring plans** | Multi-year hiring/recruitment plans, attrition-driven backfill scheduling, budget reconciliation | New-hire classes per plan; no optimisation of *when/how many* to hire to hit a target | **Medium-High** |
| **What-if scenario library** | Named, saved, comparable scenarios (volume shock, shrink spike, attrition wave) | Transient dials only | **High** (same as Anaplan gap) |
| **Intraday / shift-level scheduling bridge** | Hand-off from capacity plan to schedules/rosters | Roster is an *input* for supply, not a generated output | **Medium** |
| **Cross-skill / shrinkage optimisation** | Optimisation (LP/heuristics) for skill mix and shrinkage targets | Erlang *solve* only; borrow/lend is a heuristic suggestion, not optimised | **Medium** |
| **Predictive attrition / churn** | ML models predicting leaver risk feeding supply | Attrition is historical-input-driven only | **Medium** |

### Where we are competitive / ahead

- **Transparent, auditable formulas** — many analytics tools are black boxes; our `/help`
  documents every calculation. This is a real sales differentiator for finance/ops sign-off.
- **Effective-dated settings across all grains** — a genuinely advanced feature (changes
  apply only-forward, past periods keep prior settings) that many tools handle crudely.
- **Choice of BO capacity model** (linear-TAT vs Erlang) per setting — flexible.
- **Multi-model forecasting in one workspace** — we already run Prophet + 4 ML/stat models.

**Net:** Datanitiv-class products win on **integration, accuracy/observability, and saved
scenario/hiring-plan workflow**; we win on **calculation transparency and effective-dating**.

---

## 4. Consolidated feature-gap matrix (forecasting + capacity only)

Legend: ✅ have · ⚠️ partial · ❌ missing

| Capability | CAP-CONNECT | Anaplan | Datanitiv | Priority to close |
|-----------|:-----------:|:-------:|:---------:|:-----------------:|
| Multi-model demand forecasting | ✅ | ⚠️ (build-your-own) | ✅ | — |
| Interval/daily/weekly/monthly grains | ✅ | ✅ | ✅ | — |
| Erlang C/A + occupancy/SL/ASA/PHC | ✅ | ❌ | ✅ | — |
| Shrinkage / attrition / new-hire ramp | ✅ | ⚠️ | ✅ | — |
| Effective-dated settings | ✅ | ⚠️ | ⚠️ | — |
| **Saved/branchable scenario versions** | ❌ | ✅ | ✅ | **P0** |
| **Forecast accuracy KPIs (MAPE/WAPE/bias) surfaced** | ⚠️ | ⚠️ | ✅ | **P0** |
| **Automated data integration (WFM/ACD/API)** | ❌ | ✅ | ✅ | **P0** |
| **Confidence intervals / P10-P50-P90** | ❌ | ⚠️ | ✅ | **P1** |
| **Long-range hiring-plan optimisation** | ⚠️ | ✅ | ✅ | **P1** |
| Driver-based / flexible dimensions | ❌ | ✅ | ⚠️ | **P1** |
| Top-down target allocation | ❌ | ✅ | ⚠️ | **P2** |
| Planning workflow / approvals | ❌ | ✅ | ⚠️ | **P2** |
| Multi-user collaboration / cell audit | ❌ | ✅ | ⚠️ | **P2** |
| Cross-skill optimisation (LP/heuristic) | ⚠️ | ⚠️ | ⚠️ | **P2** |
| Predictive attrition (ML) | ❌ | ❌ | ⚠️ | **P3** |
| Monte Carlo / risk simulation | ❌ | ⚠️ | ⚠️ | **P3** |

---

## 5. Recommended roadmap (forecasting & capacity only)

Ordered by leverage — what closes the most competitively-relevant gap for the least build:

### P0 — table-stakes we are visibly missing
1. **Saved scenario versions.** Persist what-if dials as named scenarios with a
   scenario table keyed by plan; reuse the existing `compare` endpoint for N-way compare.
   Mostly storage + UI; the calc engine already keys caches by what-if params.
2. **Forecast-accuracy dashboard.** We already backtest in `daily_interval.py` — compute
   and *store* MAPE/WAPE/bias per model per run and surface a "which model is winning"
   view. Drives credibility and auto-model-selection.
3. **One inbound integration.** Even a single scheduled WFM/ACD actuals import (or a
   documented REST ingest endpoint) removes the "everything is a manual upload" objection.

### P1 — depth that buyers compare on
4. **Prediction intervals.** Prophet/SARIMAX already expose uncertainty; extract P10/P50/P90
   and feed a risk-based required-FTE band (e.g. staff to P75 demand).
5. **Hiring-plan solver.** Given a required-FTE curve + attrition + ramp, recommend
   class sizes/start weeks to hit coverage — turns new-hire input into an *output*.
6. **Flexible dimensions.** Allow a configurable extra dimension (tenure/language/LOB)
   on rollups without code changes.

### P2 — planning-platform maturity
7. Top-down target allocation (spread a target down the hierarchy).
8. Plan approval workflow + cell-level audit/versioning.
9. Cross-skill optimisation (upgrade borrow/lend from heuristic to LP/greedy optimiser).

### P3 — differentiators
10. Predictive attrition (leaver-risk model feeding supply).
11. Monte Carlo on volume/AHT/shrink for probabilistic coverage.

---

## 6. Positioning summary

> **We are a deep, transparent, contact-centre capacity *engine*. The competitors are
> broad planning *platforms*.**

- vs **Anaplan**: we win on built-in queueing math and time-to-value; we lose on scenario
  versioning, flexible dimensions, workflow, and integrations.
- vs **Datanitiv**: closest domain competitor — we match the core math and beat it on
  transparency/effective-dating, but lose on integrations, surfaced forecast accuracy,
  and saved scenario/hiring-plan workflow.

**The three highest-leverage investments are P0: saved scenarios, surfaced forecast
accuracy, and at least one automated data integration.** None require changing the
calculation core — they wrap workflow, observability, and connectivity around an engine
that is already strong.

*Note on existing capability:* "cross-skill borrow/lend" already exists in a basic form on
the Ops dashboard (suggested moves + org-hiring), so it is marked ⚠️ partial, not missing —
the gap is *optimisation*, not presence.
