# Flexible Dimensions — Design

Status: **proposal** · Author: planning team · Scope: forecasting + capacity planning

## Goal

Today the planning hierarchy is fixed: **Business Area → Sub-Business-Area →
Channel → Site**. We want planners to define and use **additional dimensions**
(e.g. tenure band, language, line-of-business, customer/SLA tier, work type) to
tag, filter, group, and eventually roll up plans and demand — without forking the
calculation engine.

This is the "configurable N-dimensions" option. It is a **multi-PR, staged**
effort because dimensions touch every layer: ingest → storage → scope keys →
rollups → UI. This doc fixes the data model and the rollout order so each stage
ships independently and safely.

## Use cases

1. **Tag & filter:** label plans by `segment=Tenured` and filter the Planning list.
   *(shipped — see Phase 0.)*
2. **Multiple custom tags per plan:** `tenure=Tenured, language=Spanish`.
3. **Group/aggregate:** "show required FTE by tenure band across this BA".
4. **Dimensioned demand:** ingest volume tagged by a custom dimension and roll it
   up by that dimension.

## Current model (what we build on)

- **Plan entity** (`planning_plans`): fixed columns `business_area`,
  `sub_business_area`, `channel`, `site`, plus a free-form `hierarchy_json` blob
  for everything else. `plan_key = lower("BA|SBA|Channel|Site")`.
- **Timeseries** (`timeseries_store`): a `scope_key` string
  `"BA|SBA|Channel|Site"` (pipe-joined, lowercased) keys every volume/AHT series;
  matching and the BA-rollup walk this exact shape.
- **Rollups** (`ba_rollup_plan`, `ops_dashboard`): aggregate child scopes by the
  fixed hierarchy.

The fixed dimensions are **system dimensions**; the work is to add **custom
dimensions** alongside them, reusing `hierarchy_json` so no destructive migration
is required.

## Data model

A custom dimension has a **definition** (registry) and **values** (on entities).

### 1. Dimension registry (config)

A small ordered list, stored as a global setting (JSON), editable in Settings:

```json
[
  { "key": "tenure",   "label": "Tenure band", "order": 1, "values": ["New", "Nesting", "Tenured"] },
  { "key": "language", "label": "Language",    "order": 2, "values": [] }
]
```

- `key` is a slug (`[a-z0-9_]`), immutable once created.
- `values` optional (free-text allowed when empty).
- System dimensions (BA/SBA/Channel/Site) are **implicit** and not in the registry.

### 2. Dimension values on a plan

Stored in `hierarchy_json.dimensions` (a `{key: value}` map). Phase 0 already
writes one key (`segment`); generalising it to a map is backward-compatible:

```json
{ "dimensions": { "segment": "Premier", "tenure": "Tenured", "language": "Spanish" } }
```

`load_plan` / `list_plans` surface `dimensions` as a top-level field (Phase 0
already does this for `segment`).

### 3. Dimension values on demand (timeseries) — the hard part

`scope_key` is the bottleneck: it is a positional `BA|SBA|Channel|Site` string
baked into storage filenames, hashing, and the rollup walk. Three options:

| Option | Idea | Pros | Cons |
|--------|------|------|------|
| **A. Extend scope_key** | Append `|k=v|k=v` sorted | Reuses all matching logic | Changes every key; migration of existing series; key explosion |
| **B. Sidecar dimension map** | Keep `scope_key` as-is; store a separate `scope_dimensions` map keyed by `scope_key` | Non-destructive; existing series untouched | Rollup must join the sidecar; two sources of truth |
| **C. Dimensions only at plan level (no dimensioned demand)** | Demand stays BA/SBA/Channel/Site; dimensions are a plan-organising layer only | Smallest blast radius; no timeseries change | Can't roll up *demand* by a custom dimension |

**Recommendation:** ship **C** first (Phases 0–1: plan-level dimensions + filter +
group), then **B** (Phase 2) when dimensioned-demand rollups are actually needed.
Avoid **A** — rewriting every scope_key is high-risk for low incremental value.

## Phased rollout

Each phase is an independent, reviewable PR.

- **Phase 0 — segment tag (DONE, PR #28).** One free-text dimension on a plan +
  Planning-list filter. Proves the storage path (`hierarchy_json`) and the UI
  surface with zero calc impact.
- **Phase 1 — dimension registry + multi-dimension plan tags.**
  - Registry CRUD (Settings) + `GET /api/planning/dimensions`.
  - Generalise the plan `segment` to a `dimensions` map (keep `segment` as an
    alias for back-compat).
  - Plan detail: edit all registered dimensions. Planning list: filter by any
    dimension; group the kanban by a chosen dimension.
  - **No calc/rollup change.** Pure plan-organising layer (Option C).
- **Phase 2 — dimensioned demand (Option B). DONE.**
  - A `scope_dimensions` sidecar (`scope_dimensions_store`) maps a normalized
    timeseries `scope_key` to a dimension map. Populated at ingest (the REST
    ingest API accepts an optional `dimensions` object) or via
    `POST /api/planning/scope/dimensions`. The `scope_key` itself is **not**
    rewritten.
  - An **opt-in** rollup (`POST /api/planning/plan/scope-balance-by-dimension`,
    pure helper `scope_dimension_rollup`) groups the per-scope FTE balance by a
    registered dimension. It only re-buckets numbers the engine already computed —
    no calc change — and the default BA/SBA/Channel/Site rollup is untouched.
- **Phase 3 — dimension-aware analytics. DONE.** The Demand-by-Dimension group-by
  is surfaced at the org level (the BA rollup view, not just per-plan), with a
  coverage % column, and a **Scope tags** manager in the Planning workspace lets
  planners assign scope→dimension values from the UI (closing the API-only gap so
  the group-by is usable end-to-end).

## Backward compatibility & migration

- No column drops, no destructive migration: custom dimensions live in JSON.
- `segment` (Phase 0) is preserved as `dimensions.segment`; old reads keep working.
- Plans without any `dimensions` map behave exactly as today.
- The registry defaults to empty; with no registered dimensions the UI is unchanged.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| scope_key rewrite breaks existing series | Don't (Option A rejected); use the sidecar (B) |
| Rollup correctness with custom group-bys | Phase 2 is opt-in; default rollup untouched; add tests per group-by |
| UI complexity from arbitrary dimensions | Registry caps what's shown; dimensions are optional everywhere |
| Cardinality blow-up (many dimension values) | Free-text dims are filters/tags only; group-by limited to registered dims |

## Out of scope (for now)

- Dimensioned *settings* (per-dimension SL/shrinkage) — would touch the
  effective-dated settings resolver; revisit after Phase 2.
- Cross-dimension Erlang solves — the capacity engine stays dimension-agnostic;
  dimensions only group already-computed results.

## Next step

Implement **Phase 1** (dimension registry + multi-dimension plan tags + group-by
filter) as the next PR, building directly on the Phase 0 `segment` plumbing.
