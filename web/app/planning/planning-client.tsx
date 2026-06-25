"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AppShell from "../_components/AppShell";
import Icon, { channelIconName } from "../_components/Icon";
import MultiSelect from "../_components/MultiSelect";
import { useGlobalLoader } from "../_components/GlobalLoader";
import { useToast } from "../_components/ToastProvider";
import { apiGet, apiPost } from "../../lib/api";

type HeadcountOptions = {
  business_areas?: string[];
  sub_business_areas?: string[];
  channels?: string[];
  locations?: string[];
  sites?: string[];
};

type PlanRecord = {
  id: number;
  plan_name?: string;
  business_area?: string;
  sub_business_area?: string;
  channel?: string;
  location?: string;
  site?: string;
  segment?: string;
  dimensions?: Record<string, string>;
};

type DimensionDef = {
  key: string;
  label: string;
  order?: number;
  values?: string[];
};

const ALPHABET = ["All", ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ"];
const CHANNEL_OPTIONS = ["Voice", "Back Office", "Chat", "MessageUs", "Outbound", "Blended"];
const PLAN_TYPES = [
  "Volume Based",
  "Billable Hours Based",
  "FTE Based",
  "FTE Based Billable Transaction"
];
const DEFAULT_LOCATIONS = ["India", "UK"];
const CHAN_ALIASES: Record<string, string> = {
  "back office": "Backoffice",
  "back-office": "Backoffice",
  backoffice: "Backoffice",
  voice: "Voice",
  phone: "Voice",
  telephony: "Voice",
  call: "Voice",
  chat: "Chat",
  messageus: "MessageUs",
  "message us": "MessageUs",
  outbound: "Outbound",
  blended: "Blended",
  email: "Email",
  mail: "Email",
  omni: "Omni"
};

const DEFAULT_NEW_PLAN = {
  org: "Barclays",
  business_entity: "Barclays",
  business_area: "",
  sub_business_area: "",
  plan_name: "",
  plan_type: "",
  channels: [] as string[],
  location: "",
  site: "",
  start_week: "",
  end_week: "",
  ft_weekly_hours: 40,
  pt_weekly_hours: 20,
  is_current: true,
  fw_lower_options: [] as string[],
  upper_options: [] as string[]
};

const ADMIN_DELETE_ENABLED = false;

function canonicalChannel(label?: string) {
  if (!label) return "Backoffice";
  const key = label.trim().toLowerCase();
  return CHAN_ALIASES[key] || label.trim().replace(/\b\w/g, (m) => m.toUpperCase());
}

export default function PlanningClient() {
  const { setLoading } = useGlobalLoader();
  const { notify } = useToast();
  const kanbanRef = useRef<HTMLDivElement | null>(null);

  const [statusFilter, setStatusFilter] = useState<"current" | "history">("current");
  const [alphaFilter, setAlphaFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [businessAreas, setBusinessAreas] = useState<string[]>([]);
  const [selectedBa, setSelectedBa] = useState<string>("");
  const [plans, setPlans] = useState<PlanRecord[]>([]);
  const [segmentFilter, setSegmentFilter] = useState<string>("All");
  const [dimRegistry, setDimRegistry] = useState<DimensionDef[]>([]);
  const [dimFilterKey, setDimFilterKey] = useState<string>("");
  const [dimFilterVal, setDimFilterVal] = useState<string>("All");
  const [dimEditorOpen, setDimEditorOpen] = useState(false);
  const [dimEditorRows, setDimEditorRows] = useState<DimensionDef[]>([]);
  const [dimEditorBusy, setDimEditorBusy] = useState(false);
  const [scopeTagsOpen, setScopeTagsOpen] = useState(false);
  const [scopeTagsRows, setScopeTagsRows] = useState<{ scope_key: string; dims: Record<string, string> }[]>([]);
  const [scopeTagsOrig, setScopeTagsOrig] = useState<string[]>([]);
  const [scopeTagsBusy, setScopeTagsBusy] = useState(false);
  const [headcountSbas, setHeadcountSbas] = useState<string[]>([]);
  const [message, setMessage] = useState("");
  const [newPlanMsg, setNewPlanMsg] = useState("");

  const [newPlanOpen, setNewPlanOpen] = useState(false);
  const [deletePlanId, setDeletePlanId] = useState<number | null>(null);
  const [planForm, setPlanForm] = useState({ ...DEFAULT_NEW_PLAN });
  const [formOptions, setFormOptions] = useState<Required<HeadcountOptions>>({
    business_areas: [],
    sub_business_areas: [],
    channels: CHANNEL_OPTIONS,
    locations: [],
    sites: []
  });

  const loadBusinessAreas = useCallback(async () => {
    setLoading(true);
    try {
      const planRes = await apiGet<{ business_areas?: string[] }>(
        `/api/planning/business-areas?status=${statusFilter}`
      );
      const list = [...(planRes.business_areas ?? [])].sort((a, b) => a.localeCompare(b));
      setBusinessAreas(list);
      setSelectedBa((prev) => (prev && list.includes(prev) ? prev : list[0] || ""));
    } catch (error: any) {
      notify("error", error?.message || "Could not load business areas.");
    } finally {
      setLoading(false);
    }
  }, [notify, setLoading, statusFilter]);

  const loadPlans = useCallback(
    async (ba: string, statusOverride?: "current" | "history") => {
      if (!ba) {
        setPlans([]);
        return;
      }
      setLoading(true);
      try {
        const statusParam = statusOverride ?? statusFilter;
        const res = await apiGet<{ plans?: PlanRecord[] }>(
          `/api/planning/plan?ba=${encodeURIComponent(ba)}&status=${statusParam}`
        );
        setPlans(res.plans ?? []);
      } catch (error: any) {
        notify("error", error?.message || "Could not load plans.");
      } finally {
        setLoading(false);
      }
    },
    [notify, setLoading, statusFilter]
  );

  useEffect(() => {
    void loadBusinessAreas();
  }, [loadBusinessAreas]);

  useEffect(() => {
    if (!selectedBa) return;
    void loadPlans(selectedBa);
  }, [loadPlans, selectedBa]);

  const loadHeadcountSbas = useCallback(async (ba: string) => {
    if (!ba) {
      setHeadcountSbas([]);
      return;
    }
    try {
      const res = await apiGet<HeadcountOptions>(`/api/forecast/headcount/options?ba=${encodeURIComponent(ba)}`);
      setHeadcountSbas(res.sub_business_areas ?? []);
    } catch {
      setHeadcountSbas([]);
    }
  }, []);

  useEffect(() => {
    void loadHeadcountSbas(selectedBa);
  }, [loadHeadcountSbas, selectedBa]);

  const filteredAreas = useMemo(() => {
    let list = [...businessAreas];
    if (alphaFilter !== "All") {
      list = list.filter((ba) => ba.toUpperCase().startsWith(alphaFilter));
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter((ba) => ba.toLowerCase().includes(q));
    }
    return list;
  }, [alphaFilter, businessAreas, search]);

  useEffect(() => {
    if (!filteredAreas.length) {
      if (selectedBa) setSelectedBa("");
      return;
    }
    if (!selectedBa || !filteredAreas.includes(selectedBa)) {
      setSelectedBa(filteredAreas[0]);
    }
  }, [filteredAreas, selectedBa]);

  const loadDimRegistry = useCallback(async () => {
    try {
      const res = await apiGet<{ dimensions?: DimensionDef[] }>("/api/planning/dimensions");
      setDimRegistry(res.dimensions ?? []);
    } catch {
      setDimRegistry([]);
    }
  }, []);

  useEffect(() => {
    void loadDimRegistry();
  }, [loadDimRegistry]);

  const segmentOptions = useMemo(() => {
    const set = new Set<string>();
    plans.forEach((p) => {
      const seg = String(p.segment || "").trim();
      if (seg) set.add(seg);
    });
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [plans]);

  // Custom dimensions available to filter by (registry minus 'segment', which
  // has its own filter) that actually appear on the loaded plans.
  const filterableDims = useMemo(
    () => dimRegistry.filter((d) => d.key !== "segment"),
    [dimRegistry]
  );

  const dimValueOptions = useMemo(() => {
    if (!dimFilterKey) return [] as string[];
    const set = new Set<string>();
    plans.forEach((p) => {
      const v = String((p.dimensions || {})[dimFilterKey] || "").trim();
      if (v) set.add(v);
    });
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [plans, dimFilterKey]);

  const visiblePlans = useMemo(() => {
    return plans.filter((p) => {
      if (segmentFilter !== "All" && String(p.segment || "").trim() !== segmentFilter) return false;
      if (dimFilterKey && dimFilterVal !== "All") {
        if (String((p.dimensions || {})[dimFilterKey] || "").trim() !== dimFilterVal) return false;
      }
      return true;
    });
  }, [plans, segmentFilter, dimFilterKey, dimFilterVal]);

  const openDimEditor = useCallback(() => {
    setDimEditorRows(dimRegistry.map((d) => ({ ...d, values: [...(d.values || [])] })));
    setDimEditorOpen(true);
  }, [dimRegistry]);

  const saveDimRegistry = useCallback(async () => {
    const payload = dimEditorRows
      .map((r) => ({
        key: String(r.key || r.label || "").trim(),
        label: String(r.label || "").trim(),
        values: Array.isArray(r.values) ? r.values : []
      }))
      .filter((r) => r.key || r.label);
    setDimEditorBusy(true);
    try {
      const res = await apiPost<{ dimensions?: DimensionDef[] }>("/api/planning/dimensions", {
        dimensions: payload
      });
      setDimRegistry(res.dimensions ?? []);
      setDimEditorOpen(false);
      notify("success", "Dimensions saved.");
    } catch (error: any) {
      notify("error", error?.message || "Could not save dimensions.");
    } finally {
      setDimEditorBusy(false);
    }
  }, [dimEditorRows, notify]);

  // Scope-dimension tags (the Phase 2 demand sidecar) — assign dimension values
  // to a timeseries scope so demand can be grouped by them.
  const openScopeTags = useCallback(async () => {
    try {
      const res = await apiGet<{ scopes?: Record<string, Record<string, string>> }>("/api/planning/scope/dimensions");
      const rows = Object.entries(res.scopes || {}).map(([scope_key, dims]) => ({ scope_key, dims: { ...dims } }));
      setScopeTagsRows(rows.length ? rows : [{ scope_key: "", dims: {} }]);
      setScopeTagsOrig(rows.map((r) => r.scope_key.trim().toLowerCase()).filter(Boolean));
    } catch {
      setScopeTagsRows([{ scope_key: "", dims: {} }]);
      setScopeTagsOrig([]);
    }
    setScopeTagsOpen(true);
  }, []);

  const saveScopeTags = useCallback(async () => {
    const rows = scopeTagsRows.filter((r) => r.scope_key.trim());
    const present = new Set(rows.map((r) => r.scope_key.trim().toLowerCase()));
    // Scopes that were loaded but the user removed a row for → clear them (empty map).
    const removed = scopeTagsOrig.filter((k) => !present.has(k));
    setScopeTagsBusy(true);
    try {
      for (const r of rows) {
        await apiPost("/api/planning/scope/dimensions", { scope_key: r.scope_key.trim(), dimensions: r.dims });
      }
      for (const key of removed) {
        await apiPost("/api/planning/scope/dimensions", { scope_key: key, dimensions: {} });
      }
      setScopeTagsOpen(false);
      notify("success", "Scope tags saved.");
    } catch (error: any) {
      notify("error", error?.message || "Could not save scope tags.");
    } finally {
      setScopeTagsBusy(false);
    }
  }, [scopeTagsRows, scopeTagsOrig, notify]);

  const groupedKanban = useMemo(() => {
    const grouped: Record<string, Record<string, PlanRecord[]>> = {};
    visiblePlans.forEach((plan) => {
      const sba = (plan.sub_business_area || "Overall").trim() || "Overall";
      const channels = (plan.channel || "").split(",").map((c) => c.trim()).filter(Boolean);
      const channelList = channels.length ? channels : ["Unspecified"];
      channelList.forEach((raw) => {
        const canonical = canonicalChannel(raw);
        grouped[sba] = grouped[sba] || {};
        grouped[sba][canonical] = grouped[sba][canonical] || [];
        grouped[sba][canonical].push(plan);
      });
    });
    return grouped;
  }, [visiblePlans]);

  const sbaOrder = useMemo(() => {
    const order: string[] = [];
    const seen = new Set<string>();
    headcountSbas.forEach((sba) => {
      const label = (sba || "").trim() || "Overall";
      if (!seen.has(label)) {
        seen.add(label);
        order.push(label);
      }
    });
    const planSeen = new Set<string>();
    const planOrder: string[] = [];
    visiblePlans.forEach((plan) => {
      const sba = (plan.sub_business_area || "Overall").trim() || "Overall";
      if (!planSeen.has(sba)) {
        planSeen.add(sba);
        planOrder.push(sba);
      }
    });
    planOrder.forEach((sba) => {
      if (!seen.has(sba)) {
        seen.add(sba);
        order.push(sba);
      }
    });
    return order.length ? order : ["Overall"];
  }, [headcountSbas, visiblePlans]);

  const openNewPlan = () => {
    setMessage("");
    setNewPlanMsg("");
    setPlanForm((prev) => ({
      ...DEFAULT_NEW_PLAN,
      business_area: selectedBa || prev.business_area
    }));
    setNewPlanOpen(true);
  };

  const closeNewPlan = () => {
    setNewPlanOpen(false);
    setNewPlanMsg("");
  };

  const refreshFormOptions = useCallback(async () => {
    if (!newPlanOpen) return;
    const params = new URLSearchParams();
    if (planForm.business_area) params.set("ba", planForm.business_area);
    if (planForm.location) params.set("location", planForm.location);
    try {
      const res = await apiGet<HeadcountOptions>(
        `/api/forecast/headcount/options${params.toString() ? `?${params.toString()}` : ""}`
      );
      const headcountBas = res.business_areas ?? [];
      const unionBas = [...new Set([...businessAreas, ...headcountBas])].sort((a, b) => a.localeCompare(b));
      const hasBa = Boolean(planForm.business_area);
      const nextSubBas = hasBa ? res.sub_business_areas ?? [] : [];
      const nextLocations =
        hasBa && (res.locations ?? []).length ? (res.locations ?? []) : DEFAULT_LOCATIONS;
      const nextSites = hasBa ? res.sites ?? [] : [];
      setFormOptions((prev) => ({
        ...prev,
        business_areas: unionBas.length ? unionBas : prev.business_areas,
        sub_business_areas: nextSubBas,
        locations: nextLocations,
        sites: nextSites,
        channels: res.channels ?? CHANNEL_OPTIONS
      }));

      setPlanForm((prev) => {
        const next = { ...prev };
        if (prev.business_area) {
          if (!next.sub_business_area || !nextSubBas.includes(next.sub_business_area)) {
            next.sub_business_area = nextSubBas[0] || "";
          }
          if (!next.location || !nextLocations.includes(next.location)) {
            next.location = nextLocations[0] || "";
          }
          if (!next.site || !nextSites.includes(next.site)) {
            next.site = nextSites[0] || "";
          }
        } else {
          next.sub_business_area = "";
          next.location = "";
          next.site = "";
        }
        return next;
      });
    } catch (error: any) {
      notify("error", error?.message || "Could not load headcount options.");
    }
  }, [businessAreas, newPlanOpen, notify, planForm.business_area, planForm.location]);

  useEffect(() => {
    void refreshFormOptions();
  }, [refreshFormOptions]);

  const createPlan = async () => {
    if (!planForm.business_area || !planForm.plan_name || !planForm.start_week) {
      setNewPlanMsg("Business Area, Plan Name and Start Week are required.");
      return;
    }
    setLoading(true);
    try {
      const payload = {
        org: planForm.org,
        business_entity: planForm.business_entity,
        business_area: planForm.business_area,
        sub_business_area: planForm.sub_business_area,
        channel: planForm.channels,
        location: planForm.location,
        site: planForm.site,
        plan_name: planForm.plan_name,
        plan_type: planForm.plan_type,
        start_week: planForm.start_week,
        end_week: planForm.end_week,
        ft_weekly_hours: planForm.ft_weekly_hours,
        pt_weekly_hours: planForm.pt_weekly_hours,
        tags: [],
        is_current: planForm.is_current,
        status: planForm.is_current ? "current" : "draft",
        hierarchy_json: {
          business_area: planForm.business_area,
          sub_business_area: planForm.sub_business_area,
          channels: planForm.channels,
          location: planForm.location,
          site: planForm.site,
          fw_lower_options: planForm.fw_lower_options,
          upper_options: planForm.upper_options
        }
      };
      const res = await apiPost<{ id?: number }>("/api/planning/plan", payload);
      notify("success", `Created plan '${planForm.plan_name}'`);
      setMessage(`Created plan '${planForm.plan_name}' (ID ${res.id ?? "—"})`);
      setNewPlanOpen(false);
      const nextStatus = planForm.is_current ? "current" : "history";
      setStatusFilter(nextStatus);
      setSelectedBa(planForm.business_area);
      setAlphaFilter("All");
      setSearch("");
      await loadBusinessAreas();
      await loadPlans(planForm.business_area, nextStatus);
    } catch (error: any) {
      const errMsg = String(error?.message || "");
      const friendly = errMsg.includes("(409)") ? "That plan already exists for this scope." : errMsg;
      setNewPlanMsg(friendly || "Could not create plan.");
      notify("error", friendly || "Could not create plan.");
    } finally {
      setLoading(false);
    }
  };

  const confirmDelete = async () => {
    if (!deletePlanId) return;
    setLoading(true);
    try {
      await apiPost("/api/planning/plan/delete", { plan_id: deletePlanId });
      notify("success", "Plan deleted.");
      setDeletePlanId(null);
      await loadBusinessAreas();
      if (selectedBa) {
        await loadPlans(selectedBa);
      }
    } catch (error: any) {
      notify("error", error?.message || "Could not delete plan.");
    } finally {
      setLoading(false);
    }
  };

  const scrollKanban = (direction: "left" | "right") => {
    const el = kanbanRef.current;
    if (!el) return;
    const step = el.querySelector(".ws-kanban-col")?.getBoundingClientRect().width || 420;
    const delta = direction === "left" ? -step - 16 : step + 16;
    el.scrollBy({ left: delta, behavior: "smooth" });
  };

  return (
    <AppShell crumbs="CAP-CONNECT / Planning Workspace">
      <div className="ws-root">
        <div className="ws-toolbar">
          <div className="ws-tabs">
            <button
              type="button"
              className={`btn ws-tab ${statusFilter === "current" ? "ws-tab--active" : ""}`}
              onClick={() => setStatusFilter("current")}
            >
              Current
            </button>
            <button
              type="button"
              className={`btn ws-tab ${statusFilter === "history" ? "ws-tab--active" : ""}`}
              onClick={() => setStatusFilter("history")}
            >
              History
            </button>
          </div>
          <button type="button" className="btn btn-primary" onClick={openNewPlan}>
            + New Cap Plan
          </button>
          {segmentOptions.length ? (
            <select
              className="input ws-segment-filter"
              value={segmentFilter}
              onChange={(e) => setSegmentFilter(e.target.value)}
              aria-label="Filter by segment"
            >
              <option value="All">All segments</option>
              {segmentOptions.map((seg) => (
                <option key={seg} value={seg}>
                  {seg}
                </option>
              ))}
            </select>
          ) : null}
          {filterableDims.length ? (
            <select
              className="input ws-segment-filter"
              value={dimFilterKey}
              onChange={(e) => {
                setDimFilterKey(e.target.value);
                setDimFilterVal("All");
              }}
              aria-label="Filter by dimension"
            >
              <option value="">No dimension</option>
              {filterableDims.map((d) => (
                <option key={d.key} value={d.key}>
                  {d.label}
                </option>
              ))}
            </select>
          ) : null}
          {dimFilterKey && dimValueOptions.length ? (
            <select
              className="input ws-segment-filter"
              value={dimFilterVal}
              onChange={(e) => setDimFilterVal(e.target.value)}
              aria-label="Filter by dimension value"
            >
              <option value="All">All values</option>
              {dimValueOptions.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          ) : null}
          <button type="button" className="btn btn-light" onClick={openDimEditor}>
            Manage dimensions
          </button>
          {filterableDims.length ? (
            <button type="button" className="btn btn-light" onClick={openScopeTags}>
              Scope tags
            </button>
          ) : null}
          <div className="ws-search">
            <input
              className="input"
              type="text"
              placeholder="Search Business Area"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
          <div className="ws-message">{message}</div>
        </div>

        <div className="ws-workspace">
          <div className="ws-left-card">
            <div className="card">
              <div className="card-body">
                <div className="ws-caption">{statusFilter === "current" ? "Current" : "History"}</div>
                <div className="ws-left-grid">
                  <div className="ws-alpha-col">
                    <div className="ws-alpha-wrapper">
                      {ALPHABET.map((letter) => (
                        <label key={letter} className="ws-alpha-option">
                          <input
                            className="ws-alpha-input"
                            type="radio"
                            name="alpha-filter"
                            value={letter}
                            checked={alphaFilter === letter}
                            onChange={() => setAlphaFilter(letter)}
                          />
                          <span className="ws-alpha-label">{letter}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  <div className="ws-ba-col">
                    {filteredAreas.length ? (
                      <ul className="ws-ba-list">
                        {filteredAreas.map((ba) => (
                          <li
                            key={ba}
                            className={`ws-ba-item ${selectedBa === ba ? "active" : ""}`}
                            onClick={() => setSelectedBa(ba)}
                          >
                            <span className="ba-ico"><Icon name="briefcase" size={16} /></span>
                            <span>{ba}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className="ws-card-empty">No Business Areas found.</div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="ws-right-card">
            <div className="card">
              <div className="card-body">
                <div className="ws-right-actions">
                  <button type="button" className="btn btn-light" onClick={() => scrollKanban("left")}>
                    <Icon name="chevron-left" size={16} title="Scroll left" />
                  </button>
                  <button type="button" className="btn btn-light" onClick={() => scrollKanban("right")}>
                    <Icon name="chevron-right" size={16} title="Scroll right" />
                  </button>
                </div>
                <div className="ws-right-stack">
                  {selectedBa ? (
                    <Link href={`/plan/ba/${encodeURIComponent(selectedBa)}`} className="ws-ba-card-link">
                      <div className="ws-ba-card">
                        <span className="ba-ico"><Icon name="briefcase" size={16} /></span>
                        <span className="fw-semibold">{selectedBa}</span>
                      </div>
                    </Link>
                  ) : null}
                  <div ref={kanbanRef} className="ws-kanban">
                    {sbaOrder.map((sba) => {
                      const channels = groupedKanban[sba] || {};
                      const channelKeys = Object.keys(channels).sort((a, b) => a.localeCompare(b));
                      return (
                        <div key={sba} className="ws-kanban-col">
                          <div className="ws-col-head">{sba || "Overall"}</div>
                          <div className="ws-col-body">
                            {channelKeys.length ? (
                              channelKeys.map((ch) => {
                                const channelPlans = channels[ch] || [];
                                const siteGroups: Record<string, PlanRecord[]> = {};
                                channelPlans.forEach((plan) => {
                                  const siteLabel = plan.site?.trim() || "Sites not specified";
                                  siteGroups[siteLabel] = siteGroups[siteLabel] || [];
                                  siteGroups[siteLabel].push(plan);
                                });
                                const siteKeys = Object.keys(siteGroups).sort((a, b) => a.localeCompare(b));
                                return (
                                  <div key={ch} className="ws-kanban-card">
                                    <div className="ws-card-title">
                                      <span className="me-2"><Icon name={channelIconName(ch)} size={14} /></span>
                                      <span className="fw-semibold">{canonicalChannel(ch)}</span>
                                    </div>
                                    <div className="ws-card-body">
                                      {siteKeys.map((siteLabel) => {
                                        const sitePlans = siteGroups[siteLabel] || [];
                                        const seen = new Set<string>();
                                        return (
                                          <div key={siteLabel}>
                                            <div className="ws-card-row ws-l1">
                                              <span className="me-2"><Icon name={channelIconName(ch)} size={14} /></span>
                                              <span>{canonicalChannel(ch)}</span>
                                            </div>
                                            <div className="ws-card-row ws-l2">
                                              <span className="me-2"><Icon name="pin" size={14} /></span>
                                              <span>{siteLabel}</span>
                                            </div>
                                            {sitePlans.map((plan) => {
                                              const name = plan.plan_name?.trim() || "Untitled";
                                              const key = `${name.toLowerCase()}-${canonicalChannel(ch)}-${siteLabel.toLowerCase()}`;
                                              if (seen.has(key)) return null;
                                              seen.add(key);
                                              return (
                                                <div key={`${plan.id}-${name}`} className="ws-card-row ws-l3">
                                                  <span className="me-2"><Icon name="edit" size={14} /></span>
                                                  <Link href={`/plan/${plan.id}`} className="ws-plan-link">
                                                    {name}
                                                  </Link>
                                                  {String(plan.segment || "").trim() ? (
                                                    <span className="ws-segment-tag">{plan.segment}</span>
                                                  ) : null}
                                                  {filterableDims.map((d) => {
                                                    const v = String((plan.dimensions || {})[d.key] || "").trim();
                                                    return v ? (
                                                      <span key={d.key} className="ws-segment-tag" title={d.label}>
                                                        {v}
                                                      </span>
                                                    ) : null;
                                                  })}
                                                  {ADMIN_DELETE_ENABLED ? (
                                                    <button
                                                      type="button"
                                                      className="ws-del-btn"
                                                      onClick={() => setDeletePlanId(plan.id)}
                                                    >
                                                      <Icon name="trash" size={16} title="Delete plan" />
                                                    </button>
                                                  ) : null}
                                                </div>
                                              );
                                            })}
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </div>
                                );
                              })
                            ) : (
                              <div className="ws-card-empty">No plans yet</div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {newPlanOpen ? (
          <div className="ws-modal-backdrop">
            <div className="ws-modal">
              <div className="ws-modal-header">
                <h3>Add New Plan</h3>
                <button type="button" className="btn-close" onClick={closeNewPlan}>
                  <Icon name="close" size={14} />
                </button>
              </div>
              <div className="ws-modal-body">
                <div className="ws-form-grid">
                  <div>
                    <label className="label">Organizations</label>
                    <input
                      className="input"
                      value={planForm.org}
                      onChange={(event) => setPlanForm((prev) => ({ ...prev, org: event.target.value }))}
                    />
                  </div>
                  <div>
                    <label className="label">Business Entity</label>
                    <input
                      className="input"
                      value={planForm.business_entity}
                      onChange={(event) => setPlanForm((prev) => ({ ...prev, business_entity: event.target.value }))}
                    />
                  </div>
                  <div>
                    <label className="label">Verticals (Business Area)</label>
                    <select
                      className="select"
                      value={planForm.business_area}
                      onChange={(event) =>
                        setPlanForm((prev) => ({
                          ...prev,
                          business_area: event.target.value,
                          sub_business_area: "",
                          site: ""
                        }))
                      }
                    >
                      <option value="">Select Business Area</option>
                      {formOptions.business_areas.map((ba) => (
                        <option key={ba} value={ba}>
                          {ba}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="label">Sub Business Area</label>
                    <select
                      className="select"
                      value={planForm.sub_business_area}
                      onChange={(event) => setPlanForm((prev) => ({ ...prev, sub_business_area: event.target.value }))}
                    >
                      <option value="">Select Sub Business Area</option>
                      {formOptions.sub_business_areas.map((sba) => (
                        <option key={sba} value={sba}>
                          {sba}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="label">Plan Name</label>
                    <input
                      className="input"
                      value={planForm.plan_name}
                      onChange={(event) => setPlanForm((prev) => ({ ...prev, plan_name: event.target.value }))}
                    />
                  </div>
                  <div>
                    <label className="label">Plan Type</label>
                    <select
                      className="select"
                      value={planForm.plan_type}
                      onChange={(event) => setPlanForm((prev) => ({ ...prev, plan_type: event.target.value }))}
                    >
                      <option value="">Select Plan Type</option>
                      {PLAN_TYPES.map((planType) => (
                        <option key={planType} value={planType}>
                          {planType}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="label">Channels</label>
                    <MultiSelect
                      options={formOptions.channels.map((val) => ({ label: val, value: val }))}
                      values={planForm.channels}
                      onChange={(next) => setPlanForm((prev) => ({ ...prev, channels: next }))}
                      placeholder="Select channels"
                    />
                  </div>
                  <div>
                    <label className="label">Location</label>
                    <select
                      className="select"
                      value={planForm.location}
                      onChange={(event) => setPlanForm((prev) => ({ ...prev, location: event.target.value, site: "" }))}
                    >
                      <option value="">Select Location</option>
                      {formOptions.locations.map((loc) => (
                        <option key={loc} value={loc}>
                          {loc}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="label">Site</label>
                    <select
                      className="select"
                      value={planForm.site}
                      onChange={(event) => setPlanForm((prev) => ({ ...prev, site: event.target.value }))}
                    >
                      <option value="">Select Site</option>
                      {formOptions.sites.map((site) => (
                        <option key={site} value={site}>
                          {site}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="label">Start Week</label>
                    <input
                      className="input"
                      type="date"
                      value={planForm.start_week}
                      onChange={(event) => setPlanForm((prev) => ({ ...prev, start_week: event.target.value }))}
                    />
                  </div>
                  <div>
                    <label className="label">End Week</label>
                    <input
                      className="input"
                      type="date"
                      value={planForm.end_week}
                      onChange={(event) => setPlanForm((prev) => ({ ...prev, end_week: event.target.value }))}
                    />
                  </div>
                  <div>
                    <label className="label">Full-time Weekly Hours</label>
                    <input
                      className="input"
                      type="number"
                      value={planForm.ft_weekly_hours}
                      onChange={(event) =>
                        setPlanForm((prev) => ({ ...prev, ft_weekly_hours: Number(event.target.value) }))
                      }
                    />
                  </div>
                  <div>
                    <label className="label">Part-time Weekly Hours</label>
                    <input
                      className="input"
                      type="number"
                      value={planForm.pt_weekly_hours}
                      onChange={(event) =>
                        setPlanForm((prev) => ({ ...prev, pt_weekly_hours: Number(event.target.value) }))
                      }
                    />
                  </div>
                  <div>
                    <label className="label">Upper Grid: Include</label>
                    <MultiSelect
                      options={[{ label: "FTE Required @ Queue", value: "req_queue" }]}
                      values={planForm.upper_options}
                      onChange={(next) => setPlanForm((prev) => ({ ...prev, upper_options: next }))}
                      placeholder="Select additional upper metrics"
                    />
                  </div>
                  <div>
                    <label className="label">Lower Grid: Include</label>
                    <MultiSelect
                      options={[
                        { label: "Backlog (Items)", value: "backlog" },
                        { label: "Queue (Items)", value: "queue" }
                      ]}
                      values={planForm.fw_lower_options}
                      onChange={(next) => setPlanForm((prev) => ({ ...prev, fw_lower_options: next }))}
                      placeholder="Select FW rows to include"
                    />
                  </div>
                  <div className="ws-checkbox-row">
                    <label className="ws-checkbox">
                      <input
                        type="checkbox"
                        checked={planForm.is_current}
                        onChange={(event) => setPlanForm((prev) => ({ ...prev, is_current: event.target.checked }))}
                      />
                      <span>Is Current Plan?</span>
                    </label>
                  </div>
                </div>
                {newPlanMsg ? <div className="ws-error">{newPlanMsg}</div> : null}
              </div>
              <div className="ws-modal-footer">
                <button type="button" className="btn btn-primary" onClick={createPlan}>
                  Create Plan
                </button>
                <button type="button" className="btn" onClick={closeNewPlan}>
                  Cancel
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {deletePlanId ? (
          <div className="ws-modal-backdrop">
            <div className="ws-modal ws-modal-sm">
              <div className="ws-modal-header">
                <h3>Delete capacity plan?</h3>
              </div>
              <div className="ws-modal-body">
                <div className="text-muted">This will remove the selected capacity plan.</div>
                <div className="ws-delete-label">Plan ID: {deletePlanId}</div>
              </div>
              <div className="ws-modal-footer">
                <button type="button" className="btn btn-danger" onClick={confirmDelete}>
                  Delete
                </button>
                <button type="button" className="btn" onClick={() => setDeletePlanId(null)}>
                  Cancel
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {dimEditorOpen ? (
          <div className="ws-modal-backdrop">
            <div className="ws-modal">
              <div className="ws-modal-header">
                <h3>Manage dimensions</h3>
              </div>
              <div className="ws-modal-body">
                <div className="text-muted" style={{ marginBottom: 10 }}>
                  Custom dimensions are optional tags you can set on plans and filter by. They organise plans and do not
                  affect any calculation. Values are optional (leave blank to allow free text).
                </div>
                {dimEditorRows.map((row, idx) => (
                  <div key={idx} className="ws-dim-row">
                    <input
                      className="input"
                      type="text"
                      placeholder="Label (e.g. Tenure band)"
                      value={row.label || ""}
                      onChange={(e) =>
                        setDimEditorRows((prev) =>
                          prev.map((r, i) => (i === idx ? { ...r, label: e.target.value } : r))
                        )
                      }
                    />
                    <input
                      className="input"
                      type="text"
                      placeholder="Allowed values, comma-separated (optional)"
                      value={(row.values || []).join(", ")}
                      onChange={(e) =>
                        setDimEditorRows((prev) =>
                          prev.map((r, i) =>
                            i === idx
                              ? { ...r, values: e.target.value.split(",").map((v) => v.trim()).filter(Boolean) }
                              : r
                          )
                        )
                      }
                    />
                    <button
                      type="button"
                      className="btn btn-light"
                      aria-label="Remove dimension"
                      onClick={() => setDimEditorRows((prev) => prev.filter((_, i) => i !== idx))}
                    >
                      <Icon name="close" size={14} />
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  className="btn btn-light"
                  onClick={() => setDimEditorRows((prev) => [...prev, { key: "", label: "", values: [] }])}
                >
                  + Add dimension
                </button>
              </div>
              <div className="ws-modal-footer">
                <button type="button" className="btn btn-primary" onClick={saveDimRegistry} disabled={dimEditorBusy}>
                  {dimEditorBusy ? "Saving…" : "Save"}
                </button>
                <button type="button" className="btn" onClick={() => setDimEditorOpen(false)}>
                  Cancel
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {scopeTagsOpen ? (
          <div className="ws-modal-backdrop">
            <div className="ws-modal">
              <div className="ws-modal-header">
                <h3>Scope tags</h3>
              </div>
              <div className="ws-modal-body">
                <div className="text-muted" style={{ marginBottom: 10 }}>
                  Tag a demand scope (<code>business_area|sub_business_area|channel|site</code>) with custom dimension
                  values so capacity can be grouped by them (Plan Detail / BA rollup → Demand by Dimension). This does
                  not change any calculation.
                </div>
                {scopeTagsRows.map((row, idx) => (
                  <div key={idx} className="ws-scopetag-row">
                    <input
                      className="input"
                      type="text"
                      placeholder="Cards|Servicing|Voice|TX"
                      value={row.scope_key}
                      onChange={(e) =>
                        setScopeTagsRows((prev) =>
                          prev.map((r, i) => (i === idx ? { ...r, scope_key: e.target.value } : r))
                        )
                      }
                    />
                    {filterableDims.map((d) => (
                      <input
                        key={d.key}
                        className="input"
                        type="text"
                        placeholder={d.label}
                        value={row.dims[d.key] || ""}
                        onChange={(e) =>
                          setScopeTagsRows((prev) =>
                            prev.map((r, i) =>
                              i === idx ? { ...r, dims: { ...r.dims, [d.key]: e.target.value } } : r
                            )
                          )
                        }
                      />
                    ))}
                    <button
                      type="button"
                      className="btn btn-light"
                      aria-label="Remove scope tag"
                      onClick={() => setScopeTagsRows((prev) => prev.filter((_, i) => i !== idx))}
                    >
                      <Icon name="close" size={14} />
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  className="btn btn-light"
                  onClick={() => setScopeTagsRows((prev) => [...prev, { scope_key: "", dims: {} }])}
                >
                  + Add scope
                </button>
              </div>
              <div className="ws-modal-footer">
                <button type="button" className="btn btn-primary" onClick={saveScopeTags} disabled={scopeTagsBusy}>
                  {scopeTagsBusy ? "Saving…" : "Save"}
                </button>
                <button type="button" className="btn" onClick={() => setScopeTagsOpen(false)}>
                  Cancel
                </button>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </AppShell>
  );
}
