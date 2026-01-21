"use client";

import { useEffect, useState } from "react";
import { ActivityEntry, getActivity } from "../../lib/activity";

export default function UserTimeline() {
  const [expanded, setExpanded] = useState(false);
  const [items, setItems] = useState<ActivityEntry[]>([]);

  useEffect(() => {
    const refresh = () => setItems(getActivity());
    refresh();
    const handler = () => refresh();
    window.addEventListener("storage", handler);
    window.addEventListener("cap-activity", handler as EventListener);
    return () => {
      window.removeEventListener("storage", handler);
      window.removeEventListener("cap-activity", handler as EventListener);
    };
  }, []);

  const visible = expanded ? items.slice(0, 24) : items.slice(0, 6);

  return (
    <section className={`home-card timeline-card ${expanded ? "expanded" : "collapsed"}`}>
      <button type="button" className="timeline-header" onClick={() => setExpanded((prev: boolean) => !prev)}>
        <span>User Timeline</span>
        <span className="timeline-chevron">{expanded ? "▴" : "▾"}</span>
      </button>
      <div className="timeline-body">
        {visible.length === 0 ? (
          <div className="timeline-empty">No activity yet.</div>
        ) : (
          <ul className="timeline-list">
            {visible.map((item) => (
              <li key={item.id} className="timeline-item">
                <span className={`timeline-dot ${item.type}`} />
                <div className="timeline-content">
                  <div className="timeline-label">{item.label}</div>
                  <div className="timeline-meta">{formatTime(item.at)}</div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="timeline-legend">
        <span className="legend-item">
          <span className="timeline-dot visit" /> Pages visited
        </span>
        <span className="legend-item">
          <span className="timeline-dot change" /> Changes saved
        </span>
      </div>
    </section>
  );
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}
