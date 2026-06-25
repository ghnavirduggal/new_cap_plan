"use client";

import type { CSSProperties } from "react";

/**
 * Lightweight inline-SVG icon set (Lucide-style, 24×24, stroke = currentColor).
 * Replaces emoji "icons" across the app for a crisp, professional, scalable look
 * that inherits text color and respects accessibility. Add new glyphs to PATHS.
 */
export type IconName =
  | "forecast"
  | "target"
  | "calendar"
  | "wallet"
  | "dashboard"
  | "user-plus"
  | "users"
  | "database"
  | "settings"
  | "upload"
  | "info"
  | "phone"
  | "briefcase"
  | "chat"
  | "message"
  | "megaphone"
  | "shuffle"
  | "mail"
  | "globe"
  | "search"
  | "close"
  | "plus"
  | "check"
  | "alert"
  | "trend-up"
  | "trend-down"
  | "sparkles"
  | "bolt"
  | "layers"
  | "filter"
  | "tag"
  | "chevron-right"
  | "external"
  | "menu"
  | "user"
  | "logout"
  | "home"
  | "save"
  | "trash"
  | "edit"
  | "refresh"
  | "download"
  | "pin"
  | "compare"
  | "arrow-left"
  | "chevron-left";

const PATHS: Record<IconName, string> = {
  forecast: "M3 3v18h18 M7 14l3-3 3 3 5-6",
  target: "M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0-18 0 M12 12m-5 0a5 5 0 1 0 10 0a5 5 0 1 0-10 0 M12 12m-1 0a1 1 0 1 0 2 0a1 1 0 1 0-2 0",
  calendar: "M3 8h18 M8 3v4 M16 3v4 M5 5h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2",
  wallet: "M3 7a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v0H5a2 2 0 0 0-2 2 M3 9a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z M16 13h2",
  dashboard: "M4 13h6V4H4z M14 9h6V4h-6z M14 20h6v-9h-6z M4 20h6v-5H4z",
  "user-plus": "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2 M9 11m-4 0a4 4 0 1 0 8 0a4 4 0 1 0-8 0 M19 8v6 M22 11h-6",
  users: "M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2 M9 7m-4 0a4 4 0 1 0 8 0a4 4 0 1 0-8 0 M23 21v-2a4 4 0 0 0-3-3.87 M16 3.13a4 4 0 0 1 0 7.75",
  database: "M12 5m-8 0a8 3 0 1 0 16 0a8 3 0 1 0-16 0 M4 5v6a8 3 0 0 0 16 0V5 M4 11v6a8 3 0 0 0 16 0v-6",
  settings: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z",
  upload: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M17 8l-5-5-5 5 M12 3v12",
  info: "M12 12m-10 0a10 10 0 1 0 20 0a10 10 0 1 0-20 0 M12 16v-4 M12 8h.01",
  phone: "M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.81.36 1.6.7 2.34a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.74-1.27a2 2 0 0 1 2.11-.45c.74.34 1.53.57 2.34.7A2 2 0 0 1 22 16.92z",
  briefcase: "M4 7h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2 M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2 M2 12h20",
  chat: "M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z",
  message: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",
  megaphone: "M3 11l18-5v12L3 14v-3z M11.6 16.8a3 3 0 1 1-5.8-1.6",
  shuffle: "M16 3h5v5 M4 20L21 3 M21 16v5h-5 M15 15l6 6 M4 4l5 5",
  mail: "M4 4h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2 M22 6l-10 7L2 6",
  globe: "M12 12m-10 0a10 10 0 1 0 20 0a10 10 0 1 0-20 0 M2 12h20 M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z",
  search: "M11 11m-8 0a8 8 0 1 0 16 0a8 8 0 1 0-16 0 M21 21l-4.35-4.35",
  close: "M18 6L6 18 M6 6l12 12",
  plus: "M12 5v14 M5 12h14",
  check: "M20 6L9 17l-5-5",
  alert: "M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z M12 9v4 M12 17h.01",
  "trend-up": "M23 6l-9.5 9.5-5-5L1 18 M17 6h6v6",
  "trend-down": "M23 18l-9.5-9.5-5 5L1 6 M17 18h6v-6",
  sparkles: "M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z M19 4v3 M5 17v2 M20 18h-2",
  bolt: "M13 2L3 14h7l-1 8 10-12h-7z",
  layers: "M12 2l9 5-9 5-9-5 9-5z M3 12l9 5 9-5 M3 17l9 5 9-5",
  filter: "M22 3H2l8 9.46V19l4 2v-8.54z",
  tag: "M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z M7 7h.01",
  "chevron-right": "M9 18l6-6-6-6",
  external: "M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6 M15 3h6v6 M10 14L21 3",
  menu: "M3 6h18 M3 12h18 M3 18h18",
  user: "M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2 M12 7m-4 0a4 4 0 1 0 8 0a4 4 0 1 0-8 0",
  logout: "M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4 M16 17l5-5-5-5 M21 12H9",
  home: "M3 11l9-8 9 8 M5 9v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9",
  save: "M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z M17 21v-8H7v8 M7 3v5h8",
  trash: "M3 6h18 M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2 M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6 M10 11v6 M14 11v6",
  edit: "M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7 M18.5 2.5a2.12 2.12 0 0 1 3 3L12 15l-4 1 1-4z",
  refresh: "M23 4v6h-6 M1 20v-6h6 M3.51 9a9 9 0 0 1 14.85-3.36L23 10 M1 14l4.64 4.36A9 9 0 0 0 20.49 15",
  download: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M7 10l5 5 5-5 M12 15V3",
  pin: "M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z M12 10m-3 0a3 3 0 1 0 6 0a3 3 0 1 0-6 0",
  compare: "M9 3v18 M15 3v18 M3 12h18 M5 7l2 0 M5 17l2 0 M17 7l2 0 M17 17l2 0",
  "arrow-left": "M19 12H5 M12 19l-7-7 7-7",
  "chevron-left": "M15 18l-6-6 6-6",
};

type IconProps = {
  name: IconName;
  size?: number;
  strokeWidth?: number;
  className?: string;
  style?: CSSProperties;
  title?: string;
};

export default function Icon({ name, size = 18, strokeWidth = 2, className, style, title }: IconProps) {
  const d = PATHS[name];
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      style={style}
      role={title ? "img" : "presentation"}
      aria-hidden={title ? undefined : true}
      aria-label={title}
      focusable="false"
    >
      {title ? <title>{title}</title> : null}
      {d.split(" M").map((seg, i) => (
        <path key={i} d={i === 0 ? seg : `M${seg}`} />
      ))}
    </svg>
  );
}

/** Map a channel label to its icon (used by planning / plan cards). */
export function channelIconName(label?: string): IconName {
  const key = String(label || "").trim().toLowerCase();
  if (key.includes("voice") || key.includes("phone") || key.includes("call")) return "phone";
  if (key.includes("back")) return "briefcase";
  if (key.includes("chat")) return "chat";
  if (key.includes("message")) return "message";
  if (key.includes("out")) return "megaphone";
  if (key.includes("blend")) return "shuffle";
  if (key.includes("mail") || key.includes("email")) return "mail";
  if (key.includes("omni")) return "globe";
  return "users";
}
