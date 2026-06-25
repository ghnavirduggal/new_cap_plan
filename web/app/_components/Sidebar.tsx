"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import Icon, { type IconName } from "./Icon";

const navItems: { href: string; label: string; icon: IconName }[] = [
  { href: "/forecast", label: "Forecasting", icon: "forecast" },
  { href: "/forecast/accuracy", label: "Forecast Accuracy", icon: "target" },
  { href: "/planning", label: "Planning", icon: "calendar" },
  { href: "/budget", label: "Budget", icon: "wallet" },
  { href: "/ops", label: "Operational Dashboard", icon: "dashboard" },
  { href: "/new-hire", label: "New Hire Summary", icon: "user-plus" },
  { href: "/roster", label: "Employee Roster", icon: "users" },
  { href: "/dataset", label: "Planner Dataset", icon: "database" },
  { href: "/settings", label: "Default Settings", icon: "settings" },
  { href: "/shrinkage", label: "Upload Shrinkage", icon: "upload" },
  { href: "/help", label: "Help & Docs", icon: "info" }
];

export default function Sidebar({ collapsed }: { collapsed: boolean }) {
  const pathname = usePathname();
  return (
    <aside className={`sidebar ${collapsed ? "collapsed" : "expanded"}`}>
      <div className="brand" aria-label="Cap-Connect">
        <div className="logo-full">
          <img src="/assets/logo2.png" alt="Cap-Connect" />
        </div>
        <img src="/assets/logo1.png" alt="Cap-Connect" className="logo-eagle" />
      </div>
      <nav className="nav-list">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={clsx("nav-item", pathname === item.href && "active")}
            title={item.label}
          >
            <span className="nav-icon" aria-hidden>
              <Icon name={item.icon} size={20} />
            </span>
            {/* Always rendered so the label fades/clips with the slide instead
                of popping in and out when the sidebar collapses. */}
            <span className="nav-text">{item.label}</span>
          </Link>
        ))}
      </nav>
    </aside>
  );
}
