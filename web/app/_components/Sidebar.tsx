"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

const navItems = [
  { href: "/forecast", label: "Forecasting", icon: "📈" },
  { href: "/planning", label: "Planning", icon: "📅" },
  { href: "/budget", label: "Budget", icon: "💰" },
  { href: "/ops", label: "Operational Dashboard", icon: "📊" },
  { href: "/new-hire", label: "New Hire Summary", icon: "🧑‍💼" },
  { href: "/roster", label: "Employee Roster", icon: "🗂️" },
  { href: "/dataset", label: "Planner Dataset", icon: "🧮" },
  { href: "/settings", label: "Default Settings", icon: "⚙️" },
  { href: "/shrinkage", label: "Upload Shrinkage", icon: "📤" },
  { href: "/help", label: "Help & Docs", icon: "ℹ️" }
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
              {item.icon}
            </span>
            {!collapsed && <span className="nav-text">{item.label}</span>}
          </Link>
        ))}
      </nav>
    </aside>
  );
}
