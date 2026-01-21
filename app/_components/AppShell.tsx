"use client";

import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Sidebar from "./Sidebar";
import { logPageVisit } from "../../lib/activity";
import { apiGet } from "../../lib/api";

interface AppShellProps {
  title?: string;
  crumbs?: string;
  crumbIcon?: string;
  userLabel?: string;
  children: ReactNode;
}

export default function AppShell({ title, crumbs, crumbIcon, userLabel, children }: AppShellProps) {
  const crumbText = crumbs ?? title ?? "";
  const [collapsed, setCollapsed] = useState<boolean>(true);
  const [resolvedUser, setResolvedUser] = useState<string>("");
  const pathname = usePathname();

  useEffect(() => {
    const label = title || crumbText || pathname;
    logPageVisit(pathname, label || "Page");
  }, [pathname, title, crumbText]);

  useEffect(() => {
    if (userLabel) {
      setResolvedUser(userLabel);
      return;
    }
    let active = true;
    apiGet<{ name: string }>("/api/user")
      .then((data) => {
        if (!active) return;
        const nextName = data?.name?.trim();
        setResolvedUser(nextName || window.location.hostname || "system");
      })
      .catch(() => {
        if (!active) return;
        setResolvedUser(window.location.hostname || "system");
      });
    return () => {
      active = false;
    };
  }, [userLabel]);

  const toggleSidebar = () => {
    setCollapsed((prev: boolean) => !prev);
  };
  return (
    <div className={`app-shell ${collapsed ? "sidebar-collapsed" : "sidebar-expanded"}`}>
      <Sidebar collapsed={collapsed} />
      <div className="main">
        <div className="topbar">
          <div className="topbar-left">
            <button className="burger" type="button" aria-label="Toggle sidebar" onClick={toggleSidebar}>
              ☰
            </button>
            <div>
              <div className="crumbs-line">
                {crumbIcon ? <span className="crumb-icon">{crumbIcon}</span> : null}
                <Link className="crumbs-link" href="/">
                  {crumbText || "CAP-CONNECT"}
                </Link>
              </div>
            </div>
          </div>
          <div className="topbar-user">{resolvedUser || "system"}</div>
        </div>
        {children}
      </div>
    </div>
  );
}
