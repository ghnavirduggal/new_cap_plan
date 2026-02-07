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
  crumbLinks?: Record<string, string>;
  children: ReactNode;
}

export default function AppShell({ title, crumbs, crumbIcon, userLabel, crumbLinks, children }: AppShellProps) {
  const crumbText = crumbs ?? title ?? "";
  const crumbParts = crumbText
    .split("/")
    .map((part) => part.trim())
    .filter(Boolean);
  const [collapsed, setCollapsed] = useState<boolean>(true);
  const [resolvedUser, setResolvedUser] = useState<string>("");
  const [resolvedEmail, setResolvedEmail] = useState<string>("");
  const [resolvedPhoto, setResolvedPhoto] = useState<string>("");
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
    apiGet<{ name?: string; email?: string; photo_url?: string }>("/api/user")
      .then((data) => {
        if (!active) return;
        const nextName = data?.name?.trim() || data?.email?.trim();
        setResolvedUser(nextName || "system");
        setResolvedEmail((data?.email || "").trim());
        setResolvedPhoto((data?.photo_url || "").trim());
      })
      .catch(() => {
        if (!active) return;
        setResolvedUser("system");
        setResolvedEmail("");
        setResolvedPhoto("");
      });
    return () => {
      active = false;
    };
  }, [userLabel]);

  const toggleSidebar = () => {
    setCollapsed((prev: boolean) => !prev);
  };
  
  const initials = (resolvedUser || "S").split(/\s+/).map(s => s[0]?.toUpperCase() || "").slice(0, 2).join("");
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
                {crumbParts.length === 0 ? (
                  <Link className="crumbs-link" href="/">
                    CAP-CONNECT
                  </Link>
                ) : (
                  <div className="crumbs-links">
                    {crumbParts.map((part, idx) => {
                      const isFirst = idx === 0;
                      const isLast = idx === crumbParts.length - 1;
                      const key = `${part}-${idx}`;
                      if (isFirst) {
                        return (
                          <span key={key}>
                            <Link className="crumbs-link" href="/">
                              {part}
                            </Link>
                            {!isLast ? " / " : null}
                          </span>
                        );
                      }
                      const mappedLink = crumbLinks?.[part];
                      if (mappedLink) {
                        return (
                          <span key={key}>
                            <Link className="crumbs-link" href={mappedLink}>
                              {part}
                            </Link>
                            {!isLast ? " / " : null}
                          </span>
                        );
                      }
                      if (part.toLowerCase() === "planning workspace") {
                        return (
                          <span key={key}>
                            <Link className="crumbs-link" href="/planning">
                              {part}
                            </Link>
                            {!isLast ? " / " : null}
                          </span>
                        );
                      }
                      return (
                        <span key={key} className="crumbs-label">
                          {part}
                          {!isLast ? " / " : null}
                        </span>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="topbar-user" title={resolvedEmail || resolvedUser || "system"}>
            {resolvedPhoto ? (
              <img src={resolvedPhoto} alt={resolvedUser || "user"} className="avatar" referrerPolicy="no-referrer"/>
            ) : (
              <div className="avatar avatar-fallback">{initials}</div>
            )}
            <div className="user-meta">
              <div className="user-name">{resolvedUser || "system"}</div>
              {resolvedEmail ? <div className="user-email">{resolvedEmail}</div> : null}
            </div>
          </div>
        </div>
        {children}
      </div>
      <style jsx global>{`
        .topbar-user {display: flex; align-items: center; gap: 0.5rem; }
        .avatar {width: 28px; height: 28px; border-radius: 50%; object-fit: cover; }
        .avatar-fallback {width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; background: #0b5fff20; color: #0b5fff; font-size: 0.8rem; }
        .user-meta {display: flex; flex-direction: column; line-height: 1.1; }
        .user-name {font-weight: 600; }
        .user-email {font-size: 0.8rem; opacity: 0.75; }
        @media (min-width: 640px) {
          .user-meta {display: flex; }
        }
      `}</style>
    </div>
  );
}
