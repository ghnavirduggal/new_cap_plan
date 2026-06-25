"use client";

import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Sidebar from "./Sidebar";
import Icon from "./Icon";
import { logPageVisit } from "../../lib/activity";
import { apiGet, apiPatch, clearAuthToken } from "../../lib/api";

interface AppShellProps {
  title?: string;
  crumbs?: string;
  crumbIcon?: ReactNode;
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
  // Full /api/user payload so the profile panel can show org details that were
  // populated at onboarding (business area, role, location, manager, …).
  const [userInfo, setUserInfo] = useState<Record<string, any>>({});
  // Client-side profile overrides (display name + photo). There is no
  // user-update backend, so edits persist in localStorage keyed by email.
  const [localPhoto, setLocalPhoto] = useState<string>("");
  const [localName, setLocalName] = useState<string>("");
  const [menuOpen, setMenuOpen] = useState<boolean>(false);
  const [profileOpen, setProfileOpen] = useState<boolean>(false);
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
    apiGet<Record<string, any>>("/api/user")
      .then((data) => {
        if (!active) return;
        const nextName = data?.name?.trim?.() || data?.email?.trim?.();
        setResolvedUser(nextName || "system");
        setResolvedEmail((data?.email || "").trim());
        setResolvedPhoto((data?.photo_url || "").trim());
        setUserInfo(data && typeof data === "object" ? data : {});
      })
      .catch(() => {
        if (!active) return;
        setResolvedUser("system");
        setResolvedEmail("");
        setResolvedPhoto("");
        setUserInfo({});
      });
    return () => {
      active = false;
    };
  }, [userLabel]);

  const [profileSaving, setProfileSaving] = useState<boolean>(false);
  const [profileError, setProfileError] = useState<string>("");

  const toggleSidebar = () => {
    setCollapsed((prev: boolean) => !prev);
  };
  
  const effectiveName = resolvedUser || "system";
  const effectivePhoto = resolvedPhoto;
  const initials = (effectiveName || "S").split(/\s+/).map(s => s[0]?.toUpperCase() || "").slice(0, 2).join("");
  // Preview avatar inside the profile modal uses the in-progress edit buffer.
  const previewPhoto = localPhoto;
  const previewInitials = ((localName || effectiveName) || "S").split(/\s+/).map(s => s[0]?.toUpperCase() || "").slice(0, 2).join("");

  // Org details captured at onboarding, shown read-only in the profile panel.
  const orgFields: Array<{ label: string; value: string }> = [
    { label: "Business Area", value: userInfo.business_area || userInfo.vertical || "" },
    { label: "Sub Business Area", value: userInfo.sub_business_area || userInfo.sub_ba || "" },
    { label: "Role", value: userInfo.role || userInfo.title || userInfo.designation || "" },
    { label: "Manager", value: userInfo.manager || userInfo.reports_to || "" },
    { label: "Location", value: userInfo.location || userInfo.site || userInfo.country || "" }
  ].filter((f) => String(f.value || "").trim());

  const onPickPhoto = (file?: File | null) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setLocalPhoto(String(reader.result || ""));
    reader.readAsDataURL(file);
  };

  const openProfile = () => {
    setLocalName(resolvedUser);
    setLocalPhoto(resolvedPhoto);
    setProfileError("");
    setMenuOpen(false);
    setProfileOpen(true);
  };

  const saveProfile = async () => {
    setProfileSaving(true);
    setProfileError("");
    try {
      const updated = await apiPatch<Record<string, any>>("/api/user", {
        name: localName,
        photo_url: localPhoto
      });
      // Reflect the server's persisted truth in the topbar immediately.
      setResolvedUser((updated?.name || localName || resolvedUser || "system").trim());
      setResolvedPhoto((updated?.photo_url || "").trim());
      if (updated && typeof updated === "object") setUserInfo(updated);
      setProfileOpen(false);
    } catch (err: any) {
      setProfileError(err?.message || "Could not save profile.");
    } finally {
      setProfileSaving(false);
    }
  };

  const signOut = () => {
    setMenuOpen(false);
    if (typeof window === "undefined") return;
    const logoutUrl = process.env.NEXT_PUBLIC_LOGOUT_URL;
    try {
      window.sessionStorage.clear();
      clearAuthToken();
    } catch {
      /* ignore */
    }
    // Reverse-proxy SSO deployments set NEXT_PUBLIC_LOGOUT_URL (e.g.
    // /oauth2/sign_out). Without it, just return to the home page.
    window.location.href = logoutUrl || "/";
  };

  return (
    <div className={`app-shell ${collapsed ? "sidebar-collapsed" : "sidebar-expanded"}`}>
      <Sidebar collapsed={collapsed} />
      <div className="main">
        <div className="topbar">
          <div className="topbar-left">
            <button className="burger" type="button" aria-label="Toggle sidebar" onClick={toggleSidebar}>
              <Icon name="menu" size={20} />
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
          <div className="topbar-user-wrap">
            <button
              type="button"
              className="topbar-user"
              title={resolvedEmail || effectiveName}
              onClick={() => setMenuOpen((v) => !v)}
              aria-haspopup="menu"
              aria-expanded={menuOpen}
            >
              {effectivePhoto ? (
                <img src={effectivePhoto} alt={effectiveName} className="avatar" referrerPolicy="no-referrer" />
              ) : (
                <div className="avatar avatar-fallback">{initials}</div>
              )}
              <div className="user-meta">
                <div className="user-name">{effectiveName}</div>
                {resolvedEmail ? <div className="user-email">{resolvedEmail}</div> : null}
              </div>
              <svg className="user-caret" width="12" height="12" viewBox="0 0 12 12" aria-hidden>
                <path d="M2 4 L6 8 L10 4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            {menuOpen ? (
              <>
                <div className="user-menu-backdrop" onClick={() => setMenuOpen(false)} />
                <div className="user-menu" role="menu">
                  <button type="button" className="user-menu-item" role="menuitem" onClick={openProfile}>
                    <span className="user-menu-icon"><Icon name="user" size={16} /></span> Profile
                  </button>
                  <button type="button" className="user-menu-item user-menu-item--danger" role="menuitem" onClick={signOut}>
                    <span className="user-menu-icon"><Icon name="logout" size={16} /></span> Sign out
                  </button>
                </div>
              </>
            ) : null}
          </div>
        </div>

        {profileOpen ? (
          <div className="ws-modal-backdrop" onClick={() => setProfileOpen(false)}>
            <div className="ws-modal ws-modal-sm" onClick={(e) => e.stopPropagation()}>
              <div className="ws-modal-header">
                <h3>My Profile</h3>
                <button type="button" className="btn btn-light closeOptions" onClick={() => setProfileOpen(false)}>
                  <svg width="16" height="16" viewBox="0 0 16 16">
                    <line x1="2" y1="2" x2="14" y2="14" stroke="white" strokeWidth="2" />
                    <line x1="14" y1="2" x2="2" y2="14" stroke="white" strokeWidth="2" />
                  </svg>
                </button>
              </div>
              <div className="ws-modal-body">
                <div className="profile-photo-row">
                  {previewPhoto ? (
                    <img src={previewPhoto} alt={localName || effectiveName} className="profile-photo" />
                  ) : (
                    <div className="profile-photo profile-photo--fallback">{previewInitials}</div>
                  )}
                  <div className="profile-photo-actions">
                    <label className="btn btn-light">
                      Change photo
                      <input
                        type="file"
                        accept="image/*"
                        className="file-input"
                        onChange={(e) => onPickPhoto(e.target.files?.[0])}
                      />
                    </label>
                    {localPhoto ? (
                      <button type="button" className="btn btn-ghost" onClick={() => setLocalPhoto("")}>
                        Remove
                      </button>
                    ) : null}
                  </div>
                </div>

                <label className="profile-field">
                  Display Name
                  <input
                    className="input"
                    value={localName}
                    onChange={(e) => setLocalName(e.target.value)}
                    placeholder={resolvedUser}
                  />
                </label>
                <label className="profile-field">
                  Email
                  <input className="input" value={resolvedEmail} readOnly disabled />
                </label>

                {orgFields.length ? (
                  <div className="profile-org">
                    <div className="profile-org-title">From your organization (onboarding)</div>
                    {orgFields.map((f) => (
                      <div className="profile-org-row" key={f.label}>
                        <span className="profile-org-label">{f.label}</span>
                        <span className="profile-org-value">{f.value}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="profile-org-empty">Organization details will appear here once provided at onboarding.</div>
                )}
              </div>
              <div className="ws-modal-footer">
                {profileError ? <span className="plan-message plan-message--error">{profileError}</span> : null}
                <button type="button" className="btn btn-light" onClick={() => setProfileOpen(false)} disabled={profileSaving}>
                  Cancel
                </button>
                <button type="button" className="btn btn-primary" onClick={saveProfile} disabled={profileSaving}>
                  {profileSaving ? "Saving…" : "Save"}
                </button>
              </div>
            </div>
          </div>
        ) : null}
        {children}
      </div>
      <style jsx global>{`
        .topbar-user {display: flex; align-items: center; gap: 0.5rem; background: transparent; border: 1px solid transparent; cursor: pointer; padding: 4px 8px; border-radius: 10px; color: inherit; font: inherit; transition: background 150ms ease, border-color 150ms ease; }
        .topbar-user:hover {background: #f8fafc; border-color: var(--border); }
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
