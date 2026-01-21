"use client";

import type { ReactNode } from "react";
import { createContext, useContext, useMemo, useState } from "react";

type GlobalLoaderContextValue = {
  loading: boolean;
  setLoading: (value: boolean) => void;
};

const GlobalLoaderContext = createContext<GlobalLoaderContextValue | null>(null);

function GlobalLoaderOverlay({ active }: { active: boolean }) {
  return (
    <div className={`global-loader ${active ? "active" : ""}`}>
      <div className="global-loader-card">
        <span className="global-loader-spinner" />
        <div>
          <div className="global-loader-title">Processing</div>
          <div className="global-loader-subtitle">Please wait...</div>
        </div>
      </div>
    </div>
  );
}

export function GlobalLoaderProvider({ children }: { children: ReactNode }) {
  const [loading, setLoading] = useState(false);
  const value = useMemo(() => ({ loading, setLoading }), [loading]);

  return (
    <GlobalLoaderContext.Provider value={value}>
      {children}
      <GlobalLoaderOverlay active={loading} />
    </GlobalLoaderContext.Provider>
  );
}

export function useGlobalLoader() {
  const ctx = useContext(GlobalLoaderContext);
  if (!ctx) {
    throw new Error("useGlobalLoader must be used within GlobalLoaderProvider.");
  }
  return ctx;
}
