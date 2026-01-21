"use client";

import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useMemo, useState } from "react";

type ToastType = "success" | "error" | "warning";

type Toast = {
  id: string;
  type: ToastType;
  message: string;
};

type ToastContextValue = {
  notify: (type: ToastType, message: string) => void;
};

const ToastContext = createContext<ToastContextValue>({
  notify: () => {}
});

function buildId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function useToast() {
  return useContext(ToastContext);
}

export default function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const notify = useCallback((type: ToastType, message: string) => {
    const id = buildId();
    setToasts((prev) => [...prev, { id, type, message }]);
    window.setTimeout(() => removeToast(id), 4000);
  }, [removeToast]);

  const value = useMemo(() => ({ notify }), [notify]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="toast-container" role="status" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast toast-${toast.type}`}>
            <div className="toast-message">{toast.message}</div>
            <button
              className="toast-close"
              type="button"
              aria-label="Dismiss notification"
              onClick={() => removeToast(toast.id)}
            >
              x
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
