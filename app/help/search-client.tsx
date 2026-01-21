"use client";

import { useEffect, useRef, useState } from "react";

export default function HelpSearch() {
  const [open, setOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const timer = window.setTimeout(() => {
      inputRef.current?.focus();
    }, 120);
    return () => window.clearTimeout(timer);
  }, [open]);

  return (
    <div className={`help-search-wrap ${open ? "open" : ""}`}>
      <button
        type="button"
        className="help-search-btn"
        aria-label="Search help"
        onClick={() => setOpen((prev) => !prev)}
      >
        🔍
      </button>
      <div className="help-search-panel">
        <input ref={inputRef} type="text" placeholder="Search help..." />
      </div>
    </div>
  );
}
