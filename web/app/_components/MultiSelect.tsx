"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Option = {
  label: string;
  value: string;
};

type MultiSelectProps = {
  options: Option[];
  values: string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
};

export default function MultiSelect({ options, values, onChange, placeholder, className, disabled }: MultiSelectProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  const labelsByValue = useMemo(() => new Map(options.map((opt) => [opt.value, opt.label])), [options]);
  const selectedLabels = useMemo(
    () => values.map((val) => labelsByValue.get(val) || val).filter(Boolean),
    [labelsByValue, values]
  );

  const displayText = useMemo(() => {
    if (!selectedLabels.length) return placeholder || "Select";
    if (selectedLabels.length <= 2) return selectedLabels.join(", ");
    return `${selectedLabels.length} selected`;
  }, [selectedLabels, placeholder]);

  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      if (!ref.current || ref.current.contains(event.target as Node)) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const toggleValue = (value: string) => {
    const nextSet = new Set(values);
    if (nextSet.has(value)) {
      nextSet.delete(value);
    } else {
      nextSet.add(value);
    }
    const ordered = options.map((opt) => opt.value).filter((val) => nextSet.has(val));
    onChange(ordered);
  };

  return (
    <div ref={ref} className={`multi-select ${className ?? ""}`.trim()}>
      <button
        type="button"
        className={`multi-select-button${open ? " open" : ""}`}
        onClick={() => setOpen((prev) => !prev)}
        disabled={disabled}
      >
        <span className={selectedLabels.length ? "multi-select-value" : "multi-select-placeholder"}>
          {displayText}
        </span>
        <span className="multi-select-caret">▾</span>
      </button>
      {open ? (
        <div className="multi-select-menu">
          {options.length ? (
            options.map((opt) => (
              <label key={opt.value} className="multi-select-option">
                <input
                  type="checkbox"
                  checked={values.includes(opt.value)}
                  onChange={() => toggleValue(opt.value)}
                />
                <span>{opt.label}</span>
              </label>
            ))
          ) : (
            <div className="multi-select-empty">No options</div>
          )}
        </div>
      ) : null}
    </div>
  );
}
