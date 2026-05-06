"use client";

import { Btn } from "@/components/ui/btn";
import { useEffect, useState } from "react";

interface QueryInputProps {
  initialValue?: string;
  onSubmit?: (query: string) => void;
  disabled?: boolean;
}

export function QueryInput({
  initialValue = "",
  onSubmit,
  disabled = false,
}: QueryInputProps) {
  const [value, setValue] = useState(initialValue);

  // Sync internal state when parent passes a new initialValue
  // (e.g., user clicked a sample query, or input was cleared after submit)
  useEffect(() => {
    setValue(initialValue);
  }, [initialValue]);

  const charCount = value.length;
  const maxChars = 4000;

  const handleSubmit = () => {
    if (!value.trim() || disabled) return;
    onSubmit?.(value.trim());
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Cmd/Ctrl + Enter submits
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-slate-200 bg-white p-4">
      <div className="max-w-3xl mx-auto">
        <div className="rounded-xl border border-slate-200 bg-white shadow-brand overflow-hidden focus-within:border-navy focus-within:shadow-brand-md transition-all">
          <textarea
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder="Ask a legal question in Roman-Urdu, English, or German..."
            rows={3}
            maxLength={maxChars}
            className="w-full px-4 py-3 text-[15px] resize-none focus:outline-none placeholder:text-slate-400 disabled:bg-slate-50 disabled:cursor-not-allowed"
          />
          <div className="flex items-center justify-between px-3 py-2 border-t border-slate-100 bg-slate-50">
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-white border border-slate-200 font-medium">
                🌐 Auto-detect
              </span>
              <span className="font-mono">
                {charCount} / {maxChars}
              </span>
            </div>
            <Btn
              variant="primary"
              size="sm"
              onClick={handleSubmit}
              disabled={!value.trim() || disabled}
              iconRight={<span>→</span>}
            >
              Ask Adalat
            </Btn>
          </div>
        </div>
        <p className="text-[11px] text-slate-400 mt-2 text-center font-mono">
          Press Cmd/Ctrl + Enter to send · Replies typically take 10–30 seconds
        </p>
      </div>
    </div>
  );
}