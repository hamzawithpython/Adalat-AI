"use client";

import { cn } from "@/lib/utils";

interface ConfidenceIndicatorProps {
  value: number; // 0.0 – 1.0
}

function tone(pct: number) {
  if (pct >= 80) return { ring: "stroke-green-600", text: "text-green-700", bg: "bg-green-50", label: "HIGH" };
  if (pct >= 55) return { ring: "stroke-amber-500", text: "text-amber-700", bg: "bg-amber-50", label: "MEDIUM" };
  return { ring: "stroke-red-600", text: "text-red-700", bg: "bg-red-50", label: "LOW" };
}

export function ConfidenceIndicator({ value }: ConfidenceIndicatorProps) {
  const pct = Math.round(value * 100);
  const t = tone(pct);
  const r = 20;
  const circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;

  return (
    <div className={cn("inline-flex items-center gap-3 rounded-lg px-3.5 py-2", t.bg)}>
      <svg width="48" height="48" viewBox="0 0 48 48">
        <circle cx="24" cy="24" r={r} fill="none" className="stroke-slate-200" strokeWidth="4" />
        <circle
          cx="24"
          cy="24"
          r={r}
          fill="none"
          className={cn(t.ring, "transition-all duration-700")}
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          transform="rotate(-90 24 24)"
        />
      </svg>
      <div>
        <div className={cn("text-lg font-semibold leading-none", t.text)}>{pct}%</div>
        <div className={cn("text-[10px] font-mono uppercase tracking-widest mt-1", t.text)}>
          {t.label} CONFIDENCE
        </div>
      </div>
    </div>
  );
}
