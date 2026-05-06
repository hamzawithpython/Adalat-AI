"use client";

import { useState } from "react";
import type { Right } from "@/types/legal";
import { cn } from "@/lib/utils";

interface RightsCardsProps {
  rights: Right[];
}

export function RightsCards({ rights }: RightsCardsProps) {
  const [expanded, setExpanded] = useState<number>(0);

  if (rights.length === 0) {
    return (
      <p className="text-sm text-slate-400 italic">
        No structured rights extracted for this query.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {rights.map((r, i) => {
        const isOpen = expanded === i;
        return (
          <div
            key={i}
            className={cn(
              "border rounded-lg bg-white transition-all overflow-hidden",
              isOpen ? "border-navy shadow-brand" : "border-slate-200"
            )}
          >
            <button
              onClick={() => setExpanded(isOpen ? -1 : i)}
              className="w-full flex items-center gap-3 px-4 py-3.5 text-left hover:bg-slate-50 transition-colors"
            >
              <span
                className={cn(
                  "flex items-center justify-center w-7 h-7 rounded-full text-xs font-mono font-bold shrink-0",
                  isOpen ? "bg-navy text-white" : "bg-gold-faint text-gold-dark"
                )}
              >
                {i + 1}
              </span>
              <span className="flex-1 text-[15px] font-semibold text-navy">
                {r.right}
              </span>
              <span
                className={cn(
                  "text-slate-400 transition-transform shrink-0",
                  isOpen && "rotate-180"
                )}
              >
                ▾
              </span>
            </button>

            {isOpen && (
              <div className="px-4 pb-4 pt-1 space-y-3 border-t border-slate-100">
                <RightField label="Legal basis" value={r.legal_basis} mono />
                {r.deadline && (
                  <RightField label="Deadline" value={r.deadline} highlight />
                )}
                <RightField label="Recourse" value={r.recourse} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function RightField({
  label,
  value,
  mono,
  highlight,
}: {
  label: string;
  value: string;
  mono?: boolean;
  highlight?: boolean;
}) {
  return (
    <div>
      <div className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-1">
        {label}
      </div>
      <div
        className={cn(
          "text-sm leading-relaxed",
          mono && "font-mono text-navy",
          highlight && "text-amber-800 font-medium",
          !mono && !highlight && "text-slate-700"
        )}
      >
        {value}
      </div>
    </div>
  );
}
