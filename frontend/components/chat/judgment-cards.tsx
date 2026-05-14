"use client";

import { useState } from "react";
import type { Judgment } from "@/types/legal";
import { GavelIcon } from "@/components/icons/legal-icons";
import { Markdown } from "@/components/ui/markdown";
import { cn } from "@/lib/utils";

interface JudgmentCardsProps {
  judgments: Judgment[];
}

export function JudgmentCards({ judgments }: JudgmentCardsProps) {
  if (!judgments || judgments.length === 0) return null;

  return (
    <div className="space-y-2.5">
      {judgments.map((j, i) => (
        <JudgmentCard key={`${j.principle}-${i}`} judgment={j} index={i} />
      ))}
    </div>
  );
}

function JudgmentCard({ judgment, index }: { judgment: Judgment; index: number }) {
  const [open, setOpen] = useState(false);

  return (
    <div
      className={cn(
        "bg-white border rounded-lg overflow-hidden transition-all",
        open ? "border-navy shadow-brand" : "border-slate-200"
      )}
    >
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full text-left flex items-start gap-3 px-3 sm:px-4 py-3.5 hover:bg-slate-50 transition-colors"
      >
        <div
          className={cn(
            "shrink-0 flex items-center justify-center w-8 h-8 rounded-md border",
            open
              ? "bg-navy border-navy text-white"
              : "bg-gold-faint border-gold-soft text-gold-dark"
          )}
        >
          <GavelIcon size={16} className="text-current" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-[10px] font-mono uppercase tracking-widest text-slate-500">
              Principle {String(index + 1).padStart(2, "0")}
            </span>
          </div>
          <h4 className="font-serif font-bold text-[16px] text-navy leading-snug">
            {judgment.principle}
          </h4>
          {judgment.typical_outcome && (
            <div className="text-[12px] text-slate-600 mt-1.5 line-clamp-2">
              <span className="font-semibold text-slate-700">Typical outcome:</span>{" "}
              {judgment.typical_outcome}
            </div>
          )}
        </div>

        <span
          className={cn(
            "shrink-0 text-slate-400 transition-transform mt-1.5",
            open && "rotate-180"
          )}
          aria-hidden="true"
        >
          ▾
        </span>
      </button>

      {open && (
        <div className="px-3 sm:px-4 pb-3 sm:pb-4 pt-1 border-t border-slate-100 space-y-3">
          <JudgmentField label="Summary">
            <Markdown content={judgment.summary} />
          </JudgmentField>

          {judgment.typical_outcome && (
            <JudgmentField label="Typical outcome">
              <p className="text-sm text-slate-700 leading-relaxed">
                {judgment.typical_outcome}
              </p>
            </JudgmentField>
          )}

          {judgment.relevant_sections && judgment.relevant_sections.length > 0 && (
            <JudgmentField label="Relevant sections">
              <div className="flex flex-wrap gap-1.5">
                {judgment.relevant_sections.map((s, i) => (
                  <span
                    key={i}
                    className="inline-block px-2 py-1 text-[11px] font-mono bg-slate-100 text-slate-700 rounded border border-slate-200"
                  >
                    {s}
                  </span>
                ))}
              </div>
            </JudgmentField>
          )}
        </div>
      )}
    </div>
  );
}

function JudgmentField({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-1.5">
        {label}
      </div>
      {children}
    </div>
  );
}