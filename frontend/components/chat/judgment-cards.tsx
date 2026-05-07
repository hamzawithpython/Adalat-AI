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
        <JudgmentCard key={`${j.citation}-${i}`} judgment={j} index={i} />
      ))}
    </div>
  );
}

function JudgmentCard({ judgment, index }: { judgment: Judgment; index: number }) {
  const [open, setOpen] = useState(false);

  const outcomeLower = judgment.outcome.toLowerCase();
  const outcomeColor =
    outcomeLower.includes("allowed") || outcomeLower.includes("granted")
      ? "bg-green-50 text-green-700 border-green-200"
      : outcomeLower.includes("dismissed") || outcomeLower.includes("rejected")
        ? "bg-red-50 text-red-700 border-red-200"
        : "bg-slate-50 text-slate-700 border-slate-200";

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
              Judgment {String(index + 1).padStart(2, "0")}
            </span>
            <span className="text-slate-300">·</span>
            <span className="text-[11px] font-mono text-slate-500 truncate">
              {judgment.citation}
            </span>
          </div>
          <h4 className="font-serif font-bold text-[16px] text-navy leading-snug truncate">
            {judgment.case_title}
          </h4>
          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            <span className="text-[11px] text-slate-600">{judgment.court}</span>
            {judgment.outcome && (
              <>
                <span className="text-slate-300">·</span>
                <span
                  className={cn(
                    "text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border",
                    outcomeColor
                  )}
                >
                  {judgment.outcome}
                </span>
              </>
            )}
          </div>
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
          {judgment.sections && judgment.sections.length > 0 && (
            <JudgmentField label="Sections invoked">
              <div className="flex flex-wrap gap-1.5">
                {judgment.sections.map((s, i) => (
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

          <JudgmentField label="Summary">
            <Markdown content={judgment.summary} />
          </JudgmentField>

          {judgment.cited_cases && judgment.cited_cases.length > 0 && (
            <JudgmentField label="Cited cases">
              <ul className="space-y-1">
                {judgment.cited_cases.map((c, i) => (
                  <li
                    key={i}
                    className="text-[12px] font-mono text-slate-600 leading-relaxed"
                  >
                    · {c}
                  </li>
                ))}
              </ul>
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