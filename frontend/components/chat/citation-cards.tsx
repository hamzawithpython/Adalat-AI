"use client";

import type { Citation } from "@/types/legal";
import { Flag } from "@/components/brand/flag";
import { cn } from "@/lib/utils";

interface CitationCardsProps {
  citations: Citation[];
}

export function CitationCards({ citations }: CitationCardsProps) {
  if (citations.length === 0) {
    return (
      <p className="text-sm text-slate-400 italic">No citations available.</p>
    );
  }

  // NOTE: keep original order so [^1], [^2] in the answer match visual #1, #2.
  // The order returned by hybrid_search is already by fused score (most relevant first).
  const sorted = citations;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
      {sorted.map((c, i) => (
        <CitationCard key={i} citation={c} index={i} />
      ))}
    </div>
  );
}

function CitationCard({ citation, index }: { citation: Citation; index: number }) {
  const pct = Math.round(citation.relevance_score * 100);
  const scoreColor =
    pct >= 80
      ? "bg-green-500"
      : pct >= 60
        ? "bg-amber-500"
        : "bg-slate-400";

  return (
    <div
      id={`citation-${index + 1}`}
      className="border border-slate-200 rounded-lg bg-white p-3.5 hover:border-navy hover:shadow-brand-sm transition-all scroll-mt-24"
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <Flag code={citation.jurisdiction} size={20} />
          <span className="text-[10px] font-mono font-bold text-slate-500 shrink-0">
            #{index + 1}
          </span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="text-xs font-mono font-semibold text-slate-700">
            {pct}%
          </span>
        </div>
      </div>

      <div className="text-sm font-semibold text-navy leading-snug mb-1.5 truncate">
        {citation.source}
      </div>

      <div className="text-[11px] font-mono text-slate-500 mb-2.5">
        {citation.breadcrumb}
        {citation.page !== null && (
          <span className="ml-1">· p. {citation.page}</span>
        )}
      </div>

      {/* Score bar */}
      <div className="h-1 rounded-full bg-slate-100 overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all", scoreColor)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
