"use client";

import { ScalesIcon } from "@/components/icons/legal-icons";
import { SAMPLE_QUERIES } from "@/lib/jurisdictions";
import { Flag } from "@/components/brand/flag";

interface EmptyStateProps {
  onSampleClick: (query: string) => void;
}

export function EmptyState({ onSampleClick }: EmptyStateProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-8 py-12">
      <div className="max-w-2xl w-full text-center">
        {/* Icon */}
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gold-faint mb-6 text-navy">
          <ScalesIcon size={44} />
        </div>

        {/* Headline */}
        <h1 className="font-serif text-4xl font-bold text-navy mb-3 tracking-tight">
          What legal question can I help you with?
        </h1>
        <p className="text-base text-slate-600 mb-10">
          Ask in Roman-Urdu, English, or German. Get a structured answer with
          article-level citations from Pakistani, UK, and German law.
        </p>

        {/* Sample queries */}
        <div className="text-[11px] uppercase tracking-widest font-semibold text-slate-500 mb-3 font-mono">
          Try a sample query
        </div>
        <div className="flex flex-col gap-2">
          {SAMPLE_QUERIES.map((s) => (
            <button
              key={s.code}
              onClick={() => onSampleClick(s.query)}
              className="flex items-center gap-3 px-4 py-3 rounded-lg border border-slate-200 bg-white hover:border-navy hover:bg-slate-50 transition-colors text-left group"
            >
              <Flag code={s.code} size={28} />
              <div className="flex-1">
                <div className="text-[10px] uppercase tracking-widest font-semibold text-slate-400 font-mono mb-0.5">
                  {s.language}
                </div>
                <div className="text-sm text-slate-700 group-hover:text-navy">
                  {s.query}
                </div>
              </div>
              <span className="text-slate-300 group-hover:text-navy transition-colors">
                →
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}