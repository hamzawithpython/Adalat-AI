"use client";

import type { AnswerSection } from "@/types/legal";
import { Markdown } from "@/components/ui/markdown";
import { SectionIcon } from "./section-icon";

interface SectionCardsProps {
  sections: AnswerSection[];
}

export function SectionCards({ sections }: SectionCardsProps) {
  if (sections.length === 0) return null;

  return (
    <div className="space-y-3">
      {sections.map((section, i) => (
        <SectionCard key={i} section={section} index={i} />
      ))}
    </div>
  );
}

function SectionCard({
  section,
  index,
}: {
  section: AnswerSection;
  index: number;
}) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg overflow-hidden hover:border-slate-300 transition-colors">
      {/* Header bar */}
      <div className="flex items-center gap-3 px-3 sm:px-5 py-3 bg-gradient-to-r from-gold-faint to-transparent border-b border-slate-100">
        <div className="flex items-center justify-center w-9 h-9 rounded-md bg-white border border-gold-soft text-navy">
          <SectionIcon hint={section.icon_hint} size={18} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark">
            Section {String(index + 1).padStart(2, "0")}
          </div>
          <h3 className="font-serif font-bold text-[17px] text-navy leading-tight">
            {section.heading}
          </h3>
        </div>
      </div>

      {/* Body */}
      <div className="px-3 sm:px-5 py-3 sm:py-4">
        <Markdown content={section.content} />
      </div>
    </div>
  );
}