"use client";

import type { LegalResponse } from "@/types/legal";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Flag } from "@/components/brand/flag";
import { ConfidenceIndicator } from "./confidence-indicator";
import { RightsCards } from "./rights-cards";
import { CitationCards } from "./citation-cards";
import { Markdown } from "@/components/ui/markdown";

interface AnswerViewProps {
  response: LegalResponse;
}

export function AnswerView({ response }: AnswerViewProps) {
  const langLabel = response.language.replace("_", "-").toUpperCase();

  return (
    <div className="space-y-5">
      {/* Header card: meta + confidence */}
      <Card padding="lg">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3 flex-wrap">
            <Flag code={response.jurisdiction} size={32} />
            <Badge tone="navy">{response.jurisdiction}</Badge>
            <Badge tone="goldSoft">{langLabel}</Badge>
          </div>
          <ConfidenceIndicator value={response.confidence} />
        </div>

        {response.translated_query &&
          response.translated_query !== response.query && (
            <div className="mt-4 pt-4 border-t border-slate-100">
              <div className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-1">
                Translated query
              </div>
              <p className="text-sm text-slate-700 italic">
                “{response.translated_query}”
              </p>
            </div>
          )}
      </Card>

      {/* Main answer */}
      <Card padding="lg">
        <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark mb-3">
          Answer
        </div>
        <Markdown content={response.answer} />
      </Card>

      {/* Rights */}
      <Card padding="lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark mb-1">
              Your Rights
            </div>
            <h3 className="font-serif text-xl font-bold text-navy">
              {response.rights.length} structured right
              {response.rights.length !== 1 ? "s" : ""} extracted
            </h3>
          </div>
        </div>
        <RightsCards rights={response.rights} />
      </Card>

      {/* Citations */}
      <Card padding="lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark mb-1">
              Citations
            </div>
            <h3 className="font-serif text-xl font-bold text-navy">
              {response.citations.length} source
              {response.citations.length !== 1 ? "s" : ""} retrieved
            </h3>
          </div>
        </div>
        <CitationCards citations={response.citations} />
      </Card>

      {/* Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-xs text-amber-900 leading-relaxed">
        <span className="font-semibold mr-1">⚠️ Disclaimer:</span>
        {response.disclaimer}
      </div>
    </div>
  );
}
