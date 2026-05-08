"use client";

interface FollowUpQuestionsProps {
  questions: string[];
  onSelect: (q: string) => void;
}

export function FollowUpQuestions({ questions, onSelect }: FollowUpQuestionsProps) {
  if (!questions || questions.length === 0) return null;

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 sm:p-5">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-base">💭</span>
        <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark font-semibold">
          You might also ask
        </div>
      </div>
      <div className="flex flex-col gap-2">
        {questions.map((q, i) => (
          <button
            key={i}
            onClick={() => onSelect(q)}
            className="text-left flex items-center gap-3 px-3 py-2.5 rounded-md border border-slate-200 bg-white hover:border-navy hover:bg-slate-50 transition-colors group"
          >
            <span className="text-[10px] font-mono font-bold text-slate-400 group-hover:text-navy shrink-0">
              {String(i + 1).padStart(2, "0")}
            </span>
            <span className="text-sm text-slate-700 group-hover:text-navy flex-1">
              {q}
            </span>
            <span className="text-slate-300 group-hover:text-navy shrink-0">→</span>
          </button>
        ))}
      </div>
    </div>
  );
}