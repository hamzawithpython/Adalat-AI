"use client";

import { useElapsedSeconds } from "@/hooks/use-elapsed-seconds";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface LoadingStateProps {
  startedAt: number;
}

const STAGES = [
  { from: 0, to: 4, label: "Detecting language & jurisdiction", emoji: "🌐" },
  { from: 4, to: 12, label: "Searching legal documents", emoji: "📚" },
  { from: 12, to: 22, label: "Reading relevant articles", emoji: "📖" },
  { from: 22, to: 999, label: "Drafting your answer", emoji: "✍️" },
];

export function LoadingState({ startedAt }: LoadingStateProps) {
  const elapsed = useElapsedSeconds(startedAt);
  const currentStageIdx = STAGES.findIndex(
    (s) => elapsed >= s.from && elapsed < s.to
  );

  return (
    <Card padding="lg" className="border-slate-200">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <span className="inline-block w-4 h-4 border-2 border-slate-200 border-t-navy rounded-full animate-spin" />
          <span className="text-sm font-medium text-navy">
            Adalat-AI is thinking…
          </span>
        </div>
        <span className="text-xs font-mono text-slate-500">
          {elapsed}s elapsed
        </span>
      </div>

      <div className="space-y-2.5">
        {STAGES.map((stage, idx) => {
          const isDone = idx < currentStageIdx;
          const isActive = idx === currentStageIdx;
          return (
            <div
              key={stage.label}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-md transition-colors",
                isActive && "bg-gold-faint",
                isDone && "opacity-50"
              )}
            >
              <span className="w-5 text-base">
                {isDone ? "✓" : isActive ? stage.emoji : "○"}
              </span>
              <span
                className={cn(
                  "text-sm",
                  isActive
                    ? "text-navy font-medium"
                    : isDone
                      ? "text-slate-500"
                      : "text-slate-400"
                )}
              >
                {stage.label}
                {isActive && (
                  <span className="ml-1 inline-flex">
                    <span
                      className="animate-pulse"
                      style={{ animationDelay: "0ms" }}
                    >
                      .
                    </span>
                    <span
                      className="animate-pulse"
                      style={{ animationDelay: "200ms" }}
                    >
                      .
                    </span>
                    <span
                      className="animate-pulse"
                      style={{ animationDelay: "400ms" }}
                    >
                      .
                    </span>
                  </span>
                )}
              </span>
            </div>
          );
        })}
      </div>

      {elapsed > 30 && (
        <p className="text-xs text-slate-500 mt-5 italic">
          Taking a bit longer than usual — complex queries can take up to 60
          seconds.
        </p>
      )}
    </Card>
  );
}
