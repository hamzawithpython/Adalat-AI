"use client";

import type { ChatTurn } from "@/types/chat";
import { Card } from "@/components/ui/card";
import { LoadingState } from "./loading-state";
import { AnswerView } from "./answer-view";

interface TurnViewProps {
  turn: ChatTurn;
}

export function TurnView({ turn }: TurnViewProps) {
  if (turn.status === "idle") return null;

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-8 py-6 sm:py-8 space-y-5 sm:space-y-6">
      <UserQueryBubble query={turn.query} />

      {turn.status === "loading" && <LoadingState startedAt={turn.startedAt} />}
      {turn.status === "error" && <ErrorState error={turn.error} />}
      {turn.status === "answered" && <AnswerView response={turn.response} />}
    </div>
  );
}

function UserQueryBubble({ query }: { query: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] sm:max-w-[80%] bg-navy text-white rounded-2xl rounded-tr-sm px-4 sm:px-5 py-3 sm:py-3.5">
        <p className="text-[14px] sm:text-[15px] leading-relaxed whitespace-pre-wrap">
          {query}
        </p>
      </div>
    </div>
  );
}

function ErrorState({ error }: { error: string }) {
  return (
    <Card padding="lg" className="border-red-200 bg-red-50">
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-full bg-red-100 flex items-center justify-center text-red-600 text-sm font-bold shrink-0">
          !
        </div>
        <div>
          <h3 className="font-semibold text-red-900 mb-1">Something went wrong</h3>
          <p className="text-sm text-red-700">{error}</p>
        </div>
      </div>
    </Card>
  );
}