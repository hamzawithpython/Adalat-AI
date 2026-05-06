"use client";

import { useState } from "react";
import { Sidebar } from "@/components/chat/sidebar";
import { EmptyState } from "@/components/chat/empty-state";
import { QueryInput } from "@/components/chat/query-input";
import { TurnView } from "@/components/chat/turn-view";
import { useChat } from "@/hooks/use-chat";
import type { Jurisdiction } from "@/types/legal";

export default function ChatPage() {
  const [jurisdiction, setJurisdiction] = useState<Jurisdiction | "ALL">("ALL");
  const [pendingQuery, setPendingQuery] = useState<string>("");
  const { turn, submit, reset } = useChat();

  const isLoading = turn.status === "loading";

  const handleSampleClick = (query: string) => {
    setPendingQuery(query);
  };

  const handleNewChat = () => {
    reset();
    setPendingQuery("");
  };

  const handleSubmit = (query: string) => {
    setPendingQuery(""); // clear input after submit
    submit(query);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      <Sidebar
        activeJurisdiction={jurisdiction}
        onJurisdictionChange={setJurisdiction}
        onNewChat={handleNewChat}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto">
          {turn.status === "idle" ? (
            <EmptyState onSampleClick={handleSampleClick} />
          ) : (
            <TurnView turn={turn} />
          )}
        </div>
        <QueryInput
          initialValue={pendingQuery}
          onSubmit={handleSubmit}
          disabled={isLoading}
        />
      </main>
    </div>
  );
}