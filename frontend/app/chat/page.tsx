"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Sidebar } from "@/components/chat/sidebar";
import { EmptyState } from "@/components/chat/empty-state";
import { QueryInput } from "@/components/chat/query-input";
import { TurnView } from "@/components/chat/turn-view";
import { Wordmark } from "@/components/brand/wordmark";
import { useChat } from "@/hooks/use-chat";
import type { Jurisdiction } from "@/types/legal";

function ChatPageInner() {
  const [jurisdiction, setJurisdiction] = useState<Jurisdiction | "ALL">("ALL");
  const [pendingQuery, setPendingQuery] = useState<string>("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { turn, submit, reset } = useChat();
  const searchParams = useSearchParams();

  const isLoading = turn.status === "loading";

  useEffect(() => {
    const q = searchParams.get("q");
    if (q && turn.status === "idle") {
      setPendingQuery(q);
    }
  }, [searchParams, turn.status]);

  const handleSampleClick = (query: string) => {
    setPendingQuery(query);
  };

  const handleNewChat = () => {
    reset();
    setPendingQuery("");
  };

  const handleSubmit = (query: string) => {
    setPendingQuery("");
    submit(query);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      <Sidebar
        activeJurisdiction={jurisdiction}
        onJurisdictionChange={setJurisdiction}
        onNewChat={handleNewChat}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Mobile header */}
        <div className="lg:hidden flex items-center justify-between px-4 py-3 bg-white border-b border-slate-200">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 -ml-2 text-slate-700 hover:text-navy"
            aria-label="Open sidebar"
          >
            <svg
              width="22"
              height="22"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
          <Wordmark size="sm" />
          <button
            onClick={handleNewChat}
            className="text-xs font-mono uppercase tracking-wider text-slate-500 hover:text-navy"
          >
            New
          </button>
        </div>

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

export default function ChatPage() {
  return (
    <Suspense fallback={null}>
      <ChatPageInner />
    </Suspense>
  );
}