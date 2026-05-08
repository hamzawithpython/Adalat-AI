"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Sidebar } from "@/components/chat/sidebar";
import { EmptyState } from "@/components/chat/empty-state";
import { QueryInput } from "@/components/chat/query-input";
import { TurnView } from "@/components/chat/turn-view";
import { Wordmark } from "@/components/brand/wordmark";
import { useChat } from "@/hooks/use-chat";
import { useHistory } from "@/hooks/use-history";
import { getSession } from "@/lib/api";
import type { Jurisdiction, LegalResponse } from "@/types/legal";

function ChatPageInner() {
  const [jurisdiction, setJurisdiction] = useState<Jurisdiction | "ALL">("ALL");
  const [pendingQuery, setPendingQuery] = useState<string>("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [loadingSession, setLoadingSession] = useState(false);

  const { turn, submit, reset, setAnswered } = useChat();
  const {
    sessions,
    loading: historyLoading,
    error: historyError,
    refresh: refreshHistory,
    remove: removeSession,
  } = useHistory();
  const searchParams = useSearchParams();

  const isLoading = turn.status === "loading";

  useEffect(() => {
    const q = searchParams.get("q");
    if (q && turn.status === "idle") {
      setPendingQuery(q);
    }
  }, [searchParams, turn.status]);

  // Only re-run when status flips to "answered" — refreshHistory is stable
  useEffect(() => {
    if (turn.status === "answered") {
      refreshHistory();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [turn.status]);

  const handleSampleClick = (query: string) => {
    setPendingQuery(query);
  };

  const handleNewChat = () => {
    reset();
    setPendingQuery("");
    setActiveSessionId(null);
  };

  const handleSubmit = (query: string) => {
    setPendingQuery("");
    submit(query, activeSessionId ?? undefined).then((newSessionId) => {
      if (newSessionId) setActiveSessionId(newSessionId);
    });
  };

  const handleSessionSelect = async (id: string) => {
    if (id === activeSessionId) return;
    setLoadingSession(true);
    try {
      const detail = await getSession(id);
      const lastTurn = detail.turns[detail.turns.length - 1];
      if (lastTurn) {
        const response: LegalResponse = {
          session_id: detail.id,
          query: lastTurn.query,
          translated_query: lastTurn.translated_query,
          language: lastTurn.language,
          jurisdiction: lastTurn.jurisdiction,
          answer: lastTurn.answer,
          rights: lastTurn.rights,
          citations: lastTurn.citations,
          sections: lastTurn.sections,
          judgments: lastTurn.judgments,
          confidence: lastTurn.confidence,
          response_language: lastTurn.response_language,
          follow_up_questions: lastTurn.follow_up_questions,
          disclaimer:
            "This is informational only. Consult a qualified lawyer for legal advice.",
          schema_valid: true,
        };
        setAnswered(lastTurn.query, response);
        setActiveSessionId(id);
      }
    } catch (err) {
      console.error("Failed to load session:", err);
    } finally {
      setLoadingSession(false);
    }
  };

  const handleSessionDelete = async (id: string) => {
    await removeSession(id);
    if (id === activeSessionId) handleNewChat();
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      <Sidebar
        activeJurisdiction={jurisdiction}
        onJurisdictionChange={setJurisdiction}
        onNewChat={handleNewChat}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        sessions={sessions}
        historyLoading={historyLoading}
        historyError={historyError}
        activeSessionId={activeSessionId}
        onSessionSelect={handleSessionSelect}
        onSessionDelete={handleSessionDelete}
        onHistoryRefresh={refreshHistory}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        <div className="lg:hidden flex items-center justify-between px-4 py-3 bg-white border-b border-slate-200">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 -ml-2 text-slate-700 hover:text-navy"
            aria-label="Open sidebar"
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
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
          {loadingSession ? (
            <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
              Loading chat…
            </div>
          ) : turn.status === "idle" ? (
            <EmptyState onSampleClick={handleSampleClick} />
          ) : (
            <TurnView turn={turn} onFollowUp={handleSubmit} />
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