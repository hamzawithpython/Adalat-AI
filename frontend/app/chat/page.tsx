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
import { ApiKeysPanel } from "@/components/chat/api-keys-panel";

function ChatPageInner() {
  const [jurisdiction, setJurisdiction] = useState<Jurisdiction | "ALL">("ALL");
  const [pendingQuery, setPendingQuery] = useState<string>("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [loadingSession, setLoadingSession] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [inputKey, setInputKey] = useState(0);

  const {
    completed,
    active,
    submit,
    reset,
    promoteActiveToCompleted,
    loadSession,
  } = useChat();
  const {
  sessions,
  loading: historyLoading,
  error: historyError,
  refresh: refreshHistory,
  remove: removeSession,
  clearAll: clearAllSessions,
  clearing,
  } = useHistory();
  const searchParams = useSearchParams();

  const isLoading = active.status === "loading";
  const hasContent = completed.length > 0 || active.status !== "idle";

  useEffect(() => {
    const q = searchParams.get("q");
    if (q && !hasContent) {
      setPendingQuery(q);
    }
  }, [searchParams, hasContent]);

  // Refresh history when a turn finishes
  useEffect(() => {
    if (active.status === "answered") {
      refreshHistory();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active.status]);

  const handleSampleClick = (query: string) => {
    setPendingQuery(query);
  };

  const handleNewChat = () => {
  reset();
  setPendingQuery("");
  setActiveSessionId(null);
  setInputKey((k) => k + 1);  // force QueryInput to remount with empty state
};

  const handleSubmit = (query: string) => {
    setPendingQuery("");
    // Push the previous answered turn into completed history first
    promoteActiveToCompleted();
    submit(query, activeSessionId ?? undefined).then((newSessionId) => {
      if (newSessionId) setActiveSessionId(newSessionId);
    });
  };

  const handleSessionSelect = async (id: string) => {
    if (id === activeSessionId) return;
    setLoadingSession(true);
    setPendingQuery("");
    setInputKey((k) => k + 1);
    try {
      const detail = await getSession(id);
      // Build CompletedTurn[] from all turns in the session
      const turns = detail.turns.map((t) => ({
        query: t.query,
        response: {
          session_id: detail.id,
          query: t.query,
          translated_query: t.translated_query,
          language: t.language,
          jurisdiction: t.jurisdiction,
          answer: t.answer,
          rights: t.rights,
          citations: t.citations,
          sections: t.sections,
          judgments: t.judgments,
          confidence: t.confidence,
          response_language: t.response_language,
          follow_up_questions: t.follow_up_questions,
          disclaimer:
            "This is informational only. Consult a qualified lawyer for legal advice.",
          schema_valid: true,
        } satisfies LegalResponse,
      }));
      loadSession(turns);
      setActiveSessionId(id);
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

  const handleClearAll = async () => {
    const deletedIds = await clearAllSessions();
    // If the active session was wiped, reset the view
    if (activeSessionId && deletedIds.includes(activeSessionId)) {
      handleNewChat();
    }
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
        onClearAll={handleClearAll}
        clearing={clearing}
        onOpenSettings={() => setSettingsOpen(true)}
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
            <div className="flex items-center justify-center min-h-full text-slate-500 text-sm">
              Loading chat…
            </div>
          ) : !hasContent ? (
            <EmptyState onSampleClick={handleSampleClick} />
          ) : (
            <ConversationView
              completed={completed}
              active={active}
              onFollowUp={handleSubmit}
            />
          )}
        </div>
        <QueryInput
          key={inputKey}
          initialValue={pendingQuery}
          onSubmit={handleSubmit}
          disabled={isLoading}
        />
      </main>

      <ApiKeysPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}

import type { CompletedTurn, ActiveTurn } from "@/types/chat";

function ConversationView({
  completed,
  active,
  onFollowUp,
}: {
  completed: CompletedTurn[];
  active: ActiveTurn;
  onFollowUp: (q: string) => void;
}) {
  // Auto-scroll to bottom when a new turn arrives
  useAutoScroll([completed.length, active.status]);

  const isLast = (idx: number) =>
    idx === completed.length - 1 && active.status === "idle";

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-8 py-6 sm:py-8 space-y-8">
      {completed.map((t, i) => (
        <CompletedTurnBlock
          key={i}
          turn={t}
          // Only show follow-ups on the very last completed turn (when no active turn)
          showFollowUps={isLast(i)}
          onFollowUp={onFollowUp}
        />
      ))}
      {active.status !== "idle" && (
        <TurnView turn={active} onFollowUp={onFollowUp} />
      )}
    </div>
  );
}

function CompletedTurnBlock({
  turn,
  showFollowUps,
  onFollowUp,
}: {
  turn: CompletedTurn;
  showFollowUps: boolean;
  onFollowUp: (q: string) => void;
}) {
  // Reuse TurnView for visual consistency by faking an "answered" turn
  return (
    <TurnView
      turn={{ status: "answered", query: turn.query, response: turn.response }}
      onFollowUp={showFollowUps ? onFollowUp : undefined}
    />
  );
}

// Minimal auto-scroll hook
function useAutoScroll(deps: unknown[]) {
  // The container is the parent <div className="flex-1 overflow-y-auto"> in ChatPageInner.
  // We scroll the document's scrolling container by querying the ref via getElementById not viable,
  // so we just scroll the nearest scrollable ancestor by setTimeout to next frame.
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      // Find the scrollable parent — the main scroll container in our layout
      const container = document.querySelector("main > div.flex-1.overflow-y-auto");
      if (container) {
        container.scrollTo({
          top: container.scrollHeight,
          behavior: "smooth",
        });
      }
    });
    return () => cancelAnimationFrame(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

export default function ChatPage() {
  return (
    <Suspense fallback={null}>
      <ChatPageInner />
    </Suspense>
  );
}