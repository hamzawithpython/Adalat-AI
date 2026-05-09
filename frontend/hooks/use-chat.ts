"use client";

import { useState, useCallback } from "react";
import { askAdalat, AdalatApiError } from "@/lib/api";
import type { ActiveTurn, CompletedTurn } from "@/types/chat";
import type { LegalResponse } from "@/types/legal";

export function useChat() {
  const [completed, setCompleted] = useState<CompletedTurn[]>([]);
  const [active, setActive] = useState<ActiveTurn>({ status: "idle" });

  const submit = useCallback(
    async (query: string, sessionId?: string): Promise<string | null> => {
      if (!query.trim()) return null;

      setActive({
        status: "loading",
        query,
        startedAt: Date.now(),
      });

      try {
        const response = await askAdalat({ query, session_id: sessionId });
        // Append previous "active" answered turn to completed history,
        // then set the new one as active so it auto-scrolls.
        setCompleted((prev) => {
          // We don't mutate prev — just return a new list.
          return prev;
        });
        setActive({ status: "answered", query, response });
        return response.session_id;
      } catch (err) {
        const message =
          err instanceof AdalatApiError
            ? `${err.message}${err.detail ? ` — ${err.detail.slice(0, 200)}` : ""}`
            : err instanceof Error
              ? err.message
              : "Something went wrong. Please try again.";
        setActive({ status: "error", query, error: message });
        return null;
      }
    },
    []
  );

  // When the active turn becomes "answered" and the user submits a NEW one,
  // we should push the previous one into completed[].
  // We do this in a separate function called by the page when it submits.
  const promoteActiveToCompleted = useCallback(() => {
  setActive((prev) => {
    if (prev.status === "answered") {
      // Use functional setCompleted with a guard against double-append
      setCompleted((c) => {
        // Guard against React strict-mode double invocation
        const last = c[c.length - 1];
        if (last && last.query === prev.query && last.response === prev.response) {
          return c;
        }
        return [...c, { query: prev.query, response: prev.response }];
      });
    }
    return { status: "idle" };
  });
}, []);

  const reset = useCallback(() => {
    setCompleted([]);
    setActive({ status: "idle" });
  }, []);

  // Load a session from history — populate completed[] with all turns,
  // and clear active.
  const loadSession = useCallback((turns: { query: string; response: LegalResponse }[]) => {
    setCompleted(turns);
    setActive({ status: "idle" });
  }, []);

  return {
    completed,
    active,
    submit,
    reset,
    promoteActiveToCompleted,
    loadSession,
  };
}