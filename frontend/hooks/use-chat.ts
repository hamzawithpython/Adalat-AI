"use client";

import { useState, useCallback } from "react";
import { askAdalat, AdalatApiError } from "@/lib/api";
import type { ChatTurn } from "@/types/chat";
import type { LegalResponse } from "@/types/legal";

export function useChat() {
  const [turn, setTurn] = useState<ChatTurn>({ status: "idle" });

  const submit = useCallback(
    async (query: string, sessionId?: string): Promise<string | null> => {
      if (!query.trim()) return null;

      setTurn({
        status: "loading",
        query,
        startedAt: Date.now(),
      });

      try {
        const response = await askAdalat({ query, session_id: sessionId });
        setTurn({ status: "answered", query, response });
        return response.session_id;
      } catch (err) {
        const message =
          err instanceof AdalatApiError
            ? `${err.message}${err.detail ? ` — ${err.detail.slice(0, 200)}` : ""}`
            : err instanceof Error
              ? err.message
              : "Something went wrong. Please try again.";
        setTurn({ status: "error", query, error: message });
        return null;
      }
    },
    []
  );

  const reset = useCallback(() => {
    setTurn({ status: "idle" });
  }, []);

  // Used to load an existing turn from history
  const setAnswered = useCallback((query: string, response: LegalResponse) => {
    setTurn({ status: "answered", query, response });
  }, []);

  return { turn, submit, reset, setAnswered };
}