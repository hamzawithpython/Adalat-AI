"use client";

import { useState, useCallback } from "react";
import { askAdalat, AdalatApiError } from "@/lib/api";
import type { ChatTurn } from "@/types/chat";

export function useChat() {
  const [turn, setTurn] = useState<ChatTurn>({ status: "idle" });

  const submit = useCallback(async (query: string) => {
    if (!query.trim()) return;

    setTurn({
      status: "loading",
      query,
      startedAt: Date.now(),
    });

    try {
      const response = await askAdalat({ query });
      setTurn({ status: "answered", query, response });
    } catch (err) {
      const message =
        err instanceof AdalatApiError
          ? `${err.message}${err.detail ? ` — ${err.detail.slice(0, 200)}` : ""}`
          : err instanceof Error
            ? err.message
            : "Something went wrong. Please try again.";
      setTurn({ status: "error", query, error: message });
    }
  }, []);

  const reset = useCallback(() => {
    setTurn({ status: "idle" });
  }, []);

  return { turn, submit, reset };
}