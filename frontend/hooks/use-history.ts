"use client";

import { useState, useEffect, useCallback } from "react";
import { listSessions, deleteSession, type SessionListItem } from "@/lib/api";

export function useHistory() {
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listSessions(30);
      setSessions(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, []);

  const remove = useCallback(async (id: string) => {
    try {
      await deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
    } catch (err) {
      console.error("Delete failed:", err);
    }
  }, []);

  const clearAll = useCallback(async (): Promise<string[]> => {
    setClearing(true);
    const deletedIds: string[] = [];
    const currentSessions = sessions;
    try {
      // Delete in parallel; collect successes
      const results = await Promise.allSettled(
        currentSessions.map((s) => deleteSession(s.id))
      );
      results.forEach((r, i) => {
        if (r.status === "fulfilled") {
          deletedIds.push(currentSessions[i].id);
        }
      });
      // Remove successfully-deleted sessions from local state
      setSessions((prev) => prev.filter((s) => !deletedIds.includes(s.id)));
    } catch (err) {
      console.error("Clear all failed:", err);
    } finally {
      setClearing(false);
    }
    return deletedIds;
  }, [sessions]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { sessions, loading, error, clearing, refresh, remove, clearAll };
}