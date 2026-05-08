import type { LegalResponse } from "./legal";

// One Q+A pair within the conversation
export type CompletedTurn = {
  query: string;
  response: LegalResponse;
};

// The CURRENT in-flight or just-finished turn (the one in progress)
export type ActiveTurn =
  | { status: "idle" }
  | { status: "loading"; query: string; startedAt: number }
  | { status: "answered"; query: string; response: LegalResponse }
  | { status: "error"; query: string; error: string };

// Backwards-compat alias (Phase 3 code still uses this name)
export type ChatTurn = ActiveTurn;