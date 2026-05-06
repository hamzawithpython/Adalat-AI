import type { LegalResponse } from "./legal";

/**
 * The state of a single query/answer turn in the chat.
 * Real apps with multi-turn conversation would have an array of these,
 * but for v1 we only show one turn at a time (matching your hi-fi design).
 */
export type ChatTurn =
  | { status: "idle" }
  | { status: "loading"; query: string; startedAt: number }
  | { status: "answered"; query: string; response: LegalResponse }
  | { status: "error"; query: string; error: string };