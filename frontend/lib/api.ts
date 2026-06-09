import type {
  LegalResponse,
  AskRequest,
  Jurisdiction,
  Language,
  AnswerSection,
  Judgment,
  Right,
  Citation,
} from "@/types/legal";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_URL) {
  throw new Error("NEXT_PUBLIC_API_URL is not set in .env.local");
}

function getVisitorId(): string {
  if (typeof window === "undefined") return "";
  let id = window.localStorage.getItem("adalat_visitor_id");
  if (!id) {
    id = crypto.randomUUID();
    window.localStorage.setItem("adalat_visitor_id", id);
  }
  return id;
}

/**
 * Custom error class so UI components can distinguish API errors
 * from generic JS errors and show appropriate messages.
 */
export class AdalatApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string
  ) {
    super(message);
    this.name = "AdalatApiError";
  }
}

/**
 * Ask Adalat-AI a legal question.
 * Calls POST /ask on the FastAPI backend.
 */
export async function askAdalat(req: AskRequest): Promise<LegalResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  headers["X-Visitor-Id"] = getVisitorId();

  // BYOK: send user-supplied keys from localStorage if present
  if (typeof window !== "undefined") {
    const stored = window.localStorage.getItem("adalat_api_keys");
    if (stored) {
      try {
        const keys = JSON.parse(stored);
        const clean: Record<string, string> = {};
        for (const k of ["groq", "cerebras", "gemini"]) {
          if (keys[k] && typeof keys[k] === "string" && keys[k].length >= 16) {
            clean[k] = keys[k];
          }
        }
        if (Object.keys(clean).length > 0) {
          headers["X-Adalat-API-Keys"] = JSON.stringify(clean);
        }
      } catch {
        // ignore malformed JSON
      }
    }
  }

  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers,
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new AdalatApiError(
      `Request failed (${res.status})`,
      res.status,
      text
    );
  }

  return res.json() as Promise<LegalResponse>;
}

/**
 * Health check — useful for showing API status in dev.
 */
export async function getHealth(): Promise<{ status: string; version: string }> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) {
    throw new AdalatApiError(`Health check failed`, res.status);
  }
  return res.json();
}

// ── Sessions / history ──────────────────────────────────────────────

export interface SessionListItem {
  id: string;
  title: string | null;
  jurisdiction: Jurisdiction | null;
  language: Language | null;
  turn_count: number;
  created_at: string;
  updated_at: string;
}

export interface SessionTurn {
  id: number;
  turn_index: number;
  query: string;
  translated_query: string | null;
  language: Language;
  jurisdiction: Jurisdiction;
  answer: string;
  sections: AnswerSection[];
  judgments: Judgment[];
  rights: Right[];
  citations: Citation[];
  confidence: number;
  response_language: Language | null;
  follow_up_questions: string[];
  created_at: string;
}

export interface SessionDetail {
  id: string;
  title: string | null;
  jurisdiction: Jurisdiction | null;
  language: Language | null;
  created_at: string;
  updated_at: string;
  turns: SessionTurn[];
}

export async function listSessions(limit = 30): Promise<SessionListItem[]> {
  const res = await fetch(`${API_URL}/history?limit=${limit}`, {
    headers: { "X-Visitor-Id": getVisitorId() },
  });
  if (!res.ok) throw new AdalatApiError(`Failed to load history`, res.status);
  const data = await res.json();
  return data.sessions || [];
}

export async function getSession(id: string): Promise<SessionDetail> {
  const res = await fetch(`${API_URL}/sessions/${id}`, {
    headers: { "X-Visitor-Id": getVisitorId() },
  });
  if (!res.ok) throw new AdalatApiError(`Failed to load session`, res.status);
  return res.json();
}

export async function deleteSession(id: string): Promise<void> {
  const res = await fetch(`${API_URL}/sessions/${id}`, {
    method: "DELETE",
    headers: { "X-Visitor-Id": getVisitorId() },
  });
  if (!res.ok) throw new AdalatApiError(`Failed to delete session`, res.status);
}