import type { LegalResponse, AskRequest, HistoryItem } from "@/types/legal";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_URL) {
  throw new Error("NEXT_PUBLIC_API_URL is not set in .env.local");
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
 *
 * Note: Queries can take 10–30s due to LLM generation.
 * The caller should show a loading state.
 */
export async function askAdalat(request: AskRequest): Promise<LegalResponse> {
  const response = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new AdalatApiError(
      `Request failed (${response.status})`,
      response.status,
      text
    );
  }

  return response.json() as Promise<LegalResponse>;
}

/**
 * Fetch list of past queries.
 * Calls GET /history on the FastAPI backend.
 */
export async function getHistory(): Promise<HistoryItem[]> {
  const response = await fetch(`${API_URL}/history`);
  if (!response.ok) {
    throw new AdalatApiError(
      `Failed to fetch history (${response.status})`,
      response.status
    );
  }
  return response.json() as Promise<HistoryItem[]>;
}

/**
 * Health check — useful for showing API status in dev.
 */
export async function getHealth(): Promise<{ status: string; version: string }> {
  const response = await fetch(`${API_URL}/health`);
  if (!response.ok) {
    throw new AdalatApiError(`Health check failed`, response.status);
  }
  return response.json();
}