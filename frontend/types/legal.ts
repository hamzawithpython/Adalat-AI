// Mirrors src/schemas/legal_response.py (Pydantic) on the backend.
// Keep these in sync if you change the API contract.

export type Jurisdiction = "PK" | "UK" | "DE";
export type Language = "roman_urdu" | "english" | "german";

export interface Right {
  right: string;
  legal_basis: string;
  deadline: string | null;
  recourse: string;
}

export interface Citation {
  source: string;
  page: number | null;
  jurisdiction: Jurisdiction;
  relevance_score: number; // 0.0 – 1.0
  breadcrumb: string;
}

export interface LegalResponse {
  session_id: string;
  query: string;
  translated_query: string | null;
  language: Language;
  jurisdiction: Jurisdiction;
  answer: string;
  rights: Right[];
  citations: Citation[];
  confidence: number; // 0.0 – 1.0
  disclaimer: string;
  schema_valid: boolean;
}

// Request body for POST /ask
export interface AskRequest {
  query: string;
  session_id?: string;
}

// History list item from GET /history
export interface HistoryItem {
  id: string;
  session_id: string;
  query: string;
  jurisdiction: Jurisdiction;
  language: Language;
  created_at: string; // ISO datetime
}