// Mirrors src/schemas/legal_response.py (Pydantic) on the backend.
// Keep these in sync if you change the API contract.

export type Jurisdiction = "PK" | "UK" | "DE";
export type Language = "roman_urdu" | "english" | "german";
export type IconHint = "scales" | "book" | "shield" | "gavel" | "globe";

export interface Right {
  right: string;
  legal_basis: string;
  obligation?: string | null;
  deadline: string | null;
  recourse: string;
}

export interface Citation {
  source: string;
  page: number | null;
  jurisdiction: Jurisdiction;
  relevance_score: number; // 0.0 – 1.0
  breadcrumb?: string;
}

// NEW: structured section of the answer
export interface AnswerSection {
  heading: string;
  content: string; // markdown
  icon_hint?: IconHint | null;
}

// Judicial principle (LLM-generated, NOT a fabricated case citation).
// Renamed from the earlier "Judgment" shape — we no longer fabricate case names.
export interface Judgment {
  principle: string;
  summary: string;
  typical_outcome: string;
  relevant_sections: string[];
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
  confidence: number;
  sections: AnswerSection[];
  judgments: Judgment[];
  follow_up_questions: string[];
  response_language?: Language | null;
  judgments_disclaimer?: string;
  disclaimer: string;
  schema_valid: boolean;
  // Set to true when the assistant decided to ask clarifying questions
  // instead of producing a substantive answer.
  is_clarification?: boolean;
}

export interface AskRequest {
  query: string;
  session_id?: string;
}

export interface HistoryItem {
  id: string;
  session_id: string;
  query: string;
  jurisdiction: Jurisdiction;
  language: Language;
  created_at: string;
}