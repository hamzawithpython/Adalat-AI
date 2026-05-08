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

// NEW: illustrative judgment (LLM-suggested, not verified retrieval)
export interface Judgment {
  case_title: string;
  citation: string;
  court: string;
  outcome: string;
  sections: string[];
  summary: string;
  cited_cases: string[];
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
  // NEW
  sections: AnswerSection[];
  judgments: Judgment[];
  follow_up_questions: string[];
  response_language?: Language | null;
  judgments_disclaimer?: string;
  // existing
  disclaimer: string;
  schema_valid: boolean;
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