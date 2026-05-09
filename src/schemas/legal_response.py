from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Jurisdiction(str, Enum):
    PK = "PK"
    UK = "UK"
    DE = "DE"
    UNKNOWN = "UNKNOWN"


class Language(str, Enum):
    ENGLISH = "english"
    ROMAN_URDU = "roman_urdu"
    GERMAN = "german"


class Citation(BaseModel):
    source: str = Field(..., description="PDF filename")
    page: int = Field(..., description="Page number in document")
    jurisdiction: str = Field(..., description="PK/UK/DE")
    relevance_score: float = Field(..., description="Cosine similarity score 0-1")


class RightsRecord(BaseModel):
    right: str = Field(..., description="The specific legal right")
    legal_basis: str = Field(..., description="Article/Section/Paragraph reference")
    obligation: Optional[str] = Field(None, description="What the other party must do")
    deadline: Optional[str] = Field(None, description="Time limit if any")
    recourse: str = Field(..., description="What action the user can take")


# NEW: Rich answer section (LLM picks 3-6 per query)
class AnswerSection(BaseModel):
    heading: str = Field(..., description="Section title, e.g. 'Statutory Framework'")
    content: str = Field(..., description="Markdown-formatted section content")
    icon_hint: Optional[str] = Field(
        None,
        description="Icon hint: scales|book|shield|gavel|globe"
    )


# Illustrative judicial principle (LLM-generated, NOT a fabricated case citation).
# We do NOT generate case names/citations until we ingest a verified case-law corpus.
class JudgmentSection(BaseModel):
    principle: str = Field(..., description="Short title, e.g. 'Burden of proof in deposit recovery'")
    summary: str = Field(..., description="2-3 sentences on how courts in this jurisdiction generally treat this issue")
    typical_outcome: str = Field(..., description="What kind of decision usually results when these facts are present")
    relevant_sections: list[str] = Field(default_factory=list, description="Statutory sections that would apply")


class LegalResponse(BaseModel):
    query: str = Field(..., description="Original user query")
    translated_query: Optional[str] = Field(None, description="Translated if Roman-Urdu")
    language: Language = Field(..., description="Detected language")
    jurisdiction: Jurisdiction = Field(..., description="Detected jurisdiction")

    # KEPT for fallback / backwards compat with existing frontend
    answer: str = Field(..., description="Full LLM generated answer (markdown)")

    rights: list[RightsRecord] = Field(default_factory=list, description="Structured rights")
    citations: list[Citation] = Field(default_factory=list, description="Source citations")
    confidence: float = Field(..., description="Average retrieval confidence 0-1")

    # NEW FIELDS
    sections: list[AnswerSection] = Field(
        default_factory=list,
        description="Structured answer sections (3-6 per query)"
    )
    judgments: list[JudgmentSection] = Field(
        default_factory=list,
        description="Illustrative judgments (LLM-suggested, NOT verified retrievals)"
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="LLM-suggested follow-up questions in user's language"
    )
    response_language: Optional[str] = Field(
        None,
        description="Language code the answer is written in (matches user's query language)"
    )

    disclaimer: str = Field(
        default="This is informational only. Consult a qualified lawyer for legal advice.",
        description="Legal disclaimer"
    )
    judgments_disclaimer: str = Field(
        default=(
            "These are general judicial principles, not specific case citations. "
            "They illustrate how courts in this jurisdiction typically approach similar issues, "
            "but should be verified with a qualified lawyer before relying on them."
        ),
        description="Disclaimer specifically for the judicial principles section"
    )
    schema_valid: bool = Field(default=True, description="Pydantic validation passed")


def build_legal_response(router_result: dict, rights: list[dict]) -> LegalResponse:
    """Convert raw router output into validated LegalResponse."""
    citations = [Citation(**c) for c in (router_result.get("citations") or [])]
    confidence = (
        sum(c.relevance_score for c in citations) / len(citations)
        if citations else 0.0
    )

    rights_records = []
    for r in rights:
        try:
            rights_records.append(RightsRecord(**r))
        except Exception:
            pass

    # NEW: parse sections (safe — empty list if missing or malformed)
    sections = []
    for s in (router_result.get("sections") or []):
        try:
            sections.append(AnswerSection(**s))
        except Exception:
            pass

    # NEW: parse judgments (safe — empty list if missing or malformed)
    judgments = []
    for j in (router_result.get("judgments") or []):
        try:
            judgments.append(JudgmentSection(**j))
        except Exception:
            pass

    lang_map = {
        "roman_urdu": Language.ROMAN_URDU,
        "german": Language.GERMAN,
        "english": Language.ENGLISH,
    }
    jur_map = {"PK": Jurisdiction.PK, "UK": Jurisdiction.UK, "DE": Jurisdiction.DE}

    translated = router_result.get("translated_query")
    if translated and translated == router_result.get("query"):
        translated = None

    detected_lang = router_result.get("language", "english")

    return LegalResponse(
        query=router_result["query"],
        translated_query=translated,
        language=lang_map.get(detected_lang, Language.ENGLISH),
        jurisdiction=jur_map.get(router_result.get("jurisdiction", "PK"), Jurisdiction.PK),
        answer=router_result.get("answer", ""),
        rights=rights_records,
        citations=citations,
        confidence=round(confidence, 4),
        # NEW
        sections=sections,
        judgments=judgments,
        follow_up_questions=router_result.get("follow_up_questions", []),
        response_language=router_result.get("response_language", detected_lang),
    )