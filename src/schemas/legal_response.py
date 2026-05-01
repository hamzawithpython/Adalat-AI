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


class LegalResponse(BaseModel):
    query: str = Field(..., description="Original user query")
    translated_query: Optional[str] = Field(None, description="Translated if Roman-Urdu")
    language: Language = Field(..., description="Detected language")
    jurisdiction: Jurisdiction = Field(..., description="Detected jurisdiction")
    answer: str = Field(..., description="Full LLM generated answer")
    rights: list[RightsRecord] = Field(default_factory=list, description="Structured rights extracted")
    citations: list[Citation] = Field(default_factory=list, description="Source citations")
    confidence: float = Field(..., description="Average retrieval confidence 0-1")
    disclaimer: str = Field(
        default="⚠️ This is informational only. Consult a qualified lawyer for legal advice.",
        description="Legal disclaimer"
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

    lang_map = {"roman_urdu": Language.ROMAN_URDU, "german": Language.GERMAN, "english": Language.ENGLISH}
    jur_map = {"PK": Jurisdiction.PK, "UK": Jurisdiction.UK, "DE": Jurisdiction.DE}

    translated = router_result.get("translated_query")
    if translated and translated == router_result.get("query"):
        translated = None

    return LegalResponse(
        query=router_result["query"],
        translated_query=translated,
        language=lang_map.get(router_result.get("language", "english"), Language.ENGLISH),
        jurisdiction=jur_map.get(router_result.get("jurisdiction", "PK"), Jurisdiction.PK),
        answer=router_result.get("answer", ""),
        rights=rights_records,
        citations=citations,
        confidence=round(confidence, 4),
    )