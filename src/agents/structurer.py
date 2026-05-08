"""Structures a prose legal answer into sections and generates illustrative judgments."""

import os
import sys
import json
import re
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logger = logging.getLogger(__name__)


def _llm():
    from src.agents.llms import fast_llm
    return fast_llm(max_tokens=3000, temperature=0.2)


# Robust JSON-array extractor (handles markdown fences, prose preambles)
def _parse_json_array(text: str) -> list:
    """Parse a JSON array from LLM output, repairing common LLM mistakes."""
    # Strip markdown fence if present
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    raw = match.group()

    # First try strict parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Repair: escape raw newlines/tabs inside string literals.
    # Walk char-by-char, tracking whether we're inside a string,
    # and replace literal \n / \r / \t with their escaped forms.
    repaired_chars = []
    in_string = False
    escape = False
    for ch in raw:
        if escape:
            repaired_chars.append(ch)
            escape = False
            continue
        if ch == "\\":
            repaired_chars.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            repaired_chars.append(ch)
            continue
        if in_string and ch == "\n":
            repaired_chars.append("\\n")
            continue
        if in_string and ch == "\r":
            repaired_chars.append("\\r")
            continue
        if in_string and ch == "\t":
            repaired_chars.append("\\t")
            continue
        repaired_chars.append(ch)

    repaired = "".join(repaired_chars)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON repair failed: {e}; first 200 chars: {repaired[:200]}")
        return []


# ── Sections ──────────────────────────────────────────────────────────
SECTIONS_PROMPT = ChatPromptTemplate.from_template("""You are restructuring a legal answer into 3-6 well-organized sections.

LANGUAGE: {response_language}
- Section HEADINGS stay in English ("Statutory Framework", "Practical Strategy", etc.) — universal labels.
- Section CONTENT must be in {response_language}.

If response_language is "roman_urdu":
✓ Use Pakistani Roman-Urdu words: koshish, taluq, muqadma, adalat, faisla, haq, hukum, qanoon, hukumat
✗ Avoid Hindi-leaning: prayas, sambandh, vivad, nyayalay, nirnay, adhikaar, aagya, kanoon, sarkar

INPUT ANSWER:
{answer}

Pick 3-6 of these section types that best fit (or invent similar):
- "Legal Context" — background and applicable laws
- "Statutory Framework" — specific statutes and provisions
- "Key Legal Points" — bulleted analysis
- "Practical Legal Strategy" — concrete steps the user can take
- "Realistic Position" — honest assessment of outcomes
- "Important Warnings" — risks, deadlines, what NOT to do

Each section content should be markdown:
- **bold** for emphasis
- bullet points (- )
- numbered lists (1. ) for steps
- 2-5 short paragraphs per section
- DO NOT REPEAT content already in another section — each section must add NEW information

Return ONLY a JSON array. No prose, no markdown fence, no preamble:
[
  {{
    "heading": "Statutory Framework",
    "content": "**Section 13** of the Punjab Rented Premises Act 2009 provides...",
    "icon_hint": "book"
  }}
]

icon_hint must be one of: scales, book, shield, gavel, globe
- scales: balance/rights
- book: statutes/legislation
- shield: warnings/protections
- gavel: court/procedure
- globe: jurisdiction/context
""")


def generate_sections(answer: str, response_language: str = "english") -> list[dict]:
    chain = SECTIONS_PROMPT | _llm() | StrOutputParser()
    try:
        raw = chain.invoke({"answer": answer, "response_language": response_language})
        sections = _parse_json_array(raw)
        valid = [
            s for s in sections
            if isinstance(s, dict) and s.get("heading") and s.get("content")
        ]
        logger.info(f"Generated {len(valid)} sections")
        return valid
    except Exception as e:
        logger.error(f"Section generation failed: {e}")
        return []


# ── Judgments ──────────────────────────────────────────────────────────
JUDGMENTS_PROMPT = ChatPromptTemplate.from_template("""You are providing 3-5 ILLUSTRATIVE judgments showing how courts in {jurisdiction} have approached issues similar to the user's query. These are educational examples from your training knowledge — they will be displayed with a clear "illustrative only, verify before relying" disclaimer.

USER QUERY: {query}
JURISDICTION: {jurisdiction}

CRITICAL — RELEVANCE:
Every judgment you suggest MUST be directly relevant to the legal issue in the user's query.
- A query about deposit recovery → judgments about tenancy deposits, rent recovery, landlord-tenant disputes
- A query about divorce dowry → judgments about dowry, gifts to bride, matrimonial property
- DO NOT suggest unrelated cases just to fill the list. If you cannot find 3 relevant cases, return fewer (minimum 2).

Return ONLY a JSON array, no preamble, no markdown fence:
[
  {{
    "case_title": "Realistic case naming for the jurisdiction",
    "citation": "Realistic citation format",
    "court": "Actual court name",
    "outcome": "Brief outcome",
    "sections": ["Statutory section invoked"],
    "summary": "2-4 sentence factual neutral summary of the case and its holding",
    "cited_cases": ["Other case 1", "Other case 2"]
  }}
]

Citation conventions for {jurisdiction}:
- PK: "X v. Y" with citations like "PLD 1980 SC 9", "2025 SCMR 1142", "2018 CLC 100"
- UK: "Smith v Jones [2020] EWCA Civ 123", "Re X (1995) 1 WLR 100"
- DE: "BGH VIII ZR 71/05", "OLG München 5 U 123/22"

- summary: 2-4 sentences, factual and neutral
- sections: list of statutory sections invoked (max 4)
- cited_cases: list of other cases referenced (max 5, can be empty)
- outcome: brief, e.g. "Appeal Allowed", "Petition Dismissed", "Eviction Upheld"
- court must be the actual court relevant to the issue
""")


def generate_judgments(query: str, jurisdiction: str) -> list[dict]:
    chain = JUDGMENTS_PROMPT | _llm() | StrOutputParser()
    try:
        raw = chain.invoke({"query": query, "jurisdiction": jurisdiction})
        judgments = _parse_json_array(raw)
        valid = [
            j for j in judgments
            if isinstance(j, dict)
            and j.get("case_title")
            and j.get("citation")
            and j.get("summary")
        ]
        # Ensure list fields exist even if LLM omitted them
        for j in valid:
            j.setdefault("sections", [])
            j.setdefault("cited_cases", [])
            j.setdefault("court", "")
            j.setdefault("outcome", "")
        logger.info(f"Generated {len(valid)} judgments")
        return valid
    except Exception as e:
        logger.error(f"Judgment generation failed: {e}")
        return []


# ── Combined ──────────────────────────────────────────────────────────
def structure_response(answer: str, query: str, jurisdiction: str, response_language: str) -> dict:
    """Run sections + judgments + follow-ups generation."""
    sections = generate_sections(answer, response_language=response_language)
    judgments = generate_judgments(query, jurisdiction)
    follow_ups = generate_followups(query, answer, jurisdiction, response_language)
    return {
        "sections": sections,
        "judgments": judgments,
        "follow_up_questions": follow_ups,
    }

# ── Follow-up Questions ───────────────────────────────────────────────
FOLLOWUPS_PROMPT = ChatPromptTemplate.from_template("""You are suggesting 3-4 SHORT follow-up questions a user is likely to ask after reading a legal answer. These will be displayed as clickable suggestions.

LANGUAGE: {response_language}
- Questions MUST be in {response_language}.
- For Roman-Urdu, use Pakistani Roman-Urdu words (koshish, qanoon, adalat, muqadma, haq), NOT Hindi-leaning ones.

USER'S ORIGINAL QUERY:
{query}

THE ANSWER THEY JUST READ:
{answer}

JURISDICTION: {jurisdiction}

Generate 3-4 follow-up questions that:
1. Are SHORT (under 12 words each)
2. Drill DEEPER into specifics not yet covered (deadlines, evidence, costs, edge cases)
3. Anticipate the user's likely real-world situation
4. Are CONCRETE, not vague ("How long does it take?" not "What else?")
5. Each must explore a DIFFERENT angle — no two questions overlapping

Examples of good follow-ups (deposit query):
- "Kitna time lagta hai Rent Controller ka faisla aane mein?"
- "Agar landlord court mein aaye hi nahi to kya hota hai?"
- "Lawyer hire karna zaroori hai ya khud case kar sakta hoon?"

Examples of BAD follow-ups (avoid):
- "Kya aur kuch bata sakte hain?" (too vague)
- "Yeh process kaisa hai?" (already covered)

Return ONLY a JSON array of strings, no preamble, no markdown fence:
["question 1", "question 2", "question 3"]
""")


def generate_followups(query: str, answer: str, jurisdiction: str, response_language: str) -> list[str]:
    chain = FOLLOWUPS_PROMPT | _llm() | StrOutputParser()
    try:
        raw = chain.invoke({
            "query": query,
            "answer": answer,
            "jurisdiction": jurisdiction,
            "response_language": response_language,
        })
        items = _parse_json_array(raw)
        # Filter to non-empty strings, max 4
        valid = [s.strip() for s in items if isinstance(s, str) and s.strip()][:4]
        logger.info(f"Generated {len(valid)} follow-up questions")
        return valid
    except Exception as e:
        logger.error(f"Follow-up generation failed: {e}")
        return []