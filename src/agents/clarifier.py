"""Decides whether a query has enough facts for a confident legal answer.
If not, returns lawyer-style clarifying questions instead of running RAG."""

import os
import sys
import json
import re
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


CLARIFIER_PROMPT = ChatPromptTemplate.from_template("""You are a senior Pakistani lawyer triaging a client's question. Your job is to decide ONE thing: do you have enough facts to give a real legal opinion, or do you need to ask clarifying questions first?

USER QUERY ({response_language}): {query}
JURISDICTION: {jurisdiction}

Apply this judgment:

DECISION = "answer" if:
- The query describes a complete enough situation to identify the legal issue and recommend an action.
- It's a general legal question that doesn't need personal facts (e.g. "What are the permitted fees a UK landlord can charge?", "What are my rights if arrested in Pakistan?").
- The query is specific enough to identify jurisdiction, type of issue, and what the user wants to do (e.g. "My landlord hasn't returned my deposit after 6 months in Lahore" — jurisdiction, duration, issue all present).

DECISION = "clarify" if:
- Critical facts are missing that would change the legal analysis — e.g. which city/province (since tenancy laws differ between Punjab/Sindh/KP), whether FIR was registered, whether a formal arrest happened vs. just questioning, whether a written agreement exists, whether a deadline has already passed.
- The query is so vague that any answer would be hedged into uselessness (e.g. "Police ne mujhe roka", "Mera masla hai", "Mujhe help chahiye").
- The user describes a police/criminal interaction without saying whether they were detained, whether documents were seized, or whether they were charged.

Be a real lawyer about this. Do NOT clarify trivially — only when the missing fact would meaningfully change the advice. If you can give a useful general answer with a "depending on whether X" qualifier, prefer ANSWER.

Return ONLY a JSON object, no preamble, no markdown fence:
{{
  "decision": "answer",
  "reason": "one short sentence in English explaining the decision",
  "questions": []
}}

OR

{{
  "decision": "clarify",
  "reason": "one short sentence in English explaining the decision",
  "questions": ["question 1 in {response_language}", "question 2 in {response_language}", "question 3 in {response_language}"]
}}

Rules for "questions":
- Empty list [] if decision is "answer".
- 2-4 short questions if decision is "clarify".
- Questions MUST be in {response_language}. Match the user's language exactly. Use Pakistani Roman-Urdu (NOT Hindi register) for roman_urdu.
- Each question targets ONE missing fact. No compound questions.
- Specific, not generic.

Examples of good clarifying questions for a vague Roman-Urdu police query "Police ne mujhe roka":
- "Incident kis shehar mein hua?"
- "Kya police ne sirf checking ki ya gari thanay bhi le gaye?"
- "Kya FIR register hui ya sirf zabani pooch-gachh thi?"
- "Kya koi cheez seize ki gayi?"
- "Kya aap ko formally detain kiya gaya tha ya jaane diya?"

Examples of good clarifying questions for a vague English query "I have a landlord problem":
- "Which country are you in (UK, Pakistan, Germany)?"
- "What specifically is the landlord doing or not doing?"
- "Is there a written tenancy agreement?"
- "How long has the issue been ongoing?"
""")


def _llm():
    from src.agents.llms import fast_llm
    return fast_llm(max_tokens=600, temperature=0.1)


def assess_completeness(query: str, jurisdiction: str, response_language: str,
                         conversation_history: list = None) -> dict:
    """Returns: {decision: 'answer'|'clarify', reason: str, questions: list[str]}"""
    # If the user is already in a multi-turn conversation, skip clarification — they're iterating
    if conversation_history and len(conversation_history) >= 2:
        return {"decision": "answer", "reason": "multi-turn context available", "questions": []}

    chain = CLARIFIER_PROMPT | _llm() | StrOutputParser()
    try:
        raw = chain.invoke({
            "query": query,
            "jurisdiction": jurisdiction,
            "response_language": response_language,
        })
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            logger.warning(f"Clarifier returned no JSON, defaulting to answer. Raw: {raw[:200]}")
            return {"decision": "answer", "reason": "parse failure, defaulting", "questions": []}
        result = json.loads(match.group())
        # Sanity: never block answer if questions list is empty
        if result.get("decision") == "clarify" and not result.get("questions"):
            return {"decision": "answer", "reason": "clarify with no questions", "questions": []}
        logger.info(f"Clarifier decision: {result.get('decision')} — {result.get('reason')}")
        return result
    except Exception as e:
        logger.warning(f"Clarifier failed, defaulting to answer: {e}")
        return {"decision": "answer", "reason": "clarifier error", "questions": []}


if __name__ == "__main__":
    # Smoke tests for the clarifier
    test_cases = [
        ("Police ne mujhe roka", "PK", "roman_urdu"),
        ("My landlord hasn't returned my deposit after 6 months in Lahore", "PK", "english"),
        ("I have a problem", "PK", "english"),
        ("What fees can my landlord charge me in the UK?", "UK", "english"),
        ("Mera masla hai", "PK", "roman_urdu"),
    ]

    for query, jur, lang in test_cases:
        print("\n" + "="*60)
        print(f"QUERY: {query}")
        print(f"JURISDICTION: {jur}  LANGUAGE: {lang}")
        print("="*60)
        result = assess_completeness(query, jur, lang)
        print(f"Decision: {result['decision']}")
        print(f"Reason: {result['reason']}")
        if result.get("questions"):
            for q in result["questions"]:
                print(f"  • {q}")