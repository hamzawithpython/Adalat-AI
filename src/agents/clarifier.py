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

{history_context}USER QUERY ({response_language}): {query}
JURISDICTION: {jurisdiction}

Apply this judgment LIKE A CONFIDENT SENIOR LAWYER. A real lawyer answers most questions directly and only asks for clarification when the facts are genuinely missing in a way that would change the legal analysis. Lawyers do NOT interrogate clients on every question — they answer, and ask only when truly necessary.

YOUR STRONG DEFAULT IS "answer". Choose "clarify" ONLY when one of these is true:
1. The query has no identifiable legal subject at all. Examples: "Mera masla hai", "Mujhe help chahiye", "I have a problem", "Police ne mujhe roka" (police did what?).
2. The user mentions an event without any context. Example: "Mujhe arrest kiya gaya" — but doesn't say where, what for, when, or what they need to know.
3. The query mixes multiple unrelated issues such that one answer can't address all of them.

OTHERWISE, choose "answer". Specifically, choose "answer" even when:
- The query is a hypothetical ("what if the judge does X?", "agar landlord court mein nahi aaye to?", "what happens if I miss the deadline?") — answer the hypothetical directly.
- The query is procedural ("how do I file a constitutional petition?", "CNIC misuse hua hai, kya karoon?") — answer the procedure; you don't need every personal detail to explain a process.
- The query mentions a legal topic and asks a reasonable question about it ("Kisi ne mera CNIC misuse kiya hai", "deposit wapas nahi mil raha") — answer with the standard legal pathway. If specific facts would change advice, mention them as "depending on whether X" qualifiers IN THE ANSWER itself, do not ask separately.
- The query lacks city/province but the legal answer is roughly the same across provinces — answer generally and note "exact procedure varies by province".
- The query is in plain language with one clear question — answer it.

Crucially: a confident lawyer would rather give a 90%-accurate answer with qualifiers than ask 4 questions before saying anything. Clarify ONLY when answering is genuinely impossible without more facts.

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

Examples requiring "clarify" (genuinely too vague to answer):
- "Mera masla hai" → ask: what type of issue, which jurisdiction, what outcome do you want
- "Police ne mujhe roka" → ask: what for, what did they do, were you detained
- "I have a problem" → ask: nature of problem, which jurisdiction
- "Mujhe help chahiye" → ask: with what issue, which jurisdiction

Examples that should "answer" (not clarify, even if details are missing):
- "High Court me constitutional petition kaise file hoti hai?" → answer: explain the Article 199 writ procedure directly. Don't ask "constitutional or civil?" — they already said constitutional.
- "Kisi ne mera CNIC misuse kiya hai" → answer: explain the standard pathway (NADRA verification, FIR under PPC, etc.). Don't ask who, when, where — those don't change the legal procedure.
- "Agar judge opposite interpretation le to kya ho sakta hai?" → answer: explain appellate remedies, review petitions, the legal hierarchy. It's a hypothetical, not a personal fact pattern.
- "My landlord won't return my deposit" → answer: explain the deposit recovery process. Don't ask which country if jurisdiction is already detected upstream.
- "What happens if I miss the appeal deadline?" → answer: explain the consequences. It's clearly hypothetical.
""")


def _llm():
    from src.agents.llms import fast_llm
    return fast_llm(max_tokens=600, temperature=0.1)


def assess_completeness(query: str, jurisdiction: str, response_language: str,
                         conversation_history: list = None) -> dict:
    """Returns: {decision: 'answer'|'clarify', reason: str, questions: list[str]}"""
    # If the user is already in a multi-turn conversation, skip clarification — they're iterating
    # on an established context. Even one prior turn is usually enough context.
    if conversation_history and len(conversation_history) >= 1:
        return {"decision": "answer", "reason": "follow-up turn — prior context available", "questions": []}

    # Build history context (only used if auto-skip didn't trigger above)
    history_context = ""
    if conversation_history:
        recent = conversation_history[-2:]
        history_context = "═══ PREVIOUS CONVERSATION (use to interpret the current query) ═══\n"
        for turn in recent:
            history_context += f"USER: {turn.get('query', '')}\nADALAT: {turn.get('answer', '')[:300]}...\n"
        history_context += "═══ The CURRENT QUERY below is a follow-up. Resolve any references using the above context before deciding. ═══\n\n"

    chain = CLARIFIER_PROMPT | _llm() | StrOutputParser()
    try:
        raw = chain.invoke({
            "query": query,
            "jurisdiction": jurisdiction,
            "response_language": response_language,
            "history_context": history_context,
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