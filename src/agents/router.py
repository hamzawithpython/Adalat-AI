import os
import sys
import logging
from typing import TypedDict, Optional
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── State Schema ──────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    query: str
    language: Optional[str]
    jurisdiction: Optional[str]
    translated_query: Optional[str]
    answer: Optional[str]
    citations: Optional[list]
    sections: Optional[list]
    judgments: Optional[list]
    follow_up_questions: Optional[list]
    response_language: Optional[str]
    error: Optional[str]
    # Clarifier additions
    clarification: Optional[dict]
    is_clarification: Optional[bool]
    conversation_history: Optional[list]


# ── LLM ──────────────────────────────────────────────────────
def get_llm():
    from src.agents.llms import fast_llm
    return fast_llm(max_tokens=512)


# ── Node 1: Detect Language ───────────────────────────────────
def detect_language(state: AgentState) -> AgentState:
    query = state["query"]

    prompt = ChatPromptTemplate.from_template("""
You are a language detector for legal queries. Determine the language the user actually WROTE IN. Return ONLY a JSON object.

Text: "{query}"

CRITICAL RULE: Judge by the SENTENCE STRUCTURE and connecting words, NOT by individual nouns. English loanwords like "landlord", "deposit", "rent", "tenancy", "police", "contract" appear in ALL THREE languages and must be IGNORED for detection — they are NOT evidence of any particular language.

Decide using the GRAMMAR and FUNCTION WORDS (verbs, pronouns, connectors):

1. GERMAN — if the sentence is built with German grammar/function words: ist, der, die, das, mein, nicht, und, wenn, kann, möchte, wird, Vermieter, Kaution, Miete, zurück.
   Example: "Mein Vermieter erhöht die Miete." → german

2. ROMAN_URDU — ONLY if the sentence uses Roman-Urdu grammar and connecting words: mera/meri/mujhe, hai/hain, nahi, kya, karoon/karna, raha/rahi, ke/ka/ki, ko, mein, sakta, agar, toh, wapas.
   Example: "Mera landlord deposit wapas nahi de raha" → roman_urdu (note: "landlord" and "deposit" are English words, but "Mera...wapas nahi de raha" is Roman-Urdu grammar)

3. ENGLISH — if the sentence is built with English grammar and function words: my, is, are, the, what, can, should, was, were, my landlord, I want, what are my rights.
   Example: "My landlord in Germany wants to increase my rent" → english (it is English grammar — the words "landlord" and "rent" do NOT make it Roman-Urdu)

Default to "english" if the sentence structure is English, regardless of which legal nouns appear.

Return ONLY one of these three exact JSON objects:
{{"language": "german"}}
{{"language": "roman_urdu"}}
{{"language": "english"}}
""")

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query})

    import json, re
    try:
        match = re.search(r'\{.*?\}', result, re.DOTALL)
        if match:
            data = json.loads(match.group())
            language = data.get("language", "english")
        else:
            language = "english"
    except:
        language = "english"

    logger.info(f"Detected language: {language}")
    return {**state, "language": language}


# ── Node 2: Detect Jurisdiction ───────────────────────────────
def detect_jurisdiction(state: AgentState) -> AgentState:
    query = state["query"]
    language = state["language"]

    prompt = ChatPromptTemplate.from_template("""
You are a legal jurisdiction classifier. Analyze the query and return ONLY a JSON object.

Query: "{query}"
Language: "{language}"

Jurisdiction Rules:
- PK: Pakistan law, constitutional rights, arrest, PPC, Pakistani tenant issues, Roman-Urdu queries about Pakistani issues
- UK: UK tenant rights, UK landlord, deposit UK, Tenant Fees Act, England/Wales rental
- DE: German law, Vermieter, Kaution, BGB, German rental, Miete

Respond ONLY with:
{{"jurisdiction": "PK"}}
or
{{"jurisdiction": "UK"}}
or
{{"jurisdiction": "DE"}}
""")

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query, "language": language})

    import json, re
    try:
        match = re.search(r'\{.*?\}', result, re.DOTALL)
        if match:
            data = json.loads(match.group())
            jurisdiction = data.get("jurisdiction", "PK")
        else:
            jurisdiction = "PK"
    except:
        jurisdiction = "PK"

    logger.info(f"Detected jurisdiction: {jurisdiction}")
    return {**state, "jurisdiction": jurisdiction}


# ── Node 3: Translate Roman-Urdu ──────────────────────────────
def translate_query(state: AgentState) -> AgentState:
    query = state["query"]
    language = state["language"]

    if language == "english":
        return {**state, "translated_query": query}

    prompt = ChatPromptTemplate.from_template("""
Translate this text to English. Return ONLY the English translation, nothing else.
No explanations, no prefixes, no quotes.

Text: "{query}"

English translation:
""")

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    translated = chain.invoke({"query": query}).strip()

    # Remove any prefix patterns the LLM might add
    prefixes = [
        "the roman-urdu query translates to:",
        "the translation of the roman-urdu query is:",
        "the translation is:",
        "translation:",
        "english translation:",
    ]
    for prefix in prefixes:
        if translated.lower().startswith(prefix):
            translated = translated[len(prefix):].strip()

    translated = translated.strip('"').strip("'").strip()
    logger.info(f"Translated: '{query}' → '{translated}'")
    return {**state, "translated_query": translated}


# ── Node 4: RAG Answer ────────────────────────────────────────
def run_rag_node(state: AgentState) -> AgentState:
    from src.retrieval.rag_chain import run_rag

    query_to_use = state.get("translated_query") or state["query"]
    jurisdiction = state["jurisdiction"]
    response_language = state.get("language") or "english"
    conversation_history = state.get("conversation_history") or []

    try:
        result = run_rag(
            query_to_use,
            jurisdiction=jurisdiction,
            response_language=response_language,
            conversation_history=conversation_history,
        )
        return {
            **state,
            "answer": result["answer"],
            "citations": result["citations"],
            "response_language": response_language,
        }
    except BaseException as e:
        logger.exception(f"RAG error: {e}")
        return {
            **state,
            "answer": "An error occurred. Please consult a qualified lawyer.",
            "citations": [],
            "error": str(e)
        }

# ── Node 5: Structure Response ──────────────────────────────────────────
def structure_response_node(state: AgentState) -> AgentState:
    from src.agents.structurer import structure_response

    if not state.get("answer") or state.get("error"):
        return {**state, "sections": [], "judgments": [], "follow_up_questions": []}

    try:
        result = structure_response(
            answer=state["answer"],
            query=state["query"],
            jurisdiction=state["jurisdiction"],
            response_language=state.get("response_language") or "english",
        )
        return {
            **state,
            "sections": result["sections"],
            "judgments": result["judgments"],
            "follow_up_questions": result.get("follow_up_questions", []),
        }
    except BaseException as e:
        logger.exception(f"Structurer error: {e}")
        return {**state, "sections": [], "judgments": [], "follow_up_questions": []}

# ── Node: Assess (decides whether to clarify or answer) ─────────────
def assess_node(state: AgentState) -> AgentState:
    """Decides whether the query has enough facts for a real legal answer."""
    from src.agents.clarifier import assess_completeness
    decision = assess_completeness(
        query=state.get("translated_query") or state["query"],
        jurisdiction=state["jurisdiction"],
        response_language=state.get("language") or "english",
        conversation_history=state.get("conversation_history") or [],
    )
    return {**state, "clarification": decision}


def route_after_assess(state: AgentState) -> str:
    """Conditional edge: if clarification needed, skip RAG."""
    if state.get("clarification", {}).get("decision") == "clarify":
        return "clarify_only"
    return "run_rag"


# ── Node: Clarify Only (returns clarifying questions instead of answer) ─
def clarify_only_node(state: AgentState) -> AgentState:
    """Skips RAG. Returns a clarification message in user's language."""
    questions = state["clarification"]["questions"]
    lang = state.get("language") or "english"

    if lang == "roman_urdu":
        intro = "Aap ke maamlay ko theek se assess karne ke liye, mujhe kuch baatein clear karni hain:"
    elif lang == "german":
        intro = "Um Ihre Situation richtig einzuschätzen, brauche ich noch ein paar Angaben:"
    else:
        intro = "To assess your situation properly, I need a few clarifications:"

    answer = intro + "\n\n" + "\n".join(f"• {q}" for q in questions)

    return {
        **state,
        "answer": answer,
        "citations": [],
        "sections": [],
        "judgments": [],
        "follow_up_questions": questions,  # so the UI can render them as clickable chips
        "response_language": lang,
        "is_clarification": True,
    }

# ── Build LangGraph ───────────────────────────────────────────
def build_router():
    graph = StateGraph(AgentState)

    graph.add_node("detect_language", detect_language)
    graph.add_node("detect_jurisdiction", detect_jurisdiction)
    graph.add_node("translate_query", translate_query)
    graph.add_node("assess", assess_node)
    graph.add_node("clarify_only", clarify_only_node)
    graph.add_node("run_rag", run_rag_node)
    graph.add_node("structure_response", structure_response_node)

    graph.set_entry_point("detect_language")
    graph.add_edge("detect_language", "detect_jurisdiction")
    graph.add_edge("detect_jurisdiction", "translate_query")
    graph.add_edge("translate_query", "assess")
    graph.add_conditional_edges("assess", route_after_assess, {
        "clarify_only": "clarify_only",
        "run_rag": "run_rag",
    })
    graph.add_edge("clarify_only", END)
    graph.add_edge("run_rag", "structure_response")
    graph.add_edge("structure_response", END)

    return graph.compile()


def ask(query: str, conversation_history: list = None) -> dict:
    """Main entry point — just pass any query, router handles everything.
    
    Optional `conversation_history` is a list of prior turns:
        [{"query": "...", "answer": "..."}, ...]
    Used to skip clarification when the user is already iterating in a session.
    """
    from src.schemas.extractor import extract_rights
    from src.schemas.legal_response import build_legal_response

    router = build_router()
    result = router.invoke({
        "query": query,
        "conversation_history": conversation_history or [],
    })

    # If this is a clarification response, skip rights extraction (no real answer to extract from)
    if result.get("is_clarification"):
        rights = []
    else:
        rights = extract_rights(result.get("answer", ""))

    # Build validated Pydantic response
    response = build_legal_response(result, rights)

    # Surface the clarification flag and questions to the API consumer
    response_dict = response.model_dump()
    response_dict["is_clarification"] = result.get("is_clarification", False)
    return response_dict


if __name__ == "__main__":
    test_queries = [
        "What fees can my landlord charge me in the UK?",          # answer
        "mera landlord deposit wapas nahi de raha",                 # answer
        "Police ne mujhe roka",                                     # CLARIFY — vague police interaction
        "I have a problem",                                         # CLARIFY — way too vague
    ]

    for q in test_queries:
        print("\n" + "="*60)
        print(f"QUERY: {q}")
        print("="*60)
        result = ask(q)
        print(f"Language:     {result['language']}")
        print(f"Jurisdiction: {result['jurisdiction']}")
        print(f"Confidence:   {result['confidence']}")
        if result.get('translated_query'):
            print(f"Translated:   {result['translated_query']}")
        print(f"\nANSWER:\n{result['answer'][:300]}...")
        print(f"\nRIGHTS EXTRACTED: {len(result['rights'])}")
        for r in result['rights']:
            print(f"  Right:       {r['right']}")
            print(f"  Legal Basis: {r['legal_basis']}")
            print(f"  Deadline:    {r['deadline']}")
            print(f"  Recourse:    {r['recourse']}")
            print()
        print(f"CITATIONS: {len(result['citations'])}")
        for c in result['citations']:
            print(f"  - {c['source']} | Page {c['page']} | Score: {c['relevance_score']}")