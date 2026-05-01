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
class AgentState(TypedDict):
    query: str
    language: Optional[str]
    jurisdiction: Optional[str]
    translated_query: Optional[str]
    answer: Optional[str]
    citations: Optional[list]
    error: Optional[str]


# ── LLM ──────────────────────────────────────────────────────
def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=512
    )


# ── Node 1: Detect Language ───────────────────────────────────
def detect_language(state: AgentState) -> AgentState:
    query = state["query"]

    prompt = ChatPromptTemplate.from_template("""
Analyze this text and respond with ONLY a JSON object, nothing else.

Text: "{query}"

Rules:
- If text contains Roman-Urdu (Urdu written in English letters like "mera", "kya", "hai", "nahi", "wapas", "landlord", "deposit"), language = "roman_urdu"
- If text is in German (words like "Vermieter", "Kaution", "Miete"), language = "german"  
- Otherwise language = "english"

Respond ONLY with this exact format:
{{"language": "roman_urdu"}}
or
{{"language": "english"}}
or
{{"language": "german"}}
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

    if language != "roman_urdu":
        return {**state, "translated_query": query}

    prompt = ChatPromptTemplate.from_template("""
Translate this Roman-Urdu legal query to clear English.
Keep legal terms intact. Be concise.

Roman-Urdu: "{query}"

English translation:
""")

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    translated = chain.invoke({"query": query}).strip()

    logger.info(f"Translated: '{query}' → '{translated}'")
    return {**state, "translated_query": translated}


# ── Node 4: RAG Answer ────────────────────────────────────────
def run_rag_node(state: AgentState) -> AgentState:
    from src.retrieval.rag_chain import run_rag

    query_to_use = state.get("translated_query") or state["query"]
    jurisdiction = state["jurisdiction"]

    try:
        result = run_rag(query_to_use, jurisdiction=jurisdiction)
        return {
            **state,
            "answer": result["answer"],
            "citations": result["citations"]
        }
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return {
            **state,
            "answer": "An error occurred. Please consult a qualified lawyer.",
            "citations": [],
            "error": str(e)
        }


# ── Build LangGraph ───────────────────────────────────────────
def build_router():
    graph = StateGraph(AgentState)

    graph.add_node("detect_language", detect_language)
    graph.add_node("detect_jurisdiction", detect_jurisdiction)
    graph.add_node("translate_query", translate_query)
    graph.add_node("run_rag", run_rag_node)

    graph.set_entry_point("detect_language")
    graph.add_edge("detect_language", "detect_jurisdiction")
    graph.add_edge("detect_jurisdiction", "translate_query")
    graph.add_edge("translate_query", "run_rag")
    graph.add_edge("run_rag", END)

    return graph.compile()


def ask(query: str) -> dict:
    """Main entry point — just pass any query, router handles everything."""
    router = build_router()
    result = router.invoke({"query": query})
    return result


if __name__ == "__main__":
    test_queries = [
        "What fees can my landlord charge me in the UK?",
        "mera landlord deposit wapas nahi de raha",
        "Mein Vermieter gibt meine Kaution nicht zurück",
        "Can police detain me without charge in Pakistan?",
    ]

    for q in test_queries:
        print("\n" + "="*60)
        print(f"QUERY: {q}")
        print("="*60)
        result = ask(q)
        print(f"Language:     {result['language']}")
        print(f"Jurisdiction: {result['jurisdiction']}")
        if result.get('translated_query') != q:
            print(f"Translated:   {result['translated_query']}")
        print(f"\nANSWER:\n{result['answer']}")
        print("\nCITATIONS:")
        for c in (result.get('citations') or []):
            print(f"  - {c['source']} | Page {c['page']} | Score: {c['relevance_score']}")