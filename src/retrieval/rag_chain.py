import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.retrieval.embedder import search

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Prompt Template ───────────────────────────────────────────
LEGAL_PROMPT = ChatPromptTemplate.from_template("""
You are Adalat-AI, a precise legal assistant specializing in:
- Pakistan Constitutional Law & Penal Code
- UK Tenant Fees Act
- German Rental Law (BGB §§535-548)

STRICT RULES:
1. Answer ONLY from the provided legal context below
2. Every claim must cite its source (document + page number)
3. If the answer is not in the context, say: "I cannot find this in the available legal documents. Please consult a qualified lawyer."
4. Always end with: "⚠️ This is informational only. Consult a qualified lawyer for legal advice."
5. Structure your answer clearly with: Legal Basis, Your Rights, Your Recourse

LEGAL CONTEXT:
{context}

USER QUESTION: {question}

ANSWER (with citations):
""")

def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into context string with citations."""
    context_parts = []
    for i, r in enumerate(results):
        meta = r["metadata"]
        context_parts.append(
            f"[Source {i+1}: {meta['source']} | Page {meta['page_num']} | {meta['jurisdiction']}]\n"
            f"{r['text']}\n"
        )
    return "\n---\n".join(context_parts)


def format_citations(results: list[dict]) -> list[dict]:
    """Extract citation list from results."""
    citations = []
    for r in results:
        meta = r["metadata"]
        citations.append({
            "source": meta["source"],
            "page": meta["page_num"],
            "jurisdiction": meta["jurisdiction"],
            "relevance_score": r["score"]
        })
    return citations


def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024
    )


def run_rag(query: str, jurisdiction: str = None, top_k: int = 5) -> dict:
    """
    Full RAG pipeline:
    query → retrieve → format context → LLM → answer + citations
    """
    logger.info(f"Query: {query} | Jurisdiction: {jurisdiction}")

    # Step 1: Retrieve relevant chunks
    results = search(query, jurisdiction=jurisdiction, top_k=top_k)

    if not results:
        return {
            "query": query,
            "answer": "No relevant legal documents found. Please consult a qualified lawyer.",
            "citations": [],
            "jurisdiction": jurisdiction
        }

    # Step 2: Format context
    context = format_context(results)
    citations = format_citations(results)

    # Step 3: Run LLM
    llm = get_llm()
    prompt = LEGAL_PROMPT
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": query
    })

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "jurisdiction": jurisdiction,
        "chunks_used": len(results)
    }


if __name__ == "__main__":
    # Test 1 - English query (UK)
    print("\n" + "="*60)
    print("TEST 1: UK Tenant Rights")
    print("="*60)
    result = run_rag(
        "What fees can my landlord charge me?",
        jurisdiction="UK"
    )
    print(result["answer"])
    print("\nCITATIONS:")
    for c in result["citations"]:
        print(f"  - {c['source']} | Page {c['page']} | Score: {c['relevance_score']}")

    # Test 2 - Pakistan Constitutional
    print("\n" + "="*60)
    print("TEST 2: Pakistan Constitutional Rights")
    print("="*60)
    result = run_rag(
        "What are my fundamental rights if I am arrested?",
        jurisdiction="PK"
    )
    print(result["answer"])
    print("\nCITATIONS:")
    for c in result["citations"]:
        print(f"  - {c['source']} | Page {c['page']} | Score: {c['relevance_score']}")

    # Test 3 - Roman Urdu
    print("\n" + "="*60)
    print("TEST 3: Roman Urdu Query")
    print("="*60)
    result = run_rag(
        "mera landlord deposit wapas nahi de raha",
        jurisdiction="PK"
    )
    print(result["answer"])
    print("\nCITATIONS:")
    for c in result["citations"]:
        print(f"  - {c['source']} | Page {c['page']} | Score: {c['relevance_score']}")