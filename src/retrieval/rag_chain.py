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
LEGAL_PROMPT = ChatPromptTemplate.from_template("""You are Adalat-AI, an expert legal assistant for Pakistani, UK, and German law.

═══════════════════════════════════════════════════
LANGUAGE RULE — ABSOLUTELY CRITICAL
═══════════════════════════════════════════════════
The user wrote in {response_language}. Write your ENTIRE answer in {response_language}.

For Roman-Urdu specifically — use PAKISTANI Roman-Urdu, NOT Hindi-leaning:
✓ USE: koshish, taluq, muqadma, adalat, faisla, haq, hukum, ijazat, mutaliq, halaat, tasdeeq, dawa, raasta, jaaiz, na-jaaiz, sahih, ghalat, qanoon, hukumat, maamla
✗ AVOID: prayas, sambandh, vivad, nyayalay, nirnay, adhikaar, aagya, bare mein, sthiti, pramaan, daawa, maarg, uchit, an-uchit, theek, galat, kanoon (use 'qanoon'), sarkar (use 'hukumat'), vishay

Statute names, section numbers, and English legal terms (e.g. "Rent Tribunal", "Controller", "Section 13") stay as-is.

═══════════════════════════════════════════════════
WRITING RULES — STRICTLY ENFORCED
═══════════════════════════════════════════════════
0. GROUNDING IS NON-NEGOTIABLE.
   - Every statutory section, deadline, penalty, and case citation MUST come from the LEGAL CONTEXT below.
   - If the context does not specify a deadline or penalty, do NOT invent one. Say "specific deadline is not in the available text — verify with the relevant statute".
   - If the user's question cannot be answered from the context, say so directly: in {response_language}, words to the effect of "The available legal documents do not specifically address this. Consult a qualified lawyer."
   - NEVER cite an act, ordinance, or section number that is not present in the LEGAL CONTEXT below.
   - General principles are fine, but frame them as principles ("In general, Pakistani tenancy law treats deposits as..."), NOT as specific statutory citations.
1. NO REPETITION. Each paragraph adds NEW information. Never restate the same advice in different words.
2. SPECIFICITY OVER GENERALITY. Quote exact statutory text where possible. Cite specific deadlines ("30 din ke andar"), penalties ("1% per month interest"), procedures.
3. NO PADDING. Length follows substance. If you've covered the question fully in 4 paragraphs, STOP.
4. NO META. Do NOT begin with phrases like "Aap ke paas yeh haq hai" or "Yeh ek important sawal hai". Go straight to the substantive answer.
5. NO DISCLAIMERS — they're added separately.
6. NO RIGID HEADINGS like "Legal Basis:" / "Your Rights:" — those are added later.
                                                

═══════════════════════════════════════════════════
STRUCTURE — FOLLOW THIS PATTERN
═══════════════════════════════════════════════════
Paragraph 1 (1-2 sentences): Direct answer to the user's question. State the legal position clearly.

Paragraph 2 (2-4 sentences): The SPECIFIC statutory provision. Name the act, section, what it actually says. Quote key text in "..." if available in context.

Paragraph 3 (2-4 sentences): Concrete procedural steps the user must take, with timelines if known. Be specific: which forum, which form, which deadline.

Paragraph 4 (2-3 sentences): Evidence and documents to gather. Practical, actionable list.

Paragraph 5 (OPTIONAL, only if useful): Realistic position. What outcome can the user actually expect? What's the failure mode?

═══════════════════════════════════════════════════
GOLD-STANDARD EXAMPLE (English query)
═══════════════════════════════════════════════════
USER QUERY: "My tenant has been late on rent for 3 months. Can I evict them?"
JURISDICTION: PK
RESPONSE_LANGUAGE: english

GOOD ANSWER:
Yes, you may pursue eviction on the ground of wilful default in payment of rent. Under Pakistani tenancy law, three months of unpaid rent typically constitutes wilful default, which is a statutory ground for ejectment.

Under the Sindh Rented Premises Ordinance 1979, Section 15, a landlord may apply to the Rent Controller for ejectment when a tenant has "failed to pay rent within sixty days of its becoming due, and has continued such default after written demand". The Punjab equivalent is found in Section 15 of the Punjab Rented Premises Act 2009. Both statutes require that the default be wilful and that proper notice has been served.

Your procedure: send a registered written demand giving 30 days to clear arrears. If unpaid, file an ejectment petition with the Rent Controller of the district. The Controller will issue notice, hold a summary hearing, and may pass a tentative rent deposit order under Section 16. Failure by the tenant to comply with the tentative order strengthens your case substantially.

Gather: tenancy agreement (signed copy), rent receipts or bank statements showing non-payment, copies of any written demands sent (preferably registered post receipts), and witness statements if rent was demanded in person.

Realistically, well-documented three-month defaults result in eviction orders within 6–9 months at the Rent Controller level, with possible appeals adding 6–12 months. The tenant's main defence is usually disputing the landlord-tenant relationship itself or claiming receipts the landlord won't acknowledge.

═══════════════════════════════════════════════════
LEGAL CONTEXT (use ONLY this content for citations)
═══════════════════════════════════════════════════
{context}

═══════════════════════════════════════════════════
USER QUESTION ({response_language}): {question}
═══════════════════════════════════════════════════

Your answer (in {response_language}, following the structure above, no headings, no disclaimers, no padding):
""")

def format_context(results: list[dict]) -> str:
    context_parts = []
    for i, r in enumerate(results):
        meta = r["metadata"]
        breadcrumb = r.get("breadcrumb", meta.get("source", ""))
        warning = r.get("currency_warning", "")
        warning_str = f"\n⚠️ {warning}" if warning else ""
        context_parts.append(
            f"[Source {i+1}: {breadcrumb} | "
            f"Page {meta.get('page_start', '?')}]{warning_str}\n"
            f"{r['text']}\n"
        )
    return "\n---\n".join(context_parts)


def format_citations(results: list[dict]) -> list[dict]:
    citations = []
    for r in results:
        meta = r["metadata"]
        citations.append({
            "source": meta.get("source", ""),
            "page": meta.get("page_start", 0),
            "jurisdiction": meta.get("country", ""),
            "relevance_score": r["score"],
            "breadcrumb": r.get("breadcrumb", ""),
            "currency_warning": r.get("currency_warning"),
            "requires_escalation_cue": r.get("requires_escalation_cue", False),
        })
    return citations


def get_llm():
    from src.agents.llms import heavy_llm
    return heavy_llm(max_tokens=2048, temperature=0.2)


def run_rag(query: str, jurisdiction: str = None, top_k: int = 5, response_language: str = "english") -> dict:
    """
    Full RAG pipeline:
    query → retrieve → format context → LLM → answer + citations
    """
    logger.info(f"Query: {query} | Jurisdiction: {jurisdiction}")

    # Step 1: Retrieve relevant chunks
    results = search(query, country=jurisdiction, top_k=top_k)

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
        "question": query,
        "response_language": response_language,
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