"""
Adalat-AI — RAGAS Evaluation
Evaluates the full RAG pipeline (retrieval + generation) on a labelled test set.

Metrics:
  - faithfulness        : is the answer grounded in retrieved context? (no hallucination)
  - answer_relevancy    : does the answer actually address the question?
  - context_precision   : are retrieved chunks relevant + well-ranked?
  - context_recall      : did retrieval capture what the ground-truth answer needs?

Judge LLM : Groq (llama-3.3-70b-versatile)
Embeddings: fastembed MiniLM (same model the app uses)

Usage:
    python configs/ragas_eval.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from src.agents.router import ask
from src.retrieval.embedder import search

# ── Judge LLM + embeddings for RAGAS ─────────────────────────────────────
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding

JUDGE_MODEL = "llama-3.3-70b-versatile"


class FastEmbedAdapter(Embeddings):
    """Wrap fastembed so RAGAS can use the same embeddings as the app."""
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self._m = TextEmbedding(model_name)

    def embed_documents(self, texts):
        return [e.tolist() for e in self._m.embed([f"passage: {t}" for t in texts])]

    def embed_query(self, text):
        return list(self._m.embed([f"query: {text}"]))[0].tolist()


# ── Test set: 12 labelled questions across PK / UK / DE ──────────────────
# Each is detailed enough to pass the clarifier and route to run_rag.
# ground_truth = a short reference answer (what a correct answer should contain).
# >>> SANITY-CHECK THE LEGAL ACCURACY OF EACH ground_truth BEFORE TRUSTING SCORES <<<
TEST_SET = [
    # ---- Pakistan ----
    {
        "question": "My landlord in Pakistan is refusing to return my security deposit after I vacated the rented house. What are my legal options?",
        "jurisdiction": "PK",
        "ground_truth": "Under Pakistani provincial rent laws (e.g. Sindh Rented Premises Ordinance / Punjab Rented Premises Act), a tenant who has vacated and cleared dues is entitled to the return of the security deposit. The tenant can serve a legal notice and, if refused, file an application before the Rent Controller / Rent Tribunal for recovery of the deposit.",
    },
    {
        "question": "I was arrested by police in Pakistan. What fundamental rights does the Constitution give me on arrest and detention?",
        "jurisdiction": "PK",
        "ground_truth": "The Constitution of Pakistan (Article 10) guarantees that an arrested person must be informed of the grounds of arrest, has the right to consult and be defended by a lawyer of choice, and must be produced before a magistrate within 24 hours of arrest. Detention beyond 24 hours without magistrate authorisation is unlawful.",
    },
    {
        "question": "In Pakistan, can the police detain me without formally charging me, and for how long?",
        "jurisdiction": "PK",
        "ground_truth": "Under the Constitution (Article 10) and the Criminal Procedure Code, an arrested person must be produced before a magistrate within 24 hours. Police cannot lawfully detain a person beyond 24 hours without the magistrate authorising further remand.",
    },
    {
        "question": "My employer in Pakistan terminated my employment without any notice or stated reason. What protection do I have under Pakistani labour law?",
        "jurisdiction": "PK",
        "ground_truth": "Under Pakistani labour law (e.g. Industrial and Commercial Employment (Standing Orders) Ordinance 1968), a workman is entitled to written reasons for termination and to notice or pay in lieu. Wrongful or unjust termination can be challenged before a Labour Court, which may order reinstatement or compensation.",
    },
    # ---- United Kingdom ----
    {
        "question": "What fees is my landlord legally allowed to charge me as a tenant in England under the Tenant Fees Act?",
        "jurisdiction": "UK",
        "ground_truth": "Under the Tenant Fees Act 2019, landlords and agents in England may only charge permitted payments: rent, a refundable tenancy deposit (capped, generally five weeks' rent), a refundable holding deposit (capped at one week's rent), and limited default fees. Most other fees (admin, referencing, renewal) are banned.",
    },
    {
        "question": "My landlord in the UK is trying to evict me without giving proper notice. What are my rights regarding eviction notice periods?",
        "jurisdiction": "UK",
        "ground_truth": "Under the Housing Act 1988 (assured shorthold tenancies), a landlord must follow a lawful process — serving a valid Section 21 or Section 8 notice with the correct statutory notice period, and obtaining a court possession order. A tenant cannot lawfully be evicted without proper notice and a court order.",
    },
    {
        "question": "I bought a faulty laptop from a UK retailer and they are refusing a refund. What are my rights under UK consumer law?",
        "jurisdiction": "UK",
        "ground_truth": "Under the Consumer Rights Act 2015, goods must be of satisfactory quality, fit for purpose, and as described. If faulty, the consumer has a short-term right to reject for a full refund within 30 days, and after that the right to repair, replacement, or a price reduction/refund.",
    },
    {
        "question": "I believe a UK employer refused to hire me because of my religion. What legal protection do I have against discrimination?",
        "jurisdiction": "UK",
        "ground_truth": "Under the Equality Act 2010, religion or belief is a protected characteristic, and it is unlawful for an employer to discriminate against a job applicant on that basis. A claim of direct discrimination can be brought to an employment tribunal; the Equality and Human Rights Commission (EHRC) can provide guidance.",
    },
    # ---- Germany ----
    {
        "question": "My landlord in Germany wants to increase my rent by 20% in one year. Is that allowed under German tenancy law (BGB)?",
        "jurisdiction": "DE",
        "ground_truth": "Under the German BGB, rent increases are restricted. A rent increase to the local comparative rent generally cannot exceed a capping limit (Kappungsgrenze) of 20% (15% in many areas) over three years, and the landlord must follow the formal procedure under §558 BGB. A sudden 20% increase in one year would generally be impermissible.",
    },
    {
        "question": "My German landlord is not returning my security deposit (Kaution) after I moved out. What does German law say?",
        "jurisdiction": "DE",
        "ground_truth": "Under the German BGB, the landlord may hold the security deposit (Kaution) for a reasonable period after tenancy end to check for claims (commonly up to 3-6 months), after which the remaining deposit, with interest, must be returned to the tenant. The tenant can demand return and pursue a civil claim if refused.",
    },
    {
        "question": "In Germany, what notice period must I give my landlord if I want to terminate my rental contract as a tenant?",
        "jurisdiction": "DE",
        "ground_truth": "Under the German BGB (§573c), a tenant terminating an open-ended residential lease must generally give three months' notice, with termination effective at the end of the relevant month if served by the third working day of that month.",
    },
    {
        "question": "Can my German landlord charge me for operating/ancillary costs (Betriebskosten) on top of rent, and what rules apply?",
        "jurisdiction": "DE",
        "ground_truth": "Under the German BGB and the Betriebskostenverordnung, a landlord may pass on defined operating costs (Betriebskosten) to the tenant only if this is agreed in the lease, and must provide an annual itemised statement. The tenant has the right to inspect supporting documents and dispute incorrect charges.",
    },
]


def build_dataset():
    """Run each test question through the real pipeline; capture answer + contexts."""
    from ragas.dataset_schema import SingleTurnSample

    samples = []
    rows_for_log = []

    for i, item in enumerate(TEST_SET, 1):
        q = item["question"]
        jur = item["jurisdiction"]
        print(f"\n[{i}/{len(TEST_SET)}] ({jur}) {q[:60]}...")

        # 1) contexts — retrieved chunk texts (same retrieval the app uses)
        hits = search(q, jurisdiction=jur, top_k=5)
        contexts = [h["text"] for h in hits if h.get("text")]
        if not contexts:
            print("   ⚠ no contexts retrieved — skipping this question")
            continue

        # 2) answer — full pipeline
        try:
            result = ask(q)
        except Exception as e:
            print(f"   ⚠ ask() failed: {e} — skipping")
            continue

        if result.get("is_clarification"):
            print("   ⚠ pipeline asked for clarification instead of answering — skipping")
            continue

        answer = result.get("answer", "")
        if not answer or "An error occurred" in answer:
            print("   ⚠ empty/error answer — skipping")
            continue

        samples.append(
            SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=contexts,
                reference=item["ground_truth"],
            )
        )
        rows_for_log.append({
            "question": q,
            "jurisdiction": jur,
            "answer": answer[:300],
            "n_contexts": len(contexts),
            "top_score": hits[0]["score"] if hits else None,
        })
        print(f"   ✓ captured ({len(contexts)} contexts, top score {hits[0]['score']})")
        time.sleep(1)  # gentle on Groq rate limits

    return samples, rows_for_log


def main():
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    print("=" * 64)
    print("ADALAT-AI — RAGAS EVALUATION (full pipeline)")
    print("=" * 64)

    # Build dataset by running the real pipeline
    samples, rows = build_dataset()
    if not samples:
        print("\nNo valid samples collected. Aborting.")
        sys.exit(1)

    print(f"\nCollected {len(samples)} valid samples. Running RAGAS...\n")

    # Configure judge + embeddings
    judge = LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(FastEmbedAdapter())

    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    dataset = EvaluationDataset(samples=samples)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
    )

    print("\n" + "=" * 64)
    print("RAGAS RESULTS")
    print("=" * 64)
    print(scores)

    # Persist
    os.makedirs("logs", exist_ok=True)
    df = scores.to_pandas()
    df.to_csv("logs/ragas_per_question.csv", index=False)

    summary = {}
    for m in metrics:
        name = getattr(m, "name", str(m))
        try:
            summary[name] = float(df[name].mean())
        except Exception:
            summary[name] = None

    with open("logs/ragas_summary.json", "w") as f:
        json.dump({"n_samples": len(samples), "scores": summary}, f, indent=2)

    print("\nPer-metric averages:")
    for k, v in summary.items():
        print(f"  {k:20s}: {v:.4f}" if v is not None else f"  {k:20s}: n/a")

    print("\nSaved:")
    print("  logs/ragas_per_question.csv")
    print("  logs/ragas_summary.json")


if __name__ == "__main__":
    main()
