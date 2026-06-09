"""
Adalat-AI — RAGAS data capture (Step 1 of 2)

Runs the 12-question test set through the REAL pipeline (ask + hybrid_search),
captures everything RAGAS needs, and writes it to logs/eval_capture.json.

This script imports NO ragas — so it runs cleanly in your main project venv
without the langchain version conflict. RAGAS scoring happens separately in
score_ragas.py (run in an isolated venv).

Usage:
    python configs/capture_eval_data.py
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
from src.retrieval.hybrid_search import hybrid_search


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
        "ground_truth": "Under the German BGB, rent increases are restricted. A rent increase to the local comparative rent generally cannot exceed a capping limit (Kappungsgrenze) of 20% (15% in many areas) over three years, and the landlord must follow the formal procedure under section 558 BGB. A sudden 20% increase in one year would generally be impermissible.",
    },
    {
        "question": "My German landlord is not returning my security deposit (Kaution) after I moved out. What does German law say?",
        "jurisdiction": "DE",
        "ground_truth": "Under the German BGB, the landlord may hold the security deposit (Kaution) for a reasonable period after tenancy end to check for claims (commonly up to 3-6 months), after which the remaining deposit, with interest, must be returned to the tenant. The tenant can demand return and pursue a civil claim if refused.",
    },
    {
        "question": "In Germany, what notice period must I give my landlord if I want to terminate my rental contract as a tenant?",
        "jurisdiction": "DE",
        "ground_truth": "Under the German BGB (section 573c), a tenant terminating an open-ended residential lease must generally give three months' notice, with termination effective at the end of the relevant month if served by the third working day of that month.",
    },
    {
        "question": "Can my German landlord charge me for operating/ancillary costs (Betriebskosten) on top of rent, and what rules apply?",
        "jurisdiction": "DE",
        "ground_truth": "Under the German BGB and the Betriebskostenverordnung, a landlord may pass on defined operating costs (Betriebskosten) to the tenant only if this is agreed in the lease, and must provide an annual itemised statement. The tenant has the right to inspect supporting documents and dispute incorrect charges.",
    },
]


def main():
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    print("=" * 64)
    print("ADALAT-AI — CAPTURE EVAL DATA (real pipeline, no ragas)")
    print("=" * 64)

    captured = []
    for i, item in enumerate(TEST_SET, 1):
        q = item["question"]
        jur = item["jurisdiction"]
        print(f"\n[{i}/{len(TEST_SET)}] ({jur}) {q[:60]}...")

        # contexts — exactly what production retrieves
        try:
            hits = hybrid_search(q, country=jur, top_k=5)
        except Exception as e:
            print(f"   x hybrid_search failed: {e} — skipping")
            continue
        contexts = [h["text"] for h in hits if h.get("text")]
        if not contexts:
            print("   x no contexts retrieved — skipping")
            continue

        # answer — full pipeline
        try:
            result = ask(q)
        except Exception as e:
            print(f"   x ask() failed: {e} — skipping")
            continue

        if result.get("is_clarification"):
            print("   x pipeline asked for clarification — skipping")
            continue

        answer = result.get("answer", "")
        if not answer or "An error occurred" in answer:
            print("   x empty/error answer — skipping")
            continue

        captured.append({
            "question": q,
            "jurisdiction": jur,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "top_score": hits[0].get("score"),
            "n_contexts": len(contexts),
        })
        print(f"   ok captured ({len(contexts)} contexts, top score {hits[0].get('score')})")
        time.sleep(1)  # gentle on Groq

    os.makedirs("logs", exist_ok=True)
    out = "logs/eval_capture.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(captured, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 64}")
    print(f"Captured {len(captured)}/{len(TEST_SET)} questions -> {out}")
    print("=" * 64)
    print("\nNext: run score_ragas.py in an isolated venv to compute RAGAS metrics.")


if __name__ == "__main__":
    main()
