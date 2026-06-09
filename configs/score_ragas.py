"""
Adalat-AI — RAGAS scoring (Step 2 of 2)

Reads logs/eval_capture.json (produced by capture_eval_data.py) and computes
RAGAS metrics. Runs in the ISOLATED ragas-venv (ragas 0.2.15 + langchain 0.3.x),
so it never touches the main project's langchain version.

Judge LLM: Groq (llama-3.3-70b-versatile)

Usage (from the ragas-venv):
    python configs/score_ragas.py
"""
import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

CAPTURE_PATH = "logs/eval_capture.json"
JUDGE_MODEL = "llama-3.3-70b-versatile"


def main():
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    if not os.path.exists(CAPTURE_PATH):
        print(f"ERROR: {CAPTURE_PATH} not found. Run capture_eval_data.py first (in main venv).")
        sys.exit(1)

    with open(CAPTURE_PATH, encoding="utf-8") as f:
        captured = json.load(f)

    print("=" * 64)
    print(f"ADALAT-AI — RAGAS SCORING  ({len(captured)} samples)")
    print("=" * 64)

    # ── Build the RAGAS dataset ──────────────────────────────────────────
    # RAGAS 0.2.x expects these column names:
    #   user_input, response, retrieved_contexts, reference
    from ragas import EvaluationDataset

    samples = []
    for item in captured:
        samples.append({
            "user_input": item["question"],
            "response": item["answer"],
            "retrieved_contexts": item["contexts"],
            "reference": item["ground_truth"],
        })

    dataset = EvaluationDataset.from_list(samples)

    # ── Judge LLM (Groq) wrapped for RAGAS ───────────────────────────────
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    judge = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", temperature=0))

    # ── Embeddings for answer_relevancy / context metrics ────────────────
    # Use a lightweight HF embedding via langchain-community to avoid OpenAI.
    # fastembed isn't in this venv, so we use a small sentence-transformers
    # model through HuggingFaceEmbeddings. If torch isn't present this will
    # error — in that case we fall back to skipping embedding-based metrics.
    embeddings = None
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        )
        print("Embeddings: HuggingFace MiniLM loaded.")
    except Exception as e:
        print(f"NOTE: embedding model unavailable ({e}).")
        print("      Will run metrics that don't need embeddings.")

    # ── Metrics ──────────────────────────────────────────────────────────
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    # faithfulness, context_precision, context_recall need only the judge LLM.
    # answer_relevancy needs embeddings too.
    if embeddings is not None:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    else:
        metrics = [faithfulness, context_precision, context_recall]

    print(f"\nRunning {len(metrics)} metrics over {len(samples)} samples...")
    print("(this makes many Groq judge calls — expect rate-limit retries)\n")

    from ragas import evaluate
    from ragas.run_config import RunConfig

    run_config = RunConfig(
        timeout=300,      # 5 min per call — Cerebras free tier is slow
        max_workers=2,    # low concurrency so calls don't pile up and time out
        max_retries=5,
    )

    kwargs = {"dataset": dataset, "metrics": metrics, "llm": judge, "run_config": run_config}
    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    result = evaluate(**kwargs)

    print("\n" + "=" * 64)
    print("RAGAS RESULTS")
    print("=" * 64)
    print(result)

    # ── Persist ──────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    df = result.to_pandas()
    df.to_csv("logs/ragas_per_question.csv", index=False)

    summary = {}
    for col in df.columns:
        if col in ("user_input", "response", "retrieved_contexts", "reference"):
            continue
        try:
            summary[col] = round(float(df[col].mean()), 4)
        except Exception:
            pass

    with open("logs/ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump({"n_samples": len(samples), "scores": summary}, f, indent=2)

    print("\nPer-metric averages:")
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")

    print("\nSaved:")
    print("  logs/ragas_per_question.csv")
    print("  logs/ragas_summary.json")


if __name__ == "__main__":
    main()
