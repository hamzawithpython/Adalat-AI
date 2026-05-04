"""
Adalat-AI Retrieval Evaluation Script
Run after any model or document change to track performance.

Usage:
    python configs/evaluate.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.mlflow_config import EVALUATION_QUERIES
from src.retrieval.embedder import search


def run_evaluation():
    print("\n" + "=" * 60)
    print("ADALAT-AI RETRIEVAL EVALUATION")
    print("=" * 60)

    results = []
    for item in EVALUATION_QUERIES:
        start = time.time()
        hits = search(item["query"], jurisdiction=item["jurisdiction"], top_k=5)
        elapsed = (time.time() - start) * 1000

        top_score = hits[0]["score"] if hits else 0.0
        avg_score = sum(h["score"] for h in hits) / len(hits) if hits else 0.0
        passed = top_score >= item["expected_min_score"]

        results.append({
            "query": item["query"][:50],
            "jurisdiction": item["jurisdiction"],
            "top_score": round(top_score, 4),
            "avg_score": round(avg_score, 4),
            "expected": item["expected_min_score"],
            "passed": passed,
            "time_ms": round(elapsed, 1)
        })

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {item['jurisdiction']} | {item['query'][:45]}")
        print(f"       Top: {top_score:.4f} (min: {item['expected_min_score']}) | "
              f"Avg: {avg_score:.4f} | {elapsed:.0f}ms")

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"RESULT: {passed_count}/{total} passed")
    print(f"{'=' * 60}\n")

    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to logs/eval_results.json")
    return results


if __name__ == "__main__":
    run_evaluation()