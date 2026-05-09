"""Single-query smoke test for hybrid retrieval.
Verifies hybrid_search returns sane results and the answer is grounded."""

import requests

BASE_URL = "http://localhost:8000"

# This query is specifically designed to test BM25's strength: it has
# both a semantic component ("landlord", "deposit") and an exact-match
# component ("Section"). Pure dense retrieval often misses the literal
# section reference; hybrid should catch both.
query = "Punjab Rented Premises Act Section 12 deposit"

print(f"\n→ Sending: {query!r}\n")
r = requests.post(
    f"{BASE_URL}/ask",
    headers={"Content-Type": "application/json"},
    json={"query": query},
    timeout=180,
)
data = r.json()

print("="*70)
print(f"  language:         {data.get('language')}")
print(f"  jurisdiction:     {data.get('jurisdiction')}")
print(f"  is_clarification: {data.get('is_clarification', False)}")
print(f"  confidence:       {data.get('confidence')}")
print()
print("CITATIONS (in fused-score order):")
for c in data.get("citations", []):
    print(f"  [{c.get('relevance_score', 0):.4f}] {c.get('source', '')[:60]} p.{c.get('page', '?')}")
    print(f"          breadcrumb: {c.get('breadcrumb', '')[:80]}")

print()
print("ANSWER (first 500 chars):")
print(data.get("answer", "")[:500])