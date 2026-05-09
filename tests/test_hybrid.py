"""End-to-end smoke test for hybrid retrieval.
Uses a query that won't trip language detection or clarifier."""

import requests

BASE_URL = "http://localhost:8000"

# This query is clearly English (no Pakistani statute names that confuse
# the language detector) but specific enough to test BM25's exact-match
# strength via "Tenant Fees Act" and "permitted payments".
query = "What does the Tenant Fees Act 2019 say about permitted payments and prohibited fees?"

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
print("ANSWER (first 600 chars):")
print(data.get("answer", "")[:600])

# ── Quality check ──────────────────────────────────────────────
print()
print("="*70)
print("  QUALITY CHECK")
print("="*70)
sources = [c.get("source", "") for c in data.get("citations", [])]
tenant_fees_hits = sum(1 for s in sources if "tenant_fees" in s.lower())
print(f"Citations from uk_tenant_fees_act.pdf: {tenant_fees_hits} of {len(sources)}")
if tenant_fees_hits >= 2:
    print("✅ PASS — hybrid retrieval correctly surfaced Tenant Fees Act chunks")
elif tenant_fees_hits == 1:
    print("⚠️  PARTIAL — only one Tenant Fees Act chunk; check if relevant")
else:
    print("❌ FAIL — hybrid retrieval did not surface the right statute")