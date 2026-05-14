"""Hybrid retrieval: BM25 (sparse) + dense embeddings, fused via RRF.

Significantly improves precision on exact statute references like 'Section 13'
or 'Article 10-A' that pure dense retrieval smooths over."""

import os
import sys
import json
import re
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_bm25_index = None
_bm25_chunks = None


def _tokenize(text: str) -> list[str]:
    """Tokenize while preserving section/article numbers as single tokens.
    
    'Section 13' → 'section_13' (instead of ['section', '13'] which loses precision).
    This means a query for 'section 13' will literally match chunks containing
    'Section 13' even though the surface forms differ.
    """
    text = re.sub(r'\bSection\s+(\d+[A-Z]?)', r'section_\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bArticle\s+(\d+[A-Z]?)', r'article_\1', text, flags=re.IGNORECASE)
    text = re.sub(r'§\s*(\d+[a-z]?)', r'section_\1', text)
    return re.findall(r"\w+", text.lower())


def build_bm25_index():
    """Lazily build the BM25 index from chunk JSON files. Called once on first search."""
    global _bm25_index, _bm25_chunks
    chunk_files = sorted(Path("data/processed/chunks").glob("*.chunks.json"))
    if not chunk_files:
        logger.warning("No chunk files found at data/processed/chunks/*.chunks.json")
        _bm25_index = None
        _bm25_chunks = []
        return 0

    all_chunks = []
    for f in chunk_files:
        try:
            all_chunks.extend(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning(f"Failed to load {f.name}: {e}")

    if not all_chunks:
        _bm25_index = None
        _bm25_chunks = []
        return 0

    tokenized = [_tokenize(c["text"]) for c in all_chunks]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_chunks = all_chunks
    logger.info(f"BM25 index built: {len(all_chunks)} chunks indexed")
    return len(all_chunks)


def bm25_search(query: str, country: str = None, top_k: int = 20) -> list[dict]:
    """Pure sparse retrieval over the chunk corpus."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is None:
        build_bm25_index()
    if _bm25_index is None:  # build failed
        return []

    scores = _bm25_index.get_scores(_tokenize(query))
    indexed = [(s, i) for i, s in enumerate(scores)]
    indexed.sort(reverse=True)

    results = []
    # Over-fetch then filter by country, since country filtering can drop many results
    for score, idx in indexed[:top_k * 5]:
        chunk = _bm25_chunks[idx]
        if country and chunk.get("country") != country:
            continue
        results.append({
            "text": chunk["text"][:1000],
            "metadata": {
                "source": chunk.get("source", ""),
                "country": chunk.get("country", ""),
                "page_start": chunk.get("page_start", 0),
                "breadcrumb": chunk.get("breadcrumb", "")[:200],
                "currency_warning": chunk.get("currency_warning") or "",
                "requires_escalation_cue": str(chunk.get("requires_escalation_cue", False)),
            },
            "bm25_score": float(score),
            "breadcrumb": chunk.get("breadcrumb", ""),
            "source": chunk.get("source", ""),
            "page_start": chunk.get("page_start", 0),
        })
        if len(results) >= top_k:
            break
    return results


def reciprocal_rank_fusion(dense_results: list[dict], bm25_results: list[dict],
                            k: int = 60, top_k: int = 5) -> list[dict]:
    """Combine two ranked lists into one via RRF.
    
    No score normalization needed — RRF only uses ranks. The constant k=60 is the
    standard default from the original RRF paper; raise k to flatten the ranking
    influence, lower it to amplify top results.
    """
    scores = {}
    objects = {}

    for rank, r in enumerate(dense_results):
        key = (r.get("source", ""), r.get("page_start", 0), r["text"][:100])
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        objects[key] = r

    for rank, r in enumerate(bm25_results):
        key = (r.get("source", ""), r.get("page_start", 0), r["text"][:100])
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in objects:
            objects[key] = r

    sorted_keys = sorted(scores.keys(), key=lambda k_: scores[k_], reverse=True)
    fused = []
    for key in sorted_keys[:top_k]:
        obj = objects[key]
        obj["fused_score"] = scores[key]
        fused.append(obj)
    return fused


def hybrid_search(query: str, country: str = None, top_k: int = 5) -> list[dict]:
    """Public entry: dense + BM25, fused via RRF.
    
    Drop-in replacement for embedder.search(). Same return shape.
    """
    from src.retrieval.embedder import search as dense_search
    dense = dense_search(query, country=country, top_k=20)
    sparse = bm25_search(query, country=country, top_k=20)
    fused = reciprocal_rank_fusion(dense, sparse, top_k=top_k)
    # Normalize: rag_chain expects 'score' field for citations
    for r in fused:
        if "score" not in r:
            r["score"] = round(r.get("fused_score", 0), 4)
    return fused


if __name__ == "__main__":
    # Smoke tests
    print("Building BM25 index...")
    n = build_bm25_index()
    print(f"Indexed {n} chunks\n")

    test_queries = [
        ("landlord deposit return", "PK"),
        ("Section 13 deposit", "PK"),
        ("Article 10 fundamental rights", "PK"),
        ("Tenant Fees Act prohibited payments", "UK"),
    ]

    for q, country in test_queries:
        print("=" * 60)
        print(f"QUERY: {q}  COUNTRY: {country}")
        print("=" * 60)
        results = hybrid_search(q, country=country, top_k=5)
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r.get('score', 0):.4f}  source={r['source'][:50]}  page={r['page_start']}")
            print(f"      breadcrumb: {r['breadcrumb'][:80]}")
        print()