import re
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Article/Section patterns per jurisdiction
ARTICLE_PATTERNS = {
    "PK": [
        r"(Article\s+\d+[\w\-]*\.?)",        # Article 25, Article 25A
        r"(CHAPTER\s+[IVXLC]+\.?\s+\w+)",    # CHAPTER I - Preliminary
        r"(Section\s+\d+[\w\-]*\.?)",         # Section 1
    ],
    "UK": [
        r"(\d+\s+[A-Z][a-z]+.*?\n)",          # numbered sections
        r"(Section\s+\d+[\w\-]*\.?)",
        r"(PART\s+\d+)",
    ],
    "DE": [
        r"(§\s*\d+[\w\-]*)",                  # § 535, § 536a
        r"(Article\s+\d+[\w\-]*\.?)",
    ]
}

def clean_text(text: str) -> str:
    """Remove excessive whitespace and noise."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def chunk_by_article(pages: list[dict], chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """
    Split pages into chunks. Tries structural (article-based) first,
    falls back to sliding window chunking.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = clean_text(page["text"])
        jurisdiction = page["jurisdiction"]
        patterns = ARTICLE_PATTERNS.get(jurisdiction, [])

        # Try structural splitting
        split_texts = _structural_split(text, patterns)

        if len(split_texts) <= 1:
            # Fallback: sliding window
            split_texts = _sliding_window(text, chunk_size, overlap)

        for chunk_text in split_texts:
            if len(chunk_text.strip()) < 50:
                continue

            chunks.append({
                "chunk_id": f"{page['doc_name']}_{chunk_id}",
                "text": chunk_text.strip(),
                "source": page["source"],
                "doc_name": page["doc_name"],
                "jurisdiction": page["jurisdiction"],
                "page_num": page["page_num"],
                "char_count": len(chunk_text)
            })
            chunk_id += 1

    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks


def _structural_split(text: str, patterns: list) -> list[str]:
    """Split text on legal article/section markers."""
    if not patterns:
        return [text]

    combined = "|".join(patterns)
    parts = re.split(combined, text)
    return [p for p in parts if p and len(p.strip()) > 30]


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """Fallback: split by character window."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def save_chunks(chunks: list[dict], output_path: str = "data/processed/chunks.json"):
    """Save chunks to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(path: str = "data/processed/chunks.json") -> list[dict]:
    """Load chunks from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    from pdf_loader import load_all_pdfs
    pages = load_all_pdfs("data/raw")
    chunks = chunk_by_article(pages)
    save_chunks(chunks)

    # Preview
    for c in chunks[:3]:
        print(f"\n--- {c['chunk_id']} | {c['jurisdiction']} | page {c['page_num']} ---")
        print(c['text'][:300])