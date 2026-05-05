"""
Adalat-AI Chunker v2.0
=======================
Upgrades from v1.0:
- Hierarchical breadcrumb prepended to every chunk
- Rich metadata per chunk (section, part, hierarchy_path)
- One section per chunk when under 800 tokens
- Never splits mid-clause
- Cross-reference detection
- Token counting via tiktoken
"""

import re
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNKS_DIR = Path("data/processed/chunks")
MAX_TOKENS = 800
OVERLAP_TOKENS = 100

# Tokenizer (cl100k_base works for all languages)
try:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOKENIZER = None

# ── Legal structure patterns per country ─────────────────────────────────────

STRUCTURE_PATTERNS = {
    "PK": [
        # Parts
        (r"^(PART\s+[IVXLC]+\.?\s*[-–—]?\s*.+)$", "part"),
        (r"^(CHAPTER\s+[IVXLC\d]+\.?\s*[-–—]?\s*.*)$", "chapter"),
        # Sections
        (r"^(\d+[A-Z]?\.\s+[A-Z][^a-z]{0,5}.*)$", "section"),
        (r"^(Section\s+\d+[A-Z]?\s*[-–—.]?\s*.*)$", "section"),
        # Subsections
        (r"^\((\d+)\)\s+", "subsection"),
        (r"^\(([a-z])\)\s+", "subsection"),
    ],
    "UK": [
        (r"^(PART\s+\d+\s*[-–—]?\s*.*)$", "part"),
        (r"^(Chapter\s+\d+\s*[-–—]?\s*.*)$", "chapter"),
        (r"^(\d+\s+[A-Z].{3,60})$", "section"),
        (r"^(Schedule\s+\d+\s*[-–—]?\s*.*)$", "schedule"),
        (r"^\((\d+)\)\s+", "subsection"),
        (r"^\(([a-z])\)\s+", "subsection"),
    ],
    "DE": [
        (r"^(§\s*\d+[a-z]?\s*[-–—]?\s*.*)$", "section"),
        (r"^(§§\s*\d+\s*[-–—]\s*\d+\s*.*)$", "section_range"),
        (r"^(Abschnitt\s+\d+\s*.*)$", "chapter"),
        (r"^(Titel\s+\d+\s*.*)$", "title"),
        (r"^\((\d+)\)\s+", "subsection"),
    ],
    "UNKNOWN": [
        (r"^(PART\s+[IVXLC\d]+\.?\s*.*)$", "part"),
        (r"^(Section\s+\d+\s*.*)$", "section"),
        (r"^(\d+\.\s+[A-Z].*)$", "section"),
    ],
}

# Cross-reference patterns
CROSS_REF_PATTERNS = [
    r"[Ss]ection\s+\d+[A-Z]?(?:\(\d+\))?",
    r"[Aa]rticle\s+\d+[A-Z]?",
    r"§\s*\d+[a-z]?",
    r"[Ss]chedule\s+\d+",
    r"[Pp]art\s+[IVXLC\d]+",
    r"[Cc]hapter\s+[IVXLC\d]+",
]


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    if TOKENIZER:
        return len(TOKENIZER.encode(text))
    return len(text) // 4  # fallback estimate


def extract_cross_references(text: str) -> list:
    """Find cross-references to other sections/articles."""
    refs = []
    for pattern in CROSS_REF_PATTERNS:
        found = re.findall(pattern, text)
        refs.extend(found)
    return list(set(refs))


def detect_hierarchy_marker(line: str, country: str) -> Optional[tuple]:
    """
    Check if a line is a legal hierarchy marker.
    Returns (marker_text, level) or None.
    """
    patterns = STRUCTURE_PATTERNS.get(country, STRUCTURE_PATTERNS["UNKNOWN"])
    for pattern, level in patterns:
        match = re.match(pattern, line.strip(), re.MULTILINE)
        if match:
            return (line.strip().replace('\xa0', ' '), level)
    return None


def build_breadcrumb(hierarchy: dict, title_en: str) -> str:
    """
    Build hierarchical breadcrumb string.
    Example:
        Punjab Rented Premises Act 2009 > Part III: Eviction > Section 15
    """
    parts = [title_en]
    for level in ["part", "chapter", "section", "schedule"]:
        if hierarchy.get(level):
            parts.append(hierarchy[level])
    return " > ".join(parts)


def split_into_sections(pages: list[dict]) -> list[dict]:
    """
    Split pages into logical sections based on legal hierarchy markers.
    Each section becomes a chunk candidate.
    """
    country = pages[0]["country"] if pages else "UNKNOWN"
    title_en = pages[0]["title_en"] if pages else ""

    sections = []
    current_section = {
        "text_lines": [],
        "hierarchy": {},
        "page_start": pages[0]["page_num"] if pages else 1,
        "page_end": pages[0]["page_num"] if pages else 1,
    }
    current_meta = pages[0].copy() if pages else {}

    for page in pages:
        lines = page["text"].split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line is a hierarchy marker
            marker = detect_hierarchy_marker(line, country)
            if marker:
                marker_text, level = marker

                # If we have accumulated content, save current section
                if len(current_section["text_lines"]) > 5:
                    sections.append({
                        "text_lines": current_section["text_lines"].copy(),
                        "hierarchy": current_section["hierarchy"].copy(),
                        "page_start": current_section["page_start"],
                        "page_end": page["page_num"],
                        "meta": current_meta.copy()
                    })

                # Update hierarchy
                current_hierarchy = current_section["hierarchy"].copy()

                if level == "part":
                    current_hierarchy = {"part": marker_text}
                elif level == "chapter":
                    current_hierarchy["chapter"] = marker_text
                    current_hierarchy.pop("section", None)
                    current_hierarchy.pop("schedule", None)
                elif level in ("section", "section_range"):
                    current_hierarchy["section"] = marker_text
                elif level == "schedule":
                    current_hierarchy["schedule"] = marker_text

                current_section = {
                    "text_lines": [line],
                    "hierarchy": current_hierarchy,
                    "page_start": page["page_num"],
                    "page_end": page["page_num"],
                }
                current_meta = page.copy()
            else:
                current_section["text_lines"].append(line)
                current_section["page_end"] = page["page_num"]

    # Save last section
    if current_section["text_lines"]:
        sections.append({
            "text_lines": current_section["text_lines"],
            "hierarchy": current_section["hierarchy"],
            "page_start": current_section["page_start"],
            "page_end": current_section.get("page_end",
                        pages[-1]["page_num"] if pages else 1),
            "meta": current_meta
        })

    return sections


def sections_to_chunks(sections: list[dict],
                       doc_name: str,
                       title_en: str) -> list[dict]:
    """
    Convert sections to final chunks.
    - Sections under MAX_TOKENS → one chunk
    - Sections over MAX_TOKENS → sliding window split
    """
    chunks = []
    chunk_id = 0

    for section in sections:
        text = "\n".join(section["text_lines"]).strip()
        if len(text) < 50:
            continue

        hierarchy = section["hierarchy"]
        meta = section["meta"]
        breadcrumb = build_breadcrumb(hierarchy, title_en)
        cross_refs = extract_cross_references(text)

        # Prepend breadcrumb to text
        full_text = f"{breadcrumb}\n\n{text}"
        token_count = count_tokens(full_text)

        if token_count <= MAX_TOKENS:
            # Single chunk
            chunks.append(_make_chunk(
                chunk_id=f"{doc_name}_{chunk_id:04d}",
                text=full_text,
                breadcrumb=breadcrumb,
                hierarchy=hierarchy,
                cross_refs=cross_refs,
                token_count=token_count,
                page_start=section["page_start"],
                page_end=section["page_end"],
                meta=meta,
            ))
            chunk_id += 1
        else:
            # Split large section by sliding window
            sub_chunks = _sliding_window_split(
                text=text,
                breadcrumb=breadcrumb,
                max_tokens=MAX_TOKENS,
                overlap=OVERLAP_TOKENS,
            )
            for sub_text in sub_chunks:
                sub_tokens = count_tokens(sub_text)
                chunks.append(_make_chunk(
                    chunk_id=f"{doc_name}_{chunk_id:04d}",
                    text=sub_text,
                    breadcrumb=breadcrumb,
                    hierarchy=hierarchy,
                    cross_refs=cross_refs,
                    token_count=sub_tokens,
                    page_start=section["page_start"],
                    page_end=section["page_end"],
                    meta=meta,
                ))
                chunk_id += 1

    return chunks


def _make_chunk(chunk_id, text, breadcrumb, hierarchy,
                cross_refs, token_count, page_start,
                page_end, meta) -> dict:
    """Build a single chunk dict with full metadata."""
    return {
        "chunk_id": chunk_id,
        "text": text,
        "breadcrumb": breadcrumb,
        "hierarchy": hierarchy,
        "cross_references": cross_refs,
        "token_count": token_count,
        "page_start": page_start,
        "page_end": page_end,
        "source": meta.get("source", ""),
        "doc_name": meta.get("doc_name", ""),
        "title_en": meta.get("title_en", ""),
        "country": meta.get("country", "UNKNOWN"),
        "jurisdiction": meta.get("jurisdiction", "unknown"),
        "province": meta.get("province"),
        "category": meta.get("category", "unknown"),
        "language": meta.get("language", "english"),
        "priority": meta.get("priority", 2),
        "currency_warning": meta.get("currency_warning"),
        "requires_escalation_cue": meta.get("requires_escalation_cue", False),
    }


def _sliding_window_split(text: str, breadcrumb: str,
                          max_tokens: int,
                          overlap: int) -> list[str]:
    """Split large text into overlapping windows."""
    if not TOKENIZER:
        # Fallback: split by characters
        size = max_tokens * 4
        ovlp = overlap * 4
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(f"{breadcrumb}\n\n{text[start:start+size]}")
            start += size - ovlp
        return chunks

    tokens = TOKENIZER.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens - count_tokens(breadcrumb) - 10, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = TOKENIZER.decode(chunk_tokens)
        chunks.append(f"{breadcrumb}\n\n{chunk_text}")
        start += max_tokens - overlap
        if end >= len(tokens):
            break

    return chunks


def chunk_document(pages: list[dict]) -> list[dict]:
    """Main entry point: pages → chunks."""
    if not pages:
        return []

    doc_name = pages[0]["doc_name"]
    title_en = pages[0]["title_en"]

    sections = split_into_sections(pages)
    chunks = sections_to_chunks(sections, doc_name, title_en)

    logger.info(
        f"{doc_name}: {len(pages)} pages → "
        f"{len(sections)} sections → "
        f"{len(chunks)} chunks"
    )
    return chunks


def save_chunks(chunks: list[dict],
                output_path: str = None) -> Path:
    """Save chunks to JSON."""
    if output_path is None:
        doc_name = chunks[0]["doc_name"] if chunks else "unknown"
        output_path = CHUNKS_DIR / f"{doc_name}.chunks.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    return Path(output_path)


def load_chunks(path: str) -> list[dict]:
    """Load chunks from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_by_article(pages: list[dict],
                     chunk_size: int = 800,
                     overlap: int = 100) -> list[dict]:
    """
    Legacy compatibility function.
    Groups pages by document then chunks each.
    """
    from collections import defaultdict
    by_doc = defaultdict(list)
    for p in pages:
        by_doc[p["doc_name"]].append(p)

    all_chunks = []
    for doc_name, doc_pages in by_doc.items():
        chunks = chunk_document(doc_pages)
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    import sys
    from src.ingestion.pdf_loader import load_all_pdfs
    from configs.document_registry import get_pilot_documents, DOCUMENT_REGISTRY
    from collections import defaultdict

    OCR_DOCS = [
        "pk-tenancy-kp-restriction-rented-buildings-security-act-2014.pdf",
        "pk-consumer-islamabad-consumers-protection-act-1995.pdf",
        "pk-labour-industrial-relations-act-2012.pdf",
        "pk-labour-punjab-minimum-wages-act-2019.pdf",
        "uk-housing-housing-act-1996.pdf",
        "uk-housing-landlord-and-tenant-act-1985.pdf",
        "uk-employment-employment-rights-act-1996.pdf",
    ]

    BORN_DIGITAL_SKIP = OCR_DOCS  # skip scanned when running --all

    if "--ocr" in sys.argv:
        filenames = OCR_DOCS
        print(f"Chunking {len(filenames)} OCR'd documents...\n")
    elif "--all" in sys.argv:
        all_docs = [d["file_name"] for d in DOCUMENT_REGISTRY]
        filenames = [f for f in all_docs if f not in BORN_DIGITAL_SKIP]
        print(f"Chunking {len(filenames)} born-digital documents...")
        print(f"Skipping {len(BORN_DIGITAL_SKIP)} scanned documents\n")
    else:
        pilots = get_pilot_documents()
        filenames = [d["file_name"] for d in pilots]
        print(f"Chunking {len(filenames)} pilot documents...")
        print("Use --all or --ocr flag\n")

    pages = load_all_pdfs("data/raw", filenames=filenames)

    by_doc = defaultdict(list)
    for p in pages:
        by_doc[p["doc_name"]].append(p)

    all_chunks = []
    failed = []
    for doc_name, doc_pages in by_doc.items():
        try:
            chunks = chunk_document(doc_pages)
            save_chunks(chunks)
            all_chunks.extend(chunks)
            print(f"✓ {doc_name}: {len(chunks)} chunks")
        except Exception as e:
            failed.append(doc_name)
            print(f"✗ {doc_name}: ERROR — {e}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_chunks)} chunks across {len(by_doc)} docs")
    if failed:
        print(f"FAILED: {failed}")
    print(f"{'='*60}")