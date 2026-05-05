"""
Adalat-AI Metadata Builder
===========================
Generates a companion .json sidecar for every PDF.
Run once per document. Output goes to data/processed/metadata/

Each sidecar contains:
- Document identity (title, country, category)
- Technical info (pages, size, SHA-256 hash)
- PDF type detection (born-digital vs scanned)
- Currency warnings for partially-in-force laws
- Amendment metadata from legislation.gov.uk footnotes
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import fitz  # PyMuPDF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from configs.document_registry import get_by_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METADATA_DIR = Path("data/processed/metadata")
RAW_DIR = Path("data/raw")

# Documents with known currency warnings
CURRENCY_WARNINGS = {
    "de-civil-bgb-english-translation-2021.pdf":
        "BGB English translation current to 10 August 2021. "
        "Recent amendments may not be reflected.",
    "bgb_german_tenancy.pdf":
        "BGB English translation current to 10 August 2021. "
        "Recent amendments may not be reflected.",
    "uk-housing-renters-rights-act-2025.pdf":
        "Renters Rights Act 2025 is partially in force. "
        "Phased commencement — not all provisions apply yet.",
    "uk-housing-renters-rights-act-2025-information-sheet.pdf":
        "This document describes provisions of the Renters Rights Act 2025 "
        "which is subject to phased commencement.",
    "uk-housing-renters-rights-act-2025-implementation-roadmap.pdf":
        "Implementation roadmap for Renters Rights Act 2025. "
        "Commencement dates subject to change.",
}

# Documents flagged as sensitive (add escalation cue)
SENSITIVE_DOCUMENTS = [
    "pk-criminal-anti-terrorism-act-1997.pdf",
    "pk-criminal-code-criminal-procedure-crpc-1898.pdf",
    "uk-immigration-british-nationality-act-1981.pdf",
    "uk-immigration-immigration-act-1971.pdf",
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_pdf_type(filepath: Path) -> dict:
    """
    Detect whether PDF is born-digital or scanned.

    Strategy:
    - Sample first 5 pages
    - Count extractable text characters
    - If avg chars per page > 100 → born-digital
    - If avg chars per page < 50  → scanned
    - Between 50-100              → mixed/uncertain
    """
    doc = fitz.open(str(filepath))
    total_pages = len(doc)
    sample_pages = min(5, total_pages)

    char_counts = []
    for i in range(sample_pages):
        page = doc[i]
        text = page.get_text("text").strip()
        char_counts.append(len(text))

    doc.close()

    avg_chars = sum(char_counts) / len(char_counts) if char_counts else 0

    if avg_chars > 100:
        pdf_type = "born_digital"
        extraction_method = "pymupdf"
    elif avg_chars < 50:
        pdf_type = "scanned"
        extraction_method = "ocr_tesseract"
    else:
        pdf_type = "mixed"
        extraction_method = "pymupdf_with_ocr_fallback"

    return {
        "pdf_type": pdf_type,
        "extraction_method": extraction_method,
        "avg_chars_per_page_sample": round(avg_chars, 1),
        "sample_pages_checked": sample_pages,
    }


def get_page_count(filepath: Path) -> int:
    """Get total page count."""
    doc = fitz.open(str(filepath))
    count = len(doc)
    doc.close()
    return count


def get_file_size_kb(filepath: Path) -> float:
    """Get file size in KB."""
    return round(filepath.stat().st_size / 1024, 1)


def build_metadata(filename: str) -> dict:
    """
    Build full metadata sidecar for a document.
    Combines registry info + technical analysis.
    """
    filepath = RAW_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    # Get registry entry
    registry = get_by_filename(filename)
    if not registry:
        logger.warning(f"No registry entry for {filename} — using defaults")

    # Technical analysis
    pdf_type_info = detect_pdf_type(filepath)
    page_count = get_page_count(filepath)
    file_size_kb = get_file_size_kb(filepath)
    sha256 = compute_sha256(filepath)

    # Build metadata
    metadata = {
        "schema_version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",

        # Identity
        "file_name": filename,
        "title_en": registry.get("title_en", filename),
        "title_local": registry.get("title_local"),
        "act_number": None,

        # Jurisdiction
        "country": registry.get("country", "UNKNOWN"),
        "jurisdiction": registry.get("jurisdiction", "unknown"),
        "province": registry.get("province"),
        "category": registry.get("category", "unknown"),
        "language": registry.get("language", "english"),

        # Temporal
        "enactment_year": registry.get("enactment_year"),
        "last_amended_date": None,
        "amendment_history": [],
        "in_force_status": "in_force",

        # Source
        "source_url": registry.get("source_url"),
        "license": registry.get("license", "public_domain"),
        "downloaded_date": datetime.utcnow().strftime("%Y-%m-%d"),

        # Technical
        "page_count": page_count,
        "file_size_kb": file_size_kb,
        "sha256": sha256,
        "pdf_type": pdf_type_info["pdf_type"],
        "extraction_method": pdf_type_info["extraction_method"],
        "avg_chars_per_page_sample": pdf_type_info["avg_chars_per_page_sample"],

        # Warnings
        "currency_warning": CURRENCY_WARNINGS.get(filename),
        "requires_escalation_cue": filename in SENSITIVE_DOCUMENTS,
        "partially_in_force": "2025" in filename and "renters" in filename.lower(),

        # Topics (to be enriched later)
        "key_topics": [],
        "related_documents": [],
        "priority": registry.get("priority", 2),
        "notes": registry.get("notes"),
    }

    # Special handling for partially-in-force laws
    if metadata["partially_in_force"]:
        metadata["in_force_status"] = "partially_in_force"

    return metadata


def save_metadata(metadata: dict, output_dir: Path = METADATA_DIR) -> Path:
    """Save metadata sidecar to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(metadata["file_name"]).stem
    output_path = output_dir / f"{stem}.meta.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return output_path


def build_all_metadata(filenames: list = None) -> list:
    """Build metadata for multiple documents."""
    if filenames is None:
        from configs.document_registry import DOCUMENT_REGISTRY
        filenames = [d["file_name"] for d in DOCUMENT_REGISTRY]

    results = []
    for filename in filenames:
        try:
            logger.info(f"Building metadata: {filename}")
            meta = build_metadata(filename)
            path = save_metadata(meta)
            results.append({
                "file": filename,
                "pdf_type": meta["pdf_type"],
                "pages": meta["page_count"],
                "size_kb": meta["file_size_kb"],
                "status": "ok",
                "saved_to": str(path)
            })
        except Exception as e:
            logger.error(f"Failed: {filename} — {e}")
            results.append({
                "file": filename,
                "status": "error",
                "error": str(e)
            })

    return results


if __name__ == "__main__":
    import sys

    # Run on pilot documents only by default
    from configs.document_registry import get_pilot_documents, DOCUMENT_REGISTRY

    if "--all" in sys.argv:
        filenames = [d["file_name"] for d in DOCUMENT_REGISTRY]
        print(f"Building metadata for ALL {len(filenames)} documents...")
    else:
        pilots = get_pilot_documents()
        filenames = [d["file_name"] for d in pilots]
        print(f"Building metadata for {len(filenames)} pilot documents...")
        print("Use --all flag to process all documents\n")

    results = build_all_metadata(filenames)

    print("\n=== METADATA BUILD RESULTS ===")
    print(f"{'File':<55} {'Type':<15} {'Pages':<8} {'Size KB':<10} {'Status'}")
    print("-" * 100)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['file']:<55} {r['pdf_type']:<15} "
                  f"{r['pages']:<8} {r['size_kb']:<10} {r['status']}")
        else:
            print(f"{r['file']:<55} {'ERROR':<15} {r.get('error', '')}")