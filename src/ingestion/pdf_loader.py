import fitz  # PyMuPDF
import os
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JURISDICTION_MAP = {
    "pakistan_constitution": "PK",
    "pakistan_penal_code": "PK",
    "uk_tenant_fees_act": "UK",
    "bgb_german_tenancy": "DE"
}

def load_pdf(pdf_path: str) -> list[dict]:
    """
    Load a PDF and return list of pages with metadata.
    Each page = { text, page_num, source, jurisdiction }
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_name = path.stem  # filename without extension
    jurisdiction = JURISDICTION_MAP.get(doc_name, "UNKNOWN")

    logger.info(f"Loading: {path.name} | Jurisdiction: {jurisdiction}")

    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if len(text) < 50:  # skip near-empty pages
            continue

        pages.append({
            "text": text,
            "page_num": page_num + 1,
            "source": path.name,
            "doc_name": doc_name,
            "jurisdiction": jurisdiction
        })

    doc.close()
    logger.info(f"Loaded {len(pages)} pages from {path.name}")
    return pages


def load_all_pdfs(raw_dir: str = "data/raw") -> list[dict]:
    """Load all PDFs from the raw directory."""
    all_pages = []
    raw_path = Path(raw_dir)

    pdf_files = list(raw_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {raw_dir}")
        return []

    for pdf_file in pdf_files:
        try:
            pages = load_pdf(str(pdf_file))
            all_pages.extend(pages)
        except Exception as e:
            logger.error(f"Failed to load {pdf_file.name}: {e}")

    logger.info(f"Total pages loaded: {len(all_pages)}")
    return all_pages


if __name__ == "__main__":
    pages = load_all_pdfs("data/raw")
    for p in pages[:3]:
        print(f"\n--- {p['source']} | Page {p['page_num']} | {p['jurisdiction']} ---")
        print(p['text'][:300])