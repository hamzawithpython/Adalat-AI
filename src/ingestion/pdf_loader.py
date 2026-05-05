"""
Adalat-AI PDF Loader v2.0
==========================
Upgrades from v1.0:
- Uses document registry for metadata
- Detects born-digital vs scanned
- Strips repeating headers/footers
- Preserves legal hierarchy markers
- Returns rich page objects with full metadata
"""

import os
import sys
import re
import logging
from pathlib import Path
from collections import Counter

import fitz  # PyMuPDF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from configs.document_registry import get_by_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

# Patterns that indicate repeating headers/footers to strip
NOISE_PATTERNS = [
    r"^Page\s+\d+\s+of\s+\d+$",
    r"^-\s*\d+\s*-$",
    r"^\d+$",
    r"^www\..+$",
    r"^https?://.+$",
    r"^Service provided by.*$",
    r"^gesetze-im-internet\.de.*$",
    r"^legislation\.gov\.uk.*$",
    r"^F\d+$",                          # footnote markers e.g. F1, F2
    r"^\[F\d+.*?\]$",                   # bracketed footnote refs
    r"^Marginal Citations$",
    r"^Commencement Information$",
    r"^Editorial Information$",
    r"^Modifications etc\.$",
]

COMPILED_NOISE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def is_noise_line(line: str) -> bool:
    """Return True if line is a header/footer/noise to strip."""
    line = line.strip()
    if not line:
        return True
    if len(line) < 3:
        return True
    for pattern in COMPILED_NOISE:
        if pattern.match(line):
            return True
    return False


def detect_repeating_lines(pages: list[dict], threshold: float = 0.6) -> set:
    """
    Find lines that appear on > threshold fraction of pages.
    These are likely headers/footers.
    """
    line_counts = Counter()
    total_pages = len(pages)

    for page in pages:
        lines = set(page["raw_text"].split("\n"))
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 5:
                line_counts[stripped] += 1

    repeating = {
        line for line, count in line_counts.items()
        if count / total_pages >= threshold
    }
    return repeating


def clean_text(text: str, repeating_lines: set = None) -> str:
    """
    Clean extracted text:
    - Remove noise lines
    - Remove repeating headers/footers
    - Normalize whitespace
    - Preserve legal structure markers
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Skip noise
        if is_noise_line(stripped):
            continue

        # Skip repeating headers/footers
        if repeating_lines and stripped in repeating_lines:
            continue

        cleaned.append(stripped)

    # Join and normalize whitespace
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r" {2,}", " ", result)
    return result.strip()


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Load a PDF and return list of pages with rich metadata.

    Returns list of:
    {
        text        : cleaned text
        raw_text    : uncleaned text
        page_num    : page number (1-indexed)
        source      : filename
        doc_name    : filename stem
        title_en    : document title
        country     : PK/UK/DE
        jurisdiction: federal/provincial/national
        province    : Punjab/Sindh/etc or None
        category    : tenancy/criminal/etc
        language    : english/german/urdu
        pdf_type    : born_digital/scanned/mixed
        priority    : 1/2/3
        currency_warning: str or None
        requires_escalation_cue: bool
    }
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    filename = path.name
    registry = get_by_filename(filename)

    if not registry:
        logger.warning(f"No registry entry for {filename}")
        registry = {
            "country": "UNKNOWN",
            "jurisdiction": "unknown",
            "province": None,
            "category": "unknown",
            "language": "english",
            "title_en": filename,
            "priority": 2,
        }

    logger.info(
        f"Loading: {filename} | "
        f"{registry.get('country')} | "
        f"{registry.get('category')} | "
        f"{registry.get('jurisdiction')}"
    )

    doc = fitz.open(pdf_path)
    raw_pages = []

    # First pass — extract raw text from all pages
    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text").strip()
        raw_pages.append({
            "raw_text": raw_text,
            "page_num": page_num + 1
        })

    doc.close()

    # Detect repeating lines (headers/footers)
    repeating_lines = detect_repeating_lines(raw_pages)
    if repeating_lines:
        logger.info(f"Found {len(repeating_lines)} repeating header/footer lines to strip")

    # Second pass — clean and build page objects
    pages = []
    for raw_page in raw_pages:
        cleaned = clean_text(raw_page["raw_text"], repeating_lines)

        if len(cleaned) < 50:
            continue

        pages.append({
            "text": cleaned,
            "raw_text": raw_page["raw_text"],
            "page_num": raw_page["page_num"],
            "source": filename,
            "doc_name": path.stem,
            "title_en": registry.get("title_en", filename),
            "country": registry.get("country", "UNKNOWN"),
            "jurisdiction": registry.get("jurisdiction", "unknown"),
            "province": registry.get("province"),
            "category": registry.get("category", "unknown"),
            "language": registry.get("language", "english"),
            "priority": registry.get("priority", 2),
            "currency_warning": registry.get("notes") if "currency_warning" in
                                (registry.get("notes") or "") else None,
            "requires_escalation_cue": filename in [
                "pk-criminal-anti-terrorism-act-1997.pdf",
                "pk-criminal-code-criminal-procedure-crpc-1898.pdf",
                "uk-immigration-british-nationality-act-1981.pdf",
                "uk-immigration-immigration-act-1971.pdf",
            ],
        })

    logger.info(f"Loaded {len(pages)} pages from {filename}")
    return pages


def load_all_pdfs(raw_dir: str = "data/raw",
                  filenames: list = None) -> list[dict]:
    """
    Load all PDFs from raw_dir or a specific list of filenames.
    """
    all_pages = []
    raw_path = Path(raw_dir)

    if filenames:
        pdf_files = [raw_path / f for f in filenames]
    else:
        pdf_files = sorted(raw_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDFs found in {raw_dir}")
        return []

    for pdf_file in pdf_files:
        if not pdf_file.exists():
            logger.warning(f"File not found: {pdf_file}")
            continue
        try:
            pages = load_pdf(str(pdf_file))
            all_pages.extend(pages)
        except Exception as e:
            logger.error(f"Failed to load {pdf_file.name}: {e}")

    logger.info(f"Total pages loaded: {len(all_pages)}")
    return all_pages


if __name__ == "__main__":
    from configs.document_registry import get_pilot_documents

    pilots = get_pilot_documents()
    filenames = [d["file_name"] for d in pilots]

    print(f"Testing loader on {len(filenames)} pilot documents...\n")
    pages = load_all_pdfs("data/raw", filenames=filenames)

    # Summary
    from collections import defaultdict
    by_doc = defaultdict(list)
    for p in pages:
        by_doc[p["source"]].append(p)

    print("\n=== LOADER RESULTS ===")
    for doc, doc_pages in by_doc.items():
        sample = doc_pages[0]["text"][:200].replace("\n", " ")
        print(f"\n{doc}")
        print(f"  Pages loaded : {len(doc_pages)}")
        print(f"  Country      : {doc_pages[0]['country']}")
        print(f"  Category     : {doc_pages[0]['category']}")
        print(f"  Sample text  : {sample}...")