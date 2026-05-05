"""
Adalat-AI OCR Pipeline for Scanned Documents
=============================================
Runs ocrmypdf on scanned PDFs to create searchable text-layer PDFs.
Output goes to data/raw/ replacing the original scanned version.

Usage:
    python scripts/ocr_scanned_docs.py
    python scripts/ocr_scanned_docs.py --dry-run
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
BACKUP_DIR = Path("data/raw/scanned_originals")

SCANNED_DOCUMENTS = [
    {
        "file": "uk-housing-housing-act-1996.pdf",
        "lang": "eng",
        "country": "UK",
    },
    {
        "file": "uk-employment-employment-rights-act-1996.pdf",
        "lang": "eng",
        "country": "UK",
    },
]


def check_tesseract():
    """Verify tesseract is accessible."""
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            version = result.stdout.split("\n")[0]
            logger.info(f"Tesseract found: {version}")
            return True
    except FileNotFoundError:
        pass
    logger.error("Tesseract not found in PATH.")
    logger.error("Run: $env:PATH += ';C:\\Program Files\\Tesseract-OCR'")
    return False


def check_ocrmypdf():
    """Verify ocrmypdf is accessible."""
    try:
        result = subprocess.run(
            ["ocrmypdf", "--version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info(f"ocrmypdf found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    logger.error("ocrmypdf not found.")
    return False


def run_ocr(doc: dict, dry_run: bool = False) -> dict:
    """
    Run ocrmypdf on a single scanned document.
    Creates a searchable PDF with text layer.
    """
    input_path = RAW_DIR / doc["file"]
    output_path = RAW_DIR / f"ocr_{doc['file']}"
    backup_path = BACKUP_DIR / doc["file"]

    if not input_path.exists():
        return {
            "file": doc["file"],
            "status": "error",
            "error": "File not found"
        }

    logger.info(f"Processing: {doc['file']}")
    logger.info(f"  Language: {doc['lang']}")

    if dry_run:
        logger.info(f"  DRY RUN — would run OCR on {input_path}")
        return {"file": doc["file"], "status": "dry_run"}

    # Backup original
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(input_path, backup_path)
        logger.info(f"  Backed up to {backup_path}")

    # Build ocrmypdf command
    cmd = [
        "ocrmypdf",
        "--language", doc["lang"],
        "--rotate-pages",
        "--skip-text",
        "--output-type", "pdf",
        "--jobs", "2",
        str(input_path),
        str(output_path)
    ]

    logger.info(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900  # 5 min per doc
        )

        if result.returncode == 0:
            # Replace original with OCR version
            shutil.move(str(output_path), str(input_path))
            logger.info(f"  ✅ OCR complete — replaced original")
            return {"file": doc["file"], "status": "ok"}
        else:
            logger.error(f"  ❌ ocrmypdf failed: {result.stderr[-500:]}")
            # Clean up failed output
            if output_path.exists():
                output_path.unlink()
            return {
                "file": doc["file"],
                "status": "error",
                "error": result.stderr[-300:]
            }

    except subprocess.TimeoutExpired:
        logger.error(f"  ❌ Timeout after 5 minutes")
        return {"file": doc["file"], "status": "timeout"}
    except Exception as e:
        logger.error(f"  ❌ Exception: {e}")
        return {"file": doc["file"], "status": "error", "error": str(e)}


def main():
    dry_run = "--dry-run" in sys.argv

    print("\n" + "="*60)
    print("ADALAT-AI OCR PIPELINE")
    print("="*60)

    if dry_run:
        print("DRY RUN MODE — no files will be modified\n")

    if not check_tesseract():
        sys.exit(1)
    if not check_ocrmypdf():
        sys.exit(1)

    print(f"\nProcessing {len(SCANNED_DOCUMENTS)} scanned documents...\n")

    results = []
    for doc in SCANNED_DOCUMENTS:
        result = run_ocr(doc, dry_run=dry_run)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("OCR RESULTS")
    print("="*60)
    for r in results:
        status = r["status"]
        icon = "✅" if status == "ok" else "❌" if status == "error" else "⏭"
        print(f"{icon} {r['file']}")
        if "error" in r:
            print(f"   Error: {r['error'][:100]}")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} documents processed successfully")

    if ok > 0 and not dry_run:
        print("\nNext steps:")
        print("  python src/ingestion/chunker.py --ocr")
        print("  python src/retrieval/embedder.py --append")


if __name__ == "__main__":
    main()