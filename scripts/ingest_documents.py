"""
Adalat-AI Document Ingestion Pipeline
Run this script whenever you add new PDFs to data/raw/

Usage:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --rebuild   # force full rebuild
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion(rebuild=False):
    from src.ingestion.pdf_loader import load_all_pdfs
    from src.ingestion.chunker import chunk_by_article, save_chunks, load_chunks
    from src.retrieval.embedder import build_vector_store

    chunks_path = "data/processed/chunks.json"

    logger.info("=" * 50)
    logger.info("ADALAT-AI DOCUMENT INGESTION PIPELINE")
    logger.info("=" * 50)

    # Step 1: Load PDFs
    logger.info("Step 1/3: Loading PDFs from data/raw/...")
    pages = load_all_pdfs("data/raw")
    logger.info(f"Loaded {len(pages)} pages total")

    # Step 2: Chunk documents
    logger.info("Step 2/3: Chunking documents...")
    chunks = chunk_by_article(pages)
    save_chunks(chunks, chunks_path)
    logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")

    # Step 3: Build embeddings
    logger.info("Step 3/3: Building vector store...")
    build_vector_store(chunks_path)
    logger.info("Vector store built successfully")

    logger.info("=" * 50)
    logger.info("INGESTION COMPLETE")
    logger.info(f"Total documents: {len(set(p['source'] for p in pages))}")
    logger.info(f"Total pages:     {len(pages)}")
    logger.info(f"Total chunks:    {len(chunks)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adalat-AI Document Ingestion")
    parser.add_argument("--rebuild", action="store_true", help="Force full rebuild")
    args = parser.parse_args()
    run_ingestion(rebuild=args.rebuild)