import os
import sys
import json
import logging
import gc
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import chromadb
from fastembed import TextEmbedding

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/embeddings/chroma")
COLLECTION_NAME = "adalat_legal_docs"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        logger.info(f"Loading fastembed model: {MODEL_NAME}")
        _model = TextEmbedding(MODEL_NAME)
        logger.info("Model loaded.")
    return _model


def get_chroma_client():
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


def build_vector_store(chunks_path: str = None):
    """Embed all chunks and store in Chroma."""
    import glob

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if chunks_path is None:
        chunk_files = sorted(glob.glob("data/processed/chunks/*.chunks.json"))
    else:
        chunk_files = [chunks_path]

    logger.info(f"Found {len(chunk_files)} chunk files to embed")

    model = get_embedding_model()
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection")
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    total_embedded = 0

    for file_idx, chunk_file in enumerate(chunk_files):
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        doc_name = Path(chunk_file).stem.replace(".chunks", "")
        logger.info(
            f"[{file_idx+1}/{len(chunk_files)}] "
            f"{doc_name} ({len(chunks)} chunks)"
        )

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]

            texts = [f"passage: {c['text'][:1000]}" for c in batch]
            ids = [c["chunk_id"] for c in batch]
            metadatas = [{
                "source":       c.get("source", ""),
                "jurisdiction": c.get("jurisdiction", ""),
                "country":      c.get("country", ""),
                "category":     c.get("category", ""),
                "province":     c.get("province") or "",
                "page_start":   c.get("page_start", 0),
                "page_end":     c.get("page_end", 0),
                "doc_name":     c.get("doc_name", ""),
                "breadcrumb":   c.get("breadcrumb", "")[:200],
                "priority":     c.get("priority", 2),
                "requires_escalation_cue": str(
                    c.get("requires_escalation_cue", False)),
                "currency_warning": (
                    c.get("currency_warning") or "")[:200],
            } for c in batch]

            try:
                embeddings = list(model.embed(texts))
                embeddings = [e.tolist() for e in embeddings]

                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=[c["text"][:1000] for c in batch],
                    metadatas=metadatas
                )
                total_embedded += len(batch)

            except Exception as e:
                logger.error(f"Batch {i} failed: {e}")
                continue

        del chunks
        gc.collect()
        logger.info(f"Total embedded so far: {total_embedded}")

    logger.info(f"DONE. Vector store: {collection.count()} vectors")
    return collection


def search(query: str, jurisdiction: str = None,
           country: str = None, category: str = None,
           top_k: int = 5):
    """Search vector store for relevant chunks."""
    model = get_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = list(model.embed([f"query: {query}"]))
    query_embedding = [query_embedding[0].tolist()]

    where_filter = None
    conditions = []
    if jurisdiction:
        conditions.append({"country": jurisdiction})   # PK/UK/DE lives in the `country` metadata field
    if country:
        conditions.append({"country": country})
    if category:
        conditions.append({"category": category})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        output.append({
            "text":       results["documents"][0][i],
            "metadata":   meta,
            "score":      round(1 - results["distances"][0][i], 4),
            "breadcrumb": meta.get("breadcrumb", ""),
            "source":     meta.get("source", ""),
            "page_start": meta.get("page_start", 0),
            "currency_warning": meta.get("currency_warning") or None,
            "requires_escalation_cue":
                meta.get("requires_escalation_cue") == "True",
        })

    return output


if __name__ == "__main__":
    import sys
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if "--test" in sys.argv:
        print("TEST MODE: embedding 1 file only")
        build_vector_store(
            "data/processed/chunks/"
            "pk-tenancy-punjab-rented-premises-act-2009.chunks.json"
        )
    elif "--append" in sys.argv:
        # Embed only new OCR'd chunk files
        import glob
        ocr_docs = [
            "pk-tenancy-kp-restriction-rented-buildings-security-act-2014",
            "pk-consumer-islamabad-consumers-protection-act-1995",
            "pk-labour-industrial-relations-act-2012",
            "pk-labour-punjab-minimum-wages-act-2019",
            "uk-housing-housing-act-1996",
            "uk-housing-landlord-and-tenant-act-1985",
            "uk-employment-employment-rights-act-1996",
        ]
        chunk_files = [
            f"data/processed/chunks/{d}.chunks.json"
            for d in ocr_docs
            if Path(f"data/processed/chunks/{d}.chunks.json").exists()
        ]
        print(f"Appending {len(chunk_files)} OCR chunk files to vector store...")

        model = get_embedding_model()
        client = get_chroma_client()

        try:
            collection = client.get_collection(COLLECTION_NAME)
            logger.info(f"Existing collection: {collection.count()} vectors")
        except:
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

        total_added = 0
        for chunk_file in chunk_files:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            doc_name = Path(chunk_file).stem.replace(".chunks", "")
            logger.info(f"Embedding {doc_name} ({len(chunks)} chunks)")

            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                texts = [f"passage: {c['text'][:1000]}" for c in batch]
                ids = [c["chunk_id"] for c in batch]
                metadatas = [{
                    "source":       c.get("source", ""),
                    "jurisdiction": c.get("jurisdiction", ""),
                    "country":      c.get("country", ""),
                    "category":     c.get("category", ""),
                    "province":     c.get("province") or "",
                    "page_start":   c.get("page_start", 0),
                    "page_end":     c.get("page_end", 0),
                    "doc_name":     c.get("doc_name", ""),
                    "breadcrumb":   c.get("breadcrumb", "")[:200],
                    "priority":     c.get("priority", 2),
                    "requires_escalation_cue": str(
                        c.get("requires_escalation_cue", False)),
                    "currency_warning": (
                        c.get("currency_warning") or "")[:200],
                } for c in batch]

                try:
                    embeddings = list(model.embed(texts))
                    embeddings = [e.tolist() for e in embeddings]
                    collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=[c["text"][:1000] for c in batch],
                        metadatas=metadatas
                    )
                    total_added += len(batch)
                except Exception as e:
                    logger.error(f"Batch failed: {e}")

            import gc
            gc.collect()

        logger.info(f"Added {total_added} vectors. "
                    f"Total: {collection.count()}")
    else:
        build_vector_store()

    if "--append" not in sys.argv:
        print("\n--- TEST SEARCH ---")
        results = search("landlord deposit return", top_k=3)
        for r in results:
            print(f"Score: {r['score']} | "
                  f"{r['source']} | "
                  f"{r['breadcrumb'][:60]}")