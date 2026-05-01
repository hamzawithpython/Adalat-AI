import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/embeddings/chroma")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "adalat_legal_docs"
BATCH_SIZE = 64


# Add this at module level (after imports)
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded.")
    return _model

def get_chroma_client():
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client


def build_vector_store(chunks_path: str = "data/processed/chunks.json"):
    """Embed all chunks and store in Chroma."""

    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")

    # Load model
    model = get_embedding_model()

    # Connect to Chroma
    client = get_chroma_client()

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection")
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Embed in batches
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        texts = [f"passage: {c['text']}" for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [{
            "source": c["source"],
            "jurisdiction": c["jurisdiction"],
            "page_num": c["page_num"],
            "doc_name": c["doc_name"]
        } for c in batch]

        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=metadatas
        )

        logger.info(f"Embedded {min(i + BATCH_SIZE, total)}/{total} chunks")

    logger.info(f"Vector store built. Total: {collection.count()} vectors")
    return collection


def search(query: str, jurisdiction: str = None, top_k: int = 5):
    """Search vector store for relevant chunks."""
    model = get_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    ).tolist()

    where_filter = {"jurisdiction": jurisdiction} if jurisdiction else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for i in range(len(results["documents"][0])):
        output.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": round(1 - results["distances"][0][i], 4)
        })

    return output


if __name__ == "__main__":
    try:
        build_vector_store()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Step 2: Test search
    print("\n--- TEST QUERY 1 (PK) ---")
    results = search("fundamental rights of citizens", jurisdiction="PK", top_k=3)
    for r in results:
        print(f"\nScore: {r['score']} | {r['metadata']['source']} | Page {r['metadata']['page_num']}")
        print(r['text'][:200])

    print("\n--- TEST QUERY 2 (UK) ---")
    results = search("landlord deposit return rules", jurisdiction="UK", top_k=3)
    for r in results:
        print(f"\nScore: {r['score']} | {r['metadata']['source']} | Page {r['metadata']['page_num']}")
        print(r['text'][:200])

    print("\n--- TEST QUERY 3 (Roman Urdu) ---")
    results = search("mera landlord deposit wapas nahi de raha", jurisdiction="PK", top_k=3)
    for r in results:
        print(f"\nScore: {r['score']} | {r['metadata']['source']} | Page {r['metadata']['page_num']}")
        print(r['text'][:200])