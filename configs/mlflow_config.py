"""
MLOps Experiment Tracking Config — Adalat-AI
Tracks retrieval performance across model versions.
"""

MLFLOW_CONFIG = {
    "experiment_name": "adalat-ai-retrieval",
    "tracking_uri": "sqlite:///logs/mlflow.db",
    "model_versions": {
        "v1.0": {
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "chunk_size": 800,
            "overlap": 100,
            "top_k": 5,
            "notes": "Baseline — CPU-friendly 90MB model"
        },
        "v2.0_planned": {
            "embedding_model": "intfloat/multilingual-e5-base",
            "chunk_size": 512,
            "overlap": 50,
            "top_k": 5,
            "notes": "Planned — better multilingual support"
        },
        "v3.0_planned": {
            "embedding_model": "intfloat/multilingual-e5-large",
            "chunk_size": 512,
            "overlap": 50,
            "top_k": 5,
            "notes": "Planned — best quality, requires GPU"
        }
    },
    "metrics_to_track": [
        "retrieval_score_pk",
        "retrieval_score_uk",
        "retrieval_score_de",
        "retrieval_score_roman_urdu",
        "schema_conformance_rate",
        "avg_response_time_ms"
    ]
}

EVALUATION_QUERIES = [
    {"query": "What are my fundamental rights if arrested?",
     "jurisdiction": "PK", "expected_min_score": 0.55},
    {"query": "What fees can my landlord charge me?",
     "jurisdiction": "UK", "expected_min_score": 0.60},
    {"query": "My landlord is not returning my deposit",
     "jurisdiction": "DE", "expected_min_score": 0.45},
    {"query": "mera landlord deposit wapas nahi de raha",
     "jurisdiction": "PK", "expected_min_score": 0.30},
    {"query": "Can police detain me without charge in Pakistan?",
     "jurisdiction": "PK", "expected_min_score": 0.55},
]