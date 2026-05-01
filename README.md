# ⚖️ Adalat-AI

> **Roman-Urdu-Aware Legal Assistant for Pakistani & UK Citizenship/Tenancy Law**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)

---

## ⚠️ Disclaimer
This tool is for **informational purposes only**. It does not constitute legal advice. Always consult a qualified lawyer for legal matters.

---

## 🎯 What is Adalat-AI?

A bilingual (Roman-Urdu/English) RAG + agent system that answers legal questions grounded in:
- 🇵🇰 **Pakistan** — Constitution of Pakistan + Pakistan Penal Code
- 🇬🇧 **United Kingdom** — UK Tenant Fees Act 2019
- 🇩🇪 **Germany** — BGB §§535–548 (Rental Law)

Returns citation-anchored answers with structured "applicable rights" output.

---

## ✨ Key Features

- **Roman-Urdu input** — ask in the way Pakistanis naturally type
- **Auto jurisdiction routing** — no need to specify PK/UK/DE manually
- **Article-level citations** — every answer references exact pages
- **Structured rights output** — rights, obligations, deadlines, recourse
- **Confidence scoring** — know how reliable the answer is
- **Chat history** — stored in PostgreSQL

---

## 🏗️ Architecture

User Query (Roman-Urdu / English / German)
↓
LangGraph Router
├── Language Detection (Groq LLM)
├── Jurisdiction Detection (Groq LLM)
├── Roman-Urdu Translation (Groq LLM)
└── RAG Agent
├── Embedding (multilingual-MiniLM)
├── Vector Search (Chroma)
└── Answer Generation (Llama-3.1-8B via Groq)
↓
Pydantic Schema Validation
↓
FastAPI → PostgreSQL + Streamlit UI

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)
```bash
git clone https://github.com/hamzawithpython/Adalat-AI.git
cd adalat-ai
cp .env.example .env   # add your GROQ_API_KEY
docker-compose up --build
```
Then open: http://localhost:8501

### Option 2 — Local
```bash
git clone https://github.com/hamzawithpython/Adalat-AI.git
cd adalat-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/retrieval/embedder.py  # build vector store once
python run.py                      # start API
streamlit run src/ui/app.py       # start UI
```

---

## 📁 Project Structure
adalat-ai/
├── data/
│   ├── raw/           # Legal PDFs
│   └── processed/     # Chunked documents
├── src/
│   ├── ingestion/     # PDF loader + chunker
│   ├── retrieval/     # Embeddings + RAG chain
│   ├── agents/        # LangGraph router
│   ├── api/           # FastAPI backend
│   ├── schemas/       # Pydantic models
│   └── ui/            # Streamlit frontend
├── docker/            # Dockerfiles
├── docker-compose.yml
├── run.py
└── requirements.txt
---

## 📊 Evaluation Metrics

| Metric | Score |
|--------|-------|
| UK Retrieval Score | 0.68 |
| PK Retrieval Score | 0.60 |
| Roman-Urdu Score | 0.34 |
| Schema Conformance | 100% |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Llama-3.1-8B (Groq) |
| Embeddings | multilingual-MiniLM-L12-v2 |
| Vector DB | Chroma |
| Agent Framework | LangGraph |
| API | FastAPI |
| UI | Streamlit |
| Database | PostgreSQL |
| Containerization | Docker |

---

## 📚 Data Sources

- Pakistan Constitution (na.gov.pk)
- Pakistan Penal Code (fmu.gov.pk)
- UK Tenant Fees Act (legislation.gov.uk)
- German BGB (gesetze-im-internet.de)

---

## 🔮 Future Improvements (MLOps Roadmap)

- Fine-tune embeddings on legal Roman-Urdu pairs
- Upgrade to multilingual-e5-large on GPU
- Add BGE-reranker-v2 for better retrieval
- RAGAS evaluation pipeline
- CI/CD with GitHub Actions
- Model versioning with MLflow