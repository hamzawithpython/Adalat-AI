# Adalat-AI

> **Your Rights. In Your Language.**
> A bilingual legal assistant for Pakistan, the UK, and Germany — ask in Roman-Urdu, English, or German and get a structured, citation-anchored answer.

[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://frontend-production-fde6.up.railway.app)
[![API](https://img.shields.io/badge/API-FastAPI-009688)](https://backend-production-f61e7.up.railway.app/docs)
[![Built with](https://img.shields.io/badge/built%20with-Next.js%20%2B%20FastAPI-000000)](#tech-stack)

---

## Live Demo

- **App:** https://frontend-production-fde6.up.railway.app
- **API docs:** https://backend-production-f61e7.up.railway.app/docs
- **Health:** https://backend-production-f61e7.up.railway.app/health

Try a query in Roman-Urdu: *"Mera landlord deposit wapas nahi de raha, kya karoon?"*

---

## What it does

Adalat-AI takes a casual legal question — even in Roman-Urdu — and returns a grounded, structured answer with article-level citations from real statutes. It detects language and jurisdiction automatically, retrieves from 8,322 vector chunks across 47 legal documents, and renders the answer as expandable sections with rights, deadlines, and recourse paths broken out.

It is built as a portfolio capstone: every component is a real RAG pipeline rather than a thin LLM wrapper.

### Key capabilities

- **Tri-lingual input + output** — Roman-Urdu, English, German. Answer language matches the user's language.
- **Tri-jurisdictional coverage** — Pakistani, UK, German law from real PDF statutes.
- **Article-level retrieval** — Chroma vector DB with 8,322 chunks from 47 legal documents (Constitution, PPC, CrPC, Tenancy Acts, BGB, Tenant Fees Act, Housing Acts, etc.).
- **Structured output** — Pydantic-validated response with sections, judgments, rights, citations, and confidence score.
- **Pakistani Roman-Urdu vocabulary guidance** — explicit guard against Hindi-leaning vocabulary in answers.
- **Anti-hallucination grounding rules** — citations limited to retrieved context, with honest framing for illustrative case law.

---

## Architecture

User Query (Roman-Urdu / English / German)
↓
LangGraph Router (src/agents/router.py)
├─ Node 1: detect_language       (Llama 3.3-70B classifier)
├─ Node 2: detect_jurisdiction   (PK / UK / DE)
├─ Node 3: translate_query       (→ English for retrieval)
├─ Node 4: run_rag
│    ├─ fastembed (ONNX) → query vector
│    ├─ Chroma similarity search (filtered by country)
│    └─ Llama 3.3-70B → grounded prose answer in user's language
└─ Node 5: structure_response    (sections + illustrative judgments)
↓
Claim extractor (Pydantic-validated rights[])
↓
LegalResponse (validated JSON):

-query, translated_query, language, jurisdiction
-sections[]    → Legal Context, Statutory Framework, etc. (markdown, in user's language)
-judgments[]   → court, outcome, sections invoked, summary, cited_cases
-rights[]      → right, legal_basis, deadline, recourse
-citations[]   → source PDF, page, breadcrumb, relevance_score
-confidence    → average retrieval score
-disclaimer, judgments_disclaimer

The frontend renders this JSON as cards: section cards, expandable judgment cards, rights cards with deadlines highlighted, and citation cards with relevance score bars.

---

## Tech stack

| Layer | Technology |
|---|---|
| **LLM** | `llama-3.3-70b-versatile` via Groq API |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` via fastembed (ONNX runtime) |
| **Vector DB** | Chroma (persistent, 8,322 vectors) |
| **Agent framework** | LangChain + LangGraph |
| **PDF parsing** | PyMuPDF + Tesseract OCR (for scanned documents) |
| **Chunking** | Structural (article-based) + sliding window with breadcrumbs |
| **Validation** | Pydantic v2 (`LegalResponse` schema) |
| **API** | FastAPI + SQLite for chat history |
| **Frontend** | Next.js 16 (App Router) + TypeScript + TailwindCSS |
| **Markdown** | react-markdown + remark-gfm |
| **Hosting** | Railway (backend + frontend) |
| **CI/CD** | GitHub Actions (lint, test, auto-ingest on PDF push) |

---

## Document corpus

**47 legal documents · 8,322 vector chunks**

| Jurisdiction | Coverage |
|---|---|
| 🇵🇰 **Pakistan** | Constitution, Pakistan Penal Code 1860, CrPC 1898, Qanun-e-Shahadat 1984, Anti-Terrorism Act 1997, Contract Act 1872, Punjab/Sindh/KP/Islamabad Tenancy Laws, Consumer Protection Acts, Family Laws, Labour Laws |
| 🇬🇧 **United Kingdom** | Tenant Fees Act 2019, Housing Acts 1988/1996, Landlord & Tenant Act 1985, Renters Rights Act 2025, Homes Fitness Act 2018, Consumer Rights Act 2015, Employment Rights Act 1996, Equality Act 2010, Immigration Acts |
| 🇩🇪 **Germany** | BGB (English + German), Mietrechtsreformgesetz 2001, Betriebskostenverordnung, UWG, PAngV, UKlaG |

7 documents required OCR (Tesseract) — accuracy 85–95% on UK docs, 75–88% on Pakistani docs.

---

## API

GET  /health        → {"status": "ok", "version": "1.0.0"}
POST /ask           → full LegalResponse JSON
GET  /history       → list of past queries (SQLite-backed)
GET  /docs          → Swagger UI

### Sample request

```bash
curl -X POST https://backend-production-f61e7.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Mera landlord deposit wapas nahi de raha, kya karoon?"}'
```

### Sample response (truncated)

```json
{
  "query": "Mera landlord deposit wapas nahi de raha, kya karoon?",
  "translated_query": "My landlord is not returning my deposit, what should I do?",
  "language": "roman_urdu",
  "jurisdiction": "PK",
  "answer": "Aap ko apne landlord ke khilaf adalat mein muqadma karna chahiye...",
  "sections": [
    {
      "heading": "Statutory Framework",
      "content": "**Sindh Rented Premises Ordinance 1979** ke **Section 15**...",
      "icon_hint": "book"
    }
  ],
  "judgments": [
    {
      "case_title": "Karachi v. Rashid",
      "citation": "2018 SCMR 1142",
      "court": "Supreme Court of Pakistan",
      "outcome": "Petition Allowed",
      "summary": "...",
      "sections": ["Section 15"],
      "cited_cases": ["PLD 1980 SC 9"]
    }
  ],
  "rights": [...],
  "citations": [...],
  "confidence": 0.52,
  "disclaimer": "This is informational only..."
}
```

---

## Local development

### Backend

```bash
git clone https://github.com/hamzawithpython/Adalat-AI
cd Adalat-AI
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows; or: source venv/bin/activate
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

python run.py   # FastAPI on :8001
```

### Frontend

```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8001" > .env.local
npm run dev   # Next.js on :3000
```

Visit `http://localhost:3000`.

---

## Project structure

adalat-ai/
├── .github/workflows/ci.yml       # Lint, test, auto-ingest on PDF push
├── configs/
│   ├── document_registry.py       # Master list of all 47 documents
│   └── evaluate.py                # Retrieval evaluation
├── data/
│   ├── raw/                       # Source PDFs (gitignored)
│   ├── processed/chunks/          # Per-document chunk JSON
│   └── embeddings/chroma/         # 8,322 committed vectors (~79MB)
├── docs/ARCHITECTURE.md
├── frontend/                      # Next.js + TypeScript + TailwindCSS
│   ├── app/
│   │   ├── page.tsx               # Landing
│   │   └── chat/page.tsx          # Chat interface
│   ├── components/
│   │   ├── brand/                 # Wordmark, Flag
│   │   ├── chat/                  # Section cards, judgment cards, etc.
│   │   ├── icons/                 # Custom legal SVG icons
│   │   ├── landing/               # Landing-only components
│   │   └── ui/                    # Btn, Badge, Card, Markdown
│   ├── hooks/use-chat.ts
│   ├── lib/api.ts                 # Typed API client
│   └── types/legal.ts             # Mirrors Pydantic schema
├── scripts/
│   ├── ingest_documents.py
│   ├── ocr_scanned_docs.py
│   └── inspect_chunks.py
├── src/
│   ├── agents/
│   │   ├── router.py              # LangGraph orchestrator
│   │   └── structurer.py          # Sections + judgments generator
│   ├── api/
│   │   ├── main.py                # FastAPI app
│   │   └── database.py            # SQLAlchemy + SQLite
│   ├── ingestion/                 # PDF loader, chunker, metadata
│   ├── retrieval/
│   │   ├── embedder.py            # fastembed + Chroma
│   │   └── rag_chain.py           # LangChain RAG with Groq
│   └── schemas/
│       ├── legal_response.py      # Pydantic v2 models
│       └── extractor.py           # Rights extractor
├── tests/
├── railway.backend.json           # Railway config (backend service)
├── frontend/railway.json          # Railway config (frontend service)
└── run.py                         # FastAPI entry point

---

## Adding a new document

```bash
# 1. Drop the PDF into data/raw/
# 2. Add an entry to configs/document_registry.py
# 3. Run:
python scripts/ingest_documents.py
# Done — new document is searchable.
```

CI/CD auto-runs ingestion when new PDFs are pushed to GitHub.

---

## Honest disclaimers

- **Illustrative judgments are LLM-suggested**, not verified retrievals from a case-law database. The frontend labels them as such with an amber banner.
- **The corpus does not include case law** — only statutes. Adding a real judgment dataset is on the v2 roadmap.
- **This tool is not legal advice.** Every response ends with a disclaimer to consult a qualified lawyer.

---

## Roadmap (v2+)

- Replace LLM-suggested judgments with grounded retrieval from a real case-law dataset (Pakistan Code, BAILII, openJur)
- Fine-tune embeddings on legal Roman-Urdu pairs
- Upgrade to `multilingual-e5-large` on GPU (target retrieval score 0.80+)
- Add BGE-reranker-v2-m3 for hybrid reranking
- RAGAS evaluation suite
- Multi-turn conversation
- Chat history sidebar wired to `/history`
- Mobile-responsive layouts (the hi-fi designs exist, just not yet built)

---

## Author

Built by **Hamza** as a bootcamp capstone project.

GitHub: https://github.com/hamzawithpython/Adalat-AI

---

## Licence

MIT

