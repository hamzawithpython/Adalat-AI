# Adalat-AI Architecture

## Pipeline
User Query → Language Detection → Jurisdiction Router → RAG Agent → Pydantic Schema → Response

## Adding New Documents
1. Place PDF in `data/raw/`
2. Add entry to `JURISDICTION_MAP` in `src/ingestion/pdf_loader.py`
3. Run: `python scripts/ingest_documents.py`

## Adding New Languages
1. Add keywords to `scripts/add_language.py`
2. Update detection prompt in `src/agents/router.py`
3. Add translation logic if needed

## Adding New Jurisdictions
1. Add new PDFs to `data/raw/`
2. Update `JURISDICTION_MAP` in `pdf_loader.py`
3. Update jurisdiction detection prompt in `src/agents/router.py`
4. Re-run ingestion pipeline