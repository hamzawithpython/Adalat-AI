import os
import sys
import uuid
import logging
from datetime import datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from src.api.database import get_db, create_tables, ChatHistory
from src.agents.router import ask
from src.retrieval.embedder import get_embedding_model

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Adalat-AI API",
    description="Roman-Urdu Legal Assistant for PK/UK/DE Law",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    create_tables()
    logger.info("Database tables created")
    logger.info("Pre-loading embedding model...")
    get_embedding_model()
    logger.info("Embedding model ready")

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

import asyncio
from fastapi.concurrency import run_in_threadpool

@app.post("/ask")
async def ask_question(request: QueryRequest, db: Session = Depends(get_db)):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(request.query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 chars)")

    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Query received: {request.query[:50]}...")

    try:
        result = await run_in_threadpool(ask, request.query)
    except BaseException as e:
        logger.exception(f"Router error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    try:
        chat_record = ChatHistory(
            session_id=session_id,
            query=result["query"],
            translated_query=result.get("translated_query"),
            language=str(result["language"]),
            jurisdiction=str(result["jurisdiction"]),
            answer=result["answer"],
            rights=result.get("rights", []),
            citations=result.get("citations", []),
            confidence=result.get("confidence", 0.0)
        )
        db.add(chat_record)
        db.commit()
        db.refresh(chat_record)
    except Exception as e:
        logger.warning(f"DB save failed: {e}")

    return {"session_id": session_id, **result}


@app.get("/history")
def get_history(
    session_id: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    query = db.query(ChatHistory).order_by(ChatHistory.created_at.desc())
    if session_id:
        query = query.filter(ChatHistory.session_id == session_id)
    records = query.limit(limit).all()
    return {
        "total": len(records),
        "records": [
            {
                "id": r.id,
                "session_id": r.session_id,
                "query": r.query,
                "jurisdiction": r.jurisdiction,
                "language": r.language,
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat()
            }
            for r in records
        ]
    }

@app.get("/history/{record_id}")
def get_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(ChatHistory).filter(ChatHistory.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return {
        "id": record.id,
        "session_id": record.session_id,
        "query": record.query,
        "translated_query": record.translated_query,
        "language": record.language,
        "jurisdiction": record.jurisdiction,
        "answer": record.answer,
        "rights": record.rights,
        "citations": record.citations,
        "confidence": record.confidence,
        "created_at": record.created_at.isoformat()
    }
