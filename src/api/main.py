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

from src.api.database import get_db, create_tables, ChatSession, ChatTurn, Feedback
from src.agents.router import ask
from src.retrieval.embedder import get_embedding_model

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _enum_value(v):
    """Extract clean string from a Pydantic enum or already-stringified enum."""
    if v is None:
        return None
    if hasattr(v, "value"):
        return v.value
    s = str(v)
    # Handle case where it was already stringified as "Jurisdiction.PK"
    if "." in s and s.split(".")[0] in ("Jurisdiction", "Language"):
        return s.split(".", 1)[1].lower() if "Language" in s else s.split(".", 1)[1]
    return s

app = FastAPI(
    title="Adalat-AI API",
    description="Roman-Urdu Legal Assistant for PK/UK/DE Law",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.up\.railway\.app|http://localhost:\d+",
    allow_credentials=False,
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

class FeedbackRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    category: str  # 'bug' | 'feature' | 'praise' | 'other'
    message: str
    rating: Optional[int] = None

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

    # Persist as ChatSession + ChatTurn (new schema)
    try:
        session_obj = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session_obj:
            # First turn — create the session, derive a title from the query
            title = result["query"][:80]
            session_obj = ChatSession(
                id=session_id,
                title=title,
                jurisdiction=_enum_value(result["jurisdiction"]),
                language=_enum_value(result["language"]),
            )
            db.add(session_obj)
            db.flush()  # ensure session row exists before inserting turn
            turn_index = 0
        else:
            # Subsequent turn — increment index, refresh updated_at via onupdate
            turn_index = db.query(ChatTurn).filter(
                ChatTurn.session_id == session_id
            ).count()
            session_obj.updated_at = datetime.utcnow()

        turn = ChatTurn(
            session_id=session_id,
            turn_index=turn_index,
            query=result["query"],
            translated_query=result.get("translated_query"),
            language=_enum_value(result["language"]),
            jurisdiction=_enum_value(result["jurisdiction"]),
            answer=result["answer"],
            sections=result.get("sections", []),
            judgments=result.get("judgments", []),
            rights=result.get("rights", []),
            citations=result.get("citations", []),
            confidence=result.get("confidence", 0.0),
            response_language=result.get("response_language"),
            follow_up_questions=result.get("follow_up_questions", []),
        )
        db.add(turn)
        db.commit()
    except Exception as e:
        logger.warning(f"DB save failed: {e}")
        db.rollback()

    return {"session_id": session_id, **result}

@app.get("/history")
def list_sessions(limit: int = 30, db: Session = Depends(get_db)):
    """Return recent chat sessions, ordered by most recently updated."""
    sessions = (
        db.query(ChatSession)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "total": len(sessions),
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "jurisdiction": s.jurisdiction,
                "language": s.language,
                "turn_count": len(s.turns),
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ],
    }


@app.get("/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)):
    """Return a full chat session with all turns in chronological order."""
    session_obj = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session_obj:
        raise HTTPException(status_code=404, detail="Session not found")

    turns = (
        db.query(ChatTurn)
        .filter(ChatTurn.session_id == session_id)
        .order_by(ChatTurn.turn_index.asc())
        .all()
    )
    return {
        "id": session_obj.id,
        "title": session_obj.title,
        "jurisdiction": session_obj.jurisdiction,
        "language": session_obj.language,
        "created_at": session_obj.created_at.isoformat(),
        "updated_at": session_obj.updated_at.isoformat(),
        "turns": [
            {
                "id": t.id,
                "turn_index": t.turn_index,
                "query": t.query,
                "translated_query": t.translated_query,
                "language": t.language,
                "jurisdiction": t.jurisdiction,
                "answer": t.answer,
                "sections": t.sections or [],
                "judgments": t.judgments or [],
                "rights": t.rights or [],
                "citations": t.citations or [],
                "confidence": t.confidence,
                "response_language": t.response_language,
                "follow_up_questions": t.follow_up_questions or [],
                "created_at": t.created_at.isoformat(),
            }
            for t in turns
        ],
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete a chat session and all its turns."""
    session_obj = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session_obj:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session_obj)
    db.commit()
    return {"deleted": session_id}

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(req.message) > 5000:
        raise HTTPException(status_code=400, detail="Message too long (max 5000 chars)")
    if req.category not in {"bug", "feature", "praise", "other"}:
        raise HTTPException(status_code=400, detail="Invalid category")
    if req.rating is not None and (req.rating < 1 or req.rating > 5):
        raise HTTPException(status_code=400, detail="Rating must be 1-5")
    if req.email and "@" not in req.email:
        raise HTTPException(status_code=400, detail="Invalid email")

    fb = Feedback(
        name=(req.name or "").strip()[:100] or None,
        email=(req.email or "").strip()[:200] or None,
        category=req.category,
        message=req.message.strip(),
        rating=req.rating,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return {"id": fb.id, "ok": True}


@app.get("/feedback/admin")
def list_feedback(token: str = "", limit: int = 100, db: Session = Depends(get_db)):
    """Admin-only endpoint to read all feedback. Pass ?token=YOUR_ADMIN_TOKEN."""
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    rows = (
        db.query(Feedback)
        .order_by(Feedback.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "total": len(rows),
        "items": [
            {
                "id": r.id,
                "name": r.name,
                "email": r.email,
                "category": r.category,
                "message": r.message,
                "rating": r.rating,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
    }