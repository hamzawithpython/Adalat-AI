import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime,
    JSON, Boolean, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

load_dotenv()

DATABASE_URL = os.getenv("POSTGRES_URL", "sqlite:///./adalat_chat.db")

# Normalize legacy postgres:// scheme
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # Postgres (Neon): pool_pre_ping handles dropped idle connections
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── ChatSession (one conversation thread) ──────────────────────────────
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True)  # UUID string
    visitor_id = Column(String, nullable=True, index=True)
    title = Column(String, nullable=True)  # auto-generated from first query
    jurisdiction = Column(String, nullable=True)
    language = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    turns = relationship("ChatTurn", back_populates="session", cascade="all, delete-orphan")


# ─── ChatTurn (one query+answer within a session) ───────────────────────
class ChatTurn(Base):
    __tablename__ = "chat_turns"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    turn_index = Column(Integer, default=0)  # 0, 1, 2, ... within the session
    query = Column(Text, nullable=False)
    translated_query = Column(Text, nullable=True)
    language = Column(String, nullable=False)
    jurisdiction = Column(String, nullable=False)
    answer = Column(Text, nullable=False)
    sections = Column(JSON, nullable=True)
    judgments = Column(JSON, nullable=True)
    rights = Column(JSON, nullable=True)
    citations = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    response_language = Column(String, nullable=True)
    follow_up_questions = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    session = relationship("ChatSession", back_populates="turns")


# ─── Feedback (landing page form submissions) ───────────────────────────
class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    category = Column(String, nullable=False)  # 'bug' | 'feature' | 'praise' | 'other'
    message = Column(Text, nullable=False)
    rating = Column(Integer, nullable=True)  # 1-5
    user_agent = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# ─── Legacy ChatHistory (KEPT for backward compat with existing data) ───
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    query = Column(Text, nullable=False)
    translated_query = Column(Text, nullable=True)
    language = Column(String, nullable=False)
    jurisdiction = Column(String, nullable=False)
    answer = Column(Text, nullable=False)
    rights = Column(JSON, nullable=True)
    citations = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()