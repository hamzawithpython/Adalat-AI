"""Centralized LLM factory — controls which model each task uses."""
import os
from langchain_groq import ChatGroq

GROQ_KEY = os.getenv("GROQ_API_KEY")


def heavy_llm(max_tokens: int = 2048, temperature: float = 0.2):
    """Premium model. Use ONLY for the user-facing legal answer."""
    return ChatGroq(
        api_key=GROQ_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def fast_llm(max_tokens: int = 1024, temperature: float = 0):
    """Cheap fast model. Use for routing, classification, structuring."""
    return ChatGroq(
        api_key=GROQ_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=max_tokens,
    )