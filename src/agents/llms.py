"""Centralized LLM factory.

Supports:
  - Multiple providers (Groq primary, Cerebras fallback)
  - Per-task model selection (heavy_llm vs fast_llm)
  - User-supplied API keys (BYOK) via context variable
"""
import os
import logging
from contextvars import ContextVar
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ── FallbackLLM: tries providers in order on rate-limit / quota errors
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

load_dotenv()
logger = logging.getLogger(__name__)

# Per-request user-supplied keys (set by middleware in main.py)
_user_keys: ContextVar[dict] = ContextVar("_user_keys", default={})


def set_user_keys(keys: dict) -> None:
    """Stash per-request user keys. Recognized: groq, cerebras, gemini."""
    _user_keys.set({k: v for k, v in (keys or {}).items() if v})


def _get_key(provider: str) -> Optional[str]:
    """User-supplied key first, env-var key second."""
    user_keys = _user_keys.get()
    if provider in user_keys:
        return user_keys[provider]
    env_map = {
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    return os.getenv(env_map.get(provider, ""))


# ── Per-provider model registry ─────────────────────────────────────
HEAVY_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "cerebras": "gpt-oss-120b",
    "gemini": "gemini-1.5-pro",
}

FAST_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "cerebras": "gpt-oss-120b",
    "gemini": "gemini-1.5-flash",
}


def _make_llm(provider: str, model_name: str, max_tokens: int, temperature: float):
    """Construct a LangChain chat model. Returns None if key/SDK missing."""
    key = _get_key(provider)
    if not key:
        return None

    if provider == "groq":
        return ChatGroq(
            api_key=key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "cerebras":
        try:
            from langchain_cerebras import ChatCerebras
        except ImportError:
            logger.warning("langchain-cerebras not installed — fallback unavailable")
            return None
        return ChatCerebras(
            api_key=key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            logger.warning("langchain-google-genai not installed — Gemini unavailable")
            return None
        return ChatGoogleGenerativeAI(
            google_api_key=key,
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    return None


# ── FallbackLLM: tries providers in order on rate-limit / quota errors
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig


class FallbackLLM(Runnable):
    """Wraps multiple LLMs and tries them in order on rate-limit / failure.

    Subclasses Runnable so it works inside LangChain pipelines:
        chain = prompt | fallback_llm | parser
    """

    def __init__(self, llms: list, provider_names: list[str]):
        self.llms = llms
        self.provider_names = provider_names

    def invoke(self, input, config: Optional[RunnableConfig] = None, **kwargs):
        last_error = None
        for llm, name in zip(self.llms, self.provider_names):
            if llm is None:
                continue
            try:
                logger.info(f"LLM call via {name}")
                return llm.invoke(input, config=config, **kwargs)
            except Exception as e:
                msg = str(e).lower()
                if (
                    "rate limit" in msg
                    or "rate_limit" in msg
                    or "429" in msg
                    or "quota" in msg
                    or "too many" in msg
                    or "credit" in msg
                ):
                    logger.warning(f"{name} rate-limited/exhausted, trying next provider")
                    last_error = e
                    continue
                # Other errors — re-raise
                raise
        if last_error:
            raise last_error
        raise RuntimeError("No LLM providers available — check API keys")

# ── Public factories ────────────────────────────────────────────────
def heavy_llm(max_tokens: int = 2048, temperature: float = 0.2):
    """Premium model with fallback chain. Use ONLY for user-facing answer."""
    user_keys = _user_keys.get()

    # User-supplied key → use that provider exclusively (no fallback)
    if "gemini" in user_keys:
        return _make_llm("gemini", HEAVY_MODELS["gemini"], max_tokens, temperature)
    if "groq" in user_keys:
        return _make_llm("groq", HEAVY_MODELS["groq"], max_tokens, temperature)
    if "cerebras" in user_keys:
        return _make_llm("cerebras", HEAVY_MODELS["cerebras"], max_tokens, temperature)

    # Default chain: Groq primary, Cerebras fallback
    llms = [
        _make_llm("groq", HEAVY_MODELS["groq"], max_tokens, temperature),
        _make_llm("cerebras", HEAVY_MODELS["cerebras"], max_tokens, temperature),
    ]
    return FallbackLLM(llms, ["groq", "cerebras"])


def fast_llm(max_tokens: int = 1024, temperature: float = 0):
    """Cheap fast model with fallback chain. Use for routing/structuring."""
    user_keys = _user_keys.get()

    if "gemini" in user_keys:
        return _make_llm("gemini", FAST_MODELS["gemini"], max_tokens, temperature)
    if "groq" in user_keys:
        return _make_llm("groq", FAST_MODELS["groq"], max_tokens, temperature)
    if "cerebras" in user_keys:
        return _make_llm("cerebras", FAST_MODELS["cerebras"], max_tokens, temperature)

    llms = [
        _make_llm("groq", FAST_MODELS["groq"], max_tokens, temperature),
        _make_llm("cerebras", FAST_MODELS["cerebras"], max_tokens, temperature),
    ]
    return FallbackLLM(llms, ["groq", "cerebras"])