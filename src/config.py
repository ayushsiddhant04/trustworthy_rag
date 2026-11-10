"""Central configuration. External services are disabled unless explicitly enabled."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _resolve_upload_limit_bytes() -> int:
    """Resolve maximum upload size in bytes.
    
    Priority:
    1. MAX_UPLOAD_SIZE_BYTES (explicit byte override)
    2. MAX_UPLOAD_SIZE_MB (explicit MB override)
    3. Target is Vercel (VERCEL=1 or DEPLOYMENT_TARGET=vercel): 4 MB (enforces Vercel's 4.5 MB gateway ceiling)
    4. Default: 25 MB for local workstations and general container deployments
    """
    if os.getenv("MAX_UPLOAD_SIZE_BYTES"):
        try:
            return int(os.getenv("MAX_UPLOAD_SIZE_BYTES"))
        except (TypeError, ValueError):
            pass
    if os.getenv("MAX_UPLOAD_SIZE_MB"):
        try:
            return int(float(os.getenv("MAX_UPLOAD_SIZE_MB")) * 1024 * 1024)
        except (TypeError, ValueError):
            pass
    if os.getenv("VERCEL") or os.getenv("DEPLOYMENT_TARGET", "").lower() == "vercel":
        return 4 * 1024 * 1024
    return 25 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    chunk_size_words: int = _int("CHUNK_SIZE_WORDS", 120)
    chunk_overlap_words: int = _int("CHUNK_OVERLAP_WORDS", 30)
    retrieval_default_k: int = _int("RETRIEVAL_DEFAULT_K", 3)
    tfidf_weight: float = _float("RETRIEVAL_TFIDF_WEIGHT", 0.5)
    embedding_weight: float = _float("RETRIEVAL_EMBED_WEIGHT", 0.5)
    enable_embeddings: bool = _bool("ENABLE_EMBEDDINGS", False)
    embedding_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    enable_external_llm: bool = _bool("ENABLE_EXTERNAL_LLM", False)
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
    llm_timeout_seconds: float = _float("LLM_TIMEOUT_SECONDS", 30.0)
    llm_max_retries: int = _int("LLM_MAX_RETRIES", 2)
    cache_max_items: int = _int("CACHE_MAX_ITEMS", 256)
    cache_ttl_seconds: int = _int("CACHE_TTL_SECONDS", 1800)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # Relevance gating (Task 3: disabled by default)
    enable_relevance_gate: bool = _bool("ENABLE_RELEVANCE_GATE", False)
    relevance_threshold: float = _float("RELEVANCE_THRESHOLD", 0.30)
    relevance_model: str = os.getenv("RELEVANCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # File safeguards (Task 5)
    max_upload_size_bytes: int = _resolve_upload_limit_bytes()
    max_pdf_pages: int = _int("MAX_PDF_PAGES", 100)
    supported_file_extensions: tuple[str, ...] = (".pdf", ".txt", ".docx")


SETTINGS = Settings()


def validate_settings() -> None:
    if SETTINGS.chunk_size_words <= 0:
        raise ValueError("CHUNK_SIZE_WORDS must be greater than zero")
    if SETTINGS.chunk_overlap_words < 0 or SETTINGS.chunk_overlap_words >= SETTINGS.chunk_size_words:
        raise ValueError("CHUNK_OVERLAP_WORDS must be >= 0 and smaller than CHUNK_SIZE_WORDS")
    if SETTINGS.tfidf_weight < 0 or SETTINGS.embedding_weight < 0:
        raise ValueError("Retrieval weights cannot be negative")
    if SETTINGS.tfidf_weight + SETTINGS.embedding_weight <= 0:
        raise ValueError("At least one retrieval weight must be positive")
    if SETTINGS.max_upload_size_bytes <= 0:
        raise ValueError("MAX_UPLOAD_SIZE_BYTES must be greater than zero")
    if SETTINGS.max_pdf_pages <= 0:
        raise ValueError("MAX_PDF_PAGES must be greater than zero")


validate_settings()
