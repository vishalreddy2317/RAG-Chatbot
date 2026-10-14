"""Central configuration for the RAG Document Assistant.

Every setting can be overridden with an environment variable (see
``.env.example``).  On Streamlit Community Cloud the same keys can be placed in
``st.secrets``; ``app.py`` copies them into ``os.environ`` before the pipeline
is created so this module never needs to import Streamlit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional; the app works without a .env file.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - only hit when python-dotenv is missing
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ANSWER_MODES = ("auto", "extractive", "ollama", "hosted")
HOSTED_PROVIDERS = ("openai", "anthropic")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings.  Build with :meth:`Settings.from_env` or directly."""

    # --- embeddings / chunking / retrieval ---------------------------------
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = 800          # characters per chunk
    chunk_overlap: int = 150       # characters shared by neighbouring chunks
    top_k: int = 5                 # passages retrieved per question
    score_threshold: float = 0.30  # cosine similarity needed to count as evidence
    strong_score: float = 0.50     # above this, evidence is accepted even without a keyword match
    min_chars_per_page: int = 25   # pages with fewer characters count as "no text"

    # --- answer generation -------------------------------------------------
    answer_mode: str = "auto"      # auto | extractive | ollama | hosted
    extractive_max_sentences: int = 3
    max_context_chars: int = 6000  # context budget sent to an LLM
    request_timeout: int = 60      # seconds for LLM HTTP calls

    # --- local LLM via Ollama ----------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # --- optional hosted LLM API ------------------------------------------
    hosted_provider: str = "openai"  # openai (any OpenAI-compatible API) | anthropic
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-5-5"

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(
            embedding_model=_env_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            chunk_size=_env_int("CHUNK_SIZE", 800),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
            top_k=_env_int("TOP_K", 5),
            score_threshold=_env_float("SCORE_THRESHOLD", 0.30),
            strong_score=_env_float("STRONG_SCORE", 0.50),
            min_chars_per_page=_env_int("MIN_CHARS_PER_PAGE", 25),
            answer_mode=_env_str("ANSWER_MODE", "auto").lower(),
            extractive_max_sentences=_env_int("EXTRACTIVE_MAX_SENTENCES", 3),
            max_context_chars=_env_int("MAX_CONTEXT_CHARS", 6000),
            request_timeout=_env_int("REQUEST_TIMEOUT", 60),
            ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=_env_str("OLLAMA_MODEL", "llama3.2:3b"),
            hosted_provider=_env_str("HOSTED_PROVIDER", "openai").lower(),
            openai_api_key=_env_str("OPENAI_API_KEY", ""),
            openai_base_url=_env_str("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_model=_env_str("OPENAI_MODEL", "gpt-4o-mini"),
            anthropic_api_key=_env_str("ANTHROPIC_API_KEY", ""),
            anthropic_model=_env_str("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.answer_mode not in ANSWER_MODES:
            raise ValueError(f"answer_mode must be one of {ANSWER_MODES}, got {self.answer_mode!r}")
        if self.hosted_provider not in HOSTED_PROVIDERS:
            raise ValueError(f"hosted_provider must be one of {HOSTED_PROVIDERS}")

    @property
    def hosted_api_key(self) -> str:
        return self.anthropic_api_key if self.hosted_provider == "anthropic" else self.openai_api_key

    @property
    def hosted_model(self) -> str:
        return self.anthropic_model if self.hosted_provider == "anthropic" else self.openai_model

    @property
    def has_hosted_key(self) -> bool:
        return bool(self.hosted_api_key)
