import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv


def _project_root() -> Path:
    # Intenta cargar .env desde la raíz del proyecto (dos niveles por encima de este archivo)
    return Path(__file__).resolve().parents[1]


def _parse_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(val: str | None, default: int) -> int:
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _parse_list(val: str | None, default: List[str]) -> List[str]:
    if val is None or not val.strip():
        return default
    return [x.strip() for x in val.split(",") if x.strip()]


# Carga del .env (una sola vez)
_ENV_PATH = _project_root() / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


class Config:
    # OpenAI
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Embeddings
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")

    # PGVector
    DATABASE_URL: str | None = os.getenv("DATABASE_URL")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_collection")

    # Retriever params
    RAG_SEARCH_TYPE: str = os.getenv("RAG_SEARCH_TYPE", "similarity").lower()
    RAG_K: int = _parse_int(os.getenv("RAG_K"), 10)
    RAG_FETCH_K: int = _parse_int(os.getenv("RAG_FETCH_K"), 60)

    # RAG context limit
    MAX_CONTEXT_WORDS: int = _parse_int(os.getenv("MAX_CONTEXT_WORDS"), 6000)

    # LangSmith / LangChain tracing
    LANGCHAIN_TRACING_V2: bool = _parse_bool(os.getenv("LANGCHAIN_TRACING_V2"), False)
    LANGCHAIN_API_KEY: str | None = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str | None = os.getenv("LANGCHAIN_PROJECT")

    # Admin
    ADMIN_API_KEY: str | None = os.getenv("ADMIN_API_KEY")

    # CORS
    CORS_ORIGINS: List[str] = _parse_list(os.getenv("CORS_ORIGINS"), ["http://localhost:3000"])

    # Upload limits
    UPLOAD_MAX_SIZE_MB: int = _parse_int(os.getenv("UPLOAD_MAX_SIZE_MB"), 20)
    UPLOAD_MAX_FILES: int = _parse_int(os.getenv("UPLOAD_MAX_FILES"), 10)


config = Config()

