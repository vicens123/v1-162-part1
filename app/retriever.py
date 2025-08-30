# app/retriever.py

import os
from dotenv import load_dotenv

from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings

# ✅ Cargar variables de entorno y forzar que sobrescriba si ya hay definidas
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

# 🔍 Verificación explícita del nombre de la colección
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")
DATABASE_URL = os.getenv("DATABASE_URL")

# Uso exclusivo de OpenAI Embeddings para consistencia
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")

# Configuración de búsqueda del retriever
RAG_SEARCH_TYPE = os.getenv("RAG_SEARCH_TYPE", "similarity").lower()  # similarity | mmr | similarity_score_threshold
try:
    RAG_K = int(os.getenv("RAG_K", "10"))
except ValueError:
    RAG_K = 10
try:
    RAG_FETCH_K = int(os.getenv("RAG_FETCH_K", "60"))
except ValueError:
    RAG_FETCH_K = 60

def _make_embeddings():
    return OpenAIEmbeddings(model=EMBEDDINGS_MODEL)

def get_retriever():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL no está definido en el entorno")

    embedding_function = _make_embeddings()

    # Inicializamos el vectorstore con PGVector
    vectorstore = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=DATABASE_URL,
        embedding_function=embedding_function,
    )

    # Configurar kwargs según el tipo de búsqueda
    search_kwargs = {"k": RAG_K}
    if RAG_SEARCH_TYPE == "mmr":
        search_kwargs["fetch_k"] = RAG_FETCH_K

    retriever = vectorstore.as_retriever(
        search_type=RAG_SEARCH_TYPE,
        search_kwargs=search_kwargs,
    )

    return retriever

