# app/retriever.py

import os
from dotenv import load_dotenv

from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Cargar variables de entorno y forzar que sobrescriba si ya hay definidas
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

# 🔍 Verificación explícita del nombre de la colección
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")
DATABASE_URL = os.getenv("DATABASE_URL")

# Selector de embeddings por entorno
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")  # modelo opcional

def _make_embeddings():
    if EMBEDDINGS_PROVIDER == "huggingface":
        model_name = EMBEDDINGS_MODEL or "all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=model_name)
    # por defecto OpenAI
    model_name = EMBEDDINGS_MODEL or "text-embedding-3-small"
    return OpenAIEmbeddings(model=model_name)

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

    # Usamos MMR para mejorar diversidad + flexibilidad de resultados
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,        # número final de chunks devueltos
            "fetch_k": 40  # número de candidatos antes de aplicar MMR
        }
    )

    return retriever

