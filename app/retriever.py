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

    # Usamos MMR para mejorar diversidad + flexibilidad de resultados
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,        # número final de chunks devueltos
            "fetch_k": 40  # número de candidatos antes de aplicar MMR
        }
    )

    return retriever

