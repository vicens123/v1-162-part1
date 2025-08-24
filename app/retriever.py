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

print(f"🔁 COLLECTION_NAME from .env: {COLLECTION_NAME}")

def get_retriever():
    # Usamos los embeddings de OpenAI
    embedding_function = OpenAIEmbeddings()

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

