from pathlib import Path
from dotenv import load_dotenv
import os


# 🔥 Cargar manualmente el .env desde dos niveles por encima de este archivo
from pathlib import Path
from dotenv import load_dotenv

# Ruta correcta al .env dentro de v1-162-part1
env_path = Path(__file__).resolve().parents[1] / ".env"
print(f"🔍 Buscando .env en: {env_path}")

if load_dotenv(dotenv_path=env_path):
    print("✅ .env cargado correctamente.")
else:
    raise RuntimeError("❌ No se pudo cargar el archivo .env desde la ruta esperada.")

print(f"✅ ¿Cargó correctamente el .env?: {load_dotenv(dotenv_path=env_path)}")
print(f"🔎 DATABASE_URL = {os.getenv('DATABASE_URL')}")
print(f"🔎 OPENAI_API_KEY = {os.getenv('OPENAI_API_KEY')}")

import os
import pytest
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.schema.document import Document

@pytest.fixture(scope="module")
def db_config():
    connection_string = os.getenv("DATABASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not connection_string:
        raise ValueError("❌ Falta DATABASE_URL en el archivo .env")
    if not openai_api_key:
        raise ValueError("❌ Falta OPENAI_API_KEY en el archivo .env")
    
    return {
        "connection_string": connection_string,
        "openai_api_key": openai_api_key
    }

def test_pgvector_connection_and_similarity(db_config):
    # Crear documentos de prueba
    documents = [
        Document(page_content="LangChain es una librería para construir apps LLM", metadata={"source": "test"}),
        Document(page_content="PGVector permite búsquedas vectoriales en PostgreSQL", metadata={"source": "test"}),
    ]

    # Crear la base vectorial en PGVector
    vectorstore = PGVector.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        connection_string=db_config["connection_string"],
        collection_name="test_pgvector_connection"
    )

    # Hacer una búsqueda de similitud
    results = vectorstore.similarity_search("vectoriales en PostgreSQL", k=1)

    # Comprobar resultados
    assert len(results) == 1
    assert "PGVector" in results[0].page_content or "vectoriales" in results[0].page_content
