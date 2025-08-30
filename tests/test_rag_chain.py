import os
import pytest
import psycopg
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from app.rag_chain import create_rag_chain

from dotenv import load_dotenv
load_dotenv()


# Puerto correcto del contenedor pgvector
CONNECTION_STRING = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5433/database164")
COLLECTION_NAME = "rag_collection"

@pytest.fixture(scope="module")
def retriever():
    vectorstore = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=OpenAIEmbeddings(model=os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")),
    )
    return vectorstore.as_retriever()

@pytest.fixture(scope="module")
def llm():
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

def test_pgvector_extension_installed():
    with psycopg.connect("dbname=database164 user=postgres password=postgres host=localhost port=5433") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            result = cur.fetchone()
            assert result is not None, "❌ La extensión pgvector no está instalada correctamente."

def test_embeddings_table_not_empty():
    with psycopg.connect("dbname=database164 user=postgres password=postgres host=localhost port=5433") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding;")
            count = cur.fetchone()[0]
            assert count > 0, f"❌ No hay ningún embedding almacenado en la tabla 'langchain_pg_embedding'."

def test_rag_chain_response(retriever, llm):
    rag_chain = create_rag_chain(retriever, llm)
    question = "¿Cómo murió Kennedy?"
    result = rag_chain.invoke({"question": question})
    # La cadena devuelve un dict con 'answer' y 'sources'
    assert isinstance(result, dict), "❌ La salida de la cadena debe ser un dict."
    assert "answer" in result, "❌ Falta el campo 'answer' en la salida."
    assert isinstance(result["answer"], str), "❌ 'answer' debe ser una cadena."
    assert "sources" in result, "❌ Falta el campo 'sources' en la salida."
    assert isinstance(result["sources"], list), "❌ 'sources' debe ser una lista."

