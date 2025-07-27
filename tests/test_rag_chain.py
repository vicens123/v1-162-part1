import pytest
import psycopg
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.rag_chain import create_rag_chain
from langchain_openai import ChatOpenAI

CONNECTION_STRING = "postgresql+psycopg://postgres:postgres@localhost:5432/database164"
COLLECTION_NAME = "rag_collection"

@pytest.fixture(scope="module")
def retriever():
    vectorstore = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    return vectorstore.as_retriever()

@pytest.fixture(scope="module")
def llm():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def test_pgvector_extension_installed():
    with psycopg.connect("dbname=database164 user=postgres password=postgres host=localhost port=5432") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            result = cur.fetchone()
            assert result is not None, "❌ La extensión pgvector no está instalada correctamente."

def test_embeddings_table_not_empty():
    with psycopg.connect("dbname=database164 user=postgres password=postgres host=localhost port=5432") as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_name = %s;", (COLLECTION_NAME,))
            count = cur.fetchone()[0]
            assert count > 0, f"❌ No hay embeddings almacenados en la colección '{COLLECTION_NAME}'."

def test_rag_chain_response(retriever, llm):
    rag_chain = create_rag_chain(retriever, llm)
    question = "¿Qué dice el documento sobre las condiciones de pago?"
    response = rag_chain.invoke({"question": question})
    assert isinstance(response, str), "❌ La respuesta no es una cadena de texto."
    assert len(response.strip()) > 0, "❌ La respuesta está vacía."
