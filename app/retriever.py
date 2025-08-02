from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(override=True)

def get_retriever():
    # 🔐 Conexión a PostgreSQL con pgvector
    connection_string = (
        f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:"
        f"{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}:"
        f"{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )

    # 🧠 Embeddings semánticos con modelo robusto
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # 🧱 Vector store usando pgvector
    vectorstore = PGVector(
        collection_name="collection164",
        connection_string=connection_string,
        embedding_function=embeddings,
    )

    # 🔍 Configuración del retriever para mayor tolerancia
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 8,                  # aumenta el nº de documentos similares
            "score_threshold": 0.1   # baja el umbral para permitir más flexibilidad semántica
        }
    )

    return retriever
