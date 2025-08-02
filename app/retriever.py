from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(override=True)

def get_retriever():
    connection_string = (
        f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:" 
        f"{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}:" 
        f"{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    vectorstore = PGVector(
        collection_name="rag_app_collection",
        connection_string=connection_string,
        embedding_function=embeddings,
    )

    # 🔧 más documentos + umbral de similitud más flexible
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6,  # número de documentos
            "score_threshold": 0.3  # permite más resultados, incluso menos relevantes
        }
    )

    return retriever

