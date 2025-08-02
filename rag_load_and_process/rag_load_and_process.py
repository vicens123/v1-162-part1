import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Cargar variables desde .env en la raíz del proyecto
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)


def load_and_process_pdfs():
    loader = DirectoryLoader(
        os.path.abspath("./pdf-documents"),
        glob="**/*.pdf",
        use_multithreading=True,
        show_progress=True,
        max_concurrency=50,
        loader_cls=UnstructuredPDFLoader,
    )
    docs = loader.load()
    print(f"📄 PDFs cargados: {len(docs)} documentos")

    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    text_splitter = SemanticChunker(embeddings=embeddings)

    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Fragmentos generados: {len(chunks)} chunks")

    # 🔍 Mostrar los primeros 5 chunks (máximo 300 caracteres cada uno)
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n🔍 Chunk {i+1}:\n{chunk.page_content[:300]}")

    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="collection164",
        connection_string=os.getenv("DATABASE_URL"),
        pre_delete_collection=True,
    )
    print("✅ Chunks guardados en PGVector.")

    return vectorstore

if __name__ == "__main__":
    vectorstore = load_and_process_pdfs()
    print("✅ Proceso completado.")
    print(f"📦 Vectorstore creado: {vectorstore}")

