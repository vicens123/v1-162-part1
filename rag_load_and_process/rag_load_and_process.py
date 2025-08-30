import os
import hashlib
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import psycopg

# Cargar variables desde .env en la raíz del proyecto
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
DELETE_COLLECTION = os.getenv("DELETE_COLLECTION", "true").lower() in {"1","true","yes","y"}


def _psycopg_conn_str(conn: str) -> str:
    return conn.replace("postgresql+psycopg2://", "postgresql://")


def _delete_by_doc_id(connection_string: str, collection_name: str, doc_id: str) -> int:
    dsn = _psycopg_conn_str(connection_string)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            row = cur.fetchone()
            if not row:
                return 0
            coll_id = row[0]
            cur.execute(
                "DELETE FROM langchain_pg_embedding WHERE collection_id = %s AND cmetadata->>'doc_id' = %s",
                (coll_id, doc_id),
            )
            deleted = cur.rowcount or 0
    return deleted


def _delete_collection(connection_string: str, collection_name: str) -> int:
    dsn = _psycopg_conn_str(connection_string)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            row = cur.fetchone()
            if not row:
                return 0
            coll_id = row[0]
            cur.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = %s", (coll_id,))
            deleted = cur.rowcount or 0
            cur.execute("DELETE FROM langchain_pg_collection WHERE id = %s", (coll_id,))
    return deleted


def load_and_process_pdfs(mode: str = "update") -> Dict[str, Any]:
    """Ingest PDFs from ./pdf-documents with per-file dedup.

    mode:
      - 'full': delete entire collection, then ingest all
      - 'update': delete by doc_id per file, then add
      - 'append': just add
    Returns a summary dict.
    """
    assert mode in {"full", "update", "append"}
    pdf_dir = Path("./pdf-documents").resolve()
    files = sorted(pdf_dir.glob("**/*.pdf"))
    print(f"📄 PDFs encontrados: {len(files)} archivos en {pdf_dir}")

    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    text_splitter = SemanticChunker(embeddings=embeddings)

    vectorstore = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=DATABASE_URL,
        embedding_function=embeddings,
    )

    summary = {"mode": mode, "files": len(files), "deleted": 0, "added_chunks": 0, "processed": 0}

    if mode == "full" and files:
        deleted = _delete_collection(DATABASE_URL, COLLECTION_NAME)
        summary["deleted"] += deleted

    for file_path in files:
        # Compute stable doc_id based on file content
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        doc_id = h.hexdigest()

        # Load one PDF
        loader = UnstructuredPDFLoader(str(file_path))
        docs = loader.load()

        # Enrich metadata
        for d in docs:
            md = d.metadata or {}
            md["doc_id"] = doc_id
            md["source"] = str(file_path)
            md["source_filename"] = Path(file_path).name
            d.metadata = md

        chunks = text_splitter.split_documents(docs)

        if mode == "update":
            summary["deleted"] += _delete_by_doc_id(DATABASE_URL, COLLECTION_NAME, doc_id)

        if chunks:
            vectorstore.add_documents(chunks)
            summary["added_chunks"] += len(chunks)
        summary["processed"] += 1

    print("✅ Ingesta finalizada:", summary)
    return summary


if __name__ == "__main__":
    result = load_and_process_pdfs("update")
    print("✅ Proceso completado.")
    print(f"📦 Resumen: {result}")
