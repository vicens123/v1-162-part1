import os
import hashlib
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import psycopg
import uuid

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
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            row = cur.fetchone()
            if not row:
                return 0
            coll_uuid = row[0]
            cur.execute(
                "DELETE FROM langchain_pg_embedding WHERE collection_id = %s AND cmetadata->>'doc_id' = %s",
                (coll_uuid, doc_id),
            )
            deleted = cur.rowcount or 0
    return deleted


def _delete_collection(connection_string: str, collection_name: str) -> int:
    dsn = _psycopg_conn_str(connection_string)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            row = cur.fetchone()
            if not row:
                return 0
            coll_uuid = row[0]
            cur.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = %s", (coll_uuid,))
            deleted = cur.rowcount or 0
            cur.execute("DELETE FROM langchain_pg_collection WHERE uuid = %s", (coll_uuid,))
    return deleted


def _table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1
              FROM information_schema.tables
              WHERE table_name = %s
            )
            """,
            (table_name,),
        )
        return bool(cur.fetchone()[0])


def ensure_collection(connection_string: str, collection_name: str) -> Dict[str, Any]:
    """Ensure the collection row exists; do not attempt to create tables.
    Returns a dict with health info and whether a collection row was created.
    """
    info: Dict[str, Any] = {"db_ok": False, "vector_ext": False, "collections_table": False, "embeddings_table": False, "collection_exists": False, "collection_created": False}
    dsn = _psycopg_conn_str(connection_string)
    try:
        with psycopg.connect(dsn) as conn:
            info["db_ok"] = True
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                info["vector_ext"] = bool(cur.fetchone()[0])

            info["collections_table"] = _table_exists(conn, "langchain_pg_collection")
            info["embeddings_table"] = _table_exists(conn, "langchain_pg_embedding")

            if info["collections_table"]:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM langchain_pg_collection WHERE name = %s", (collection_name,))
                    row = cur.fetchone()
                    if row:
                        info["collection_exists"] = True
                    else:
                        # Insert collection row with a generated uuid
                        coll_uuid = str(uuid.uuid4())
                        cur.execute(
                            "INSERT INTO langchain_pg_collection (uuid, name, cmetadata) VALUES (%s, %s, '{}'::jsonb)",
                            (coll_uuid, collection_name),
                        )
                        info["collection_created"] = True
    except Exception as e:
        info["error"] = str(e)
    return info


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

    summary: Dict[str, Any] = {"mode": mode, "files": len(files), "deleted": 0, "added_chunks": 0, "processed": 0, "errors": []}

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

        # Load one PDF with Unstructured; fallback to PyPDFLoader on failure
        try:
            loader = UnstructuredPDFLoader(str(file_path))
            docs = loader.load()
        except Exception as e_un:
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            except Exception as e_py:
                summary["errors"].append({
                    "file": str(file_path),
                    "error": f"Unstructured failed: {e_un}; PyPDF failed: {e_py}",
                })
                continue

        # Enrich metadata
        for d in docs:
            md = d.metadata or {}
            md["doc_id"] = doc_id
            md["source"] = str(file_path)
            md["source_filename"] = Path(file_path).name
            d.metadata = md

        try:
            chunks = text_splitter.split_documents(docs)
        except Exception as e_chunk:
            summary["errors"].append({
                "file": str(file_path),
                "error": f"Chunking failed: {e_chunk}",
            })
            continue

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
