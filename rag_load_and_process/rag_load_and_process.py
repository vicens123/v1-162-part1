# rag_load_and_process/rag_module.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

def load_and_process_pdfs():
    load_dotenv()

    loader = DirectoryLoader(
        os.path.abspath("./pdf-documents"),  # corregido para entorno relativo a ejecución
        glob="**/*.pdf",
        use_multithreading=True,
        show_progress=True,
        max_concurrency=50,
        loader_cls=UnstructuredPDFLoader,
    )
    docs = loader.load()

    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    text_splitter = SemanticChunker(embeddings=embeddings)

    # ⚠️ Este paso depende del formato de `docs`
    flattened_docs = docs  # Ya es una lista de Document
    chunks = text_splitter.split_documents(flattened_docs)

    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="collection164",
        connection_string="postgresql+psycopg://postgres@localhost:5432/database164",
        pre_delete_collection=True,
    )

    return vectorstore
