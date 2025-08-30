# app/rag_chain.py
from __future__ import annotations

import os
from operator import itemgetter
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableMap
from langchain_openai import ChatOpenAI

from app.retriever import get_retriever


# 1) Tipo de entrada para LangServe / tu API
class RagInput(TypedDict):
    question: str


# 2) Prompt central (¡nombre consistente!)
rag_prompt = ChatPromptTemplate.from_template(
    """Eres un asistente que responde preguntas basándose únicamente en el CONTEXTO dado.
Reglas:
1) Usa exclusivamente el CONTEXTO proporcionado.
2) Si hay información suficiente en el CONTEXTO para responder, contesta en español de forma directa y concisa (1–3 frases). No hace falta citar literalmente: puedes resumir fielmente.
3) Si no hay información suficiente en el CONTEXTO para responder, escribe EXACTAMENTE: No lo sé
4) No inventes ni añadas notas, advertencias, fuentes ni epílogos.

CONTEXTO:
{context}

PREGUNTA:
{question}
"""
)


# 3) Formateo de documentos (límite aproximado por nº palabras)
def format_docs(docs: list[Document], max_words: int | None = None) -> str:
    if max_words is None:
        try:
            max_words = int(os.getenv("MAX_CONTEXT_WORDS", "6000"))
        except ValueError:
            max_words = 6000
    chunks, used = [], 0
    for d in docs:
        words = len(d.page_content.split())
        if used + words > max_words:
            break
        chunks.append(d.page_content)
        used += words
    return "\n\n".join(chunks)


# 4) Empaquetado del contexto manteniendo los docs crudos para 'sources'
def _pack_with_context(x: dict) -> dict:
    raw_docs: list[Document] = x["raw_docs"]
    return {
        "question": x["question"],
        "context": format_docs(raw_docs),
        "raw_docs": raw_docs,
    }


# 5) Conversión documento -> info de fuente para la UI
def _doc_to_source_info(doc: Document) -> dict:
    md = doc.metadata or {}
    return {
        "title": md.get("title") or md.get("file_name") or md.get("source") or "Documento",
        "page": md.get("page") or md.get("page_number"),
        "source": md.get("source") or md.get("path") or md.get("file_path") or md.get("url"),
        "metadata": md,  # por si quieres mostrar más campos
    }


# 6) Cadena RAG compatible con LangServe (answer + sources)
def create_rag_chain(retriever=None, llm=None):
    # LLM con streaming (inyectable para tests; configurable por env)
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
            streaming=True,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # Retriever inyectable; si no se pasa, usamos el por defecto
    if retriever is None:
        retriever = get_retriever()

    # Paso A: en paralelo → recuperar docs y pasar la pregunta
    initial = RunnableParallel(
        raw_docs=(itemgetter("question") | retriever),
        question=itemgetter("question"),
    )

    # Paso B: añadir 'context' (texto) y conservar 'raw_docs'
    with_context = initial | RunnableLambda(_pack_with_context)

    # Paso C: en paralelo → generar respuesta y preparar fuentes
    rag_chain = (
        with_context
        | RunnableParallel(
            # Respuesta del LLM (prompt → llm → parser)
            answer=(
                RunnableMap({"question": itemgetter("question"), "context": itemgetter("context")})
                | rag_prompt
                | llm
                | StrOutputParser()
            ),
            # Fuentes para la UI (no dependen del LLM)
            sources=(itemgetter("raw_docs") | RunnableLambda(lambda docs: [_doc_to_source_info(d) for d in docs])),
        )
    )

    # Para LangServe: tipado de entrada
    return rag_chain.with_types(input_type=RagInput)
