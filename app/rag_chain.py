# app/rag_chain.py
from __future__ import annotations

import os
from operator import itemgetter
from typing import TypedDict, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableMap
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever
from langchain.memory import ConversationBufferWindowMemory

from app.retriever import get_retriever
from app.chat_history import chat_history_manager


# 1) Tipo de entrada para LangServe / tu API
class RagInput(TypedDict):
    question: str
    session_id: Optional[str]  # Nuevo campo para chat history


# 2) Prompt central (¡nombre consistente!)
rag_prompt = ChatPromptTemplate.from_template(
    """Eres un asistente que responde preguntas basándose únicamente en el CONTEXTO dado.
Reglas:
1) Usa exclusivamente el CONTEXTO proporcionado.
2) Si hay información suficiente en el CONTEXTO para responder, contesta en español de forma directa y concisa (1–3 frases). No hace falta citar literalmente: puedes resumir fielmente.
3) Si no hay información suficiente en el CONTEXTO para responder, escribe EXACTAMENTE: No lo sé
4) No inventes ni añadas notas, advertencias, fuentes ni epílogos.
5) Si hay HISTORIAL de conversación, úsalo para dar contexto pero responde basándote en el CONTEXTO actual.

HISTORIAL de conversación:
{chat_history}

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
    """Empaquetar contexto con documentos y historial de chat"""
    raw_docs: list[Document] = x["raw_docs"]
    return {
        "question": x["question"],
        "context": format_docs(raw_docs),
        "raw_docs": raw_docs,
        "chat_history": x.get("chat_history", ""),
    }


# 5) Obtener historial de chat si existe session_id
def _get_chat_history(session_id: Optional[str]) -> str:
    if not session_id:
        return ""
    
    try:
        history = chat_history_manager.get_session_history(session_id)
        messages = history.messages
        if not messages:
            return ""
        
        # Formatear historial para el prompt
        formatted_history = []
        for msg in messages[-10:]:  # Últimos 10 mensajes
            role = "Usuario" if msg.type == "human" else "Asistente"
            formatted_history.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted_history)
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return ""


# 6) Conversión documento -> info de fuente para la UI
def _doc_to_source_info(doc: Document) -> dict:
    md = doc.metadata or {}
    return {
        "title": md.get("title") or md.get("file_name") or md.get("source") or "Documento",
        "page": md.get("page") or md.get("page_number"),
        "source": md.get("source") or md.get("path") or md.get("file_path") or md.get("url"),
        "metadata": md,  # por si quieres mostrar más campos
    }


# Fallback retriever para inicialización segura del módulo (no consulta DB)
class _NoopRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str, *, run_manager=None):
        return []
    async def aget_relevant_documents(self, query: str, *, run_manager=None):
        return []


# Instancia global exportada para tests; se redefinirá al construir la cadena real
multiquery: MultiQueryRetriever | None = None


# 7) Cadena RAG compatible con LangServe (answer + sources) + Chat History
def create_rag_chain(retriever=None, llm=None, session_id: Optional[str] = None):
    # LLM con streaming (inyectable para tests; configurable por env)
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
            streaming=True,
            callbacks=[],
        )
    
    # Obtener historial de chat si hay session_id
    chat_history = _get_chat_history(session_id)
    
    # Retriever (inyectable para tests)
    if retriever is None:
        retriever = get_retriever()
    
    # Función para guardar mensajes en el historial
    def _save_to_history(output):
        # Convertir a string si es un objeto
        if hasattr(output, 'content'):
            output_str = output.content
        elif isinstance(output, dict) and 'content' in output:
            output_str = output['content']
        else:
            output_str = str(output)
        
        if session_id:
            try:
                history = chat_history_manager.get_session_history(session_id)
                # Guardar respuesta del asistente
                history.add_ai_message(output_str)
            except Exception as e:
                print(f"Error saving to chat history: {e}")
        return output_str
    
    # Cadena principal con historial integrado
    chain = (
        RunnableLambda(lambda x: {
            "question": x["question"],
            "raw_docs": retriever.get_relevant_documents(x["question"]),
            "chat_history": chat_history,
        })
        | RunnableLambda(_pack_with_context)
        | rag_prompt
        | llm
        #| StrOutputParser()
    )
    
    # Cadena final con guardado de historial
    final_chain = chain | RunnableLambda(_save_to_history)
    
    return final_chain


# 8) Función para crear nueva sesión de chat
def create_chat_session(user_id: Optional[str] = None, title: Optional[str] = None) -> str:
    """Crear nueva sesión de chat y retornar session_id"""
    return chat_history_manager.create_session(user_id, title)


# 9) Función para obtener sesiones de usuario
def get_user_sessions(user_id: str) -> list[dict]:
    """Obtener todas las sesiones de chat de un usuario"""
    return chat_history_manager.get_user_sessions(user_id)


# 10) Función para eliminar sesión
def delete_chat_session(session_id: str) -> bool:
    """Eliminar sesión de chat y todos sus mensajes"""
    return chat_history_manager.delete_session(session_id)