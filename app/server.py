"""
RAG Application Server with Chat History Integration

This module provides a FastAPI server that integrates RAG (Retrieval-Augmented Generation)
with chat history functionality, document management, and administrative tools.

Author: Vicenç Delgado
Version: 2.1
"""

import os
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langserve import add_routes

from app.rag_chain import create_rag_chain
from app.chat_history import chat_history_manager
from app.retriever import get_retriever
from rag_load_and_process.rag_load_and_process import (
    ensure_collection,
    load_and_process_pdfs,
)

# ============================================================================
# CONFIGURACIÓN DE ENTORNO
# ============================================================================

# Cargar variables de entorno
load_dotenv(override=True)

# Configuración de LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")

langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

langsmith_project = os.getenv("LANGCHAIN_PROJECT")
if langsmith_project:
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project

# ============================================================================
# INICIALIZACIÓN DE LA APLICACIÓN
# ============================================================================

app = FastAPI(
    title="RAG Application with Chat History",
    description="Advanced RAG system with document management and conversational AI",
    version="2.1.0",
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURACIÓN DE RAG CHAIN
# ============================================================================

rag_chain = create_rag_chain()

# Agregar rutas de LangServe
add_routes(
    app,
    rag_chain,
    path="/rag",
)

# ============================================================================
# CONFIGURACIÓN DE ARCHIVOS ESTÁTICOS
# ============================================================================

# Servir archivos estáticos (PDFs)
app.mount("/rag/static", StaticFiles(directory="./pdf-documents"), name="static")

# Directorio de uploads
UPLOAD_DIR = Path("./pdf-documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ENDPOINTS DE GESTIÓN DE DOCUMENTOS
# ============================================================================

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Subir PDFs y guardarlos en el directorio de documentos.
    
    Args:
        files: Lista de archivos PDF a subir
        
    Returns:
        Dict con archivos guardados y conteo
        
    Raises:
        HTTPException: Si no hay archivos o no son PDFs
    """
    if not files:
        raise HTTPException(
            status_code=400, 
            detail="No se enviaron archivos"
        )

    saved_files = []
    for file in files:
        filename = Path(file.filename).name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, 
                detail=f"Solo PDFs permitidos: {filename}"
            )
        
        destination = UPLOAD_DIR / filename
        with destination.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(str(destination))

    return {
        "saved": saved_files, 
        "count": len(saved_files)
    }


@app.post("/admin/ingest")
async def admin_ingest(mode: str = "update"):
    """Reingesta de PDFs con diferentes modos de operación."""
    try:
        # Debug: verificar colección
        print("DEBUG: Verificando colección...")
        collection = ensure_collection(
            os.getenv("DATABASE_URL"), 
            os.getenv("COLLECTION_NAME", "rag_collection")
        )
        print(f"DEBUG: Colección resultado: {collection}")
        
        # Debug: verificar que la colección esté lista
        if not collection.get("collection_exists"):
            print("DEBUG: Colección no existe, creando...")
        
        result = load_and_process_pdfs(mode=mode)
        print(f"DEBUG: Resultado ingesta: {result}")
        
        return {
            "status": "ok", 
            "collection": collection, 
            **result
        }
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error en ingesta: {e}"
        )

# ============================================================================
# ENDPOINTS DE MONITOREO Y DEBUG
# ============================================================================

@app.get("/healthz")
async def healthz():
    """
    Healthcheck de base de datos y colección.
    
    Returns:
        Dict con estado del sistema
    """
    try:
        collection = ensure_collection(
            os.getenv("DATABASE_URL"), 
            os.getenv("COLLECTION_NAME", "rag_collection")
        )
        return {
            "status": "ok" if collection.get("db_ok") else "degraded", 
            **collection
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Healthcheck error: {e}"
        )


@app.get("/rag/debug/retrieve")
async def debug_retrieve(question: str):
    """
    Devuelve documentos recuperados y tamaño de contexto para depuración.
    
    Args:
        question: Pregunta para recuperar documentos
        
    Returns:
        Dict con documentos y métricas de contexto
    """
    retriever = get_retriever()
    documents = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in documents)
    
    return {
        "question": question,
        "docs": [
            {
                "content_preview": doc.page_content[:400],
                "metadata": doc.metadata,
            }
            for doc in documents
        ],
        "context_words": len(context.split()),
        "count": len(documents),
    }


# ============================================================================
# ENDPOINTS DE CHAT HISTORY
# ============================================================================

@app.post("/chat/session")
def create_session(user_id: str = None, title: str = None):
    """Crear nueva sesión de chat"""
    try:
        # Debug: imprimir parámetros
        print(f"DEBUG: user_id={user_id}, title={title}")
        
        # Debug: verificar que chat_history_manager existe
        print(f"DEBUG: chat_history_manager type: {type(chat_history_manager)}")
        
        session_id = chat_history_manager.create_session(user_id, title)
        print(f"DEBUG: session_id created: {session_id}")
        
        return {"session_id": session_id, "status": "created"}
    except Exception as e:
        # Debug: imprimir el error completo
        import traceback
        print(f"ERROR COMPLETO: {e}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error creating session: {e}")


@app.get("/chat/sessions/{user_id}")
def get_user_chat_sessions(user_id: str):
    """
    Obtener todas las sesiones de chat de un usuario.
    
    Args:
        user_id: Identificador del usuario
        
    Returns:
        Dict con lista de sesiones
        
    Raises:
        HTTPException: Si hay errores al obtener las sesiones
    """
    try:
        sessions = chat_history_manager.get_user_sessions(user_id)
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting sessions: {e}"
        )


@app.delete("/chat/session/{session_id}")
def delete_chat_session_endpoint(session_id: str):
    """
    Eliminar sesión de chat y todos sus mensajes.
    
    Args:
        session_id: Identificador de la sesión
        
    Returns:
        Dict con estado de la operación
        
    Raises:
        HTTPException: Si la sesión no existe o hay errores
    """
    try:
        success = chat_history_manager.delete_session(session_id)
        if success:
            return {
                "status": "deleted", 
                "session_id": session_id
            }
        else:
            raise HTTPException(
                status_code=404, 
                detail="Session not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error deleting session: {e}"
        )


@app.get("/chat/session/{session_id}/history")
def get_chat_session_history(session_id: str):
    """
    Obtener historial completo de una sesión de chat.
    
    Args:
        session_id: Identificador de la sesión
        
    Returns:
        Dict con historial de mensajes
        
    Raises:
        HTTPException: Si hay errores al obtener el historial
    """
    try:
        history = chat_history_manager.get_session_history(session_id)
        messages = history.messages
        
        formatted_messages = []
        for message in messages:
            formatted_messages.append({
                "role": "user" if message.type == "human" else "assistant",
                "content": message.content,
                "timestamp": message.additional_kwargs.get("timestamp", None)
            })
        
        return {
            "session_id": session_id,
            "messages": formatted_messages,
            "count": len(formatted_messages)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting session history: {e}"
        )


@app.get("/chat/health")
def chat_health():
    """
    Healthcheck específico para funcionalidades de chat.
    
    Returns:
        Dict con estado de las funcionalidades de chat
    """
    try:
        # Verificar que las tablas de chat existen
        # Intentar crear una sesión de prueba
        test_session_id = chat_history_manager.create_session("test_user", "test_session")
        # Eliminar la sesión de prueba
        chat_history_manager.delete_session(test_session_id)
        
        return {
            "status": "healthy",
            "chat_history": "operational",
            "database": "connected"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "chat_history": "error",
            "database": "error",
            "error": str(e)
        }


# ============================================================================
# CONFIGURACIÓN DE EVENTOS DE LA APLICACIÓN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Evento ejecutado al iniciar la aplicación."""
    print("🚀 RAG Application with Chat History starting up...")
    print(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"🔍 RAG Chain initialized: {rag_chain is not None}")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento ejecutado al cerrar la aplicación."""
    print("🛑 RAG Application shutting down...")


# ============================================================================
# INFORMACIÓN DE LA APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )