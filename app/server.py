import os
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from langserve import add_routes
from app.rag_chain import create_rag_chain

# 🌍 Cargar variables de entorno
load_dotenv(override=True)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "kennedy-app-v1")

# 🚀 Inicializar FastAPI
app = FastAPI()

# 🔐 CORS para permitir conexión desde el frontend (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Ajusta esto si usas otro origen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Modelo de entrada del usuario
class RagRequest(BaseModel):
    user_input: str

# 🔗 Cargar la cadena RAG
rag_chain = create_rag_chain()

# 🧪 Endpoint para el playground de LangServe
add_routes(app, rag_chain, path="/rag")

# 📡 Endpoint separado para el streaming SSE desde el frontend
@app.post("/rag/stream")
async def rag_stream_endpoint(request: RagRequest):
    async def event_generator() -> AsyncGenerator[dict, None]:
        response = await rag_chain.ainvoke({"question": request.user_input})
        for line in response.strip().split("\n"):
            yield {"data": line}
            await asyncio.sleep(0.1)

    return EventSourceResponse(event_generator())

