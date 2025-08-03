import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Configuración de LangSmith desde el entorno
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "kennedy-app-v1")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from app.rag_chain import create_rag_chain

app = FastAPI()

# ✅ Middleware para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # origen del frontend
    allow_credentials=True,
    allow_methods=["*"],                      # permite cualquier tipo de petición: GET, POST...
    allow_headers=["*"],                      # permite cualquier cabecera
)

rag_chain = create_rag_chain()

add_routes(
    app,
    rag_chain,
    path="/rag"
)
