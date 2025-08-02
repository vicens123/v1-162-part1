import os

# Configuración de LangSmith desde el entorno
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "kennedy-app-v1")


from fastapi import FastAPI
from langserve import add_routes
from app.rag_chain import create_rag_chain

app = FastAPI()

rag_chain = create_rag_chain()

add_routes(
    app,
    rag_chain,
    path="/rag"
)
