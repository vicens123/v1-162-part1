# server.py

from fastapi import FastAPI
from langserve import add_routes
from app.rag_chain import rag_chain

app = FastAPI()

# Añade tu cadena RAG como endpoint
add_routes(
    app,
    rag_chain,
    path="/rag"  # puedes cambiarlo si quieres
)

