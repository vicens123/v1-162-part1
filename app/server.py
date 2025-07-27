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
