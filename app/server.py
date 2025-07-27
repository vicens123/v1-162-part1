from fastapi import FastAPI
from langserve import add_routes
from app.rag_chain import create_rag_chain
from app.retriever import get_retriever
from langchain_openai import ChatOpenAI

app = FastAPI()

# 1. Crear el retriever (desde tu lógica)
retriever = get_retriever()

# 2. Crear el LLM (modelo de lenguaje de OpenAI o el que uses)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 3. Crear la cadena RAG
rag_chain = create_rag_chain(retriever, llm)

# 4. Añadir la cadena al servidor como endpoint
add_routes(
    app,
    rag_chain,
    path="/rag"
)
