import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(override=True)

# Configuración de LangSmith solo si las variables existen
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")

langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

langsmith_project = os.getenv("LANGCHAIN_PROJECT")
if langsmith_project:
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project


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
