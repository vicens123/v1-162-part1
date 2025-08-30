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
from fastapi import UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langserve import add_routes
from app.rag_chain import create_rag_chain

app = FastAPI()

# Middleware CORS para permitir llamadas desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = create_rag_chain()

add_routes(
    app,
    rag_chain,
    path="/rag"
)

# Servir ficheros estáticos (PDFs)
app.mount("/rag/static", StaticFiles(directory="./pdf-documents"), name="static")


# Endpoint para subir PDFs y guardarlos en ./pdf-documents
UPLOAD_DIR = Path("./pdf-documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    saved: list[str] = []
    for file in files:
        name = Path(file.filename).name
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Solo PDFs permitidos: {name}")
        dest = UPLOAD_DIR / name
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(str(dest))

    return {"saved": saved, "count": len(saved)}
