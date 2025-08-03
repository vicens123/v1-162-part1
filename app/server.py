import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Generator
from langserve import add_routes
from app.rag_chain import create_rag_chain

load_dotenv(override=True)

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "kennedy-app-v1")

app = FastAPI()

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

@app.get("/rag")
async def rag_endpoint(request: Request):
    async def event_generator() -> Generator[str, None, None]:
        for i in range(1, 6):
            await asyncio.sleep(1)
            yield f"data: Message {i}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

