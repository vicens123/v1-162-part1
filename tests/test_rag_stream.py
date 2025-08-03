# tests/test_rag_stream.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.server import app  # ✅ esta es la ruta correcta

pytestmark = pytest.mark.asyncio

async def test_rag_stream_response():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/rag/stream", json={"user_input": "¿Qué es LangChain?"})


    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data:" in response.text or "Processed" in response.text

