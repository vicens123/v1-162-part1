# app/rag_chain.py

from operator import itemgetter
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from app.retriever import get_retriever

# 1️⃣ Input tipo TypedDict como el del instructor
class RagInput(TypedDict):
    question: str

# 2️⃣ Prompt simple y claro
custom_rag_prompt = ChatPromptTemplate.from_template(
    """You are an expert assistant. Use the provided context to answer the question.
If the context does not contain the answer, just say: "No information found in the uploaded documents."

Context:
{context}

Question:
{question}
"""
)

# 3️⃣ Función para formatear los documentos y limitar tokens (aproximado)
def format_docs(docs: list[Document], max_tokens: int = 3000) -> str:
    output = []
    current_tokens = 0

    for doc in docs:
        tokens = len(doc.page_content.split())  # Aproximación rápida
        if current_tokens + tokens > max_tokens:
            break
        output.append(doc.page_content)
        current_tokens += tokens

    return "\n\n".join(output)

# 4️⃣ Crear la cadena RAG con formateo de contexto limitado
def create_rag_chain():
    retriever = get_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    rag_chain = (
        {
            "context": itemgetter("question")
            | retriever
            | RunnableLambda(lambda docs: format_docs(docs, max_tokens=3000)),
            "question": itemgetter("question"),
        }
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.with_types(input_type=RagInput)

