# app/rag_chain.py

from operator import itemgetter
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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

# 3️⃣ Crear la cadena RAG básica como en el ejemplo original
def create_rag_chain():
    retriever = get_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.with_types(input_type=RagInput)
