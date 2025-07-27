# app/rag_chain.py

from typing import TypedDict
from operator import itemgetter

from langchain_core.runnables import RunnableMap, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document


# 1️⃣ Estructura del input esperada
class QuestionInput(TypedDict):
    context: str
    question: str


# 2️⃣ Prompt personalizado
custom_rag_prompt = ChatPromptTemplate.from_template(
    """Answer the question using the provided context. If you don't know the answer, simply say you don't know.

Context:
{context}

Question:
{question}
"""
)


# 3️⃣ Formateo de los documentos: se usa en el retriever
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# 4️⃣ Constructor de la cadena RAG
def create_rag_chain(retriever, llm: BaseChatModel):
    """
    Construye una RAG chain completa que recibe una pregunta, recupera contexto con el retriever
    y genera la respuesta con el modelo de lenguaje (LLM).
    """
    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "question": itemgetter("question")
        }
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain