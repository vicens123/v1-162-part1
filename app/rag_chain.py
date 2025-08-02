from operator import itemgetter

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

from app.retriever import get_retriever

# ✅ Definición del input para LangServe
class RagInput(BaseModel):
    question: str = Field(..., description="The question you want to answer.")

# 🧠 Prompt personalizado con aviso si no hay contexto
custom_rag_prompt = ChatPromptTemplate.from_template(
    """You are an expert assistant. Use the provided context to answer the question. 
If the context does not contain the answer, just say: "No se ha encontrado información en los documentos cargados."

Context:
{context}

Question:
{question}
"""
)

# 🔧 Formateo de los documentos recuperados
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# 🧩 Paso para verificar si el contexto está vacío
def check_empty_context(inputs: dict) -> dict:
    if not inputs["context"].strip():
        return {
            "context": "No se ha encontrado información en los documentos cargados.",
            "question": inputs["question"],
        }
    return inputs

# 🚀 Constructor de la cadena RAG con control de contexto vacío
def create_rag_chain():
    retriever = get_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "question": itemgetter("question"),
        }
        | RunnableLambda(check_empty_context)  # 👈 añade el paso intermedio
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.with_types(input_type=RagInput)
