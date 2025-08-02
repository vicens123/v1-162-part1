from operator import itemgetter

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.retriever import get_retriever
from langchain_core.pydantic_v1 import BaseModel, Field

class RagInput(BaseModel):
    question: str = Field(..., description="The question you want to answer.")

# 🧠 PROMPT optimizado para obtener respuesta incluso con contexto débil
custom_rag_prompt = ChatPromptTemplate.from_template(
    """You are an expert assistant. Use the provided context to answer the question. 
If the context does not contain the answer, try to infer a helpful response anyway. Be precise and informative.

Context:
{context}

Question:
{question}
"""
)

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain():
    retriever = get_retriever()
    
    # 🔥 temperatura mayor → menos conservador
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "question": itemgetter("question"),
        }
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.with_types(input_type=RagInput)

