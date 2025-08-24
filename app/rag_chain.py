''' # app/rag_chain.py

from typing import TypedDict
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from app.retriever import get_retriever


# Define the expected input structure
class RagInput(TypedDict):
    question: str


# Prompt to expand vague questions
expand_question_prompt = ChatPromptTemplate.from_template(
    "Rephrase the following user question to be more explicit and complete:\n\n{question}"
)

# Prompt to generate the final answer
custom_rag_prompt = ChatPromptTemplate.from_template(
    """You are an expert assistant. Use the provided context to answer the user's question.
If the answer is not in the context, reply: "No information found in the uploaded documents."

Context:
{context}

Question:
{question}
"""
)


# Function to format retrieved documents
def format_docs(docs: list[Document], max_tokens: int = 3000) -> str:
    output = []
    current_tokens = 0

    for doc in docs:
        tokens = len(doc.page_content.split())
        if current_tokens + tokens > max_tokens:
            break
        output.append(doc.page_content)
        current_tokens += tokens

    return "\n\n".join(output)


# Create the full RAG chain
def create_rag_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = get_retriever()

    expand_chain = expand_question_prompt | llm | StrOutputParser()

    # Helper to merge expanded_question and question
    def enrich_input(input: dict) -> dict:
        return {
            "question": input["question"],
            "context_input": input["expanded_question"]
        }

    # ✅ Cadena de ejecución correctamente encadenada con `|`
    rag_chain = (
        RunnableMap({
            "question": itemgetter("question"),
            "expanded_question": itemgetter("question") | expand_chain
        }).with_config(run_name="🔍 Expand Question")
        |
        RunnableLambda(enrich_input).with_config(run_name="🔀 Enrich Input")
        |
        RunnableMap({
            "question": itemgetter("question"),
            "context": itemgetter("context_input")
            | retriever
            | RunnableLambda(
                lambda docs: format_docs(docs) if docs else "No information found in the uploaded documents."
            )
        }).with_config(run_name="📚 Retrieve & Format Context")
        |
        custom_rag_prompt.with_config(run_name="📝 Prompt")
        |
        llm.with_config(run_name="🤖 LLM")
        |
        StrOutputParser().with_config(run_name="🧾 Parse Output")
    )

    return rag_chain.with_types(input_type=RagInput)

'''
# app/rag_chain.py

import os
from typing import TypedDict
from pathlib import Path
from operator import itemgetter
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from app.retriever import get_retriever

# Carga de variables de entorno con override
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

# Estructura de input esperada por LangServe
class RagInput(TypedDict):
    question: str

# Prompt en inglés
rag_prompt = ChatPromptTemplate.from_template(
    """You are an expert assistant. Use the following context to answer the user's question. 
If the answer is not contained in the context, reply: "No information found in the uploaded documents."

Context:
{context}

Question:
{question}
"""
)

# ✅ NUEVO: formateador sin límite artificial de tokens
def format_docs(docs: list[Document]) -> str:
    combined = "\n\n".join(doc.page_content for doc in docs)
    print("📚 CONTEXTO RECUPERADO:\n", combined[:1000], "...\n")  # Mostrar solo los primeros 1000 caracteres
    return combined


# Cadena RAG compatible con LangServe
def create_rag_chain():
    llm = ChatOpenAI(
        model="gpt-4-1106-preview",
        temperature=0,
        streaming=True,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    retriever = get_retriever()

    rag_chain = (
        RunnableMap({
            "question": itemgetter("question"),
            "context": itemgetter("question") | retriever | RunnableLambda(format_docs)
        })
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.with_types(input_type=RagInput)


