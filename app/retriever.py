# retriever.py

from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from psycopg import connect
import os
from dotenv import load_dotenv

load_dotenv()

connection = connect(
    host=os.environ["DATABASE_HOST"],
    dbname=os.environ["DATABASE_NAME"],
    user=os.environ["DATABASE_USER"],
    password=os.environ["DATABASE_PASSWORD"],
    port=int(os.environ["DATABASE_PORT"]),
)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = PGVector(
    collection_name="rag_app_collection",
    connection=connection,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()
