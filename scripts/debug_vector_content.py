from app.retriever import get_retriever

retriever = get_retriever()
query = "¿quien fue jordi pujol?"

docs = retriever.get_relevant_documents(query)

print(f"🔍 Se han recuperado {len(docs)} documentos relevantes:\n")

for i, doc in enumerate(docs[:5]):
    print(f"\n🔹 Documento {i+1}:\n{doc.page_content[:700]}")