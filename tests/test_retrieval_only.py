# test_retrieval_only.py

from app.retriever import get_retriever

# Pregunta de prueba que sí debería recuperar chunks
query = "John F. Kennedy murió en Dallas"


# Recuperar chunks desde la base de datos
retriever = get_retriever()
docs = retriever.invoke(query)  


# Mostrar los resultados
if docs:
    for i, doc in enumerate(docs, 1):
        print(f"\n🔎 Chunk {i}:\n{doc.page_content}\n")
else:
    print("❌ No se recuperaron documentos relevantes.")
