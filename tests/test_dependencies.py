def test_import_dependencies():
    try:
        import os
        from dotenv import load_dotenv
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS

        print("✅ Todos los módulos se importaron correctamente.")
    except ImportError as e:
        print(f"❌ ERROR: No se pudo importar un módulo. {e}")
    except Exception as e:
        print(f"⚠️ Otro error ocurrió: {e}")

if __name__ == "__main__":
    test_import_dependencies()
