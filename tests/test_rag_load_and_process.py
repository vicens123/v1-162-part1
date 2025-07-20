import sys
import os

# Añadir ruta absoluta al módulo
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rag_load_and_process"))
sys.path.insert(0, module_path)

from rag_load_and_process import load_and_process_pdfs  # Nombre correcto del archivo

def test_load_and_process_pdfs():
    try:
        load_and_process_pdfs()
        print("✅ TEST PASADO: El procesamiento se ejecutó correctamente.")
    except Exception as e:
        print(f"❌ TEST FALLIDO: {e}")

if __name__ == "__main__":
    test_load_and_process_pdfs()
