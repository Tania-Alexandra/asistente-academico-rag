import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
# Carga variables de entorno (.env)
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
CHROMA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "chroma_db"))

def ingest_documents():
    print("Cargando documentos desde data/...")

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"No se encontró el directorio de datos: {DATA_DIR}")

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No se encontraron archivos PDF en {DATA_DIR}")

    loaders = [PyPDFLoader(os.path.join(DATA_DIR, f)) for f in pdf_files]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    print(f"✅ Se cargaron {len(docs)} páginas/documentos.")

    print("Dividiendo en fragmentos (chunking)...")
    # chunk_size=1000 y overlap=100 optimizan recuperación contextual
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✅ Se generaron {len(chunks)} fragmentos.")

    print("Generando embeddings y guardando en ChromaDB...")
    embeddings = OpenAIEmbeddings()
    os.makedirs(CHROMA_DIR, exist_ok=True)
    # Crea y persiste la base vectorial
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    vectorstore.persist()
    print(f"Base vectorial guardada en {CHROMA_DIR}")

if __name__ == "__main__":
    ingest_documents()