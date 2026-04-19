import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
CHROMA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "chroma_db"))

def ingest_documents():
    print("📚 [IE3] Cargando documentos desde data/...")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ No hay PDFs en data/")
        return
        
    loaders = [PyPDFLoader(os.path.join(DATA_DIR, f)) for f in pdf_files]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    print(f"✅ Se cargaron {len(docs)} páginas.")

    print("✂️ Dividiendo en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✅ Se generaron {len(chunks)} fragmentos.")

    print("🧠 Generando embeddings y guardando en ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # ✅ Chroma guarda automáticamente con persist_directory
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    # ❌ vectorstore.persist() YA NO EXISTE en versiones recientes
    print(f"💾 Base vectorial guardada en: {CHROMA_DIR}")
    print(f"📊 Vectores almacenados: {vectorstore._collection.count()}")

if __name__ == "__main__":
    ingest_documents()