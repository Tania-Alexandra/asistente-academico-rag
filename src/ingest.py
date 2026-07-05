import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from observability import ObservabilityRecorder, should_block_query

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
FAISS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "faiss_db"))

def ingest_documents():
    recorder = ObservabilityRecorder()
    recorder.logger.info("ingest_started")
    start_time = time.time()
    print("📚 [IE3] Cargando documentos desde data/...")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ No hay PDFs en data/")
        recorder.record_request(
            question="ingestion",
            response="No hay PDFs disponibles",
            latency_ms=(time.time() - start_time) * 1000,
            success=False,
            error="No hay PDFs en data/",
            tokens=0,
            expected_keywords=["pdf"],
        )
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

    print("🧠 Generando embeddings y guardando en FAISS...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("GITHUB_TOKEN"),
        openai_api_base="https://models.inference.ai.azure.com"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print(f"💾 Base vectorial guardada en: {FAISS_DIR}")
    print(f"📊 Vectores almacenados: {len(chunks)}")
    recorder.record_request(
        question="ingestion",
        response=f"Documentos procesados: {len(chunks)} fragmentos",
        latency_ms=(time.time() - start_time) * 1000,
        success=True,
        tokens=len(chunks),
        expected_keywords=["fragmentos"],
    )

if __name__ == "__main__":
    ingest_documents()