import os
import time
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from observability import ObservabilityRecorder, should_block_query

# Cargar variables de entorno (.env)
load_dotenv(override=True)

# Configuración de rutas
FAISS_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_db")


def get_api_credentials():
    github_token = os.getenv("GITHUB_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("GITHUB_BASE_URL")
    if github_token:
        return github_token, api_base, "github"
    return openai_key, api_base, "openai"


def main():
    recorder = ObservabilityRecorder()
    print("Iniciando Asistente Académico RAG...")
    recorder.logger.info("assistant_started")

    # Verificar credenciales
    api_key, api_base, provider = get_api_credentials()
    if not api_key:
        print("X Error: Configura GITHUB_TOKEN o OPENAI_API_KEY en .env")
        return
    if provider == "github" and not api_base:
        print("X Error: Configura GITHUB_BASE_URL en .env cuando uses GITHUB_TOKEN")
        return
    
    # 1. Cargar base de datos vectorial
    embeddings_kwargs = {
        "model": "text-embedding-3-small",
        "openai_api_key": api_key,
    }
    if api_base:
        embeddings_kwargs["openai_api_base"] = api_base
    embeddings = OpenAIEmbeddings(**embeddings_kwargs)
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # 2. Configurar el LLM
    llm_kwargs = {
        "model": "gpt-4o-mini",
        "openai_api_key": api_key,
        "temperature": 0.1,
    }
    if api_base:
        llm_kwargs["openai_api_base"] = api_base
    llm = ChatOpenAI(**llm_kwargs)
    
    # 3. Definir el prompt
    prompt_template = """Eres un asistente académico oficial. Responde SOLO con la información de los documentos.
Si la respuesta no está en el contexto, di EXACTAMENTE: "No encuentro esta información en los documentos oficiales."

Contexto: {context}
Pregunta: {question}
Respuesta:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 4. Crear el pipeline (LCEL - esto reemplaza a RetrievalQA)
    chain = (
        {"context": vectorstore.as_retriever(search_kwargs={"k": 3}), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ Listo. Escribe 'salir' para terminar.\n")
    
    # 5. Bucle de consulta
    while True:
        q = input("👤 Tú: ")
        if q.lower() in ['salir', 'exit', 'q']: 
            break
        if not q.strip(): 
            continue
        
        if should_block_query(q):
            recorder.record_request(
                question=q,
                response="Solicitud bloqueada por guardrail de seguridad.",
                latency_ms=0.0,
                success=False,
                error="Pregunta bloqueada por seguridad",
                tokens=0,
                expected_keywords=["seguridad"],
            )
            print("🛡️ Solicitud bloqueada por política de seguridad.\n")
            continue

        print("Buscando ...")
        start = time.time()
        try:
            res = chain.invoke(q)
            latency_ms = (time.time() - start) * 1000
            recorder.record_request(
                question=q,
                response=res,
                latency_ms=latency_ms,
                success=True,
                tokens=len((q + res).split()),
                expected_keywords=["reglamento", "titulación"],
            )
            print(f"🤖 IA: {res}\n")
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            recorder.record_request(
                question=q,
                response=str(e),
                latency_ms=latency_ms,
                success=False,
                error=str(e),
                tokens=0,
            )
            print(f"❌ Error al procesar: {e}\n")

if __name__ == "__main__":
    main()