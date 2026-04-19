import os
from dotenv import load_dotenv

# Importaciones correctas y modernas
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cargar variables de entorno (.env)
load_dotenv()

# Configuración de rutas
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
AZURE_ENDPOINT = os.getenv("GITHUB_BASE_URL") or os.getenv("OPENAI_BASE_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def main():
    print("🚀 Iniciando Asistente Académico RAG...")
    
    # 1. Cargar base de datos vectorial
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    # 2. Configurar el LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_base=AZURE_ENDPOINT,
        openai_api_key=GITHUB_TOKEN,
        temperature=0.1
    )
    
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
        
        print("⏳ Buscando...")
        try:
            # Ejecutar el pipeline
            res = chain.invoke(q)
            print(f"🤖 IA: {res}\n")
        except Exception as e:
            print(f"❌ Error al procesar: {e}\n")

if __name__ == "__main__":
    main()