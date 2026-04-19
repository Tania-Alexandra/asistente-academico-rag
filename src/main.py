import os
from dotenv import load_dotenv

# Importaciones correctas y modernas
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cargar variables de entorno (.env)
load_dotenv()

# Configuración de rutas
FAISS_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_db")

def main():
    print("Iniciando Asistente Académico RAG...")
    
    # Verificar credenciales
    api_key = os.getenv("GITHUB_TOKEN")
    if not api_key:
        print("X Error: Configura GITHUB_TOKEN en .env")
        return
    
    # 1. Cargar base de datos vectorial
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base="https://models.inference.ai.azure.com"
    )
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # 2. Configurar el LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        openai_api_base="https://models.inference.ai.azure.com",
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
        
        print("Buscando ...")
        try:
            # Ejecutar el pipeline
            res = chain.invoke(q)
            print(f"🤖 IA: {res}\n")
        except Exception as e:
            print(f"❌ Error al procesar: {e}\n")

if __name__ == "__main__":
    main()