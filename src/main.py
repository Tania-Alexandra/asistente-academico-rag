import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "../chroma_db"

def setup_rag_chain():
    print("🔍 [IE3] Cargando base vectorial...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # 📝 [IE2] Prompt optimizado: estructura clara, contexto, restricciones y cita obligatoria
    prompt_template = """Eres un asistente académico oficial de la institución.
Tu tarea es responder consultas de estudiantes utilizando ÚNICAMENTE la información de los documentos proporcionados.

Instrucciones:
1. Responde de forma clara, directa y en español.
2. SIEMPRE cita la fuente documental entre paréntesis al final, ej: (Reglamento Académico, p. X).
3. Si la información no está en los documentos, responde EXACTAMENTE: 
   "No encuentro esta información en los documentos oficiales. Te sugiero contactar a secretaría académica."
4. Nunca inventes datos, asumas información externa ni des consejos no documentados.

Contexto recuperado:
{context}

Pregunta:
{question}

Respuesta:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # temperature=0 reduce alucinaciones y favorece respuestas fieles [IE4]
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # recupera 3 fragmentos más relevantes
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def main():
    if not os.path.exists(CHROMA_DIR):
        print("⚠️ No se encontró la base vectorial. Ejecuta primero: python3 src/ingest.py")
        return

    qa = setup_rag_chain()
    print("\n Asistente Académico RAG listo. Escribe 'salir' para terminar.")
    
    while True:
        query = input("\n🙋 Estudiante: ")
        if query.lower() in ["salir", "exit", "q"]:
            print("👋 ¡Hasta pronto!")
            break
            
        print("🤖 Pensando...")
        result = qa.run(query)
        print(f"🤖 Asistente: {result}")

if __name__ == "__main__":
    main()