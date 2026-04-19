# asistente-academico-rag
Asistente Académico RAG - Solución con LLM para consulta de normativas institucionales (Duoc UC). Proyecto EP1 - Ingeniería de Soluciones con IA.

## Ejecutar el proyecto

1. Crear y activar un entorno virtual:

```bash
cd /workspaces/asistente-academico-rag
python3 -m venv .venv
. .venv/bin/activate
```

2. Instalar las dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Configurar variables de entorno:

- Copia `.env.example` a `.env`
- Rellena tu `GITHUB_TOKEN` y, si usas OpenAI directo, `OPENAI_API_KEY`
- El proyecto ya está configurado para usar GitHub Models / Azure OpenAI en `GITHUB_BASE_URL`

4. Ingestar los PDFs en `data/`:

```bash
python src/ingest.py
```

5. Ejecutar el asistente:

```bash
python src/main.py
```

6. Escribir preguntas en el prompt y terminar con:

```bash
salir
```

## Notas importantes

- El proyecto utiliza `FAISS` como almacén vectorial en lugar de `Chroma` para evitar problemas de compatibilidad en este entorno.
- **¿Por qué FAISS y no Chroma?** Chroma es una base de datos vectorial completa con API REST y persistencia avanzada, pero requiere `onnxruntime>=1.14.1` para optimizaciones de GPU/CPU. En este entorno Alpine Linux, no hay ruedas precompiladas de `onnxruntime`, causando errores de instalación. FAISS es una librería más ligera y pura de búsqueda vectorial que se instala fácilmente con `faiss-cpu`, manteniendo toda la funcionalidad necesaria para el RAG sin dependencias nativas problemáticas.
- Si tienes problemas de conexión con Azure, verifica que tu `.env` tenga las variables correctas:
  - `GITHUB_BASE_URL`
  - `GITHUB_TOKEN`
  - `OPENAI_API_KEY` (opcional si usas OpenAI directo)
- Los PDFs deben estar en la carpeta `data/`.
- Se quito el token de .env por seguridad y quedo comentado en donde debe ir. 
