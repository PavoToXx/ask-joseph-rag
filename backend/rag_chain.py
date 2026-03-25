# backend/rag_chain.py
"""
Pipeline RAG usando LCEL (LangChain Expression Language).

Por qué LCEL y no RetrievalQA:
- RetrievalQA está deprecado en LangChain 0.3 y será eliminado.
- LCEL es composable: chain = prompt | llm | parser es más legible y testeable.
- LCEL soporta streaming nativo (para Sprint 5 con Streamlit).

Flujo:
  pregunta → retriever → [docs relevantes] →
  format_docs → contexto como string →
  prompt template → mensaje para el LLM →
  AzureChatOpenAI → respuesta cruda →
  StrOutputParser → string limpio
"""
import logging
import time
from typing import Optional

from pydantic import SecretStr  # requerido por langchain-openai para api_key
from langchain_chroma import Chroma
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langdetect import LangDetectException, detect
import re
from backend.ingest import RETRIEVAL_INTENT_MAP, VALID_RETRIEVAL_INTENTS

from backend.config import Settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Constantes — nunca magic numbers en el código                      #
# ------------------------------------------------------------------ #

# Cuántos chunks recupera el retriever por query.
# 3 es el balance entre contexto suficiente y no superar el context window.
RETRIEVER_K: int = 3


# Temperatura baja = respuestas más deterministas y factuales.
# Para un asistente de portfolio no queremos creatividad, queremos precisión.
LLM_TEMPERATURE: float = 1

# ------------------------------------------------------------------ #
#  Prompt templates — definidos como constantes, no strings inline    #
# ------------------------------------------------------------------ #

PROMPT_ES = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Eres el asistente personal de Joseph. "
        "Responde en español, de forma clara y profesional.\n"
        "Reglas:\n"
        "- Si la pregunta trata sobre Joseph, su perfil, identidad, estudios, gustos, experiencia o proyectos, o desarrollos, trata de responder la pregunta, ya que si hay contenido disponible."
        "- Usa la información del contexto aunque esté expresada con sinónimos, pronombres o nombres completos parciales."
        "- No inventes datos."
        "- Si tienes evidencia aunque sea poca, responde"
        "- Si no hay evidencia en el contexto, responde exactamente:"
        '"No tengo esa información sobre Joseph. Sea más específico.".\n\n'
        "Contexto:\n{context}\n\n"
        "Pregunta: {question}\n"
        "Respuesta:"
    ),
)

PROMPT_EN = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Joseph's personal assistant. "
        "Answer in English, clearly and professionally.\n"
        "Rules:\n"
        "- If the question is about Joseph, his profile, identity, education, interests, experience or projects, or developments, try to answer the question, as there may be relevant information available."
        "- Use the information in the context even if it's expressed with synonyms, pronouns or partial full names."
        "- Do not make up data."
        "- If the information is not in the context, respond exactly: "
        '"I don\'t have that information about Joseph. Please be more specific."\n\n'
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)
# PROMPT_ES = PromptTemplate(
#     input_variables=["context", "question"],
#     template=(
#         "Eres el asistente personal de Joseph. "
#         "Responde en español, de forma clara y profesional.\n"
#         "Reglas:\n"
#         "- Si la pregunta trata sobre Joseph, su perfil, identidad, estudios, gustos, experiencia o proyectos, interpreta la pregunta como una consulta sobre su información personal.\n"
#         "- Usa la información del contexto aunque esté expresada con sinónimos, pronombres o nombres completos parciales.\n"
#         "- No inventes datos.\n"
#         "- Si no hay evidencia suficiente en el contexto, responde exactamente: \"No tengo esa información sobre Joseph.\"\n\n"
#         "Contexto:\n{context}\n\n"
#         "Pregunta: {question}\n"
#         "Respuesta:"
#     ),
# )

# PROMPT_EN = PromptTemplate(
#     input_variables=["context", "question"],
#     template=(
#         "You are Joseph's personal assistant. "
#         "Answer in English, clearly and professionally.\n"
#         "Rules:\n"
#         "- If the question is about Joseph, his profile, identity, education, interests, experience or projects, interpret the question as a query about his personal information.\n"
#         "- Use the information in the context even if it's expressed with synonyms, pronouns or partial full names.\n"
#         "- Do not make up data.\n"
#         "- If the information is not in the context, respond exactly: \"I don't have that information about Joseph.\"\n\n"
#         "Context:\n{context}\n\n"
#         "Question: {question}\n"
#         "Answer:"
#     ),
# )

# Respuestas de fallback cuando ChromaDB no encuentra nada relevante.
# Esto cubre el bug del EPIC-3: "edge case si ChromaDB no tiene resultados".
_FALLBACK = {
    "es": "No encontré información relevante en mis documentos para esa pregunta.",
    "en": "I couldn't find relevant information in my documents to answer that.",
}


def map_query_to_intents(query: str) -> list[str]:
    tokens = re.findall(r"\w+", query.lower())
    mapped: list[str] = []
    for t in tokens:
        if t in RETRIEVAL_INTENT_MAP:
            mapped.append(RETRIEVAL_INTENT_MAP[t])
    # mantener orden y unicidad, y filtrar por intents válidos
    normalized: list[str] = []
    for m in mapped:
        if m not in normalized and m in VALID_RETRIEVAL_INTENTS:
            normalized.append(m)
    return normalized


class RAGChain:
    """
    Encapsula el pipeline RAG completo: embeddings, vector store y LLM.

    Diseñado para ser instanciado una sola vez al startup de FastAPI
    (via lifespan) y reutilizado en cada request via dependency injection.

    Por qué clase y no funciones sueltas:
    - El vectorstore y el LLM son objetos costosos de inicializar.
    - Una clase permite inyectar settings en el constructor y
      controlar explícitamente cuándo se inicializa (initialize()).
    - Más fácil de mockear en tests.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Args:
            settings: Configuración validada de la aplicación.
                      Nunca recibe strings directos de API keys.
        """
        self._settings = settings
        self._vectorstore: Optional[Chroma] = None
        self._llm: Optional[AzureChatOpenAI] = None
        self._initialized: bool = False

    def initialize(self) -> None:
        """
        Inicializa embeddings, ChromaDB y LLM. Llama esto UNA VEZ al startup.

        Por qué separado del __init__:
        - Permite hacer dependency injection del objeto antes de que esté listo.
        - Facilita tests unitarios: puedes crear RAGChain sin conexiones reales.

        Raises:
            RuntimeError: Si ChromaDB no existe en chroma_path.
            Exception: Si las credenciales de Azure son inválidas.
        """
        logger.info("Initializing RAG chain components...")

        # Los valores vienen de Settings (pydantic-settings), que los leyó
        # de variables de entorno. Nunca llegan hardcodeados desde el código.
        # SecretStr: langchain-openai exige este tipo para api_key.
        # Pydantic lo usa para que el valor no aparezca en logs ni repr().
        secret_key = SecretStr(self._settings.azure_openai_api_key)

        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=self._settings.azure_openai_embedding_deployment,
            azure_endpoint=self._settings.azure_openai_endpoint,
            api_key=secret_key,
            api_version=self._settings.azure_openai_api_version,
        )

        self._vectorstore = Chroma(
            persist_directory=self._settings.chroma_path,
            embedding_function=embeddings,
            # Sin esto, LangChain conecta a "langchain" por default (0 docs).
            # El nombre viene de settings para no hardcodear strings en el código.
            collection_name=self._settings.chroma_collection_name,
        )

        self._llm = AzureChatOpenAI(
            azure_deployment=self._settings.azure_openai_chat_deployment,
            azure_endpoint=self._settings.azure_openai_endpoint,
            api_key=secret_key,
            api_version=self._settings.azure_openai_api_version,
            temperature=LLM_TEMPERATURE,
        )

        self._initialized = True
        logger.info("RAG chain initialized successfully.")

    # ---------------------------------------------------------------- #
    #  Métodos privados                                                  #
    # ---------------------------------------------------------------- #

    def _detect_language(self, text: str) -> str:
        """
        Detecta el idioma del texto. Defaultea a español si falla.

        Args:
            text: Texto a analizar (la pregunta del usuario).

        Returns:
            Código ISO 639-1 del idioma detectado (ej: 'es', 'en').
        """
        try:
            return detect(text)
        except LangDetectException:
            # langdetect puede fallar en textos muy cortos (1-2 palabras).
            # En ese caso, español es el default seguro para este proyecto.
            logger.warning("Language detection failed for short/ambiguous input.")
            return "es"
        
        

    @staticmethod
    def _format_docs(docs: list) -> str:
        """
        Concatena el contenido de los documentos recuperados.

        Args:
            docs: Lista de Document objects de LangChain.

        Returns:
            String con todos los page_content separados por doble salto.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _extract_sources(docs: list) -> list[str]:
        """
        Extrae nombres de fuentes únicos de los documentos recuperados.

        Deduplication via set: si 3 chunks vienen del mismo CV,
        solo aparece una vez en la respuesta.

        Args:
            docs: Lista de Document objects de LangChain.

        Returns:
            Lista de strings con nombres de fuente únicos.
        """
        return list(
            {doc.metadata.get("file_name", "Documento desconocido") for doc in docs}
        )

    # ---------------------------------------------------------------- #
    #  API pública                                                       #
    # ---------------------------------------------------------------- #

    def get_answer(self, question: str) -> dict:
        """
        Ejecuta el pipeline RAG completo para una pregunta.

        Flujo:
          1. Retrieve: busca chunks relevantes en ChromaDB
          2. Check: si no hay resultados, devuelve fallback amigable (no crash)
          3. Detect language: selecciona el prompt correcto
          4. Chain LCEL: prompt | llm | parser
          5. Track tokens con get_openai_callback
          6. Return: answer, sources, latency_ms, tokens_used

        Args:
            question: Pregunta del usuario. Ya debe venir validada por FastAPI.

        Returns:
            dict con keys: answer (str), sources (list[str]),
                           latency_ms (int), tokens_used (int).

        Raises:
            RuntimeError: Si se llama antes de initialize().
        """
        if not self._initialized:
            raise RuntimeError(
                "RAGChain.get_answer() llamado antes de initialize(). "
                "Revisa el lifespan de FastAPI."
            )

        # assert le dice a Pylance: "a partir de aquí _vectorstore y _llm
        # son definitivamente no-None". En runtime, si initialize() se llamó
        # correctamente, estos assert nunca fallan — son solo para el type checker.
        assert self._vectorstore is not None, "vectorstore no inicializado"
        assert self._llm is not None, "llm no inicializado"

        # time.monotonic() es más preciso que time.time() para medir duraciones.
        start = time.monotonic()
        lang = self._detect_language(question)

        # --- Paso 1: Retrieve (semántico + filtrado por metadata) ---
        # Mapear query a intents detectados
        intents = map_query_to_intents(question)

        # Pedir más candidatos para tener margen al filtrar por metadata
        candidate_k = max(RETRIEVER_K * 4, 12)
        raw_pairs = self._vectorstore.similarity_search_with_relevance_scores(
            question, k=candidate_k
        )

        normalized_pairs = []
        for doc, raw_score in raw_pairs:
            try:
                s = float(raw_score)
            except Exception:
                s = 0.0
            # Normalizar de [-1,1] a [0,1]
            s_norm = (s + 1.0) / 2.0
            normalized_pairs.append((doc, s_norm))

        # Umbral base para considerar candidatos semánticos
        threshold = 0.18
        normalized_pairs.sort(key=lambda t: t[1], reverse=True)

        def doc_matches_intents(doc, intents_list: list[str]) -> bool:
            if not intents_list:
                return True
            meta_intents = doc.metadata.get("retrieval_intent", [])
            if isinstance(meta_intents, str):
                meta_intents = [meta_intents]
            meta_intents = [str(x).strip().lower() for x in meta_intents]
            return any(i in meta_intents for i in intents_list)

        # Si la query mapeó a intents, intentar filtrar por esos intents
        if intents:
            filtered = [(d, s) for d, s in normalized_pairs if doc_matches_intents(d, intents) and s >= threshold]
            if filtered:
                filtered.sort(key=lambda t: t[1], reverse=True)
                docs = [d for d, _ in filtered][:RETRIEVER_K]
            else:
                # Fallback semántico si el filtrado por metadata no devolvió nada
                docs = [d for d, s in normalized_pairs if s >= threshold][:RETRIEVER_K]
        else:
            docs = [d for d, s in normalized_pairs if s >= threshold][:RETRIEVER_K]

        # --- Paso 2: Fallback si ChromaDB no encontró nada ---
        # Esto cubre el bug del EPIC-3: "edge case si no hay resultados relevantes"
        if not docs:
            logger.info("No relevant documents found. Returning fallback.")
            fallback_lang = "en" if lang == "en" else "es"
            return {
                "answer": _FALLBACK[fallback_lang],
                "sources": [],
                "latency_ms": int((time.monotonic() - start) * 1000),
                "tokens_used": 0,
            }

        # --- Paso 3: Seleccionar prompt según idioma ---
        prompt = PROMPT_EN if lang == "en" else PROMPT_ES

        # --- Paso 4: LCEL chain ---
        # Reemplaza al deprecado RetrievalQA.
        # prompt | llm | StrOutputParser() es la composición estándar en LangChain 0.3+.
        chain = prompt | self._llm | StrOutputParser()

        # --- Paso 5: Invocar con tracking de tokens ---
        tokens_used = 0
        try:
            with get_openai_callback() as cb:
                answer: str = chain.invoke(
                    {
                        "context": self._format_docs(docs),
                        "question": question,
                    }
                )
                tokens_used = cb.total_tokens
        except Exception as e:
            # Log el tipo de error pero NO el mensaje completo (puede contener
            # fragmentos de la query del usuario con PII).
            logger.error(
                "LLM invocation failed. error_type='%s'", type(e).__name__
            )
            raise  # Re-raise para que FastAPI lo maneje y devuelva 500

        # --- Paso 6: Ensamblar respuesta ---
        latency_ms = int((time.monotonic() - start) * 1000)
        sources = self._extract_sources(docs)

        # Log de operaciones normales: latencia, tokens, número de fuentes.
        # NO logueamos la query ni el answer (pueden contener PII).
        logger.info(
            "Query processed. latency_ms=%d tokens=%d sources=%d lang='%s'",
            latency_ms,
            tokens_used,
            len(sources),
            lang,
        )

        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
        }