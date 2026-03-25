# backend/ingest.py
"""
Pipeline de ingesta de documentos para el sistema RAG de Joseph.

Soporta dos formatos de entrada:
  .jsonl  → chunks pre-definidos con metadata estructurada (formato preferido)
  .txt / .md / .html → texto plano chunkeado automáticamente

Flujo:
  S3 download → process_jsonl / process_text → validate → ingest_to_chroma

Autenticación AWS:
  local          → AWS CLI profile en ~/.aws/credentials
  github_actions → OIDC, credenciales inyectadas por aws-actions
  azure          → Managed Identity + AssumeRoleWithWebIdentity

Cómo correr:
  python -m backend.ingest
"""
import json
import logging
import re
from pathlib import Path
from typing import Optional

import boto3
import chromadb
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredHTMLLoader
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from backend.config import Settings, get_settings

# ------------------------------------------------------------------ #
#  Logging                                                             #
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Constantes                                                          #
# ------------------------------------------------------------------ #

# Directorio temporal donde se descargan los docs de S3
LOCAL_DOCS_DIR: str = "./docs_temp"

# Nombre de la colección en ChromaDB — debe coincidir con config.py
COLLECTION_NAME: str = "ask-joseph-docs"

# Separadores para el splitter de texto plano.
# Orden de prioridad: párrafo → línea → oración → palabra → caracter
TEXT_SPLITTERS: list[str] = ["\n\n", "\n", ".", " ", ""]

# Formatos de archivo soportados
SUPPORTED_PLAIN_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".html"})

# Campos que DEBEN estar presentes en el metadata de cada chunk JSONL.
# Si falta alguno, el chunk se descarta con un warning.
REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "file_name",
    "doc_type",
    "section",
    "topic",
    "language",
    "visibility",
    "sensitivity",
    "technology",
    "date_range",
    "retrieval_intent",
)

# Campos opcionales con valor default explícito.
# Se aplican si no vienen en el JSONL.
OPTIONAL_METADATA_DEFAULTS: dict[str, None] = {
    "project_name": None,
    "company": None,
}

# Campos que deben ser listas en el metadata.
# ChromaDB los serializa como listas solo si llegan tipados correctamente.
LIST_METADATA_FIELDS: tuple[str, ...] = ("topic", "technology", "retrieval_intent")

# Valores válidos que acepta date_range además de fechas con formato.
# "actual" = trabajo actual, "pandemia" = proyecto durante pandemia.
VALID_DATE_RANGE_LITERALS: frozenset[str] = frozenset(
    {"actual", "futuro", "pandemia", "unknown"}
)

# Mapa de canonicalización para doc_type.
# Normaliza variantes en español/inglés a un valor único.
DOC_TYPE_MAP: dict[str, str] = {
    "perfil": "perfil",
    "profile": "perfil",
    "biografia": "biography",
    "biography": "biography",
    "cv": "experience",
    "vitae": "experience",
    "curriculum": "experience",
    "resumen": "summary",
    "summary": "summary",
    "skills": "skills",
    "habilidades": "skills",
    "tecnologias": "skills",
    "readme": "project",
    "proyecto": "project",
    "project": "project",
    "experiencia": "experience",
    "experience": "experience",
    "certificaciones": "certification",
    "certification": "certification",
    "educacion": "education",
}

# Intenciones válidas de retrieval — conjunto cerrado en español según solicitud
# del usuario. Todas las entradas de metadata se mapearán a este conjunto.
VALID_RETRIEVAL_INTENTS: frozenset[str] = frozenset(
    {"perfil", "proyectos", "skills", "experiencia", "educacion", "logros", "tecnologias", "contacto"}
)

# Mapa de variantes y sinónimos (inglés/español/variantes internas) hacia la
# intención canónica en español. Esto permite aceptar datos históricos y
# normalizarlos al conjunto cerrado.
RETRIEVAL_INTENT_MAP: dict[str, str] = {
    # Perfil / contacto
    "profile": "perfil",
    "perfil": "perfil",
    "contact": "contacto",
    "contacto": "contacto",

    # Proyectos
    "project": "proyectos",
    "proyecto": "proyectos",
    "proyectos": "proyectos",

    # Skills / technologies
    "technical": "skills",
    "tecnico": "skills",
    "skills": "skills",
    "habilidades": "skills",

    # Experiencia / cv
    "experience": "experiencia",
    "experiencia": "experiencia",
    "cv": "experiencia",

    # Educacion
    "education": "educacion",
    "educacion": "educacion",

    # Logros / achievements
    "achievements": "logros",
    "logros": "logros",

    # Tecnologías / tech
    "technology": "tecnologias",
    "tecnologia": "tecnologias",
    "tecnologias": "tecnologias",
    "tech": "tecnologias",

    # Routing / search / defaults map to 'perfil' as safe default
    "routing": "perfil",
    "search": "perfil",
    "qa": "perfil",
    "summary": "perfil",
}

# Prioridades permitidas y default
VALID_PRIORITIES: frozenset[str] = frozenset({"low", "normal", "high"})
DEFAULT_PRIORITY: str = "normal"

# Longitud mínima de chunk recomendable; los chunks más cortos intentaremos
# fusionarlos con el chunk anterior del mismo documento.
MIN_CHUNK_CHARS: int = 60

# Valor por defecto para technology cuando viene vacío o ausente
DEFAULT_TECHNOLOGY: list[str] = ["general"]


# ------------------------------------------------------------------ #
#  Helpers de metadata                                                 #
# ------------------------------------------------------------------ #

def normalize_date_range(value: Optional[str]) -> str:
    """
    Normaliza el campo date_range a un formato consistente.

    Formatos aceptados:
      - Literales: "actual", "futuro", "pandemia", "unknown"
      - Año-mes:   "2024-06"
      - Rango:     "2024-06/2025-08"
      - Año:       "2024" 
      - Rango año: "2024/2025"

    Args:
        value: Valor raw del campo date_range.

    Returns:
        Valor normalizado o "unknown" si no coincide con ningún formato.
    """
    if not value:
        return "unknown"

    normalized = str(value).strip().lower()

    if normalized in VALID_DATE_RANGE_LITERALS:
        return normalized

    # Formato año-mes: 2024-06
    if re.fullmatch(r"\d{4}-\d{2}", normalized):
        return normalized

    # Formato rango: 2024-06/2025-08
    if re.fullmatch(r"\d{4}-\d{2}/\d{4}-\d{2}", normalized):
        return normalized

    # Formato año: 2024
    if re.fullmatch(r"\d{4}", normalized):
        return normalized
    
    # Formato año rango: 2024/2025
    if re.fullmatch(r"\d{4}/\d{4}", normalized):
        return normalized

    logger.warning(
        "date_range '%s' no coincide con ningún formato válido — usando 'unknown'",
        value,
    )
    return "unknown"

def canonicalize_doc_type(metadata: dict) -> dict:
    """
    Normaliza doc_type usando el DOC_TYPE_MAP.
    Si el valor no está en el mapa, lo deja sin cambios y loguea un warning.

    Args:
        metadata: Diccionario de metadata del chunk.

    Returns:
        Metadata con doc_type normalizado.
    """
    raw = metadata.get("doc_type")
    if not raw:
        return metadata

    key = str(raw).strip().lower()
    canonical = DOC_TYPE_MAP.get(key)

    if canonical:
        metadata["doc_type"] = canonical
    else:
        logger.warning(
            "doc_type '%s' no está en DOC_TYPE_MAP — considera añadirlo", raw
        )

    return metadata


def validate_metadata(
    metadata: dict,
    source_file: str,
    line_number: int,
) -> bool:
    """
    Valida que el metadata de un chunk tenga todos los campos requeridos
    y los tipos correctos.

    Si un campo requerido falta, el chunk DEBE ser descartado (retorna False).
    Los campos opcionales se completan con sus defaults si no están presentes.

    Args:
        metadata: Diccionario de metadata a validar.
        source_file: Nombre del archivo fuente (para logging).
        line_number: Número de línea en el JSONL (para logging).

    Returns:
        True si el metadata es válido, False si debe descartarse el chunk.
    """
    # Verificar campos requeridos
    missing = [f for f in REQUIRED_METADATA_FIELDS if f not in metadata]
    if missing:
        logger.warning(
            "Chunk descartado %s:%d — campos faltantes: %s",
            source_file,
            line_number,
            ", ".join(missing),
        )
        return False

    # Verificar y coerzir campos que deben ser lista.
    for field in LIST_METADATA_FIELDS:
        value = metadata.get(field)
        # Si viene como string separado por comas, intentar convertir a lista
        if isinstance(value, str):
            items = [s.strip() for s in value.split(",") if s.strip()]
            metadata[field] = items
        # Si es None o lista vacía -> asignar defaults razonables para fields requeridos
        elif value is None:
            if field == "technology":
                metadata[field] = DEFAULT_TECHNOLOGY.copy()
            else:
                metadata[field] = []
        elif not isinstance(value, list):
            logger.warning(
                "Chunk descartado %s:%d — '%s' debe ser lista o string, recibido %s",
                source_file,
                line_number,
                field,
                type(value).__name__,
            )
            return False

        # Normalizar strings dentro de las listas
        if isinstance(metadata.get(field), list):
            cleaned = [str(x).strip() for x in metadata[field] if str(x).strip()]
            metadata[field] = cleaned

    # Completar opcionales con defaults
    for field, default in OPTIONAL_METADATA_DEFAULTS.items():
        metadata.setdefault(field, default)

    # Normalizar date_range
    metadata["date_range"] = normalize_date_range(metadata.get("date_range"))

    # Normalizar retrieval_intent a conjunto cerrado usando el mapa de variantes
    intents = metadata.get("retrieval_intent")
    normalized_intents: list[str] = []
    if isinstance(intents, list):
        for raw in intents:
            key = str(raw).strip().lower()
            mapped = RETRIEVAL_INTENT_MAP.get(key, None)
            if mapped and mapped in VALID_RETRIEVAL_INTENTS:
                if mapped not in normalized_intents:
                    normalized_intents.append(mapped)
    else:
        # intentar mapear si viene como string
        if isinstance(intents, str):
            key = intents.strip().lower()
            mapped = RETRIEVAL_INTENT_MAP.get(key, None)
            if mapped and mapped in VALID_RETRIEVAL_INTENTS:
                normalized_intents.append(mapped)

    # Si no quedó ninguna intención válida, usar 'perfil' como default seguro
    if not normalized_intents:
        metadata["retrieval_intent"] = ["perfil"]
    else:
        metadata["retrieval_intent"] = normalized_intents

    # Añadir campo priority si falta y validar su valor
    priority = metadata.get("priority")
    if not priority:
        metadata["priority"] = DEFAULT_PRIORITY
    else:
        p = str(priority).strip().lower()
        metadata["priority"] = p if p in VALID_PRIORITIES else DEFAULT_PRIORITY

    return True


def sanitize_for_chroma(metadata: dict) -> dict:
    """
    Prepara el metadata para ser almacenado en ChromaDB.

    ChromaDB no acepta listas con elementos vacíos ni valores None en
    ciertos contextos. Esta función limpia los valores problemáticos.

    Args:
        metadata: Diccionario de metadata validado.

    Returns:
        Metadata listo para ChromaDB.
    """
    sanitized = dict(metadata)

    for field in LIST_METADATA_FIELDS:
        value = sanitized.get(field)
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if cleaned:
                sanitized[field] = cleaned
            else:
                # No dejar listas vacías para campos críticos; sustituir por default
                if field == "technology":
                    sanitized[field] = DEFAULT_TECHNOLOGY.copy()
                elif field == "retrieval_intent":
                    sanitized[field] = ["perfil"]
                else:
                    # eliminar campo si no es crítico
                    sanitized.pop(field, None)

    # Asegurar que 'priority' esté presente y sea válida
    pr = sanitized.get("priority")
    if not pr or str(pr).strip().lower() not in VALID_PRIORITIES:
        sanitized["priority"] = DEFAULT_PRIORITY
    else:
        sanitized["priority"] = str(pr).strip().lower()

    # Normalizar doc_type a string simple
    if "doc_type" in sanitized:
        sanitized["doc_type"] = str(sanitized["doc_type"]).strip().lower()

    return sanitized


# ------------------------------------------------------------------ #
#  Carga de configuración                                              #
# ------------------------------------------------------------------ #

def load_ingest_config() -> dict:
    """
    Carga la configuración de chunking por archivo desde ingest_config.json.

    El archivo define chunk_size, chunk_overlap y defaults de metadata
    por nombre de archivo. Si no existe, retorna un dict vacío y el
    pipeline usa los valores por defecto de Settings.

    Returns:
        Dict con configuración por nombre de archivo, o {} si no existe.
    """
    # Busca el config relativo a la raíz del proyecto
    config_path = Path(__file__).resolve().parent.parent / "docs" / "ingest_guides" / "ingest_config.json"

    if not config_path.exists():
        logger.warning(
            "ingest_config.json no encontrado en %s — usando defaults", config_path
        )
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info("ingest_config.json cargado: %d entradas", len(config))
        return config
    except json.JSONDecodeError as e:
        logger.error(
            "ingest_config.json tiene JSON inválido: %s — usando defaults", e
        )
        return {}


# ------------------------------------------------------------------ #
#  Cliente S3                                                          #
# ------------------------------------------------------------------ #

def _build_s3_client(settings: Settings):
    """
    Crea un cliente S3 usando el método de autenticación correcto.

    Args:
        settings: Configuración validada de la aplicación.

    Returns:
        boto3 S3 client.

    Raises:
        RuntimeError: Si no se encuentran credenciales válidas.
    """
    try:
        from config import Environment

        if settings.environment == Environment.LOCAL:
            logger.debug("AWS auth: LOCAL profile='%s'", settings.aws_profile)
            session = boto3.Session(
                profile_name=settings.aws_profile,
                region_name=settings.aws_region,
            )
        else:
            # GITHUB_ACTIONS: OIDC inyecta env vars automáticamente
            # AZURE: Managed Identity → AssumeRoleWithWebIdentity automático
            logger.debug("AWS auth: %s (auto-detected)", settings.environment.value)
            session = boto3.Session(region_name=settings.aws_region)

        return session.client("s3")

    except NoCredentialsError as e:
        raise RuntimeError(
            f"No se encontraron credenciales AWS para ENVIRONMENT='{settings.environment}'. "
            "Revisa ~/.aws/credentials (local) o las variables de entorno (CI/Azure)."
        ) from e


def download_from_s3(settings: Settings, local_dir: str = LOCAL_DOCS_DIR) -> Path:
    """
    Descarga todos los documentos del bucket S3 al directorio local.

    Args:
        settings: Configuración validada con bucket y región.
        local_dir: Ruta local donde guardar los archivos.

    Returns:
        Path al directorio con los archivos descargados.

    Raises:
        RuntimeError: Si el bucket no existe o hay error de permisos.
    """
    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        s3 = _build_s3_client(settings)
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=settings.aws_bucket_name)

        count = 0
        for page in pages:
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                # Extrae solo el nombre del archivo, ignorando "carpetas" en S3
                filename = key.split("/")[-1]
                if not filename:
                    continue
                local_path = dest / filename
                logger.info("Descargando s3://%s/%s", settings.aws_bucket_name, key)
                s3.download_file(settings.aws_bucket_name, key, str(local_path))
                count += 1

        logger.info("Descarga completa: %d archivos en %s", count, dest)
        return dest

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        raise RuntimeError(
            f"Error S3 al descargar documentos. code='{error_code}' "
            f"bucket='{settings.aws_bucket_name}'"
        ) from e
    except BotoCoreError as e:
        raise RuntimeError("Error de conexión con AWS S3.") from e


# ------------------------------------------------------------------ #
#  Procesadores de archivo                                             #
# ------------------------------------------------------------------ #

def process_jsonl(path: Path, file_config: dict) -> list[Document]:
    """
    Procesa un archivo .jsonl donde cada línea es un chunk pre-definido.

    Formato esperado de cada línea:
        {"chunk_text": "...", "metadata": {"file_name": "...", ...}}

    Los chunks con metadata inválida se descartan con un warning — nunca
    se propagan silenciosamente (bug corregido: era `pass` antes).

    Args:
        path: Ruta al archivo .jsonl.
        file_config: Configuración de ingest_config.json para este archivo.

    Returns:
        Lista de Document objects válidos y listos para vectorizar.
    """
    docs: list[Document] = []
    defaults = file_config.get("defaults", {})
    discarded = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            # JSON parse con tipo explícito — no bare except
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    "JSON inválido en %s:%d — %s", path.name, line_number, e
                )
                discarded += 1
                continue

            # Extraer y validar chunk_text
            chunk_text = str(record.get("chunk_text", "") or "").strip()
            if not chunk_text:
                logger.warning(
                    "chunk_text vacío en %s:%d — descartado", path.name, line_number
                )
                discarded += 1
                continue

            # Construir metadata: defaults del config + metadata del record
            raw_metadata = record.get("metadata")
            metadata: dict = {}
            if isinstance(raw_metadata, dict):
                metadata = raw_metadata

            # Los defaults del config tienen menor prioridad que el metadata del chunk
            merged = {**defaults, **metadata}
            merged = canonicalize_doc_type(merged)

            # Validar — si falla, DESCARTAR (no `pass`)
            if not validate_metadata(merged, path.name, line_number):
                discarded += 1
                continue

            sanitized = sanitize_for_chroma(merged)

            # # Si el chunk es demasiado corto, intentar fusionarlo con el anterior
            # if len(chunk_text) < MIN_CHUNK_CHARS:
            #     if docs:
            #         last_doc = docs[-1]
            #         last_meta = last_doc.metadata or {}
            #         # Fusionar solo si son del mismo archivo y tipo de documento
            #         if (
            #             last_meta.get("file_name") == sanitized.get("file_name")
            #             and last_meta.get("doc_type") == sanitized.get("doc_type")
            #         ):
            #             combined_text = last_doc.page_content + "\n\n" + chunk_text
            #             # Mantener metadata del anterior (ya saneada)
            #             docs[-1] = Document(page_content=combined_text, metadata=last_meta)
            #             continue
            #         else:
            #             logger.info(
            #                 "Chunk corto no pudo fusionarse %s:%d — descartado",
            #                 path.name,
            #                 line_number,
            #             )
            #             discarded += 1
            #             continue
            #     else:
            #         logger.info(
            #             "Chunk corto sin previo para fusionar %s:%d — descartado",
            #             path.name,
            #             line_number,
            #         )
            #         discarded += 1
            #         continue

            docs.append(Document(page_content=chunk_text, metadata=sanitized))

    logger.info(
        "JSONL %s: %d chunks válidos, %d descartados",
        path.name,
        len(docs),
        discarded,
    )
    return docs


def process_plain_text(path: Path, file_config: dict, settings: Settings) -> list[Document]:
    """
    Procesa archivos .txt, .md y .html chunkeándolos automáticamente.

    El chunk_size y chunk_overlap se toman de ingest_config.json para
    este archivo. Si no están configurados, usa los valores de Settings.

    Args:
        path: Ruta al archivo de texto.
        file_config: Configuración de ingest_config.json para este archivo.
        settings: Configuración de la aplicación (para defaults).

    Returns:
        Lista de Document objects chunkeados.
    """
    # Seleccionar loader según extensión
    ext = path.suffix.lower()
    try:
        if ext == ".html":
            loader = UnstructuredHTMLLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        loaded_docs = loader.load()
    except Exception as e:
        logger.error("Error cargando %s: %s", path.name, type(e).__name__)
        return []

    # chunk_size del config de archivo tiene prioridad sobre Settings
    chunk_size: int = file_config.get("chunk_size", 300 )
    chunk_overlap: int = file_config.get("chunk_overlap", 10)
    defaults: dict = file_config.get("defaults", {})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=TEXT_SPLITTERS,
    )

    docs: list[Document] = []
    discarded = 0

    for loaded_doc in loaded_docs:
        splits = splitter.split_documents([loaded_doc])
        for split in splits:
            metadata = {**defaults, **(split.metadata or {})}
            metadata = canonicalize_doc_type(metadata)

            # Para texto plano, la validación es informativa — no descarta
            # porque el metadata viene del config, no del archivo.
            # Solo loguea si hay campos faltantes.
            is_valid = validate_metadata(metadata, path.name, 0)
            if not is_valid:
                logger.debug("Metadata incompleto en texto plano %s — indexando igual", path.name)

            sanitized = sanitize_for_chroma(metadata)
            docs.append(Document(page_content=split.page_content, metadata=sanitized))

    logger.info(
        "Texto plano %s: chunk_size=%d → %d chunks",
        path.name,
        chunk_size,
        len(docs),
    )
    return docs


# ------------------------------------------------------------------ #
#  ChromaDB                                                            #
# ------------------------------------------------------------------ #

def ingest_to_chroma(
    chunks: list[Document],
    settings: Settings,
    collection_name: str = COLLECTION_NAME,
    clear_existing: bool = True,
) -> Chroma:
    """
    Vectoriza los chunks y los almacena en ChromaDB.

    Siempre limpia la colección antes de ingestear para evitar duplicados.
    Si necesitas ingesta incremental, usa clear_existing=False — pero ten
    en cuenta que podrías tener duplicados si re-procesas los mismos archivos.

    Args:
        chunks: Lista de Document objects a vectorizar.
        settings: Configuración de la aplicación.
        collection_name: Nombre de la colección en ChromaDB.
        clear_existing: Si True, elimina la colección antes de ingestear.

    Returns:
        Chroma vector store inicializado.

    Raises:
        ValueError: Si chunks está vacío.
    """
    if not chunks:
        raise ValueError(
            "No hay chunks para vectorizar. "
            "Verifica que los archivos en S3 son válidos y el ingest_config.json existe."
        )

    # Limpiar colección existente para evitar duplicados
    # Bug anterior: el import de chromadb estaba dentro de esta función
    client = chromadb.PersistentClient(path=settings.chroma_path)
    existing_names = [c.name for c in client.list_collections()]

    if clear_existing and collection_name in existing_names:
        client.delete_collection(collection_name)
        logger.info("Colección '%s' eliminada para re-ingestar limpio", collection_name)

    # Construir embeddings — SecretStr protege la key desde el inicio
    secret_key = SecretStr(settings.azure_openai_api_key)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=settings.azure_openai_embedding_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=secret_key,
        api_version=settings.azure_openai_api_version
    )

    logger.info("Vectorizando %d chunks → colección '%s'...", len(chunks), collection_name)

    vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=settings.chroma_path,
    collection_name=collection_name,
    collection_metadata={"hnsw:space": "cosine"},
)

    # Verificar que realmente se guardó
    client = chromadb.PersistentClient(path=settings.chroma_path)
    saved_count = client.get_collection(collection_name).count()
    logger.info(
        "Ingest completo: %d chunks guardados en '%s'",
        saved_count,
        collection_name,
    )

    return vectorstore


# ------------------------------------------------------------------ #
#  Pipeline principal                                                  #
# ------------------------------------------------------------------ #

def run_ingest(
    collection_name: str = COLLECTION_NAME,
    clear_existing: bool = True,
) -> None:
    """
    Ejecuta el pipeline completo de ingesta.

    Flujo:
      1. Carga settings desde variables de entorno / .env
      2. Carga ingest_config.json para configuración por archivo
      3. Descarga documentos desde S3
      4. Procesa cada archivo (.jsonl o texto plano)
      5. Vectoriza y guarda en ChromaDB

    Args:
        collection_name: Nombre de la colección destino en ChromaDB.
        clear_existing: Si True, limpia la colección antes de ingestear.
    """
    # Settings se validan aquí — falla ruidosamente si falta una env var
    settings = get_settings()

    logger.info(
        "Iniciando ingest. environment='%s' bucket='%s' collection='%s'",
        settings.environment.value,
        settings.aws_bucket_name,
        collection_name,
    )

    ingest_config = load_ingest_config()    
    local_dir = download_from_s3(settings)

    all_chunks: list[Document] = []
    files_processed = 0
    files_skipped = 0

    for file_path in sorted(local_dir.iterdir()):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        file_config = ingest_config.get(file_path.name, {})

        try:
            if ext == ".jsonl":
                chunks = process_jsonl(file_path, file_config)
                all_chunks.extend(chunks)
                files_processed += 1

            elif ext in SUPPORTED_PLAIN_EXTENSIONS:
                chunks = process_plain_text(file_path, file_config, settings)
                all_chunks.extend(chunks)
                files_processed += 1

            else:
                logger.debug("Formato no soportado, saltando: %s", file_path.name)
                files_skipped += 1

        except Exception as e:
            # Log el tipo de error — no el mensaje completo (puede tener paths sensibles)
            logger.error(
                "Error procesando %s: %s", file_path.name, type(e).__name__
            )
            files_skipped += 1

    logger.info(
        "Procesamiento completo: %d archivos procesados, %d saltados, %d chunks totales",
        files_processed,
        files_skipped,
        len(all_chunks),
    )

    if not all_chunks:
        logger.error(
            "No se generaron chunks. Verifica que S3 tiene archivos y "
            "que ingest_config.json tiene los defaults correctos."
        )
        return

    ingest_to_chroma(
        chunks=all_chunks,
        settings=settings,
        collection_name=collection_name,
        clear_existing=clear_existing,
    )

    logger.info("Pipeline de ingesta finalizado correctamente.")


if __name__ == "__main__":
    run_ingest()