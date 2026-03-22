"""
Sprint 2: Ingesta de documentos desde S3 y vectorización con ChromaDB
Actualizado para langchain-community 0.3.1+, chromadb 1.5.5+, boto3 1.42.73+
Seguridad: Todas las credenciales cargadas desde variables de entorno
"""

import boto3
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# == Setup Logging ==
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Cargar variables de entorno ────────────────────────
load_dotenv()

# ── Configuración con validación ──────────────────────
try:
    S3_BUCKET = os.getenv("S3_BUCKET_NAME")
    AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
    
    # Validar variables esenciales
    if not all([S3_BUCKET, AZURE_KEY, AZURE_ENDPOINT]):
        raise ValueError("❌ Faltan variables de entorno: S3_BUCKET_NAME, AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT")
    
    logger.info(f"✅ Configuración cargada correctamente")
except Exception as e:
    logger.error(f"❌ Error en configuración: {e}")
    raise

# ── Inicializar embeddings con Azure OpenAI ───────────
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=DEPLOYMENT_NAME,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    api_version=API_VERSION
)

# ── 1. Descargar docs desde S3 ────────────────────────
def download_from_s3(local_dir: str = "./docs_temp") -> str:
    """
    Descarga todos los documentos del bucket S3 configurado.
    
    Args:
        local_dir: Directorio local donde guardar los archivos
        
    Returns:
        str: Ruta del directorio con archivos descargados
    """
    try:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        s3 = boto3.client(
            "s3",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
        
        logger.info(f"📥 Listando archivos de S3: {S3_BUCKET}")
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET)
        
        count = 0
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                key = obj['Key']
                local_path = os.path.join(local_dir, key.split('/')[-1])
                logger.info(f"  ⬇️  Descargando: {key}")
                s3.download_file(S3_BUCKET, key, local_path)
                count += 1
        
        logger.info(f"✅ {count} archivo(s) descargado(s)")
        return local_dir
        
    except Exception as e:
        logger.error(f"❌ Error descargando de S3: {e}")
        raise

# ── 2. Cargar documentos con LangChain ────────────────────
def load_documents(local_dir: str) -> list:
    """
    Carga documentos .txt, .md, y .html desde el directorio.
    Compatible con langchain-community 0.3.1+
    
    Args:
        local_dir: Directorio con archivos a cargar
        
    Returns:
        list: Lista de documentos cargados
    """
    docs = []
    supported_formats = {'.txt', '.md', '.html'}
    
    try:
        file_count = 0
        for filename in os.listdir(local_dir):
            ext = Path(filename).suffix.lower()
            path = os.path.join(local_dir, filename)
            
            if not os.path.isfile(path):
                continue
            
            if ext not in supported_formats:
                logger.debug(f"  ⏭️  Formato no soportado: {filename} ({ext})")
                continue
            
            try:
                if ext in ['.txt', '.md']:
                    loader = TextLoader(path, encoding="utf-8")
                elif ext == '.html':
                    loader = UnstructuredHTMLLoader(path)
                
                loaded = loader.load()
                docs.extend(loaded)
                file_count += 1
                logger.info(f"  📄 Cargado: {filename} ({len(loaded)} chunk(s))")
                
            except Exception as e:
                logger.warning(f"  ⚠️  Error cargando {filename}: {e}")
                continue
        
        logger.info(f"✅ Total documentos cargados: {file_count}")
        return docs
        
    except Exception as e:
        logger.error(f"❌ Error en load_documents: {e}")
        raise

# ── 3. Chunking ───────────────────────────────────────
def split_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Divide documentos en chunks para vectorización.
    Compatible con langchain-text-splitters 0.2.0+
    
    Args:
        docs: Lista de documentos Document
        chunk_size: Tamaño de cada chunk
        chunk_overlap: Solapamiento entre chunks
        
    Returns:
        list: Lista de chunks (Document objects)
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"✅ Chunking completo: {len(chunks)} chunks generados")
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Error en split_documents: {e}")
        raise

# ── 4. Vectorizar y guardar en ChromaDB ───────────────────
def ingest_to_chroma(chunks: list) -> Chroma:
    """
    Vectoriza chunks y los almacena en ChromaDB.
    Compatible con chromadb 1.5.5+ y langchain-community 0.3.1+
    
    Args:
        chunks: Lista de documentos a vectorizar
        
    Returns:
        Chroma: Vector store instance
    """
    try:
        if not chunks:
            raise ValueError("❌ No hay chunks para vectorizar")
        
        logger.info(f"🔄 Vectorizando {len(chunks)} chunks...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            collection_name="ask-joseph-docs"
        )
        
        logger.info(f"✅ ChromaDB guardado en: {CHROMA_PATH}")
        logger.info(f"📊 Collection creada: ask-joseph-docs")
        return vectorstore
        
    except Exception as e:
        logger.error(f"❌ Error en ingest_to_chroma: {e}")
        raise


# ── Main ───────────────────────────────────────────────
def main():
    """Pipeline principal de ingesta"""
    logger.info("🚀 === Iniciando Sprint 2: Ingesta RAG ===")
    
    try:
        # 1. Download
        local_dir = download_from_s3()
        
        # 2. Load
        docs = load_documents(local_dir)
        if not docs:
            logger.warning("⚠️  No hay documentos para procesar")
            return
        
        # 3. Split
        chunks = split_documents(docs)
        
        # 4. Ingest
        ingest_to_chroma(chunks)
        
        logger.info("✅ === Sprint 2: Ingesta completada exitosamente ===")
        
    except Exception as e:
        logger.error(f"❌ === Error fatal en pipeline: {e}")
        raise


if __name__ == "__main__":
    main()