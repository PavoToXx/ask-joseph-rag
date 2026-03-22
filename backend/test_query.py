"""
Sprint 2: Script de validación de ingesta.
Verifica que ChromaDB está instanciado correctamente y retorna resultados relevantes.
Actualizado para langchain 0.3.1+, chromadb 1.5.5+
"""

import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# ── Setup Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Cargar variables ──
load_dotenv()

try:
    AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
    
    if not all([AZURE_KEY, AZURE_ENDPOINT]):
        raise ValueError("❌ Faltan AZURE_OPENAI_KEY o AZURE_OPENAI_ENDPOINT")
    
    logger.info("✅ Variables de entorno cargadas")
    
except Exception as e:
    logger.error(f"❌ Error de configuración: {e}")
    raise

# ── Inicializar embeddings ──
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=DEPLOYMENT_NAME,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    api_version=API_VERSION
)

logger.info(f"🔧 Usando modelo: {DEPLOYMENT_NAME}")
logger.info(f"📍 ChromaDB path: {CHROMA_PATH}")


def test_chroma_connection():
    """Conecta a ChromaDB y valida que está disponible"""
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="ask-joseph-docs"
        )
        logger.info("✅ Conexión a ChromaDB exitosa")
        return db
    except Exception as e:
        logger.error(f"❌ Error conectando a ChromaDB: {e}")
        raise


def test_similarity_search(db, query: str, k: int = 3):
    """Ejecuta búsqueda de similaridad"""
    try:
        logger.info(f"🔍 Buscando: '{query}'")
        results = db.similarity_search(query, k=k)
        
        if not results:
            logger.warning("⚠️  No hay resultados para la query")
            return
        
        logger.info(f"✅ {len(results)} resultado(s) encontrado(s)\n")
        
        for i, doc in enumerate(results, 1):
            logger.info(f"{'─' * 70}")
            logger.info(f"📋 Resultado {i} (Score: relevante)")
            logger.info(f"{'─' * 70}")
            
            # Truncar content para legibilidad
            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            logger.info(f"Content:\n{content}\n")
            
            metadata = doc.metadata or {}
            source = metadata.get('source', 'desconocida')
            logger.info(f"📌 Fuente: {source}")
            logger.info()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error en similarity_search: {e}")
        raise


def main():
    """Main test pipeline"""
    logger.info("🚀 === Sprint 2: Test Query ===\n")
    
    # Test 1: Conexión
    logger.info("TEST 1: Conectar a ChromaDB")
    logger.info("─" * 70)
    db = test_chroma_connection()
    
    # Test 2: Query de prueba
    logger.info("\nTEST 2: Ejecutar similarity search")
    logger.info("─" * 70)
    
    test_queries = [
        "¿En qué proyectos ha trabajado Joseph?",
        "Cuáles son las habilidades técnicas?",
        "Experiencia en machine learning y cloud"
    ]
    
    for query in test_queries:
        try:
            results = test_similarity_search(db, query, k=2)
            if results:
                logger.info("✅ Query devolvió resultados\n")
        except Exception as e:
            logger.error(f"❌ Query falló: {e}\n")
            continue
    
    logger.info("✅ === Test completado ===")


if __name__ == "__main__":
    main()
