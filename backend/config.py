# backend/config.py
"""
Configuración centralizada de la aplicación.

Por qué pydantic-settings:
- Valida que todas las variables de entorno requeridas existen AL STARTUP.
- Si falta AZURE_OPENAI_API_KEY, la app falla inmediatamente con un error
  claro — no a mitad de una request del usuario.
- Nunca uses os.getenv() directo en los módulos de negocio.
"""
import logging
from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """
    Controla qué método de autenticación usa boto3 para AWS.

    - LOCAL: AWS CLI profile en ~/.aws/credentials (nunca en repo)
    - GITHUB_ACTIONS: OIDC completo vía aws-actions/configure-aws-credentials,
      inyecta credenciales temporales como env vars, boto3 las detecta solo.
    - AZURE: Managed Identity de Azure App Service asume un rol en AWS vía
      AssumeRoleWithWebIdentity. boto3 detecta AWS_WEB_IDENTITY_TOKEN_FILE
      automáticamente — cero API keys en Azure.
    """

    LOCAL = "local"
    GITHUB_ACTIONS = "github_actions"
    AZURE = "azure"


class Settings(BaseSettings):
    """Todas las configuraciones de la app, cargadas desde env vars o .env."""

    # --- Azure OpenAI ---
    # Field(...) significa "requerido". Si no existe en el entorno, falla al startup.
    azure_openai_endpoint: str = Field(..., description="URL del recurso Azure OpenAI")
    azure_openai_api_key: str = Field(..., description="API key de Azure OpenAI")
    azure_openai_api_version: str = Field(
        "2024-02-01", description="Versión de la API de Azure OpenAI"
    )
    azure_openai_embedding_deployment: str = Field(
        "text-embedding-3-small", description="Nombre del deployment de embeddings"
    )
    azure_openai_chat_deployment: str = Field(
        "gpt-4o-mini", description="Nombre del deployment del LLM"
    )

    # --- AWS ---
    aws_region: str = Field("us-east-1", description="Región de AWS")
    aws_bucket_name: str = Field(..., description="Nombre del bucket S3 para documentos")
    aws_profile: str = Field(
        "default",
        description="AWS CLI profile (solo se usa cuando ENVIRONMENT=local)",
    )

    # --- App ---
    environment: Environment = Field(
        Environment.LOCAL, description="Entorno de ejecución"
    )
    chroma_path: str = Field("./chroma_db", description="Ruta local de ChromaDB")
    # Nombre de la colección donde ingest.py guardó los documentos.
    # Cámbialo aquí o en .env si tu ingest usó otro nombre.
    chroma_collection_name: str = Field(
        "ask-joseph-docs", description="Nombre de la colección en ChromaDB"
    )
    max_question_length: int = Field(
        500, description="Máximo de caracteres permitidos en una pregunta"
    )
    rate_limit: str = Field(
        "10/hour", description="Rate limit para el endpoint /ask"
    )

    # streamlit
    backend_url: str = Field(..., description="URL del backend")

    # pydantic-settings config: lee desde .env en desarrollo local
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,  # AZURE_OPENAI_KEY y azure_openai_key son equivalentes
    }

    @field_validator("azure_openai_endpoint")
    @classmethod
    def validate_endpoint_format(cls, v: str) -> str:
        """Valida que el endpoint tenga formato correcto y normaliza trailing slash."""
        if not v.startswith("https://"):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT debe comenzar con https://. "
                f"Valor recibido: '{v[:20]}...'"
            )
        return v.rstrip("/")  # normaliza: evita doble slash en las llamadas a la API

    @field_validator("max_question_length")
    @classmethod
    def validate_max_length(cls, v: int) -> int:
        """Previene configuraciones que anulen el rate limiting por tamaño."""
        if v < 10 or v > 2000:
            raise ValueError("MAX_QUESTION_LENGTH debe estar entre 10 y 2000.")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retorna la instancia singleton de Settings.

    lru_cache garantiza que pydantic-settings lee el .env solo una vez,
    no en cada request. Úsala con FastAPI Depends().

    Returns:
        Settings: Configuración validada de la aplicación.
    """
    logger.info("Loading application settings...")
    # type: ignore[call-arg] — Pylance no sabe que pydantic-settings sobreescribe
    # __init__ para leer los valores desde variables de entorno y .env.
    # En runtime no se pasan argumentos: Settings() lee AZURE_OPENAI_API_KEY,
    # AWS_S3_BUCKET, etc. del entorno automáticamente. No es un error real.
    return Settings()  # type: ignore[call-arg]