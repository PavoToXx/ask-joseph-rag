"""Centralized application settings."""

import logging
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Execution environments supported by the app."""

    LOCAL = "local"
    GITHUB_ACTIONS = "github_actions"
    AZURE = "azure"


class Settings(BaseSettings):
    """Application settings loaded from env vars or a local .env file."""

    azure_openai_endpoint: str = Field(..., description="Azure OpenAI resource URL")
    azure_openai_api_key: str = Field(..., description="Azure OpenAI API key")
    azure_openai_api_version: str = Field(
        "2024-02-01", description="Azure OpenAI API version"
    )
    azure_openai_embedding_deployment: str = Field(
        "text-embedding-3-small",
        description="Azure OpenAI embedding deployment name",
    )
    azure_openai_chat_deployment: str = Field(
        "gpt-4o-mini", description="Azure OpenAI chat deployment name"
    )

    aws_region: str = Field("us-east-1", description="AWS region")
    aws_bucket_name: str = Field(..., description="S3 bucket for source documents")
    aws_chroma_bucket_name: Optional[str] = Field(
        None,
        description="Optional S3 bucket for persisted ChromaDB. Falls back to AWS_BUCKET_NAME.",
    )
    aws_chroma_prefix: Optional[str] = Field(
        None,
        description="S3 prefix where the persisted ChromaDB directory is stored.",
    )
    aws_profile: str = Field(
        "default",
        description="AWS CLI profile used only when ENVIRONMENT=local",
    )

    environment: Environment = Field(
        Environment.LOCAL, description="Execution environment"
    )
    require_chroma_sync: bool = Field(
        False,
        description="If true, startup.py rebuilds ChromaDB from source documents in S3 before the app starts.",
    )
    chroma_path: str = Field("./chroma_db", description="Local ChromaDB path")
    chroma_collection_name: str = Field(
        "ask-joseph-docs", description="Chroma collection name"
    )
    max_question_length: int = Field(
        500, description="Maximum question length in characters"
    )
    rate_limit: str = Field("10/hour", description="Rate limit for /ask")
    backend_url: str = Field(..., description="Backend URL used by Streamlit")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @field_validator("azure_openai_endpoint")
    @classmethod
    def validate_endpoint_format(cls, value: str) -> str:
        """Ensure the endpoint is HTTPS and normalized."""
        if not value.startswith("https://"):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT debe comenzar con https://. "
                f"Valor recibido: '{value[:20]}...'"
            )
        return value.rstrip("/")

    @field_validator("max_question_length")
    @classmethod
    def validate_max_length(cls, value: int) -> int:
        """Guardrail for user input limits."""
        if value < 10 or value > 2000:
            raise ValueError("MAX_QUESTION_LENGTH debe estar entre 10 y 2000.")
        return value

    @field_validator("aws_chroma_prefix")
    @classmethod
    def normalize_chroma_prefix(cls, value: Optional[str]) -> Optional[str]:
        """Normalize optional S3 prefixes to a stable form."""
        if value is None:
            return None

        normalized = value.strip().strip("/")
        return normalized or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated Settings instance."""
    logger.info("Loading application settings...")
    return Settings()  # type: ignore[call-arg]
