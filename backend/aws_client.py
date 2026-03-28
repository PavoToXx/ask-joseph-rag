# backend/aws_client.py
"""
Cliente S3 con los tres métodos de autenticación AWS según el entorno.

NUNCA se almacenan credenciales en código ni en el repo.

Modos de autenticación:
┌─────────────────┬─────────────────────────────────────────────────────────┐
│ ENVIRONMENT     │ Cómo se autentica boto3                                 │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ local           │ AWS CLI profile en ~/.aws/credentials                   │
│ github_actions  │ OIDC: aws-actions/configure-aws-credentials inyecta     │
│                 │ AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + SESSION_TOKEN│
│                 │ boto3 los detecta automáticamente del entorno           │
│ azure           │ Managed Identity: Azure inyecta AWS_WEB_IDENTITY_TOKEN_ │
│                 │ FILE + AWS_ROLE_ARN, boto3 hace AssumeRoleWithWebIdentity│
└─────────────────┴─────────────────────────────────────────────────────────┘
"""
import logging
from typing import Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from backend.config import Environment, Settings

logger = logging.getLogger(__name__)


def get_s3_client(settings: Settings, bucket_name: Optional[str] = None) -> Any:
    """
    Crea un cliente S3 usando el método de autenticación correcto para el entorno.

    Args:
        settings: Configuración validada de la aplicación.

    Returns:
        boto3 S3 client listo para usar.

    Raises:
        RuntimeError: Si boto3 no puede encontrar credenciales válidas.
    """
    # Asegura que `target_bucket` está siempre inicializado, incluso si ocurre
    # una excepción antes de entrar en el bloque `try` (evita advertencias
    # de linters/static analyzers sobre variable posiblemente no inicializada).
    target_bucket = bucket_name or settings.aws_bucket_name

    try:
        if settings.environment == Environment.LOCAL:
            # Usa el profile de ~/.aws/credentials — NUNCA hardcodeado en repo.
            # aws_profile viene de .env local (que también está en .gitignore).
            logger.debug(
                "AWS auth mode: LOCAL profile='%s'", settings.aws_profile
            )
            session = boto3.Session(profile_name=settings.aws_profile)

        elif settings.environment == Environment.GITHUB_ACTIONS:
            # OIDC completo: el workflow de GitHub Actions usa
            # aws-actions/configure-aws-credentials que asume un rol en AWS
            # y setea las variables de entorno temporales.
            # boto3 las detecta solo — no hay nada que configurar aquí.
            logger.debug("AWS auth mode: GITHUB_ACTIONS (OIDC, env vars auto-detected)")
            session = boto3.Session()

        else:
            # ENVIRONMENT=azure: Azure App Service Managed Identity
            # Azure inyecta AWS_WEB_IDENTITY_TOKEN_FILE y AWS_ROLE_ARN.
            # boto3 hace AssumeRoleWithWebIdentity automáticamente.
            # Cero API keys almacenadas en Azure ni en el repo.
            logger.debug(
                "AWS auth mode: AZURE (Managed Identity → AssumeRoleWithWebIdentity)"
            )
            session = boto3.Session()

        client = session.client("s3", region_name=settings.aws_region)

        # Validación temprana: verifica que las credenciales funcionan
        # antes de que llegue la primera request.
        client.head_bucket(Bucket=target_bucket)
        logger.info(
            "S3 client initialized. bucket='%s' region='%s'",
            target_bucket,
            settings.aws_region,
        )
        return client

    except NoCredentialsError as e:
        # Error claro con contexto del entorno para facilitar el debug
        raise RuntimeError(
            f"No se encontraron credenciales AWS para ENVIRONMENT='{settings.environment}'. "
            f"Revisa tu ~/.aws/credentials (local) o las variables de entorno (CI/Azure)."
        ) from e

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        # No loguear el mensaje completo: puede contener info del bucket
        logger.error("S3 ClientError al validar bucket. code='%s'", error_code)
        raise RuntimeError(
            f"No se pudo acceder al bucket S3 '{target_bucket}'. "
            f"Verifica que existe y que el rol tiene permisos s3:HeadBucket."
        ) from e

    except BotoCoreError as e:
        logger.error("Error de boto3 al inicializar S3 client: %s", type(e).__name__)
        raise RuntimeError("Error interno al conectar con AWS S3.") from e
