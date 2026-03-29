# backend/aws_client.py
"""
S3 client with environment-specific AWS auth modes.

Credentials are never hardcoded in source code.
"""

import logging
import os
from typing import Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from backend.config import Environment, Settings

logger = logging.getLogger(__name__)


def _has_env_var(name: str) -> bool:
    """Return True when an environment variable exists and is not blank."""
    value = os.getenv(name)
    return value is not None and value.strip() != ""


def get_s3_client(settings: Settings, bucket_name: Optional[str] = None) -> Any:
    """
    Create an S3 client using the expected auth flow for the current environment.

    Args:
        settings: Validated application settings.

    Returns:
        boto3 S3 client ready to use.

    Raises:
        RuntimeError: If valid credentials cannot be resolved.
    """
    target_bucket = bucket_name or settings.aws_bucket_name

    try:
        if settings.environment == Environment.LOCAL:
            logger.debug("AWS auth mode: LOCAL profile='%s'", settings.aws_profile)
            session = boto3.Session(profile_name=settings.aws_profile)

        elif settings.environment == Environment.GITHUB_ACTIONS:
            # aws-actions/configure-aws-credentials injects short-lived env creds.
            logger.debug("AWS auth mode: GITHUB_ACTIONS (OIDC env vars auto-detected)")
            session = boto3.Session()

        else:
            # ENVIRONMENT=azure expects one of:
            # 1) OIDC web identity vars for AssumeRoleWithWebIdentity
            # 2) Standard AWS access key env vars
            web_identity_ready = _has_env_var("AWS_WEB_IDENTITY_TOKEN_FILE") and _has_env_var(
                "AWS_ROLE_ARN"
            )
            static_keys_ready = _has_env_var("AWS_ACCESS_KEY_ID") and _has_env_var(
                "AWS_SECRET_ACCESS_KEY"
            )

            if not web_identity_ready and not static_keys_ready:
                raise RuntimeError(
                    "ENVIRONMENT='azure' sin credenciales AWS detectables. "
                    "Configura AWS_WEB_IDENTITY_TOKEN_FILE + AWS_ROLE_ARN (OIDC) "
                    "o AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN si aplica)."
                )

            logger.debug("AWS auth mode: AZURE (credentials from environment)")
            session = boto3.Session()

        client = session.client("s3", region_name=settings.aws_region)

        # Early validation so startup fails fast with a clear message.
        client.head_bucket(Bucket=target_bucket)
        logger.info(
            "S3 client initialized. bucket='%s' region='%s'",
            target_bucket,
            settings.aws_region,
        )
        return client

    except NoCredentialsError as exc:
        raise RuntimeError(
            f"No se encontraron credenciales AWS para ENVIRONMENT='{settings.environment}'. "
            "Revisa las variables de entorno del runtime."
        ) from exc

    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        logger.error("S3 ClientError while validating bucket. code='%s'", error_code)
        raise RuntimeError(
            f"No se pudo acceder al bucket S3 '{target_bucket}'. "
            "Verifica existencia y permisos (incluyendo s3:HeadBucket)."
        ) from exc

    except BotoCoreError as exc:
        logger.error("boto3 initialization error: %s", type(exc).__name__)
        raise RuntimeError("Error interno al conectar con AWS S3.") from exc
