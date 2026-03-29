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
            logger.debug(
                "AWS auth mode: AZURE APP SERVICE (Managed Identity -> AssumeRoleWithWebIdentity)"
            )

            aws_role_arn = os.getenv("AWS_ROLE_ARN")
            if not aws_role_arn:
                raise RuntimeError("AWS_ROLE_ARN no está configurado.")

            identity_endpoint = os.getenv("IDENTITY_ENDPOINT")
            identity_header = os.getenv("IDENTITY_HEADER")

            if not identity_endpoint or not identity_header:
                raise RuntimeError(
                    "Managed Identity no disponible. Verifica que esté habilitada en App Service."
                )

            # 👇 IMPORTANTE: este resource debe coincidir con tu trust policy en AWS
            resource = os.getenv(
                "AZURE_WEB_IDENTITY_RESOURCE",
                "api://AzureADTokenExchange"
            )

            token_url = (
                f"{identity_endpoint}"
                f"?api-version=2019-08-01"
                f"&resource={resource}"
            )

            req = Request(
                token_url,
                headers={"X-IDENTITY-HEADER": identity_header},
                method="GET",
            )

            with urlopen(req, timeout=10) as resp:
                token_data = json.loads(resp.read().decode("utf-8"))
                web_identity_token = token_data["access_token"]

            sts_client = boto3.client("sts", region_name=settings.aws_region)

            assumed = sts_client.assume_role_with_web_identity(
                RoleArn=aws_role_arn,
                RoleSessionName="azure-app-service-session",
                WebIdentityToken=web_identity_token,
            )

            creds = assumed["Credentials"]

            session = boto3.Session(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            )

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
