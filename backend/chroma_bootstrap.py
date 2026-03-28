"""Helpers to validate and synchronize persisted ChromaDB data."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

from backend.aws_client import get_s3_client
from backend.config import Settings

logger = logging.getLogger(__name__)

CHROMA_SQLITE_FILENAME = "chroma.sqlite3"


def is_chroma_ready(chroma_path: str | Path) -> bool:
    """Return True when the directory looks like a persisted Chroma database."""
    path = Path(chroma_path)
    if not path.exists() or not path.is_dir():
        return False

    sqlite_file = path / CHROMA_SQLITE_FILENAME
    if not sqlite_file.is_file():
        return False

    return any(child.name != CHROMA_SQLITE_FILENAME for child in path.iterdir())


def ensure_chroma_ready(chroma_path: str | Path) -> Path:
    """Fail loudly when the expected persisted Chroma files are not present."""
    path = Path(chroma_path)
    if not is_chroma_ready(path):
        raise RuntimeError(
            "ChromaDB no existe o esta incompleta en "
            f"'{path}'. El servicio no puede arrancar sin la base vectorial."
        )
    return path


def download_chroma_from_s3(settings: Settings) -> Path:
    """Download a persisted Chroma directory from S3 and replace the local copy."""
    bucket_name = settings.aws_chroma_bucket_name or settings.aws_bucket_name
    prefix = settings.aws_chroma_prefix
    if not prefix:
        raise RuntimeError(
            "AWS_CHROMA_PREFIX es obligatorio cuando REQUIRE_CHROMA_SYNC=true."
        )

    destination = Path(settings.chroma_path)
    parent = destination.parent if destination.parent != Path("") else Path(".")
    temp_dir = parent / f".{destination.name}.download"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    s3_client = get_s3_client(settings, bucket_name=bucket_name)
    normalized_prefix = f"{prefix}/"
    downloaded_files = 0

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=normalized_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue

                relative_key = key[len(normalized_prefix) :]
                if not relative_key:
                    continue

                local_path = temp_dir / relative_key
                local_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading s3://%s/%s", bucket_name, key)
                s3_client.download_file(bucket_name, key, str(local_path))
                downloaded_files += 1
    except ClientError as exc:
        error_code = exc.response["Error"].get("Code", "Unknown")
        raise RuntimeError(
            f"No se pudo descargar ChromaDB desde s3://{bucket_name}/{prefix}. "
            f"S3 devolvio code='{error_code}'."
        ) from exc
    except BotoCoreError as exc:
        raise RuntimeError("Error interno al descargar ChromaDB desde S3.") from exc

    if downloaded_files == 0:
        raise RuntimeError(
            f"No se encontro ningun archivo en s3://{bucket_name}/{prefix}."
        )

    if destination.exists():
        shutil.rmtree(destination)
    temp_dir.replace(destination)

    logger.info(
        "ChromaDB synchronized from S3. files=%d destination='%s'",
        downloaded_files,
        destination,
    )
    return ensure_chroma_ready(destination)
