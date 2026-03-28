"""Container startup bootstrap for runtime ChromaDB synchronization."""

import logging
import sys

from dotenv import load_dotenv

from backend.chroma_bootstrap import download_chroma_from_s3
from backend.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _is_missing_chroma_error(error: RuntimeError) -> bool:
    """Return True when a runtime error indicates missing Chroma files in S3.

    Args:
        error: RuntimeError raised during S3 download.

    Returns:
        True when the error appears to be caused by absent Chroma artifacts.
    """
    message = str(error).lower()
    missing_markers = (
        "no se encontro ningun archivo",
        "not found",
        "nosuchkey",
    )
    return any(marker in message for marker in missing_markers)


def main() -> int:
    """Validate env vars, download ChromaDB from S3, and fail loudly.

    Returns:
        Exit code where 0 means success and 1 means failure.
    """
    load_dotenv(override=False)
    get_settings.cache_clear()

    try:
        settings = get_settings()
        logger.info(
            "Startup bootstrap running. environment='%s'",
            settings.environment.value,
        )
        download_chroma_from_s3(settings)
        logger.info("Startup bootstrap completed successfully.")
        return 0
    except RuntimeError as error:
        if _is_missing_chroma_error(error):
            logger.error(
                "ChromaDB not found in S3. Run ingest.py locally first, then re-deploy."
            )
            return 1
        logger.exception("Startup bootstrap failed while downloading ChromaDB.")
        return 1
    except Exception:
        logger.exception("Startup bootstrap failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
