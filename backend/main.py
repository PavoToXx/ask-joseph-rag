# backend/main.py
"""
FastAPI app para "Pregúntale a Joseph".

Patrones aplicados:
- lifespan: inicializa RAGChain una sola vez al startup (no en cada request).
- Dependency injection: get_rag_chain() inyecta la chain ya inicializada.
- Pydantic models: todos los inputs y outputs validados con BaseModel.
- Rate limiting: 10 req/hora/IP via slowapi.
- Logging estructurado: timestamp, endpoint, status, latency. Nunca la query.
- /health endpoint: monitoreo y readiness checks en Azure.
"""
import logging
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.config import Settings, get_settings
from backend.rag_chain import RAGChain

# ------------------------------------------------------------------ #
#  Logging — configurar antes de cualquier import que loguee          #
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Rate limiter — instancia global, se conecta a la app en lifespan   #
# ------------------------------------------------------------------ #
limiter = Limiter(key_func=get_remote_address)

# ------------------------------------------------------------------ #
#  Lifespan — startup y shutdown de la app                            #
# ------------------------------------------------------------------ #
# Por qué lifespan y no @app.on_event("startup"):
# @app.on_event está deprecado en FastAPI 0.93+. lifespan es el estándar.
# Permite yield: todo antes del yield es startup, después es shutdown.

_rag_chain: RAGChain | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa y libera recursos al inicio y cierre de la app."""
    global _rag_chain
    settings = get_settings()

    logger.info(
        "Starting RAG Joseph API. environment='%s'", settings.environment.value
    )

    # Inicializa la chain UNA sola vez — no en cada request
    _rag_chain = RAGChain(settings=settings)
    _rag_chain.initialize()

    logger.info("Application startup complete.")
    yield  # La app corre aquí

    # Shutdown: cerrar conexiones si fuera necesario
    logger.info("Application shutting down.")


# ------------------------------------------------------------------ #
#  FastAPI app                                                         #
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Pregúntale a Joseph",
    description="RAG system para el portfolio de Joseph",
    version="0.1.0",
    lifespan=lifespan,
    # Deshabilitar docs en producción si quieres más seguridad
    # docs_url=None if os.getenv("ENVIRONMENT") != "local" else "/docs"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
# type: ignore[arg-type] — Pylance ve un conflicto entre la firma genérica de
# FastAPI (ExceptionHandler) y la firma específica de slowapi. En runtime
# funciona correctamente: slowapi está diseñado para usarse exactamente así.

app.add_middleware(
    CORSMiddleware,
    # En producción, reemplaza "*" con tu dominio exacto:
    # allow_origins=["https://tu-app.azurewebsites.net"]
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
#  Dependency injection                                                #
# ------------------------------------------------------------------ #


def get_rag_chain() -> RAGChain:
    """
    Dependency que retorna la RAGChain inicializada.

    FastAPI la inyecta en los endpoints via Depends().
    Nunca re-inicializa — usa la instancia global del lifespan.

    Returns:
        RAGChain lista para usar.

    Raises:
        HTTPException 503: Si la chain no está inicializada (startup fallido).
    """
    if _rag_chain is None:
        logger.error("RAGChain not initialized. Startup may have failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servicio no está listo todavía.",
        )
    return _rag_chain


# Alias tipado para usar con Annotated en los endpoints
RAGChainDep = Annotated[RAGChain, Depends(get_rag_chain)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

# ------------------------------------------------------------------ #
#  Schemas Pydantic                                                    #
# ------------------------------------------------------------------ #


class QuestionRequest(BaseModel):
    """Schema de entrada para el endpoint /ask."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Pregunta sobre Joseph",
        examples=["¿En qué proyectos ha trabajado Joseph?"],
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        """Rechaza strings que sean solo espacios."""
        if not v.strip():
            raise ValueError("La pregunta no puede estar vacía.")
        return v.strip()


class AnswerResponse(BaseModel):
    """Schema de salida del endpoint /ask."""

    answer: str
    sources: list[str]
    latency_ms: int
    tokens_used: int
    request_id: str  # para correlacionar logs


class HealthResponse(BaseModel):
    """Schema de salida del endpoint /health."""

    status: str
    environment: str
    rag_initialized: bool


# ------------------------------------------------------------------ #
#  Endpoints                                                           #
# ------------------------------------------------------------------ #


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health(settings: SettingsDep) -> HealthResponse:
    """
    Readiness check. Azure App Service y GitHub Actions usan este endpoint
    para saber si la app está lista antes de enviar tráfico.
    """
    return HealthResponse(
        status="ok",
        environment=settings.environment.value,
        rag_initialized=_rag_chain is not None and _rag_chain._initialized,
    )


@app.post(
    "/ask",
    response_model=AnswerResponse,
    tags=["RAG"],
    status_code=status.HTTP_200_OK,
)
@limiter.limit("10/hour")
async def ask(
    request: Request,  # requerido por slowapi para extraer la IP
    body: QuestionRequest,
    chain: RAGChainDep,
) -> AnswerResponse:
    """
    Responde una pregunta sobre Joseph usando el pipeline RAG.

    Rate limit: 10 requests/hora por IP.
    Input máximo: 500 caracteres (configurado en QuestionRequest).
    """
    # request_id permite correlacionar esta request en los logs
    # sin necesidad de loguear la query del usuario.
    request_id = str(uuid.uuid4())[:8]
    start = time.monotonic()

    logger.info(
        "Request received. request_id='%s' endpoint='/ask'", request_id
    )

    try:
        result = chain.get_answer(body.question)
    except Exception:
        # El error real ya fue logueado en rag_chain.py con nivel ERROR.
        # Aquí devolvemos un mensaje genérico — nunca stack traces al usuario.
        logger.error(
            "Unhandled error processing request. request_id='%s'", request_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ocurrió un error al procesar tu pregunta. Intenta de nuevo.",
        )

    elapsed = int((time.monotonic() - start) * 1000)
    logger.info(
        "Request completed. request_id='%s' status=200 total_ms=%d",
        request_id,
        elapsed,
    )

    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
        tokens_used=result["tokens_used"],
        request_id=request_id,
    )


@app.post(
    "/ask/stream",
    tags=["RAG"],
    status_code=status.HTTP_200_OK,
)
@limiter.limit("10/hour")
async def ask_stream(
    request: Request,
    body: QuestionRequest,
    chain: RAGChainDep,
) -> StreamingResponse:
    """
    Responde una pregunta sobre Joseph usando streaming de texto.

    Devuelve texto plano en chunks para que el frontend lo renderice
    progresivamente, estilo chat en tiempo real.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(
        "Streaming request received. request_id='%s' endpoint='/ask/stream'",
        request_id,
    )

    def generate():
        try:
            for event in chain.stream_answer(body.question):
                event_type = event.get("type", "chunk")
                payload = {k: v for k, v in event.items() if k != "type"}
                yield (
                    f"event: {event_type}\n"
                    f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                )
        except Exception:
            logger.error(
                "Unhandled streaming error. request_id='%s'", request_id
            )
            yield (
                "event: error\n"
                'data: {"message": "Ocurrió un error al procesar tu pregunta. Intenta de nuevo."}\n\n'
            )

    return StreamingResponse(generate(), media_type="text/event-stream")
