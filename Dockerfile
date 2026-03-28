FROM python:3.12.10-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN python -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip==24.3.1 setuptools==75.6.0 wheel==0.45.1

COPY requirements.txt .
RUN pip install -r requirements.txt


FROM python:3.12.10-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    HOME=/home/appuser \
    BACKEND_URL=http://127.0.0.1:8000 \
    REQUIRE_CHROMA_SYNC=true \
    PYTHONPATH=/app

RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser \
    && mkdir -p /app /app/chroma_db /home/appuser/.streamlit \
    && chown -R appuser:appuser /app /home/appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser backend /app/backend
COPY --chown=appuser:appuser frontend /app/frontend
COPY --chown=appuser:appuser .streamlit /app/.streamlit
COPY --chown=appuser:appuser startup.py /app/startup.py
COPY --chown=appuser:appuser requirements.txt /app/requirements.txt
COPY --chown=appuser:appuser docker/ /app/docker
RUN chmod +x /app/docker/entrypoint.sh

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); sys.exit(0)"

CMD ["/app/docker/entrypoint.sh"]
