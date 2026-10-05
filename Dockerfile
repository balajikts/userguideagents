# One image for the API, the UI, ingestion and evals; docker-compose picks the command.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    PYTHONPATH=/srv \
    VECTOR_STORE=pgvector

WORKDIR /srv

# Dependencies first for layer caching. requirements-prod.txt = requirements.txt + psycopg/pgvector.
COPY requirements.txt requirements-prod.txt ./
RUN pip install -r requirements-prod.txt

# Bake the embedding model into the image so containers start without downloading it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY app ./app
COPY ui ./ui
COPY scripts ./scripts
COPY evals ./evals
COPY db ./db

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /srv /opt/fastembed
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "--factory", "app.api.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
