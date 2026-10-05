"""Manual retrieval: one interface, backend chosen by ``VECTOR_STORE``.

* ``chroma`` (default) — embedded, on-disk Chroma; no server or native Postgres
  driver, so it runs on locked-down Windows machines.
* ``pgvector`` — Postgres + pgvector; used in docker-compose / production.

Backends live in their own modules and are imported only inside
``create_manual_store``, so selecting Chroma never imports psycopg or pgvector.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from app.schemas import DeviceQuery
from app.tools.embeddings import Embedder

if TYPE_CHECKING:
    from app.config import Settings

MODEL_MATCH_BONUS = 0.1


class ManualChunk(BaseModel):
    brand: str
    model: str | None = None
    device_type: str | None = None
    title: str
    url: str
    page: int | None = None
    content: str


class ManualHit(BaseModel):
    chunk: ManualChunk
    score: float  # cosine similarity (+ model bonus), 0..1


class ManualStore(Protocol):
    async def open(self) -> None: ...
    async def close(self) -> None: ...
    async def search(self, query: DeviceQuery, k: int = 6) -> list[ManualHit]: ...
    async def add(self, chunks: list[ManualChunk]) -> int: ...


def model_key(s: str | None) -> str:
    """Lower-case alphanumerics: "WH-1000XM5/B" → "wh1000xm5b"."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def model_bonus(query_model: str | None, chunk_model: str | None) -> float:
    """Soft boost for the same model family; manuals often cover several variants."""
    q, c = model_key(query_model), model_key(chunk_model)
    return MODEL_MATCH_BONUS if q and c and (c.startswith(q) or q.startswith(c)) else 0.0


def chunk_text_for_embedding(c: ManualChunk) -> str:
    return f"{c.title}\n{c.content}"


def create_manual_store(settings: Settings, embedder: Embedder | None = None) -> ManualStore:
    if embedder is None:
        from app.tools.embeddings import FastEmbedEmbedder

        embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)

    if settings.vector_store == "pgvector":
        from app.tools.pgvector_store import PgVectorManualStore  # imports psycopg

        return PgVectorManualStore(settings.database_url, embedder)
    if settings.vector_store == "chroma":
        from app.tools.chroma_store import ChromaManualStore

        return ChromaManualStore(settings.chroma_path, embedder)
    raise ValueError(f"unknown VECTOR_STORE {settings.vector_store!r}; expected 'chroma' or 'pgvector'")


class InMemoryManualStore:
    """Same contract as the real stores; for tests."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._rows: list[tuple[ManualChunk, list[float]]] = []

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def add(self, chunks: list[ManualChunk]) -> int:
        vecs = await self._embedder.embed([chunk_text_for_embedding(c) for c in chunks])
        self._rows.extend(zip(chunks, vecs))
        return len(chunks)

    async def search(self, query: DeviceQuery, k: int = 6) -> list[ManualHit]:
        [q] = await self._embedder.embed([query.search_text])
        hits = [
            ManualHit(chunk=chunk, score=min(1.0, _cos(q, v) + model_bonus(query.model, chunk.model)))
            for chunk, v in self._rows
            if not query.brand or chunk.brand.lower() == query.brand.lower()
        ]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
