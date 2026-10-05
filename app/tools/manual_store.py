"""Manual chunk store: pgvector in production, in-memory for dev/tests."""

from __future__ import annotations

import math
from typing import Protocol

from pydantic import BaseModel

from app.schemas import DeviceQuery
from app.tools.embeddings import Embedder


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
    score: float  # cosine similarity, 0..1


class ManualStore(Protocol):
    async def search(self, query: DeviceQuery, k: int = 6) -> list[ManualHit]: ...
    async def add(self, chunks: list[ManualChunk]) -> int: ...


def _model_key(s: str | None) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


class PgVectorManualStore:
    """Cosine search over ``manual_chunks`` (see db/init.sql).

    Brand is a hard filter when known. The model is a soft boost: manuals often
    cover a model family ("WH-1000XM5" vs "WH1000XM5/B"), so we compare
    alphanumeric-only keys and add a bonus instead of excluding near-misses.
    """

    def __init__(self, dsn: str, embedder: Embedder, *, min_size: int = 1, max_size: int = 5) -> None:
        from psycopg_pool import AsyncConnectionPool

        self._embedder = embedder
        self._pool = AsyncConnectionPool(dsn, min_size=min_size, max_size=max_size, open=False, configure=self._configure)

    @staticmethod
    async def _configure(conn) -> None:
        from pgvector.psycopg import register_vector_async

        await register_vector_async(conn)

    async def open(self) -> None:
        await self._pool.open()

    async def close(self) -> None:
        await self._pool.close()

    async def search(self, query: DeviceQuery, k: int = 6) -> list[ManualHit]:
        import numpy as np

        [vec] = await self._embedder.embed([query.search_text])
        sql = """
            SELECT brand, model, device_type, title, url, page, content,
                   1 - (embedding <=> %(v)s) AS sim,
                   (model_key <> '' AND %(mk)s <> '' AND (model_key LIKE %(mk)s || '%%' OR %(mk)s LIKE model_key || '%%')) AS model_match
            FROM manual_chunks
            WHERE (%(brand)s::text IS NULL OR lower(brand) = lower(%(brand)s))
            ORDER BY embedding <=> %(v)s
            LIMIT %(lim)s
        """
        params = {"v": np.array(vec, dtype=np.float32), "brand": query.brand, "mk": _model_key(query.model), "lim": k * 3}
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql, params)).fetchall()
        hits = [
            ManualHit(
                chunk=ManualChunk(brand=r[0], model=r[1], device_type=r[2], title=r[3], url=r[4], page=r[5], content=r[6]),
                score=min(1.0, float(r[7]) + (0.1 if r[8] else 0.0)),
            )
            for r in rows
        ]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]

    async def add(self, chunks: list[ManualChunk]) -> int:
        import numpy as np

        if not chunks:
            return 0
        vecs = await self._embedder.embed([f"{c.title}\n{c.content}" for c in chunks])
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                """INSERT INTO manual_chunks (brand, model, model_key, device_type, title, url, page, content, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (url, COALESCE(page, 0), md5(content)) DO NOTHING""",
                [
                    (c.brand, c.model, _model_key(c.model), c.device_type, c.title, c.url, c.page, c.content, np.array(v, dtype=np.float32))
                    for c, v in zip(chunks, vecs)
                ],
            )
        return len(chunks)


class InMemoryManualStore:
    """Same contract as the pgvector store; for tests and running without Postgres."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._rows: list[tuple[ManualChunk, list[float]]] = []

    async def add(self, chunks: list[ManualChunk]) -> int:
        vecs = await self._embedder.embed([f"{c.title}\n{c.content}" for c in chunks])
        self._rows.extend(zip(chunks, vecs))
        return len(chunks)

    async def search(self, query: DeviceQuery, k: int = 6) -> list[ManualHit]:
        [q] = await self._embedder.embed([query.search_text])
        mk = _model_key(query.model)
        hits = []
        for chunk, v in self._rows:
            if query.brand and chunk.brand.lower() != query.brand.lower():
                continue
            ck = _model_key(chunk.model)
            bonus = 0.1 if mk and ck and (ck.startswith(mk) or mk.startswith(ck)) else 0.0
            hits.append(ManualHit(chunk=chunk, score=min(1.0, _cos(q, v) + bonus)))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
