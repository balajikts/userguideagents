"""pgvector backend. Imports psycopg at module load — only import this module via
``app.tools.manual_rag.create_manual_store`` with ``VECTOR_STORE=pgvector``."""

from __future__ import annotations

import numpy as np
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from app.schemas import DeviceQuery
from app.tools.embeddings import Embedder
from app.tools.manual_rag import MODEL_MATCH_BONUS, ManualChunk, ManualHit, chunk_text_for_embedding, model_key


class PgVectorManualStore:
    """Cosine search over ``manual_chunks`` (see db/init.sql).

    Brand is a hard filter when known; the model is a soft boost (see ``model_bonus``),
    computed in SQL on the stored ``model_key`` column.
    """

    def __init__(self, dsn: str, embedder: Embedder, *, min_size: int = 1, max_size: int = 5) -> None:
        self._embedder = embedder
        self._pool = AsyncConnectionPool(dsn, min_size=min_size, max_size=max_size, open=False, configure=register_vector_async)

    async def open(self) -> None:
        await self._pool.open()

    async def close(self) -> None:
        await self._pool.close()

    async def search(self, query: DeviceQuery, k: int = 6) -> list[ManualHit]:
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
        params = {"v": np.array(vec, dtype=np.float32), "brand": query.brand, "mk": model_key(query.model), "lim": k * 3}
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql, params)).fetchall()
        hits = [
            ManualHit(
                chunk=ManualChunk(brand=r[0], model=r[1], device_type=r[2], title=r[3], url=r[4], page=r[5], content=r[6]),
                score=min(1.0, float(r[7]) + (MODEL_MATCH_BONUS if r[8] else 0.0)),
            )
            for r in rows
        ]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]

    async def add(self, chunks: list[ManualChunk]) -> int:
        if not chunks:
            return 0
        vecs = await self._embedder.embed([chunk_text_for_embedding(c) for c in chunks])
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                """INSERT INTO manual_chunks (brand, model, model_key, device_type, title, url, page, content, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (url, COALESCE(page, 0), md5(content)) DO NOTHING""",
                [
                    (c.brand, c.model, model_key(c.model), c.device_type, c.title, c.url, c.page, c.content, np.array(v, dtype=np.float32))
                    for c, v in zip(chunks, vecs)
                ],
            )
        return len(chunks)
