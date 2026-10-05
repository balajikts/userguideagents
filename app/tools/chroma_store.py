"""Chroma backend: embedded, persisted to a local directory, no server needed."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from app.schemas import DeviceQuery
from app.tools.embeddings import Embedder
from app.tools.manual_rag import ManualChunk, ManualHit, chunk_text_for_embedding, model_bonus

COLLECTION = "manual_chunks"


def chunk_id(c: ManualChunk) -> str:
    """Same identity as the pgvector unique index (url, page, content), so re-ingesting is a no-op."""
    return hashlib.sha256(f"{c.url}\x00{c.page or 0}\x00{c.content}".encode()).hexdigest()


def _metadata(c: ManualChunk) -> dict[str, Any]:
    md = {"brand": c.brand, "brand_lc": c.brand.lower(), "title": c.title, "url": c.url,
          "model": c.model, "device_type": c.device_type, "page": c.page}
    return {k: v for k, v in md.items() if v is not None}


class ChromaManualStore:
    def __init__(self, path: str, embedder: Embedder) -> None:
        self._path = path
        self._embedder = embedder
        self._col = None

    def _collection(self):
        if self._col is None:
            raise RuntimeError("ChromaManualStore used before open()")
        return self._col

    async def open(self) -> None:
        import chromadb

        def _open():
            client = chromadb.PersistentClient(path=self._path)
            # We pass our own embeddings, so no Chroma embedding function (and no model download).
            return client.get_or_create_collection(
                COLLECTION, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
            )

        self._col = await asyncio.to_thread(_open)

    async def close(self) -> None:
        self._col = None  # PersistentClient writes through; nothing to flush

    async def add(self, chunks: list[ManualChunk]) -> int:
        if not chunks:
            return 0
        vecs = await self._embedder.embed([chunk_text_for_embedding(c) for c in chunks])
        # Dedupe within the batch too: Chroma rejects duplicate ids in one upsert.
        unique = {chunk_id(c): (c, v) for c, v in zip(chunks, vecs)}
        col = self._collection()
        await asyncio.to_thread(
            col.upsert,
            ids=list(unique),
            embeddings=[v for _, v in unique.values()],
            documents=[c.content for c, _ in unique.values()],
            metadatas=[_metadata(c) for c, _ in unique.values()],
        )
        return len(chunks)

    async def search(self, query: DeviceQuery, k: int = 6) -> list[ManualHit]:
        col = self._collection()
        if await asyncio.to_thread(col.count) == 0:
            return []
        [vec] = await self._embedder.embed([query.search_text])
        res = await asyncio.to_thread(
            col.query,
            query_embeddings=[vec],
            n_results=k * 3,
            where={"brand_lc": query.brand.lower()} if query.brand else None,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, md, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunk = ManualChunk(
                brand=md["brand"], model=md.get("model"), device_type=md.get("device_type"),
                title=md["title"], url=md["url"], page=md.get("page"), content=doc,
            )
            sim = 1.0 - float(dist)  # cosine distance → similarity
            hits.append(ManualHit(chunk=chunk, score=max(0.0, min(1.0, sim + model_bonus(query.model, chunk.model)))))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]
