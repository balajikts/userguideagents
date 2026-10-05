"""Local text embeddings (fastembed / ONNX) so the VPS needs no embedding API."""

from __future__ import annotations

import asyncio
from typing import Protocol


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def run() -> list[list[float]]:
            return [v.tolist() for v in self._load().embed(texts)]

        return await asyncio.to_thread(run)
