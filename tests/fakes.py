"""Deterministic test doubles for retrieval backends."""

import asyncio
import hashlib
import re

from app.tools.web_search import WebHit


class HashEmbedder:
    """Bag-of-words hashed into a small vector: cosine ≈ word overlap. Deterministic, no model download."""

    dim = 64

    async def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            out.append(v)
        return out


class FakeWeb:
    def __init__(self, hits_by_scope=None, delay=0.0, error=None):
        self.hits_by_scope = hits_by_scope or {}
        self.calls = []
        self.delay = delay
        self.error = error

    async def search(self, query, *, include_domains=None, max_results=6):
        self.calls.append((query, include_domains))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        key = "official" if include_domains else "open"
        return [WebHit(**h) for h in self.hits_by_scope.get(key, [])][:max_results]
