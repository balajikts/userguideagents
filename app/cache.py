"""Redis: answer cache, clarification sessions, and per-client rate limiting."""

from __future__ import annotations

import hashlib
import json
import re

from redis.asyncio import Redis

from app.schemas import GuideAnswer

SESSION_TTL_S = 60 * 30


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


class Store:
    def __init__(self, redis: Redis, *, cache_ttl_s: int, rate_limit_per_min: int) -> None:
        self.r = redis
        self.cache_ttl_s = cache_ttl_s
        self.rate_limit = rate_limit_per_min

    # --- answer cache (single-turn questions only; key is the normalised text) ---
    @staticmethod
    def cache_key(question: str) -> str:
        return "answer:" + hashlib.sha256(_norm(question).encode()).hexdigest()

    async def get_answer(self, question: str) -> GuideAnswer | None:
        raw = await self.r.get(self.cache_key(question))
        return GuideAnswer.model_validate_json(raw) if raw else None

    async def put_answer(self, question: str, answer: GuideAnswer) -> None:
        await self.r.set(self.cache_key(question), answer.model_dump_json(), ex=self.cache_ttl_s)

    # --- clarification sessions: list of (source, text) turns ---
    async def get_session(self, session_id: str) -> list[tuple[str, str]]:
        raw = await self.r.get(f"session:{session_id}")
        return [tuple(t) for t in json.loads(raw)] if raw else []  # type: ignore[misc]

    async def save_session(self, session_id: str, turns: list[tuple[str, str]]) -> None:
        await self.r.set(f"session:{session_id}", json.dumps(turns), ex=SESSION_TTL_S)

    async def clear_session(self, session_id: str) -> None:
        await self.r.delete(f"session:{session_id}")

    # --- fixed-window rate limit ---
    async def allow(self, client_id: str) -> bool:
        key = f"rate:{client_id}"
        n = await self.r.incr(key)
        if n == 1:
            await self.r.expire(key, 60)
        return n <= self.rate_limit
