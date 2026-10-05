"""FastAPI backend.

POST /ask  {question, session_id?} → GuideAnswer + session_id
When the answer is ``needs_clarification``, send the user's reply with the same
``session_id`` and the Manager sees the whole exchange.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.agents.manager import ASSISTANT, USER, ManagerAgent
from app.cache import Store
from app.config import get_settings
from app.schemas import GuideAnswer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(None, max_length=64, pattern=r"^[A-Za-z0-9-]+$")


class AskResponse(BaseModel):
    session_id: str
    cached: bool = False
    answer: GuideAnswer


def create_app(manager: ManagerAgent | None = None, redis: Redis | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            if manager is None:
                from app.factory import build_manager

                app.state.manager = await stack.enter_async_context(build_manager(settings))
            else:
                app.state.manager = manager
            r = redis or Redis.from_url(settings.redis_url, decode_responses=True)
            app.state.store = Store(r, cache_ttl_s=settings.cache_ttl_s, rate_limit_per_min=settings.rate_limit_per_min)
            yield
            if redis is None:
                await r.aclose()

    app = FastAPI(title="Electronics User-Guide Assistant", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        try:
            await app.state.store.r.ping()
            redis_ok = True
        except Exception:
            redis_ok = False
        return {"status": "ok", "redis": redis_ok}

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest, request: Request) -> AskResponse:
        store: Store = app.state.store
        # Behind Caddy, the client IP is in X-Forwarded-For.
        client = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "anon")).split(",")[0].strip()
        if not await store.allow(client):
            raise HTTPException(429, "Too many requests; try again in a minute.")

        session_id = req.session_id or uuid.uuid4().hex
        history = await store.get_session(session_id) if req.session_id else []
        single_turn = not history

        if single_turn and (hit := await store.get_answer(req.question)):
            return AskResponse(session_id=session_id, cached=True, answer=hit)

        turns = [*history, (USER, req.question)]
        try:
            answer = await app.state.manager.ask(turns)
        except Exception:
            log.exception("pipeline failed")
            raise HTTPException(502, "The assistant failed to answer; please try again.")

        if answer.status == "needs_clarification":
            await store.save_session(session_id, [*turns, (ASSISTANT, answer.clarification_question or "")])
        else:
            await store.clear_session(session_id)
            if answer.status == "answered" and single_turn:
                await store.put_answer(req.question, answer)
        return AskResponse(session_id=session_id, answer=answer)

    return app
# Run with: uvicorn --factory app.api.main:create_app
