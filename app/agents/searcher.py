"""Agent 2 — Searcher.

No LLM: given a ``DeviceQuery`` it runs three retrievals concurrently and merges
them into ``SearchResults``:

* manual store (pgvector) — chunks of ingested official manuals
* web search restricted to the brand's official domains
* open web search for "<brand> <model> manual <task>" (catches mirrors and
  support pages on domains we don't know about; the Filter ranks them lower)

A failing or slow backend is recorded in ``errors`` and never sinks the others.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, StructuredMessage
from autogen_core import CancellationToken

from app.schemas import DeviceQuery, SearchResults, Source, SourceOrigin
from app.tools.manual_store import ManualStore
from app.tools.web_search import WebSearchClient, official_domains


class SearcherAgent(BaseChatAgent):
    def __init__(
        self,
        *,
        manual_store: ManualStore | None,
        web_search: WebSearchClient | None,
        manual_k: int = 6,
        web_k: int = 6,
        timeout_s: float = 15.0,
        name: str = "searcher",
    ) -> None:
        super().__init__(name, description="Searches official manuals (web + pgvector) in parallel.")
        self._store = manual_store
        self._web = web_search
        self._manual_k = manual_k
        self._web_k = web_k
        self._timeout = timeout_s

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (StructuredMessage[SearchResults],)

    async def _manual(self, q: DeviceQuery) -> list[Source]:
        if not self._store:
            return []
        hits = await self._store.search(q, k=self._manual_k)
        return [
            Source(
                id="", title=h.chunk.title, url=h.chunk.url, snippet=h.chunk.content,
                origin=SourceOrigin.MANUAL_STORE, retrieval_score=h.score, page=h.chunk.page,
            )
            for h in hits
        ]

    async def _web_query(self, text: str, domains: list[str] | None) -> list[Source]:
        if not self._web:
            return []
        hits = await self._web.search(text, include_domains=domains, max_results=self._web_k)
        return [
            Source(id="", title=h.title, url=h.url, snippet=h.content, origin=SourceOrigin.WEB, retrieval_score=h.score)
            for h in hits
        ]

    async def search(self, q: DeviceQuery) -> SearchResults:
        device = " ".join(p for p in (q.brand, q.model) if p) or q.device_type.value
        jobs: dict[str, asyncio.Future] = {"manual_store": self._manual(q)}
        domains = official_domains(q.brand)
        if domains:
            jobs["web_official"] = self._web_query(f"{device} {q.question}", domains)
        jobs["web_open"] = self._web_query(f"{device} user manual {q.question}", None)

        results = await asyncio.gather(
            *(asyncio.wait_for(j, self._timeout) for j in jobs.values()), return_exceptions=True
        )
        sources: list[Source] = []
        errors: list[str] = []
        for name, res in zip(jobs, results):
            if isinstance(res, BaseException):
                errors.append(f"{name}: {type(res).__name__}: {res}"[:300])
            else:
                sources.extend(res)
        # Provisional ids; the Filter re-numbers after ranking.
        sources = [s.model_copy(update={"id": f"R{i}"}) for i, s in enumerate(sources, 1)]
        return SearchResults(query=q, sources=sources, errors=errors)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        query = next(
            (m.content for m in reversed(messages) if isinstance(m, StructuredMessage) and isinstance(m.content, DeviceQuery)),
            None,
        )
        if query is None:
            raise ValueError("SearcherAgent expects a StructuredMessage[DeviceQuery]")
        results = await self.search(query)
        return Response(chat_message=StructuredMessage[SearchResults](content=results, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
