"""Wire real dependencies into the Manager (used by the API and evals)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.agents.filter import FilterAgent
from app.agents.manager import ManagerAgent
from app.agents.query_verifier import QueryVerifierAgent
from app.agents.searcher import SearcherAgent
from app.config import Settings, get_settings
from app.llm import get_model_client
from app.tools.embeddings import FastEmbedEmbedder
from app.tools.manual_store import PgVectorManualStore
from app.tools.web_search import TavilySearchClient

log = logging.getLogger(__name__)


@asynccontextmanager
async def build_manager(settings: Settings | None = None) -> AsyncIterator[ManagerAgent]:
    s = settings or get_settings()
    store = PgVectorManualStore(s.database_url, FastEmbedEmbedder(s.embedding_model, s.embedding_dim))
    try:
        await store.open()
    except Exception:
        log.exception("manual store unavailable; continuing with web search only")
        store = None  # type: ignore[assignment]
    web = TavilySearchClient(s.tavily_api_key, timeout=s.search_timeout_s) if s.tavily_api_key else None
    if web is None:
        log.warning("TAVILY_API_KEY not set; web search disabled")

    # Extraction is easy → low effort; writing grounded steps → medium.
    verifier_llm = get_model_client("low", s)
    filter_llm = get_model_client("medium", s)
    manager = ManagerAgent(
        QueryVerifierAgent(verifier_llm, clarify_threshold=s.clarify_threshold),
        SearcherAgent(manual_store=store, web_search=web, manual_k=s.manual_store_k,
                      web_k=s.web_search_results, timeout_s=s.search_timeout_s),
        FilterAgent(filter_llm, max_sources=s.max_sources),
        grounding_min_overlap=s.grounding_min_overlap,
    )
    try:
        yield manager
    finally:
        await verifier_llm.close()
        await filter_llm.close()
        if web:
            await web.aclose()
        if store:
            await store.close()
