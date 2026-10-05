import asyncio
import time

import httpx
import pytest
import respx
from autogen_agentchat.messages import StructuredMessage
from autogen_core import CancellationToken

from app.agents.searcher import SearcherAgent
from app.schemas import DeviceQuery, SearchResults, SourceOrigin
from app.tools.manual_rag import InMemoryManualStore, ManualChunk
from app.tools.web_search import TavilySearchClient, domain_matches, official_domains
from tests.fakes import FakeWeb, HashEmbedder


Q = DeviceQuery(brand="Sony", model="WH-1000XM5", device_type="headphones", question="factory reset", confidence=0.95)

OFFICIAL = [{"title": "Initializing the headset", "url": "https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/reset.html",
             "content": "Hold the power and custom buttons for 7 seconds.", "score": 0.9}]
OPEN = [{"title": "Sony WH-1000XM5 manual", "url": "https://www.manualslib.com/manual/123/sony-wh-1000xm5.html",
         "content": "Factory reset: hold power + custom.", "score": 0.7}]


async def store_with_manuals():
    store = InMemoryManualStore(HashEmbedder())
    await store.add([
        ManualChunk(brand="Sony", model="WH-1000XM5", title="WH-1000XM5 Help Guide: Initializing",
                    url="https://helpguide.sony.net/mdr/wh1000xm5/v1/en/print.pdf", page=112,
                    content="To factory reset the headset, hold the power button and custom button for 7 seconds."),
        ManualChunk(brand="Sony", model="WH-1000XM5", title="WH-1000XM5 Help Guide: Charging",
                    url="https://helpguide.sony.net/mdr/wh1000xm5/v1/en/print.pdf", page=20,
                    content="Connect the USB-C cable to charge the battery."),
        ManualChunk(brand="Bose", model="QC45", title="QC45 reset", url="https://bose.com/qc45.pdf", page=3,
                    content="factory reset headphones hold power"),
    ])
    return store


async def test_merges_all_three_backends():
    web = FakeWeb({"official": OFFICIAL, "open": OPEN})
    agent = SearcherAgent(manual_store=await store_with_manuals(), web_search=web)
    msg = StructuredMessage[DeviceQuery](content=Q, source="query_verifier")
    resp = await agent.on_messages([msg], CancellationToken())
    res = resp.chat_message.content
    assert isinstance(res, SearchResults) and not res.errors
    origins = [s.origin for s in res.sources]
    assert origins.count(SourceOrigin.MANUAL_STORE) == 2  # Bose chunk filtered out by brand
    assert origins.count(SourceOrigin.WEB) == 2
    assert [s.id for s in res.sources] == [f"R{i}" for i in range(1, 5)]
    # Official-domain query was restricted to Sony's domains.
    assert ("Sony WH-1000XM5 factory reset", official_domains("Sony")) in web.calls


async def test_manual_store_ranks_relevant_chunk_first():
    hits = await (await store_with_manuals()).search(Q, k=2)
    assert hits[0].chunk.page == 112


async def test_unknown_brand_skips_official_query():
    web = FakeWeb({"open": OPEN})
    agent = SearcherAgent(manual_store=None, web_search=web)
    res = await agent.search(Q.model_copy(update={"brand": "Obscuro"}))
    assert len(web.calls) == 1 and web.calls[0][1] is None
    assert len(res.sources) == 1


async def test_backends_run_concurrently():
    web = FakeWeb({"official": OFFICIAL, "open": OPEN}, delay=0.3)
    agent = SearcherAgent(manual_store=None, web_search=web)
    t = time.perf_counter()
    await agent.search(Q)
    assert time.perf_counter() - t < 0.55  # two 0.3s calls in parallel, not 0.6s


async def test_backend_failure_is_isolated():
    agent = SearcherAgent(manual_store=await store_with_manuals(), web_search=FakeWeb(error=RuntimeError("quota")))
    res = await agent.search(Q)
    assert len(res.errors) == 2 and all("quota" in e for e in res.errors)
    assert all(s.origin == SourceOrigin.MANUAL_STORE for s in res.sources) and res.sources


async def test_timeout_is_isolated():
    agent = SearcherAgent(manual_store=await store_with_manuals(), web_search=FakeWeb(delay=1), timeout_s=0.1)
    res = await agent.search(Q)
    assert any("TimeoutError" in e for e in res.errors)
    assert res.sources


def test_domain_matching():
    assert domain_matches("https://helpguide.sony.net/x", ["sony.net"])
    assert domain_matches("https://www.samsung.com/us/support", ["samsung.com"])
    assert not domain_matches("https://notsamsung.com/x", ["samsung.com"])
    assert official_domains("TP Link") == official_domains("tp-link") != []


@respx.mock
async def test_tavily_client_request_and_parse():
    route = respx.post(TavilySearchClient.URL).mock(return_value=httpx.Response(200, json={
        "results": [{"title": "Reset", "url": "https://sony.com/r", "content": "hold", "score": 0.8}, {"title": "no url"}]
    }))
    hits = await TavilySearchClient("tvly-key").search("q", include_domains=["sony.com"], max_results=3)
    assert [h.url for h in hits] == ["https://sony.com/r"]
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer tvly-key"
    assert b'"include_domains":["sony.com"]' in req.content.replace(b" ", b"")
