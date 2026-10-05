from autogen_agentchat.messages import StructuredMessage
from autogen_core import CancellationToken

from app.agents.filter import (
    TRUST_MANUAL_STORE, TRUST_MIRROR, TRUST_OFFICIAL, TRUST_OTHER,
    FilterAgent, build_prompt, jaccard, normalise_url, rank_and_dedupe, trust_score,
)
from app.schemas import DeviceQuery, GuideAnswer, SearchResults, Source, SourceOrigin
from tests.conftest import replay

Q = DeviceQuery(brand="Sony", model="WH-1000XM5", device_type="headphones", question="factory reset", confidence=0.95)


def src(id, url, snippet, origin=SourceOrigin.WEB, score=0.5, page=None, title="t"):
    return Source(id=id, title=title, url=url, snippet=snippet, origin=origin, retrieval_score=score, page=page)


MANUAL = src("R1", "https://helpguide.sony.net/wh1000xm5/print.pdf", "Hold the power and custom buttons for 7 seconds to initialize.",
             SourceOrigin.MANUAL_STORE, 0.8, 112)
OFFICIAL = src("R2", "https://www.sony.com/electronics/support/wh-1000xm5/reset", "Press and hold power and custom button together for 7 s.", score=0.9)
MIRROR = src("R3", "https://www.manualslib.com/manual/1/sony.html", "Sony WH-1000XM5 operating instructions reset section", score=0.95)
BLOG = src("R4", "https://randomtechblog.io/sony-reset", "I think you hold all buttons for a long time", score=0.99)


def test_trust_tiers():
    assert trust_score(MANUAL, Q) >= TRUST_MANUAL_STORE
    assert TRUST_OFFICIAL <= trust_score(OFFICIAL, Q) < TRUST_MANUAL_STORE + 0.1
    assert TRUST_MIRROR <= trust_score(MIRROR, Q) < TRUST_OFFICIAL
    assert trust_score(BLOG, Q) == TRUST_OTHER


def test_rank_prefers_trust_over_retrieval_score():
    ranked = rank_and_dedupe([BLOG, MIRROR, OFFICIAL, MANUAL], Q)
    assert [s.url for s in ranked] == [MANUAL.url, OFFICIAL.url, MIRROR.url, BLOG.url]
    assert [s.id for s in ranked] == ["S1", "S2", "S3", "S4"]


def test_dedupe_by_url_and_near_duplicate_text():
    dup_url = src("R5", "http://sony.com/electronics/support/wh-1000xm5/reset/?utm=x", "different text entirely here")
    long = "To reset the headset press and hold the power button and the custom button simultaneously for more than seven seconds"
    a = src("R6", "https://sony.com/a", long, score=0.9)
    b = src("R7", "https://sony.com/b", long + " .", score=0.1)
    ranked = rank_and_dedupe([OFFICIAL, dup_url, a, b], Q)
    urls = [s.url for s in ranked]
    assert dup_url.url not in urls and b.url not in urls and a.url in urls


def test_same_pdf_different_pages_kept():
    p2 = MANUAL.model_copy(update={"page": 20, "snippet": "Charging the battery takes 3.5 hours via USB-C."})
    assert len(rank_and_dedupe([MANUAL, p2], Q)) == 2


def test_max_sources_and_empty_snippets():
    many = [src(f"R{i}", f"https://sony.com/p{i}", f"unique text number {i} " * 5) for i in range(20)]
    many.append(src("RX", "https://sony.com/empty", "   "))
    ranked = rank_and_dedupe(many, Q, max_sources=5)
    assert len(ranked) == 5 and all(s.snippet.strip() for s in ranked)


def test_helpers():
    assert normalise_url("https://WWW.Sony.com/A/b/") == normalise_url("http://sony.com/a/b")
    assert jaccard("a b c d e f", "a b c d e f") == 1.0


def test_prompt_marks_sources_as_untrusted_data():
    ranked = rank_and_dedupe([MANUAL, BLOG], Q)
    p = build_prompt(Q, ranked)
    assert '<source id="S1" trust="official manual"' in p and "page=112" in p
    assert 'trust="other"' in p


async def test_writes_cited_answer():
    client = replay({
        "answerable": True, "summary": "Reset restores factory settings.",
        "steps": [{"instruction": "Turn on the headset.", "citation": "s1"},
                  {"instruction": "Hold power and custom buttons for 7 seconds.", "citation": "S2"}],
        "warnings": ["Pairing info will be deleted."],
    })
    agent = FilterAgent(client)
    msg = StructuredMessage[SearchResults](content=SearchResults(query=Q, sources=[BLOG, OFFICIAL, MANUAL]), source="searcher")
    resp = await agent.on_messages([msg], CancellationToken())
    ans = resp.chat_message.content
    assert isinstance(ans, GuideAnswer) and ans.status == "answered"
    assert [s.citation for s in ans.steps] == ["S1", "S2"]  # normalised to upper case
    assert [s.number for s in ans.steps] == [1, 2]
    assert ans.sources[0].origin == SourceOrigin.MANUAL_STORE
    # The LLM saw the ranked, renumbered sources.
    sent = client.create_calls[-1]["messages"][-1].content
    assert sent.index('id="S1"') < sent.index("helpguide.sony.net") < sent.index("randomtechblog")


async def test_unanswerable_returns_not_found():
    agent = FilterAgent(replay({"answerable": False, "summary": "Sources cover a different model.", "steps": [], "warnings": []}))
    ans = await agent.answer(SearchResults(query=Q, sources=[MANUAL]))
    assert ans.status == "not_found" and not ans.steps and ans.sources


async def test_no_sources_skips_llm():
    client = replay()
    ans = await FilterAgent(client).answer(SearchResults(query=Q, sources=[]))
    assert ans.status == "not_found" and client.create_calls == []
