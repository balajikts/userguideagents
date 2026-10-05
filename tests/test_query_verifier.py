import pytest
from autogen_agentchat.messages import StructuredMessage, TextMessage
from autogen_core import CancellationToken

from app.agents.query_verifier import QueryVerifierAgent
from app.schemas import DeviceCategory, DeviceQuery
from tests.conftest import replay



async def ask(agent, *texts):
    msgs = [TextMessage(content=t, source="user") for t in texts]
    resp = await agent.on_messages(msgs, CancellationToken())
    assert isinstance(resp.chat_message, StructuredMessage)
    return resp.chat_message.content


async def test_extracts_confident_query():
    agent = QueryVerifierAgent(replay({
        "brand": "Sony", "model": "WH-1000XM5", "device_type": "headphones",
        "question": "factory reset", "confidence": 0.95, "missing_fields": [],
        "clarification_question": None,
    }))
    q = await ask(agent, "How do I factory reset my Sony WH-1000XM5?")
    assert isinstance(q, DeviceQuery)
    assert (q.brand, q.model, q.device_type) == ("Sony", "WH-1000XM5", DeviceCategory.HEADPHONES)
    assert not agent.needs_clarification(q)
    assert q.clarification_question is None


async def test_low_confidence_keeps_llm_clarification():
    agent = QueryVerifierAgent(replay({
        "brand": None, "model": None, "device_type": "tv", "question": "turn on subtitles",
        "confidence": 0.3, "missing_fields": ["brand", "model"],
        "clarification_question": "Which TV brand and model do you have?",
    }))
    q = await ask(agent, "how do I turn on subtitles on my tv")
    assert agent.needs_clarification(q)
    assert q.clarification_question == "Which TV brand and model do you have?"


async def test_low_confidence_without_question_gets_default():
    agent = QueryVerifierAgent(replay({
        "brand": "Samsung", "model": "unknown", "device_type": "phone", "question": "reset",
        "confidence": 0.4, "missing_fields": [], "clarification_question": None,
    }))
    q = await ask(agent, "reset my samsung")
    assert q.model is None  # "unknown" normalised to None
    assert "model" in q.missing_fields
    assert q.clarification_question and "model" in q.clarification_question


async def test_high_confidence_clears_stray_clarification():
    agent = QueryVerifierAgent(replay({
        "brand": "Apple", "model": "iPhone 15", "device_type": "phone", "question": "force restart",
        "confidence": 0.9, "missing_fields": [], "clarification_question": "Which iPhone?",
    }))
    q = await ask(agent, "force restart iphone 15")
    assert q.clarification_question is None


async def test_threshold_is_configurable():
    payload = {"brand": "LG", "model": None, "device_type": "tv", "question": "update firmware",
               "confidence": 0.7, "missing_fields": ["model"], "clarification_question": None}
    strict = QueryVerifierAgent(replay(payload), clarify_threshold=0.8)
    assert strict.needs_clarification(await ask(strict, "update my LG tv"))
    lenient = QueryVerifierAgent(replay(payload), clarify_threshold=0.5)
    assert not lenient.needs_clarification(await ask(lenient, "update my LG tv"))


async def test_clarification_turns_are_combined():
    client = replay({
        "brand": "Bose", "model": "QuietComfort Ultra", "device_type": "headphones",
        "question": "pair with laptop", "confidence": 0.9, "missing_fields": [], "clarification_question": None,
    })
    agent = QueryVerifierAgent(client)
    await ask(agent, "how do i pair my headphones with my laptop", "Bose QuietComfort Ultra")
    sent = client.create_calls[-1]["messages"][-1].content
    assert "pair my headphones" in sent and "QuietComfort Ultra" in sent


async def test_rejects_invalid_llm_output():
    agent = QueryVerifierAgent(replay({"brand": "Sony", "confidence": 2.0}))
    with pytest.raises(Exception):
        await ask(agent, "reset sony")
