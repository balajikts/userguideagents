"""LLM_PROVIDER=openai: real AutoGen OpenAI client against a mocked OpenAI HTTP API."""

import functools
import json

import autogen_ext.models.openai as autogen_openai
import httpx2
import pytest
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from openai.lib._pydantic import to_strict_json_schema

from app.agents.filter import DraftAnswer, FilterAgent
from app.agents.query_verifier import DeviceQueryDraft, QueryVerifierAgent
from app.config import Settings
from app.llm import get_model_client
from app.schemas import DeviceQuery, SearchResults, Source, SourceOrigin

URL = "https://api.openai.com/v1/chat/completions"


def settings(**kw) -> Settings:
    return Settings(_env_file=None, llm_provider="openai", openai_api_key="sk-test", **kw)


def completion(content: dict) -> httpx2.Response:
    return httpx2.Response(200, json={
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gpt-5-2025-08-07",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps(content), "refusal": None}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })


@pytest.mark.parametrize("model", [DeviceQueryDraft, DraftAnswer])
def test_llm_schemas_are_plain_for_openai_strict_mode(model):
    # Keywords OpenAI strict mode may reject; LLM-facing models must not emit them.
    schema = json.dumps(to_strict_json_schema(model))
    for kw in ("minLength", "maxLength", "minimum", "maximum", '"default"'):
        assert kw not in schema, kw


def test_defaults_per_provider():
    assert settings().model_name == "gpt-5"
    assert Settings(_env_file=None).model_name == "claude-opus-5-5"
    assert settings(llm_model="gpt-4.1").model_name == "gpt-4.1"


@pytest.fixture
def mock_openai(monkeypatch):
    """The OpenAI SDK (v3) sends requests through httpx2, which respx doesn't patch,
    so inject a MockTransport; nothing reaches the network."""
    queue: list[dict] = []
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == URL
        requests.append(request)
        return completion(queue.pop(0))

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    real = autogen_openai.OpenAIChatCompletionClient
    monkeypatch.setattr(autogen_openai, "OpenAIChatCompletionClient", functools.partial(real, http_client=http))
    return queue, requests


async def test_verifier_and_filter_over_openai(mock_openai):
    queue, requests = mock_openai
    queue += [
        {"brand": "Sony", "model": "WH-1000XM5", "device_type": "headphones", "question": "factory reset",
         "confidence": 0.95, "missing_fields": [], "clarification_question": None},
        {"answerable": True, "summary": "Reset.", "warnings": [],
         "steps": [{"instruction": "Hold power and custom for 7 seconds.", "citation": "S1"}]},
    ]
    s = settings()

    verifier = QueryVerifierAgent(get_model_client("low", s))
    r = await verifier.on_messages([TextMessage(content="reset my sony xm5", source="user")], CancellationToken())
    q = r.chat_message.content
    assert isinstance(q, DeviceQuery) and q.model == "WH-1000XM5"

    src = Source(id="R1", title="Reset", url="https://helpguide.sony.net/r", snippet="hold power and custom 7 seconds",
                 origin=SourceOrigin.MANUAL_STORE)
    ans = await FilterAgent(get_model_client("medium", s)).answer(SearchResults(query=q, sources=[src]))
    assert ans.status == "answered" and ans.steps[0].citation == "S1"

    bodies = [json.loads(r.content) for r in requests]
    assert [b["reasoning_effort"] for b in bodies] == ["low", "medium"]
    assert [b["response_format"]["json_schema"]["name"] for b in bodies] == ["DeviceQueryDraft", "DraftAnswer"]
    assert all(b["response_format"]["json_schema"]["strict"] for b in bodies)
    assert all("temperature" not in b for b in bodies)
    assert requests[0].headers["authorization"] == "Bearer sk-test"


def test_non_reasoning_model_gets_no_reasoning_effort():
    client = get_model_client("low", settings(llm_model="gpt-4.1"))
    assert "reasoning_effort" not in client._create_args
