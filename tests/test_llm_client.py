"""ClaudeChatCompletionClient against a fake anthropic SDK (no network)."""

from types import SimpleNamespace

import pytest
from anthropic import transform_schema
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage

from app.llm import ClaudeChatCompletionClient, LLMRefusalError, to_anthropic_messages
from app.schemas import DeviceQuery


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def parse(self, **kw):
        self.calls.append(("parse", kw))
        return self.response

    async def create(self, **kw):
        self.calls.append(("create", kw))
        return self.response


def fake_client(response):
    msgs = FakeMessages(response)
    sdk = SimpleNamespace(beta=SimpleNamespace(messages=msgs))
    return ClaudeChatCompletionClient(client=sdk, effort="low"), msgs


def resp(stop="end_turn", parsed=None, text=""):
    return SimpleNamespace(
        stop_reason=stop, stop_details={"category": "cyber"} if stop == "refusal" else None,
        parsed_output=parsed, content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def test_message_conversion_merges_roles():
    system, msgs = to_anthropic_messages([
        SystemMessage(content="sys"),
        UserMessage(content="a", source="u"), UserMessage(content="b", source="u"),
        AssistantMessage(content="c", source="x"),
    ])
    assert system == "sys"
    assert msgs == [{"role": "user", "content": "a\n\nb"}, {"role": "assistant", "content": "c"}]


@pytest.mark.asyncio
async def test_structured_output_via_parse_and_request_shape():
    q = DeviceQuery(brand="Sony", model="A80L", device_type="tv", question="reset", confidence=0.9)
    client, calls = fake_client(resp(parsed=q))
    agent = AssistantAgent("a", model_client=client, system_message="sys", output_content_type=DeviceQuery)
    r = await agent.on_messages([TextMessage(content="reset my sony a80l", source="user")], CancellationToken())
    assert isinstance(r.chat_message, StructuredMessage) and r.chat_message.content == q

    kind, kw = calls.calls[0]
    assert kind == "parse" and kw["output_format"] is DeviceQuery
    assert kw["model"] == "claude-opus-5-5"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"] == {"effort": "low"}
    assert "temperature" not in kw and "thinking" not in kw
    assert client.total_usage().prompt_tokens == 10


@pytest.mark.asyncio
async def test_plain_text_create():
    client, calls = fake_client(resp(text="hello"))
    r = await client.create([UserMessage(content="hi", source="u")])
    assert r.content == "hello" and calls.calls[0][0] == "create"


@pytest.mark.asyncio
async def test_refusal_raises():
    client, _ = fake_client(resp(stop="refusal"))
    with pytest.raises(LLMRefusalError):
        await client.create([UserMessage(content="hi", source="u")])


def test_device_query_schema_is_structured_output_compatible():
    schema = transform_schema(DeviceQuery)
    assert schema["type"] == "object" and schema.get("additionalProperties") is False
