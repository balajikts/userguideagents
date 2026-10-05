"""Agent 1 — Query Verifier.

Turns free text into a ``DeviceQuery`` and decides whether we know enough about
the device to search for its manual. Below the confidence threshold it returns
a clarification question instead of letting the pipeline guess.
"""

from __future__ import annotations

from typing import Sequence

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, StructuredMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from pydantic import BaseModel

from app.schemas import DeviceCategory, DeviceQuery

SYSTEM_PROMPT = """\
You extract structured information from questions about consumer electronics.

Return the brand, the model (exact model number or marketing name if given), the
device_type, and the question restated as a short task ("factory reset",
"pair with a second phone", "enable HDR").

confidence (0-1) is how sure you are that we can find the right manual:
- 0.9+  brand and a specific model are clear
- 0.6-0.8  brand is clear and the model is implied or the task is generic to the brand
- below 0.6  brand or device is unknown, ambiguous, or the model matters and is missing

List missing fields in missing_fields ("brand", "model", "device_type").
When confidence is below 0.6, write one short, friendly clarification_question
that asks only for what is missing. Otherwise set clarification_question to null.
Never invent a model number. The conversation may include earlier clarification
turns; combine all of them.
"""


class DeviceQueryDraft(BaseModel):
    """What the LLM fills in. Every field required, no defaults or constraints, so the
    schema is accepted by both Claude and OpenAI strict structured outputs;
    ``DeviceQuery`` does the validation afterwards."""

    brand: str | None
    model: str | None
    device_type: DeviceCategory
    question: str
    confidence: float
    missing_fields: list[str]
    clarification_question: str | None


def _default_clarification(missing: list[str]) -> str:
    want = [f for f in ("brand", "model") if f in missing] or ["brand and model"]
    return f"Which {' and '.join(want)} is your device? You can usually find it on a label on the back or underside."


class QueryVerifierAgent(BaseChatAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        clarify_threshold: float = 0.6,
        name: str = "query_verifier",
    ) -> None:
        super().__init__(name, description="Extracts brand/model/device/question and asks for clarification.")
        self._threshold = clarify_threshold
        self._model_client = model_client

    def _llm(self) -> AssistantAgent:
        # A fresh AssistantAgent per call: it holds conversation state, and one
        # verifier instance serves concurrent API requests.
        return AssistantAgent(
            f"{self.name}_llm",
            model_client=self._model_client,
            system_message=SYSTEM_PROMPT,
            output_content_type=DeviceQueryDraft,
        )

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (StructuredMessage[DeviceQuery],)

    def needs_clarification(self, q: DeviceQuery) -> bool:
        return q.confidence < self._threshold

    def postprocess(self, q: DeviceQuery) -> DeviceQuery:
        missing = set(q.missing_fields)
        if q.brand is None:
            missing.add("brand")
        if q.model is None:
            missing.add("model")
        q = q.model_copy(update={"missing_fields": sorted(missing)})
        if self.needs_clarification(q):
            if not q.clarification_question:
                q = q.model_copy(update={"clarification_question": _default_clarification(q.missing_fields)})
        else:
            q = q.model_copy(update={"clarification_question": None})
        return q

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        transcript = "\n".join(
            f"{m.source}: {m.content}" for m in messages if isinstance(m, TextMessage)
        )
        if not transcript:
            raise ValueError("QueryVerifierAgent needs at least one TextMessage")
        # The Manager passes the full clarification history every time.
        result = await self._llm().on_messages([TextMessage(content=transcript, source="user")], cancellation_token)
        msg = result.chat_message
        if not isinstance(msg, StructuredMessage) or not isinstance(msg.content, DeviceQueryDraft):
            raise TypeError(f"verifier LLM returned {type(msg).__name__}, expected DeviceQueryDraft")
        query = self.postprocess(DeviceQuery.model_validate(msg.content.model_dump()))
        return Response(
            chat_message=StructuredMessage[DeviceQuery](content=query, source=self.name, models_usage=msg.models_usage),
            inner_messages=result.inner_messages,
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
