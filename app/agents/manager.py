"""Manager — orchestrates guardrails and the three agents.

The pipeline order is fixed, so the Manager calls agents directly instead of
letting an LLM pick the next speaker. It is itself a ``BaseChatAgent``: give it
the conversation (user turns plus any earlier clarification questions) and it
replies with ``StructuredMessage[GuideAnswer]``; every sub-agent message is
returned in ``inner_messages`` as a trace.
"""

from __future__ import annotations

import logging
from typing import Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, StructuredMessage, TextMessage
from autogen_core import CancellationToken

from app.agents.filter import FilterAgent
from app.agents.query_verifier import QueryVerifierAgent
from app.agents.searcher import SearcherAgent
from app.guardrails.input import REFUSAL_MESSAGES, check_input
from app.guardrails.output import BLOCKED_MESSAGE, apply_output_guardrails, is_blocked_procedure
from app.llm import LLMRefusalError
from app.schemas import DeviceQuery, GuideAnswer, SearchResults

log = logging.getLogger(__name__)

USER = "user"
ASSISTANT = "assistant"


class ManagerAgent(BaseChatAgent):
    def __init__(
        self,
        verifier: QueryVerifierAgent,
        searcher: SearcherAgent,
        filter_agent: FilterAgent,
        *,
        max_clarifications: int = 2,
        grounding_min_overlap: float = 0.2,
        name: str = "manager",
    ) -> None:
        super().__init__(name, description="Answers electronics how-to questions from official manuals.")
        self.verifier = verifier
        self.searcher = searcher
        self.filter = filter_agent
        self._max_clarifications = max_clarifications
        self._min_overlap = grounding_min_overlap

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (StructuredMessage[GuideAnswer],)

    def _reply(self, answer: GuideAnswer, trace: list) -> Response:
        return Response(chat_message=StructuredMessage[GuideAnswer](content=answer, source=self.name), inner_messages=trace)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        turns = [m for m in messages if isinstance(m, TextMessage)]
        if not turns or turns[-1].source != USER:
            raise ValueError("ManagerAgent expects the conversation to end with a user TextMessage")
        trace: list[BaseAgentEvent | BaseChatMessage] = []

        # 1. Input guardrails on the newest user turn; earlier turns were checked when they arrived.
        check = check_input(turns[-1].content)
        if not check.allowed:
            reason = check.reasons[0]
            log.info("input rejected: %s", check.reasons)
            return self._reply(GuideAnswer(status="rejected", summary=REFUSAL_MESSAGES[reason]), trace)
        if is_blocked_procedure(check.sanitized_text):
            return self._reply(GuideAnswer(status="rejected", summary=BLOCKED_MESSAGE, warnings=[BLOCKED_MESSAGE]), trace)
        turns = [*turns[:-1], TextMessage(content=check.sanitized_text, source=USER)]

        try:
            # 2. Verify / clarify.
            v = await self.verifier.on_messages(turns, cancellation_token)
            trace.append(v.chat_message)
            query: DeviceQuery = v.chat_message.content  # type: ignore[attr-defined]
            asked = sum(1 for t in turns if t.source == ASSISTANT)
            if self.verifier.needs_clarification(query) and asked < self._max_clarifications:
                return self._reply(
                    GuideAnswer(status="needs_clarification", clarification_question=query.clarification_question, query=query),
                    trace,
                )

            # 3. Search.
            s = await self.searcher.on_messages([v.chat_message], cancellation_token)
            trace.append(s.chat_message)
            results: SearchResults = s.chat_message.content  # type: ignore[attr-defined]
            if results.errors:
                log.warning("search backend errors: %s", results.errors)

            # 4. Filter + write.
            f = await self.filter.on_messages([s.chat_message], cancellation_token)
            trace.append(f.chat_message)
            draft: GuideAnswer = f.chat_message.content  # type: ignore[attr-defined]
        except LLMRefusalError:
            log.warning("model declined request", exc_info=True)
            return self._reply(GuideAnswer(status="rejected", summary="Sorry, I can't help with that request."), trace)

        # 5. Output guardrails.
        final, report = apply_output_guardrails(draft, min_overlap=self._min_overlap)
        log.info("grounding verified_ratio=%.2f passed=%s status=%s", report.verified_ratio, report.passed, final.status)
        return self._reply(final, trace)

    async def ask(self, conversation: list[tuple[str, str]], cancellation_token: CancellationToken | None = None) -> GuideAnswer:
        """Convenience wrapper: ``[(source, text), ...]`` → GuideAnswer."""
        msgs = [TextMessage(content=text, source=src) for src, text in conversation]
        r = await self.on_messages(msgs, cancellation_token or CancellationToken())
        return r.chat_message.content  # type: ignore[attr-defined]

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        for a in (self.verifier, self.searcher, self.filter):
            await a.on_reset(cancellation_token)
