"""Model clients for AutoGen agents.

AutoGen's bundled Anthropic client always sends ``temperature``, which current
Claude models reject, so Claude is wired in through a small ChatCompletionClient
built on the official ``anthropic`` SDK. When an AssistantAgent sets
``output_content_type``, AutoGen passes that Pydantic type as ``json_output`` and
we map it onto Claude structured outputs (``messages.parse``), so the JSON is
schema-valid by construction.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Literal, Mapping, Optional, Sequence

import anthropic
from autogen_core import CancellationToken
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

from app.config import Settings, get_settings

Effort = Literal["low", "medium", "high", "xhigh", "max"]

_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "refusal": "content_filter"}

CLAUDE_MODEL_INFO: ModelInfo = {
    "vision": True,
    "function_calling": False,  # these agents don't use tools; keeps the client small
    "json_output": True,
    "structured_output": True,
    "family": ModelFamily.ANY,
    "multiple_system_messages": True,
}


class LLMRefusalError(RuntimeError):
    """Claude (and its server-side fallback) declined the request."""


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c for c in content if isinstance(c, str))


def to_anthropic_messages(messages: Sequence[LLMMessage]) -> tuple[str | None, list[dict]]:
    """Split AutoGen messages into Claude's (system, messages); merge adjacent same-role turns."""
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system_parts.append(m.content)
            continue
        if isinstance(m, UserMessage):
            role, text = "user", _text(m.content)
        elif isinstance(m, AssistantMessage):
            role, text = "assistant", _text(m.content)
        else:
            raise ValueError(f"unsupported message type for this client: {type(m).__name__}")
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + text
        else:
            out.append({"role": role, "content": text})
    return ("\n\n".join(system_parts) or None), out


class ClaudeChatCompletionClient(ChatCompletionClient):
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        effort: Effort = "medium",
        max_tokens: int = 16000,
        api_key: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)
        self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._last = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        if tools:
            raise ValueError("ClaudeChatCompletionClient does not support tools")
        system, msgs = to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": msgs,
            "output_config": {"effort": extra_create_args.get("effort", self._effort)},
            # On a safety decline, re-run on Anthropic's recommended fallback model.
            "betas": [_FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            kwargs["system"] = system

        if isinstance(json_output, type) and issubclass(json_output, BaseModel):
            resp = await self._client.beta.messages.parse(output_format=json_output, **kwargs)
            if resp.stop_reason == "refusal":
                raise LLMRefusalError(str(resp.stop_details))
            content = resp.parsed_output.model_dump_json() if resp.parsed_output is not None else ""
        else:
            resp = await self._client.beta.messages.create(**kwargs)
            if resp.stop_reason == "refusal":
                raise LLMRefusalError(str(resp.stop_details))
            content = "".join(b.text for b in resp.content if b.type == "text")

        usage = RequestUsage(prompt_tokens=resp.usage.input_tokens, completion_tokens=resp.usage.output_tokens)
        self._last = usage
        self._total = RequestUsage(
            prompt_tokens=self._total.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total.completion_tokens + usage.completion_tokens,
        )
        return CreateResult(
            finish_reason=_FINISH.get(resp.stop_reason or "", "unknown"),  # type: ignore[arg-type]
            content=content,
            usage=usage,
            cached=False,
        )

    async def create_stream(self, messages: Sequence[LLMMessage], **kwargs: Any) -> AsyncGenerator[str | CreateResult, None]:
        # Agents here need whole JSON objects, so "streaming" yields the final result once.
        yield await self.create(messages, **kwargs)

    async def close(self) -> None:
        await self._client.close()

    def actual_usage(self) -> RequestUsage:
        return self._last

    def total_usage(self) -> RequestUsage:
        return self._total

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        # Rough local estimate; AutoGen only uses this for context-window bookkeeping.
        return sum(len(str(getattr(m, "content", ""))) for m in messages) // 4

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 1_000_000 - self.count_tokens(messages)

    @property
    def capabilities(self) -> ModelCapabilities:  # deprecated in AutoGen, still abstract
        return self.model_info  # type: ignore[return-value]

    @property
    def model_info(self) -> ModelInfo:
        return CLAUDE_MODEL_INFO


_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def get_model_client(effort: Effort = "medium", settings: Settings | None = None) -> ChatCompletionClient:
    s = settings or get_settings()
    if s.llm_provider == "openai":
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        kwargs: dict[str, Any] = {}
        if s.model_name.startswith(_OPENAI_REASONING_PREFIXES):
            # OpenAI reasoning models take low/medium/high; map our higher levels down.
            kwargs["reasoning_effort"] = {"xhigh": "high", "max": "high"}.get(effort, effort)
        # Structured outputs: AssistantAgent's output_content_type → response_format (strict schema).
        return OpenAIChatCompletionClient(model=s.model_name, api_key=s.openai_api_key, **kwargs)
    return ClaudeChatCompletionClient(s.model_name, effort=effort, max_tokens=s.llm_max_tokens, api_key=s.anthropic_api_key)
