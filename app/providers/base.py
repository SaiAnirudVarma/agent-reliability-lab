"""The provider seam: the minimum interface between an LLMAgent and any one
hosted model API.

Keeping this small and provider-agnostic is what lets ``LLMAgent`` swap
between OpenAI, Anthropic, or a test fake without changing its own code —
only a new module implementing ``LLMProvider`` is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from app.models.contracts import ModelProvider

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


@dataclass(frozen=True)
class ProviderMessage:
    """One message in a chat-style request. ``role`` is "system" or "user" —
    Phase 1 has no need for multi-turn conversation or tool-call messages."""

    role: str
    content: str


@dataclass(frozen=True)
class ProviderTokenUsage:
    """Raw token counts as reported by the provider. Never includes cost —
    a provider knows how many tokens it used, not what they should cost;
    pricing is applied by the agent from separately, explicitly configured
    numbers (see app.agent.pricing)."""

    input_tokens: Optional[int]
    output_tokens: Optional[int]


@dataclass(frozen=True)
class StructuredCompletion:
    """The result of one structured-output request to a provider.

    ``parsed`` is an instance of the requested response model, or ``None``
    if the provider could not produce something conforming to it (a refusal,
    or output that didn't parse). ``raw_text`` is the observable response
    text -- populated particularly when ``parsed`` is ``None``, so the
    caller can build a diagnostic ``AgentOutputError`` from it. It must never
    contain hidden provider reasoning/chain-of-thought beyond what the
    provider already exposes as visible completion text, and callers must
    bound how much of it they persist.

    ``model_name`` is the REQUESTED model identity (what the caller
    configured the provider with) -- unchanged in meaning from before.
    ``served_model_name`` is the ACTUAL model the provider's response says
    served the request (e.g. a dated snapshot behind a rolling alias), when
    the provider can confirm a successful response. On a successful
    completion, a provider should prefer the response's own reported model
    identity and fall back to the requested ``model_name`` only if the
    response doesn't include one -- so ``served_model_name`` is populated
    whenever ``parsed`` is populated. On an error/refusal path (``parsed``
    is ``None``), providers currently leave it ``None`` rather than guess.
    """

    parsed: Optional[BaseModel]
    raw_text: Optional[str]
    usage: Optional[ProviderTokenUsage]
    provider: ModelProvider
    model_name: str
    served_model_name: Optional[str] = None


@runtime_checkable
class LLMProvider(Protocol):
    """The minimal seam every concrete provider (OpenAI, Anthropic, a test
    fake) implements. Intentionally narrow: one piece of provenance, one
    method, because that's all Phase 1's single-turn structured-output
    agent needs.
    """

    provider: ModelProvider
    model_name: str

    def complete_structured(
        self, messages: list[ProviderMessage], response_model: type[ResponseModelT]
    ) -> StructuredCompletion: ...
