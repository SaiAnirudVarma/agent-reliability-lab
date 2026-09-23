"""OpenAI implementation of the LLMProvider seam.

Uses the OpenAI SDK's native structured-output parsing
(``client.beta.chat.completions.parse``), which validates the model's JSON
response against a given Pydantic model and hands back either a parsed
instance or a refusal/failure signal — the "prefer the provider's native
structured-output capability" path called for by the Component 5 design,
rather than hand-rolled JSON-mode plus manual parsing.

``openai`` is an optional dependency (see pyproject.toml's ``openai``
extra) and is imported lazily inside ``__init__``, so importing this module
does not require it to be installed unless ``OpenAIProvider`` is actually
instantiated. Nothing here reads temperature/max-output-tokens from the
prompt-configuration constants in ``app.agent.llm_agent`` directly — those
are passed in by whoever constructs this provider, keeping the dependency
direction one-way (agent code depends on the provider seam, not vice versa).
"""

from __future__ import annotations

from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from app.models.contracts import ModelProvider
from app.providers.base import ProviderMessage, ProviderTokenUsage, StructuredCompletion

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


def _usage_from_completion(completion: Any) -> Optional[ProviderTokenUsage]:
    """Extract token usage from a (possibly None) ChatCompletion-like object.

    Used both on the success path and on SDK parse-failure exceptions
    (LengthFinishReasonError/ContentFilterFinishReasonError), whose
    `.completion` attribute carries a real ChatCompletion -- including
    `.usage` -- for the non-streaming `.parse()` calls this provider makes.
    Never fabricates a value: returns None if no completion, or no usage on
    it, is available.
    """

    if completion is None:
        return None
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    return ProviderTokenUsage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens)


class OpenAIProvider:
    """LLMProvider backed by OpenAI's chat completions API."""

    provider = ModelProvider.OPENAI

    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
    ):
        try:
            import openai
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIProvider requires the 'openai' package. Install it with: "
                "pip install -e '.[openai]'"
            ) from exc

        self.model_name = model_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._client = OpenAI(api_key=api_key)
        # Held as instance attributes (rather than imported at module level)
        # since the whole `openai` import is intentionally lazy/optional.
        self._length_finish_reason_error = openai.LengthFinishReasonError
        self._content_filter_finish_reason_error = openai.ContentFilterFinishReasonError

    def complete_structured(
        self, messages: list[ProviderMessage], response_model: type[ResponseModelT]
    ) -> StructuredCompletion:
        try:
            completion = self._client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": message.role, "content": message.content} for message in messages],
                response_format=response_model,
                temperature=self._temperature,
                max_completion_tokens=self._max_output_tokens,
            )
        except self._length_finish_reason_error as exc:
            # The response was truncated before a complete, parseable object
            # was produced. This is a model/agent-level failure (nothing
            # usable came back), not an infrastructure failure. The
            # exception's `.completion` (a real, non-streaming ChatCompletion
            # for the `.parse()` calls this provider makes) reliably carries
            # `.usage` -- real tokens were still spent producing the
            # truncated output, so that usage is preserved here rather than
            # discarded.
            return StructuredCompletion(
                parsed=None,
                raw_text=f"response truncated at max_output_tokens={self._max_output_tokens}: {exc}",
                usage=_usage_from_completion(getattr(exc, "completion", None)),
                provider=self.provider,
                model_name=self.model_name,
            )
        except self._content_filter_finish_reason_error as exc:
            # `.completion` is Optional here (the SDK's own typing) -- may be
            # None if the filter fired before any completion was formed, in
            # which case there is genuinely no usage to report.
            return StructuredCompletion(
                parsed=None,
                raw_text=f"response withheld by content filter: {exc}",
                usage=_usage_from_completion(getattr(exc, "completion", None)),
                provider=self.provider,
                model_name=self.model_name,
            )

        choice = completion.choices[0]
        usage = _usage_from_completion(completion)

        refusal = getattr(choice.message, "refusal", None)
        raw_text = refusal if refusal else choice.message.content

        # completion.model is what OpenAI's response says actually served
        # the request (e.g. a dated snapshot behind a rolling alias) --
        # preferred over self.model_name (the requested identity) so a
        # historical result can distinguish "what we asked for" from "what
        # actually ran," per the requested/served model provenance design.
        # Falls back to the requested name only if the response omits it.
        served_model_name = getattr(completion, "model", None) or self.model_name

        return StructuredCompletion(
            parsed=choice.message.parsed,
            raw_text=raw_text,
            usage=usage,
            provider=self.provider,
            model_name=self.model_name,
            served_model_name=served_model_name,
        )
