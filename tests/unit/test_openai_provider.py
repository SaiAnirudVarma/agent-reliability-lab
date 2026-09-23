"""Direct, mocked unit tests for app.providers.openai_provider.OpenAIProvider.

Every test here monkeypatches the bound `client.beta.chat.completions.parse`
method so the installed OpenAI SDK's real request path is never invoked --
zero network calls. The one exception,
`TestNetworkBarrierProof::test_real_provider_call_path_is_blocked_locally`,
deliberately does NOT mock that method, to prove the tests/conftest.py
network barrier intercepts a real call attempt before any network I/O --
using an obviously fake key, never the developer's real credentials.
"""

from __future__ import annotations

import pytest

from app.agent.llm_agent import LLMFindingPayload
from app.models.contracts import Finding, ModelProvider
from app.providers.base import ProviderMessage
from app.providers.openai_provider import OpenAIProvider
from tests.conftest import NetworkBlockedError

FAKE_API_KEY = "sk-test-fake-never-real-do-not-use"


def _messages() -> list[ProviderMessage]:
    return [
        ProviderMessage(role="system", content="sys"),
        ProviderMessage(role="user", content="usr"),
    ]


def _payload(**overrides) -> LLMFindingPayload:
    defaults = dict(
        finding=Finding.PASS,
        confidence=0.9,
        reasoning_summary="ok",
        evidence=[],
        missing_information=[],
        abstain=False,
    )
    defaults.update(overrides)
    return LLMFindingPayload(**defaults)


def _make_provider(model_name: str = "gpt-5.4-mini-2026-03-17") -> OpenAIProvider:
    return OpenAIProvider(api_key=FAKE_API_KEY, model_name=model_name)


def _chat_completion(*, finish_reason: str, content, parsed=None, refusal=None, usage=None, model="gpt-5.4-mini-2026-03-17"):
    """Build a real openai SDK ParsedChatCompletion (the type `.parse()`
    actually returns), so these tests exercise the exact structure our
    provider code reads."""
    from openai.types.chat import ParsedChatCompletion, ParsedChatCompletionMessage, ParsedChoice

    message = ParsedChatCompletionMessage(role="assistant", content=content, parsed=parsed, refusal=refusal)
    choice = ParsedChoice(index=0, finish_reason=finish_reason, message=message)
    return ParsedChatCompletion(
        id="chatcmpl-test", choices=[choice], created=0, model=model,
        object="chat.completion", usage=usage,
    )


def _usage(prompt_tokens: int, completion_tokens: int):
    from openai.types.completion_usage import CompletionUsage

    return CompletionUsage(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _chat_completion_non_parsed(*, finish_reason: str, content, usage=None, model="gpt-5.4-mini-2026-03-17"):
    """A plain (non-Parsed) ChatCompletion, the type carried by
    LengthFinishReasonError/ContentFilterFinishReasonError's `.completion`."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    message = ChatCompletionMessage(role="assistant", content=content)
    choice = Choice(index=0, finish_reason=finish_reason, message=message)
    return ChatCompletion(
        id="chatcmpl-test", choices=[choice], created=0, model=model,
        object="chat.completion", usage=usage,
    )


class TestSuccessfulStructuredParse:
    def test_parsed_payload_returned(self):
        provider = _make_provider()
        payload = _payload(finding=Finding.EXCEPTION, confidence=0.77)
        fake_completion = _chat_completion(
            finish_reason="stop", content="{}", parsed=payload, usage=_usage(100, 50)
        )
        provider._client.beta.chat.completions.parse = lambda *a, **k: fake_completion

        result = provider.complete_structured(_messages(), LLMFindingPayload)

        assert result.parsed == payload
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.provider == ModelProvider.OPENAI

    def test_served_model_preferred_over_requested_when_present(self):
        provider = _make_provider(model_name="gpt-5.4-mini-2026-03-17")
        fake_completion = _chat_completion(
            finish_reason="stop", content="{}", parsed=_payload(),
            model="gpt-5.4-mini-2026-03-17-exact-snapshot",
        )
        provider._client.beta.chat.completions.parse = lambda *a, **k: fake_completion

        result = provider.complete_structured(_messages(), LLMFindingPayload)

        assert result.model_name == "gpt-5.4-mini-2026-03-17"  # requested, unchanged
        assert result.served_model_name == "gpt-5.4-mini-2026-03-17-exact-snapshot"  # served
        assert result.model_name != result.served_model_name  # the two can diverge and both survive

    def test_served_model_falls_back_to_requested_when_response_omits_it(self):
        provider = _make_provider(model_name="gpt-5.4-mini-2026-03-17")
        fake_completion = _chat_completion(finish_reason="stop", content="{}", parsed=_payload())
        # ParsedChatCompletion.model is a required string on the SDK's own
        # type, so we simulate "unavailable" the same way our code checks
        # for it: falsy (empty string), via a modified copy.
        fake_completion_with_empty_model = fake_completion.model_copy(update={"model": ""})
        provider._client.beta.chat.completions.parse = lambda *a, **k: fake_completion_with_empty_model

        result = provider.complete_structured(_messages(), LLMFindingPayload)

        assert result.served_model_name == "gpt-5.4-mini-2026-03-17"  # requested, as fallback


class TestRefusal:
    def test_refusal_returns_parsed_none_with_refusal_text(self):
        provider = _make_provider()
        fake_completion = _chat_completion(
            finish_reason="stop", content=None, parsed=None, refusal="I can't help with that."
        )
        provider._client.beta.chat.completions.parse = lambda *a, **k: fake_completion

        result = provider.complete_structured(_messages(), LLMFindingPayload)

        assert result.parsed is None
        assert result.raw_text == "I can't help with that."


class TestTruncation:
    def test_length_finish_reason_error_returns_parsed_none(self):
        import openai

        provider = _make_provider()
        completion = _chat_completion_non_parsed(finish_reason="length", content="{\"trunc", usage=_usage(500, 800))

        def raise_length(*a, **k):
            raise openai.LengthFinishReasonError(completion=completion)

        provider._client.beta.chat.completions.parse = raise_length

        result = provider.complete_structured(_messages(), LLMFindingPayload)

        assert result.parsed is None
        assert "truncated" in result.raw_text

    def test_usage_preserved_on_truncation(self):
        """The fix: exc.completion.usage carries real tokens for the
        non-streaming .parse() path -- must not be discarded as None."""
        import openai

        provider = _make_provider()
        completion = _chat_completion_non_parsed(finish_reason="length", content="{\"trunc", usage=_usage(512, 800))

        def raise_length(*a, **k):
            raise openai.LengthFinishReasonError(completion=completion)

        provider._client.beta.chat.completions.parse = raise_length

        result = provider.complete_structured(_messages(), LLMFindingPayload)

        assert result.usage is not None
        assert result.usage.input_tokens == 512
        assert result.usage.output_tokens == 800

    def test_usage_is_none_when_truncation_exception_has_no_usage(self):
        """Never fabricated: if the exception's completion has no usage,
        result.usage stays None rather than guessing."""
        import openai

        provider = _make_provider()
        completion = _chat_completion_non_parsed(finish_reason="length", content="{\"trunc", usage=None)

        def raise_length(*a, **k):
            raise openai.LengthFinishReasonError(completion=completion)

        provider._client.beta.chat.completions.parse = raise_length

        result = provider.complete_structured(_messages(), LLMFindingPayload)

        assert result.usage is None


class TestContentFilter:
    def test_content_filter_error_returns_parsed_none(self):
        import openai

        provider = _make_provider()

        def raise_content_filter(*a, **k):
            raise openai.ContentFilterFinishReasonError()

        provider._client.beta.chat.completions.parse = raise_content_filter

        result = provider.complete_structured(_messages(), LLMFindingPayload)

        assert result.parsed is None
        assert "content filter" in result.raw_text

    def test_usage_preserved_on_content_filter_when_available(self):
        import openai

        provider = _make_provider()
        completion = _chat_completion_non_parsed(finish_reason="content_filter", content=None, usage=_usage(200, 5))

        def raise_content_filter(*a, **k):
            raise openai.ContentFilterFinishReasonError(completion=completion)

        provider._client.beta.chat.completions.parse = raise_content_filter

        result = provider.complete_structured(_messages(), LLMFindingPayload)

        assert result.usage is not None
        assert result.usage.input_tokens == 200
        assert result.usage.output_tokens == 5

    def test_usage_is_none_when_content_filter_completion_is_absent(self):
        """ContentFilterFinishReasonError.completion is genuinely Optional
        per the SDK's own type -- must not fabricate usage when it's None."""
        import openai

        provider = _make_provider()

        def raise_content_filter(*a, **k):
            raise openai.ContentFilterFinishReasonError(completion=None)

        provider._client.beta.chat.completions.parse = raise_content_filter

        result = provider.complete_structured(_messages(), LLMFindingPayload)

        assert result.usage is None


class TestNetworkBarrierProof:
    def test_real_provider_call_path_is_blocked_locally(self):
        """Proves the tests/conftest.py barrier intercepts a real call
        BEFORE any network I/O: this test deliberately does NOT monkeypatch
        `.parse()`, so calling complete_structured() here would attempt a
        genuine HTTPS connection to OpenAI's servers if not for the barrier.
        Uses an obviously fake key -- never a real one -- and expects to
        never reach the point of authenticating at all."""
        provider = _make_provider()
        with pytest.raises(NetworkBlockedError):
            provider.complete_structured(_messages(), LLMFindingPayload)
