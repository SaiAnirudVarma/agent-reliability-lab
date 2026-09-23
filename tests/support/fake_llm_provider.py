"""A test-only LLMProvider that returns pre-configured responses with no
network calls, so LLMAgent can be exercised end to end in tests without
spending real API money or requiring credentials.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from app.models.contracts import ModelProvider
from app.providers.base import ProviderMessage, StructuredCompletion

CompletionOrFactory = Union[
    StructuredCompletion, Callable[[list[ProviderMessage], type], StructuredCompletion]
]


class FakeLLMProvider:
    """Implements LLMProvider by returning a fixed StructuredCompletion, or
    one produced by a callable (useful for varying the response per call,
    e.g. across the 10 dataset cases in a test loop).

    Records the messages it was called with (``last_messages``) so tests
    can assert on prompt content without needing a real provider.
    """

    provider = ModelProvider.OPENAI
    model_name = "fake-model-v1"

    def __init__(self, completion: CompletionOrFactory):
        self._completion = completion
        self.last_messages: Optional[list[ProviderMessage]] = None
        self.call_count = 0

    def complete_structured(self, messages: list[ProviderMessage], response_model: type) -> StructuredCompletion:
        self.last_messages = messages
        self.call_count += 1
        if callable(self._completion):
            return self._completion(messages, response_model)
        return self._completion
