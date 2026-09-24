"""Test-only fake Cohere client: no network, fully caller-controlled
response shape, used to test app.reranking.cohere_reranker.CohereReranker
without requiring COHERE_API_KEY or the real cohere package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


@dataclass(frozen=True)
class FakeCohereResultItem:
    index: Any
    relevance_score: Any


@dataclass(frozen=True)
class FakeCohereResponse:
    results: list
    model: Optional[str] = None
    id: Optional[str] = None


ResponseOrFactory = Union[FakeCohereResponse, Callable[..., Any]]


class FakeCohereClient:
    """Records the exact call it received (model/query/documents/top_n)
    for assertions, and returns a fixed (or callable-produced) response --
    mirrors ``tests.support.fake_llm_provider.FakeLLMProvider``'s role."""

    def __init__(self, response: ResponseOrFactory):
        self._response = response
        self.last_call: Optional[dict] = None
        self.call_count = 0

    def rerank(self, *, model: str, query: str, documents: list[str], top_n: int) -> Any:
        self.call_count += 1
        self.last_call = {"model": model, "query": query, "documents": list(documents), "top_n": top_n}
        if callable(self._response):
            return self._response(model=model, query=query, documents=documents, top_n=top_n)
        return self._response
