"""Provider-independent embedding interface, mirroring
``app.providers.base.LLMProvider``'s shape: one piece of provenance
(``provider``/``model_name``), one method per direction (documents vs.
query), so ``VectorRetriever`` can swap between a real provider and a test
fake without changing its own code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

from app.models.contracts import ModelProvider

Vector = tuple[float, ...]


def validate_vector(values: Sequence[float], *, expected_dimension: Optional[int] = None) -> Vector:
    """Normalizes ``values`` to an immutable ``Vector`` and enforces:

    - non-empty
    - every value finite (rejects NaN/inf -- a malformed embedding must
      never silently enter a corpus index or a similarity computation)
    - if ``expected_dimension`` is given, ``len(values)`` must match it
      exactly (dimension-consistency validation, called both by providers
      on their own output and by ``build_corpus_embedding_index``)
    """

    vector = tuple(float(v) for v in values)
    if not vector:
        raise ValueError("embedding vector must not be empty")
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ValueError(f"embedding vector has dimension {len(vector)}, expected {expected_dimension}")
    for value in vector:
        if not math.isfinite(value):
            raise ValueError(f"embedding vector contains a non-finite value: {value}")
    return vector


@dataclass(frozen=True)
class EmbeddedVector:
    """One embedding result plus ITS OWN served-model provenance.

    ``served_model_name`` is ``None`` whenever the provider cannot
    legitimately report which exact model served the request (e.g. a
    provider whose API response carries no such field, or a test fake with
    nothing to report) -- it must never be fabricated by copying the
    requested ``model_name`` and presenting that as an independently
    confirmed fact (see ``OpenAIEmbeddingProvider`` and
    ``FakeEmbeddingProvider``, both of which default to ``None`` here).
    """

    vector: Vector
    served_model_name: Optional[str] = None


@runtime_checkable
class EmbeddingProvider(Protocol):
    """The minimal seam every concrete embedding provider (OpenAI, a test
    fake) implements.

    ``model_name`` is the REQUESTED model identity -- what the caller
    configured the provider with. ``embedding_dimension`` is declared up
    front by the provider (never inferred lazily from the first vector it
    happens to return), so callers -- ``build_corpus_embedding_index`` in
    particular -- can validate dimension consistency before doing anything
    expensive. Served-model identity is per-call (see ``EmbeddedVector``),
    not a fixed provider attribute, since some providers/APIs may report it
    inconsistently or not at all.
    """

    provider: ModelProvider
    model_name: str
    embedding_dimension: int

    def embed_documents(self, texts: list[str]) -> list[EmbeddedVector]: ...

    def embed_query(self, text: str) -> EmbeddedVector: ...


class FakeEmbeddingProvider:
    """Test double for ``EmbeddingProvider``: deterministic, no network,
    exact caller control over which text maps to which vector -- so a test
    can engineer precise cosine-similarity relationships and assert an
    exact expected ranking, rather than merely "some deterministic result".

    ``vectors_by_text`` maps EXACT text strings (as produced by
    ``app.retrieval.serialization``'s serialization functions) to a fixed
    vector. Embedding a text not in the map is a test-authoring error, not
    a legitimate "unknown text" case a real provider would need to handle
    gracefully -- it raises ``KeyError`` immediately rather than
    fabricating a value.

    ``served_model_name`` defaults to ``None`` -- a fake has nothing
    legitimate to report either, and must not pretend otherwise -- but a
    test may pass an explicit value to exercise the requested-vs-served
    distinction deliberately.

    ``embed_call_count`` and ``embedded_texts`` record exactly how many
    times, and with what texts, this provider was called -- the hooks
    tests use to prove document embeddings are computed once and reused,
    never recomputed per case (see ``build_corpus_embedding_index`` /
    ``VectorRetriever``).
    """

    provider = ModelProvider.OPENAI  # a stand-in identity, same convention as FakeLLMProvider

    def __init__(
        self,
        vectors_by_text: dict[str, Sequence[float]],
        model_name: str = "fake-embedding-v1",
        served_model_name: Optional[str] = None,
    ):
        if not vectors_by_text:
            raise ValueError("FakeEmbeddingProvider requires at least one text->vector mapping")

        self.model_name = model_name
        self.served_model_name = served_model_name
        self._vectors: dict[str, Vector] = {}
        dimensions = set()
        for text, values in vectors_by_text.items():
            vector = validate_vector(values)
            self._vectors[text] = vector
            dimensions.add(len(vector))
        if len(dimensions) != 1:
            raise ValueError(f"all configured vectors must share one dimension, got {sorted(dimensions)}")
        self.embedding_dimension = dimensions.pop()

        self.embed_call_count = 0
        self.embedded_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[EmbeddedVector]:
        self.embed_call_count += 1
        self.embedded_texts.extend(texts)
        return [EmbeddedVector(vector=self._lookup(text), served_model_name=self.served_model_name) for text in texts]

    def embed_query(self, text: str) -> EmbeddedVector:
        self.embed_call_count += 1
        self.embedded_texts.append(text)
        return EmbeddedVector(vector=self._lookup(text), served_model_name=self.served_model_name)

    def _lookup(self, text: str) -> Vector:
        if text not in self._vectors:
            raise KeyError(f"FakeEmbeddingProvider has no configured vector for text: {text!r}")
        return self._vectors[text]
